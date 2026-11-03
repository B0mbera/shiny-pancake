import asyncio
import json
import time

import pytest

from shiny_pancake.hub import Hub, HubStore, JobWorker, JobWorkerPool, validate_payload
from shiny_pancake.tools import Tool, function_tool
from shiny_pancake.tracing import Tracer


async def test_ask_is_recorded_with_events(hub: Hub):
    result = await hub.ask("What is 6 * 7? Check it.")
    assert result.ok and "42" in result.output
    assert result.trace["llm_calls"] >= 4 and set(result.trace["agents"]) >= {"orchestrator", "analyst"}
    job = hub.store.get_job(result.run_id)
    assert job.status == "succeeded" and job.worker == "inline" and job.result["output"] == result.output
    kinds = [e["kind"] for e in hub.store.events(result.run_id)]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished" and "tool_call" in kinds
    jsonl = hub.settings.trace_dir / f"{result.run_id}.jsonl"
    assert len(jsonl.read_text().splitlines()) == len(kinds)


async def test_summarize_from_workspace_path_is_sandboxed(hub: Hub):
    result = await hub.summarize(path="notes/kickoff.md", profiles=["action_items"])
    assert "Bence" in result.output and result.data["strategy"] == "stuff"
    with pytest.raises(ValueError):
        await hub.summarize(path="../../etc/passwd")
    assert hub.store.list_jobs(status="failed")[0].kind == "summarize"


async def test_submit_and_worker_process_jobs(hub: Hub):
    first = hub.submit("ask", {"goal": "What is 2 + 2?"})
    second = hub.submit(
        "summarize",
        {"text": "Anna will send the deck by Monday. It is a risk.", "profiles": ["action_items"]},
    )
    assert hub.store.stats()["queued"] == 2
    worker = JobWorker(hub, worker_id="test-w1")
    done1, done2 = await worker.run_once(), await worker.run_once()
    assert await worker.run_once() is None
    assert (done1.id, done2.id) == (first.id, second.id)  # FIFO
    assert done1.status == done2.status == "succeeded" and done1.worker == "test-w1"
    assert "4" in done1.result["output"]
    events = hub.store.events(first.id)
    assert events[0]["kind"] == "job_claimed" and events[0]["data"]["worker"] == "test-w1"


async def test_failed_job_does_not_kill_worker(hub: Hub):
    job = hub.store.create_job("summarize", {"path": "missing.md"})  # bypasses validation on purpose
    done = await JobWorker(hub).run_once()
    assert done.id == job.id and done.status == "failed" and "not a file" in done.error


async def test_worker_pool_drains_queue(hub: Hub):
    ids = [hub.submit("ask", {"goal": f"What is {i} * 3?"}).id for i in range(4)]
    pool = JobWorkerPool(hub, concurrency=2, poll_interval=0.01, name="pool")
    await pool.start()
    deadline = time.time() + 10
    while time.time() < deadline and hub.store.stats()["succeeded"] < 4:
        await asyncio.sleep(0.02)
    await pool.stop()
    jobs = [hub.store.get_job(i) for i in ids]
    assert all(j.status == "succeeded" for j in jobs)
    assert {j.worker for j in jobs} <= {"pool-w1", "pool-w2"}


def test_store_claims_are_exclusive_and_stale_jobs_requeue(tmp_path):
    db = tmp_path / "hub.db"
    a, b = HubStore(db), HubStore(db)  # two connections, like two worker processes
    job = a.create_job("ask", {"goal": "x"})
    claimed = [s.claim_next("w") for s in (a, b)]
    assert [c.id if c else None for c in claimed].count(job.id) == 1
    assert a.requeue_stale(older_than_s=-1) == 1  # pretend the worker crashed
    assert b.claim_next("w2").id == job.id and b.get_job(job.id).attempts == 2


def test_validate_payload():
    validate_payload("ask", {"goal": "hi"})
    for kind, payload in [
        ("ask", {}),
        ("summarize", {}),
        ("summarize", {"text": "x", "strategy": "nope"}),
        ("summarize", {"text": "x", "profiles": ["poetry"]}),
        ("eval", {}),
        ("dance", {}),
    ]:
        with pytest.raises(ValueError):
            validate_payload(kind, payload)


async def test_hub_as_a_harness_with_custom_tools_and_specialists(settings):
    from shiny_pancake.agents import SPECIALISTS, SpecialistSpec

    @function_tool
    def pancake_price(count: int) -> str:
        """Price of a stack of pancakes in euros."""
        return str(count * 2.5)

    chef = SpecialistSpec(
        "chef", "Knows pancake prices.", "You are the chef. Use pancake_price.", tools=("pancake_price",)
    )
    async with Hub(settings, extra_tools=[pancake_price], specialists={**SPECIALISTS, "chef": chef}) as hub:
        info = hub.info()
        assert "pancake_price" in [t["name"] for t in info["tools"]] and "chef" in info["agents"]
        orch = hub.orchestrator(Tracer())
        assert orch.team["chef"].tools.names() == ["pancake_price"]
        assert "delegate_to_chef" in orch.agent.tools.names()
        assert isinstance(pancake_price, Tool)


async def test_eval_job_kind(hub: Hub):
    cases = [
        {
            "id": "m",
            "kind": "ask",
            "input": "What is 2 * 21?",
            "checks": [{"type": "contains", "value": "42"}],
        }
    ]
    result = await hub.run("eval", {"cases": cases})
    assert result.status == "completed" and result.data["passed"] == 1
    assert json.loads(json.dumps(result.to_dict()))["data"]["pass_rate"] == 1.0
