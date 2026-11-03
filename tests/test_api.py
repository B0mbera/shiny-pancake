import json
import time

import pytest
from fastapi.testclient import TestClient

from shiny_pancake.api import create_app


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, inline_workers=2)) as c:
        yield c


def wait_for(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_ui_health_and_info(client):
    page = client.get("/")
    assert page.status_code == 200 and "Pancake Hub" in page.text
    assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg+xml")
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["provider"] == "offline" and health["inline_workers"] == 2
    info = client.get("/api/info").json()
    assert "summarize_document" in [t["name"] for t in info["tools"]]
    assert info["settings"]["anthropic_api_key"] is None


def test_sync_run_and_summarize(client):
    run = client.post("/run", json={"goal": "What is 12 * (3 + 4)?"}).json()
    assert "84" in run["answer"] and run["summary"]["llm_calls"] > 0
    trace = client.get(f"/runs/{run['run_id']}/trace").json()
    assert trace[0]["kind"] == "run_started"
    summary = client.post(
        "/summarize", json={"path": "notes/kickoff.md", "profiles": ["action_items"]}
    ).json()
    assert "Bence" in summary["output"] and summary["data"]["strategy"] == "stuff"
    assert client.post("/summarize", json={"path": "../secret"}).status_code == 422
    assert client.post("/run", json={"goal": ""}).status_code == 422


def test_job_queue_events_and_sse_stream(client):
    created = client.post(
        "/jobs", json={"kind": "summarize", "payload": {"path": "notes/kickoff.md", "strategy": "ensemble"}}
    )
    assert created.status_code == 202
    job_id = created.json()["job"]["id"]

    trace_events, done = [], None
    with client.stream("GET", f"/jobs/{job_id}/stream") as stream:
        event = None
        for line in stream.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if event == "done":
                    done = payload
                    break
                trace_events.append(payload)
    assert done is not None and done["status"] == "succeeded"
    assert {"job_claimed", "summary_started", "llm_call", "summary_finished"} <= {
        e["kind"] for e in trace_events
    }
    assert done["worker"].startswith("inline-w")

    job = wait_for(client, job_id)
    assert "## Action items" in job["result"]["output"]
    assert len(client.get(f"/jobs/{job_id}/events").json()) == len(trace_events)
    listing = client.get("/jobs").json()
    assert listing["stats"]["succeeded"] >= 1 and "result" not in listing["jobs"][0]


def test_job_validation_and_404(client):
    assert client.post("/jobs", json={"kind": "ask", "payload": {}}).status_code == 422
    assert client.post("/jobs", json={"kind": "cook", "payload": {}}).status_code == 422
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/stream").status_code == 404


def test_api_key_auth(settings):
    secured = settings.with_overrides(api_key="s3cret")
    with TestClient(create_app(secured, inline_workers=0)) as c:
        assert c.get("/health").status_code == 200  # health stays open for probes
        assert c.get("/").status_code == 200
        assert c.get("/api/info").status_code == 401
        assert c.get("/api/info", headers={"x-api-key": "wrong"}).status_code == 401
        assert c.get("/api/info", headers={"x-api-key": "s3cret"}).status_code == 200
        assert c.get("/jobs", headers={"authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/jobs?key=s3cret").status_code == 200  # EventSource fallback
