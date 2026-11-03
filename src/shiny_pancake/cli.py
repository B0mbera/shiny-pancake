"""Command line interface: ``shiny-pancake <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .art import Style, mascot, print_banner, use_color
from .config import Settings


def _settings(args: argparse.Namespace, **extra: Any) -> Settings:
    provider = "offline" if getattr(args, "offline", False) else getattr(args, "provider", None)
    return Settings.from_env(
        provider=provider,
        model=getattr(args, "model", None),
        worker_model=getattr(args, "worker_model", None),
        **extra,
    )


def _footer(result: Any, style: Style) -> str:
    t = result.trace
    tokens = t.get("tokens", {})
    agents = ", ".join(t.get("agents", {}))
    return style.dim(
        f"run {result.run_id} · {result.status} · {t.get('llm_calls', 0)} LLM calls · "
        f"{t.get('tool_calls', 0)} tool calls · {tokens.get('total', 0)} tokens · "
        f"~${t.get('cost_usd', 0):.4f} · {t.get('duration_s', 0):.2f}s · {agents}"
    )


def _print_trace(hub: Any, run_id: str) -> None:
    from .tracing import TraceEvent, Tracer

    events = hub.store.events(run_id)
    tracer = Tracer(run_id)
    tracer.events = [
        TraceEvent(e["run_id"], e["seq"], e["ts"], e["agent"], e["kind"], e["data"]) for e in events
    ]
    if tracer.events:
        tracer.started = tracer.events[0].ts
    print("\n" + tracer.render())


# ---------------------------------------------------------------------------- commands


async def cmd_run(args: argparse.Namespace) -> int:
    from .hub.service import Hub

    style = Style(use_color())
    settings = _settings(args, approval=args.approval)
    if not args.json:
        print_banner(f"agent team · {settings.provider}:{settings.model}", mood="working")
    async with Hub(settings, mcp_targets=args.mcp or ()) as hub:
        result = await hub.ask(args.goal)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0 if result.ok else 1
        print(style.bold("answer") + "\n" + result.output + "\n")
        print(_footer(result, style))
        if args.show_trace:
            _print_trace(hub, result.run_id)
    return 0 if result.ok else 1


async def cmd_summarize(args: argparse.Namespace) -> int:
    from .hub.service import Hub

    style = Style(use_color())
    extra: dict[str, Any] = {}
    if args.workers:
        extra["summary_workers"] = args.workers
    if args.chunk_chars:
        extra["chunk_chars"] = args.chunk_chars
    settings = _settings(args, **extra)
    if args.text is not None:
        text, title = args.text, None
    elif args.file in (None, "-"):
        text, title = sys.stdin.read(), None
    else:
        path = Path(args.file)
        text, title = path.read_text(encoding="utf-8"), path.name
    profiles = [p.strip() for p in args.profiles.split(",")] if args.profiles else None
    if not args.json:
        print_banner(
            f"summary workers ×{settings.summary_workers} · {settings.provider}:{settings.worker_model}",
            mood="working",
        )
    async with Hub(settings) as hub:
        result = await hub.summarize(
            text, strategy=args.strategy, profiles=profiles, focus=args.focus, title=title
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        data = result.data
        print(result.output + "\n")
        workers = ", ".join(f"{w}: {s['calls']} calls" for w, s in data.get("workers", {}).items())
        print(
            style.dim(
                f"strategy={data['strategy']} · chunks={data['chunks']} · levels={data['levels']} · "
                f"profiles={','.join(data['profiles'])} · {workers}"
            )
        )
        print(_footer(result, style))
        if args.show_trace:
            _print_trace(hub, result.run_id)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    settings = Settings.from_env(provider="offline" if args.offline else None)
    workers = settings.inline_workers if args.inline_workers is None else args.inline_workers
    print_banner(
        f"hub on http://{args.host}:{args.port} · {settings.provider}:{settings.model} · "
        f"{workers} inline job workers",
        mood="proud",
    )
    uvicorn.run(
        create_app(settings, inline_workers=workers), host=args.host, port=args.port, log_level=args.log_level
    )
    return 0


async def cmd_worker(args: argparse.Namespace) -> int:
    from .hub.service import Hub
    from .hub.worker import JobWorkerPool

    settings = Settings.from_env(provider="offline" if args.offline else None)
    print_banner(f"job worker ×{args.concurrency} · db={settings.db_path}", mood="working")
    async with Hub(settings) as hub:
        pool = JobWorkerPool(hub, concurrency=args.concurrency, poll_interval=args.poll)
        if args.once:
            processed = 0
            while (job := await pool.workers[0].run_once()) is not None:
                processed += 1
                print(f"{job.id} {job.kind} -> {job.status}")
            print(f"processed {processed} job(s)")
            return 0
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # pragma: no cover - Windows
                pass
        await pool.run_until(stop)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    from .hub.service import validate_payload
    from .hub.store import HubStore

    payload: dict[str, Any] = json.loads(args.payload) if args.payload else {}
    if args.kind == "ask" and args.input:
        payload.setdefault("goal", args.input)
    if args.kind == "summarize" and args.input:
        payload.setdefault(
            "text", Path(args.input).read_text(encoding="utf-8") if Path(args.input).is_file() else args.input
        )
    if args.kind == "eval" and args.input:
        payload.setdefault("suite", args.input)
    validate_payload(args.kind, payload)
    store = HubStore(Settings.from_env().db_path)
    job = store.create_job(args.kind, payload)
    print(job.id)
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    from .hub.store import HubStore

    store = HubStore(Settings.from_env().db_path)
    for job in store.list_jobs(limit=args.limit):
        d = job.to_dict(include_result=False)
        print(
            f"{job.id}  {job.kind:<10} {job.status:<10} {str(d['duration_s'] or '-'):>8}s  "
            f"{job.worker or '-'}"
        )
    print(store.stats())
    return 0


async def cmd_eval(args: argparse.Namespace) -> int:
    from .evals.harness import EvalHarness, load_suite
    from .hub.service import Hub

    cases = [case for suite in args.suite for case in load_suite(Path(suite))]
    settings = _settings(args)
    print_banner(f"eval harness · {len(cases)} cases · {settings.provider}:{settings.model}", mood="working")
    async with Hub(settings) as hub:
        report = await EvalHarness(hub, concurrency=args.concurrency).run(cases)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(report.render())
    ok = report.pass_rate >= args.min_pass
    print("\n" + mascot("proud" if ok else "oops"))
    return 0 if ok else 1


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    from .hub.service import Hub
    from .mcp_server import serve

    settings = Settings.from_env(provider="offline" if args.offline else None)
    if args.transport != "stdio":  # stdout is the protocol channel in stdio mode
        print_banner(f"MCP server ({args.transport}) on {args.host}:{args.port}", mood="proud")
    serve(Hub(settings), transport=args.transport, host=args.host, port=args.port)
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from .hub.service import Hub

    hub = Hub(_settings(args))
    style = Style(use_color())
    for tool in hub.info()["tools"]:
        flags = [
            f
            for f, on in (("side-effect", tool["side_effect"]), ("untrusted", tool["untrusted_output"]))
            if on
        ]
        print(
            f"{style.bold(tool['name']):<32} {tool['description'][:90]}"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .hub.service import Hub

    print(json.dumps(Hub(_settings(args)).info(), ensure_ascii=False, indent=2))
    return 0


def cmd_art(args: argparse.Namespace) -> int:
    print_banner(mood=args.mood)
    return 0


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shiny-pancake", description="Multi-agent LLM hub 🥞")
    parser.add_argument("--version", action="version", version=f"shiny-pancake {__version__}")
    sub = parser.add_subparsers(dest="command")

    def llm_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--offline", action="store_true", help="use the deterministic offline provider")
        p.add_argument("--provider", choices=["anthropic", "openai", "offline"])
        p.add_argument("--model")
        p.add_argument("--worker-model", dest="worker_model")

    p = sub.add_parser("run", help="ask the agent team")
    p.add_argument("goal")
    llm_flags(p)
    p.add_argument("--mcp", action="append", metavar="TARGET", help="MCP server command or URL (repeatable)")
    p.add_argument("--approval", choices=["auto", "deny"], help="policy for side-effecting tools")
    p.add_argument("--show-trace", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run, is_async=True)

    p = sub.add_parser("summarize", help="multi-worker summarization of a file, stdin or --text")
    p.add_argument("file", nargs="?", help="path to a text file, or '-' for stdin")
    p.add_argument("--text")
    p.add_argument(
        "--strategy", default="auto", choices=["auto", "stuff", "map_reduce", "refine", "ensemble"]
    )
    p.add_argument("--profiles", help="comma-separated: key_points,executive,action_items,risks,tldr")
    p.add_argument("--focus")
    p.add_argument("--workers", type=int)
    p.add_argument("--chunk-chars", dest="chunk_chars", type=int)
    llm_flags(p)
    p.add_argument("--show-trace", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_summarize, is_async=True)

    p = sub.add_parser("serve", help="HTTP API + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--inline-workers", dest="inline_workers", type=int)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--log-level", dest="log_level", default="info")
    p.set_defaults(func=cmd_serve, is_async=False)

    p = sub.add_parser("worker", help="run job workers against the shared hub database")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--once", action="store_true", help="drain the queue and exit")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_worker, is_async=True)

    p = sub.add_parser("submit", help="enqueue a job for the workers")
    p.add_argument("kind", choices=["ask", "summarize", "eval"])
    p.add_argument("input", nargs="?", help="goal (ask), file or text (summarize), suite path (eval)")
    p.add_argument("--payload", help="raw JSON payload")
    p.set_defaults(func=cmd_submit, is_async=False)

    p = sub.add_parser("jobs", help="list recent jobs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_jobs, is_async=False)

    p = sub.add_parser("eval", help="run eval suites (JSONL) through the hub")
    p.add_argument("suite", nargs="+")
    llm_flags(p)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--min-pass", dest="min_pass", type=float, default=1.0)
    p.add_argument("--json", help="write the JSON report to this path")
    p.set_defaults(func=cmd_eval, is_async=True)

    p = sub.add_parser("mcp-serve", help="expose the hub as an MCP server")
    p.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_mcp_serve, is_async=False)

    p = sub.add_parser("tools", help="list the tools agents can use")
    llm_flags(p)
    p.set_defaults(func=cmd_tools, is_async=False)

    p = sub.add_parser("info", help="print hub configuration as JSON")
    llm_flags(p)
    p.set_defaults(func=cmd_info, is_async=False)

    p = sub.add_parser("art", help="say hi to Pancake-chan")
    p.add_argument("--mood", default="happy", choices=["happy", "working", "proud", "oops"])
    p.set_defaults(func=cmd_art, is_async=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        print_banner()
        parser.print_help()
        return 0
    try:
        if args.is_async:
            return asyncio.run(args.func(args))
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
