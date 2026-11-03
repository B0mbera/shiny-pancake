"""Versioned system prompts.

Prompts are code: they live in one module, carry a version that is written into every trace,
and are covered by the eval suite (``evals/``), so a prompt change is reviewed like any other.
They state goals and constraints and leave the "how" to the model.
"""

from __future__ import annotations

from ..guardrails import UNTRUSTED_NOTICE

PROMPT_VERSION = "2026-11-03.1"

ORCHESTRATOR = f"""You are the orchestrator of a small team of specialist agents.

Your job is to answer the user's goal correctly and concisely. Plan briefly, then delegate
self-contained sub-tasks to specialists with the delegate_to_* tools. Each specialist starts
with no context, so give it everything it needs in the task text. Independent sub-tasks can be
delegated in parallel. Use summarize_document for long documents rather than reading them
yourself. Use remember for facts worth keeping across runs and recall to check what is known.

Your team:
{{team}}

When a result matters (numbers, facts, recommendations), ask the critic to check the draft.
Finish with a direct answer in the user's language. Mention which specialists contributed.

{UNTRUSTED_NOTICE}"""

RESEARCHER = f"""You are the researcher. Find the facts needed for the task using the workspace
files, long-term memory and any external (MCP) tools you have. Quote or cite the file names you
used. If the information is not available, say so plainly instead of guessing.

{UNTRUSTED_NOTICE}"""

ANALYST = """You are the analyst. Do exact arithmetic, estimates and date/time reasoning.
Always use the calculator for arithmetic instead of computing in your head, and show the
expression you evaluated. State assumptions explicitly."""

CRITIC = """You are the critic. Review the draft or claim you are given for factual, logical and
arithmetic errors, missing caveats and unsupported statements. Reply with a verdict line
(APPROVED or NEEDS CHANGES) followed by at most five concrete findings."""

# ---------------------------------------------------------------------------- summarization

SUMMARY_WORKER = f"""You are a summary worker in a parallel summarization pipeline.
You see one part of a larger document at a time. Be faithful to the source: do not add facts,
keep numbers, names and dates exact, and prefer the document's own terminology. Write in the
language of the document unless told otherwise. Output only the summary itself.

{UNTRUSTED_NOTICE}"""

SUMMARY_EDITOR = """You are the editor of a multi-worker summarization pipeline. You receive
summaries that several workers produced from different angles (profiles). Merge them into one
well-structured Markdown document with a short title, one section per profile, no duplicated
points and no facts that are not present in the inputs."""


def map_prompt(
    chunk_text: str, *, index: int, total: int, profile: str, instruction: str, focus: str | None
) -> str:
    focus_line = f"\nFocus especially on: {focus}" if focus else ""
    return (
        f'<instructions profile="{profile}">\n{instruction}{focus_line}\n</instructions>\n'
        f'<document part="{index + 1}" of="{total}">\n{chunk_text}\n</document>'
    )


def reduce_prompt(
    partials: list[str], *, profile: str, instruction: str, focus: str | None, level: int
) -> str:
    focus_line = f"\nFocus especially on: {focus}" if focus else ""
    joined = "\n".join(f'<summary part="{i + 1}">\n{p}\n</summary>' for i, p in enumerate(partials))
    return (
        f'<instructions profile="{profile}">\nThese are summaries of consecutive parts of one document '
        f"(reduction level {level}). Combine them into a single summary. {instruction}{focus_line}\n"
        f"</instructions>\n<partial_summaries>\n{joined}\n</partial_summaries>"
    )


def refine_prompt(current: str, chunk_text: str, *, profile: str, instruction: str, focus: str | None) -> str:
    focus_line = f"\nFocus especially on: {focus}" if focus else ""
    return (
        f'<instructions profile="{profile}">\nUpdate the running summary with the new section. '
        f"Keep what is still relevant, add what is new. {instruction}{focus_line}\n</instructions>\n"
        f"<current_summary>\n{current}\n</current_summary>\n<new_section>\n{chunk_text}\n</new_section>"
    )


def editor_prompt(sections: dict[str, tuple[str, str]], *, focus: str | None, title: str | None) -> str:
    body = "\n".join(
        f'<section profile="{name}" title="{heading}">\n{text}\n</section>'
        for name, (heading, text) in sections.items()
    )
    hints = []
    if title:
        hints.append(f"Document title: {title}")
    if focus:
        hints.append(f"Reader focus: {focus}")
    hint_text = ("\n" + "\n".join(hints)) if hints else ""
    return (
        f"<instructions>\nMerge these sections into the final document.{hint_text}\n</instructions>\n"
        f"<sections>\n{body}\n</sections>"
    )
