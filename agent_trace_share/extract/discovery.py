"""Extractor registry and orchestration."""
from __future__ import annotations

import json
from pathlib import Path

from agent_trace_share.extract.base import Conversation, Extractor
from agent_trace_share.extract.claude_code import ClaudeCodeExtractor
from agent_trace_share.extract.codex import CodexExtractor
from agent_trace_share.extract.continue_ import ContinueExtractor
from agent_trace_share.extract.cursor import CursorExtractor
from agent_trace_share.extract.gemini import GeminiExtractor
from agent_trace_share.extract.opencode import OpenCodeExtractor
from agent_trace_share.extract.trae import TraeExtractor
from agent_trace_share.extract.windsurf import WindsurfExtractor


EXTRACTORS: dict[str, type[Extractor]] = {
    "claude_code": ClaudeCodeExtractor,
    "codex": CodexExtractor,
    "continue": ContinueExtractor,
    "cursor": CursorExtractor,
    "gemini": GeminiExtractor,
    "opencode": OpenCodeExtractor,
    "trae": TraeExtractor,
    "windsurf": WindsurfExtractor,
}


def available_sources() -> list[str]:
    return sorted(EXTRACTORS.keys())


def _write_jsonl(path: Path, records: list[Conversation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_extract(
    sources: list[str] | None,
    root: Path,
    out_dir: Path,
) -> dict[str, int]:
    """Run the requested extractors and write one JSONL per source.

    Returns {source: record_count}. Sources with no installations or no
    conversations contribute 0 and write an empty file (so downstream
    stages always find the file).
    """
    selected = sources or available_sources()
    raw_dir = out_dir / "raw_extracted"
    counts: dict[str, int] = {}

    for name in selected:
        if name not in EXTRACTORS:
            raise ValueError(f"unknown source: {name!r}. available: {available_sources()}")
        extractor = EXTRACTORS[name]()
        records: list[Conversation] = []
        for install in extractor.find_installations():
            records.extend(extractor.extract(install))
        _write_jsonl(raw_dir / f"{name}.jsonl", records)
        counts[name] = len(records)

    return counts
