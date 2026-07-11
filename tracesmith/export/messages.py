"""DistillKit `messages` variant exporter."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tracesmith.config import ExportConfig
from tracesmith.export.flatten import flatten_message
from tracesmith.export.filters import dedup_key, filter_reason
from tracesmith.export.metadata import build_metadata


def _iter_conversations(in_dir: Path):
    for src in sorted(in_dir.glob("*.jsonl")):
        with src.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def export_messages(in_dir: Path, out_file: Path, config: ExportConfig) -> dict:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    dropped_no_assistant = 0
    dropped_filter = 0
    dropped_by_reason: Counter[str] = Counter()
    dropped_dedup = 0
    seen: set[str] = set()
    by_source: Counter[str] = Counter()

    with out_file.open("w") as f:
        for conv in _iter_conversations(in_dir):
            reason = filter_reason(conv, config)
            if reason is not None:
                dropped_filter += 1
                dropped_by_reason[reason] += 1
                continue
            if config.dedup:
                k = dedup_key(conv)
                if k in seen:
                    dropped_dedup += 1
                    continue
                seen.add(k)
            msgs = conv.get("messages", [])
            if not any(m.get("role") == "assistant" for m in msgs):
                dropped_no_assistant += 1
                continue
            flattened = [
                {"role": m.get("role", "user"), "content": flatten_message(m)}
                for m in msgs
            ]
            metadata = build_metadata(conv, metadata_key=config.metadata_key)
            row = {"messages": flattened, "metadata": metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            by_source[metadata["source"]["family"]] += 1

    return {
        "rows": rows,
        "dropped_no_assistant": dropped_no_assistant,
        "dropped_filter": dropped_filter,
        "dropped_by_reason": dict(dropped_by_reason),
        "dropped_dedup": dropped_dedup,
        "by_source": dict(by_source),
    }
