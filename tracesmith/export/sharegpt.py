"""DistillKit ShareGPT variant exporter (instruction pairs)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from tracesmith.config import ExportConfig
from tracesmith.export.flatten import flatten_message
from tracesmith.export.filters import dedup_key, filter_reason
from tracesmith.export.metadata import build_metadata


def _iter_conversations(in_dir: Path) -> Iterator[dict[str, Any]]:
    for src in sorted(in_dir.glob("*.jsonl")):
        with src.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _pairs(msgs: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield (user_msg, assistant_msg) pairs. Independent (no rolling context).

    A trailing user without a following assistant is silently dropped (no yield).
    """
    pending_user: dict[str, Any] | None = None
    for m in msgs:
        role = m.get("role", "user")
        if role == "user":
            pending_user = m
        elif role == "assistant":
            if pending_user is not None:
                yield pending_user, m
                pending_user = None
    # Trailing user without assistant is dropped (no yield).


def export_sharegpt(in_dir: Path, out_file: Path, config: ExportConfig) -> dict:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pairs_written = 0
    dropped_no_assistant = 0
    dropped_trailing_user = 0
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
            system_msgs = [m for m in msgs if m.get("role") == "system"]
            non_system = [m for m in msgs if m.get("role") != "system"]
            pairs = list(_pairs(non_system))
            # A trailing user without a following assistant is an orphan: it is
            # silently dropped by `_pairs`. Detect independently so it is counted
            # even when earlier user->assistant pairs were formed.
            has_trailing_user = bool(non_system) and non_system[-1].get("role", "user") == "user"
            if has_trailing_user:
                dropped_trailing_user += 1
            if not pairs:
                if not has_trailing_user:
                    dropped_no_assistant += 1
                continue
            base_metadata = build_metadata(conv, metadata_key=config.metadata_key)
            for idx, (u, a) in enumerate(pairs):
                convs: list[dict[str, str]] = []
                if idx == 0:
                    for sm in system_msgs:
                        convs.append({"from": "system", "value": flatten_message(sm)})
                convs.append({"from": "human", "value": flatten_message(u)})
                convs.append({"from": "gpt", "value": flatten_message(a)})
                metadata = dict(base_metadata)
                metadata["pair"] = {"index": idx, "count": len(pairs)}
                row = {"conversations": convs, "metadata": metadata}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                pairs_written += 1
                by_source[base_metadata["source"]["family"]] += 1

    return {
        "pairs": pairs_written,
        "dropped_no_assistant": dropped_no_assistant,
        "dropped_trailing_user": dropped_trailing_user,
        "dropped_filter": dropped_filter,
        "dropped_by_reason": dict(dropped_by_reason),
        "dropped_dedup": dropped_dedup,
        "by_source": dict(by_source),
    }
