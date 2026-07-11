# tracesmith/stats.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tracesmith.export.metadata import build_metadata


def corpus_stats(in_dir: Path) -> dict:
    sources = Counter()
    models = Counter()
    statuses = Counter()
    token_usage = Counter()
    turn_counts = []
    asst_chars = 0
    tool_calls = 0
    traces_with_tools = 0
    traces_with_diffs = 0
    rows_by_variant = Counter()
    seen_traces: set[str] = set()
    for src in sorted(in_dir.glob("*.jsonl")):
        with src.open() as f:
            for row_index, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                conv = json.loads(line)
                rows_by_variant[src.stem] += 1
                metadata = conv.get("metadata")
                if not isinstance(metadata, dict):
                    metadata_input = conv
                    legacy_sharegpt = conv.get("conversations")
                    if isinstance(legacy_sharegpt, list):
                        role_map = {"human": "user", "gpt": "assistant"}
                        metadata_input = {
                            "source": src.stem,
                            "messages": [
                                {
                                    "role": role_map.get(item.get("from"), item.get("from")),
                                    "content": item.get("value", ""),
                                }
                                for item in legacy_sharegpt
                                if isinstance(item, dict)
                            ],
                        }
                    metadata = build_metadata(metadata_input)
                trace_id = str(metadata.get("trace_id", ""))
                if not isinstance(conv.get("metadata"), dict) and conv.get("conversations"):
                    trace_id = f"legacy:{src.name}:{row_index}"
                if trace_id and trace_id in seen_traces:
                    continue
                if trace_id:
                    seen_traces.add(trace_id)
                source = metadata.get("source", {}).get("family", src.stem)
                sources[source] += 1
                counts = metadata.get("counts", {})
                turn_counts.append(int(counts.get("messages", 0)))
                asst_chars += int(counts.get("assistant_chars", 0))
                tool_calls += int(counts.get("tool_calls", 0))
                models.update(metadata.get("models", []))
                token_usage.update(metadata.get("token_usage", {}))
                status = metadata.get("session", {}).get("status")
                if status:
                    statuses[str(status)] += 1
                flags = metadata.get("flags", {})
                traces_with_tools += int(bool(flags.get("has_tools")))
                traces_with_diffs += int(bool(flags.get("has_diffs")))
    return {
        "conversations_by_source": dict(sources),
        "total_conversations": sum(sources.values()),
        "turn_count_min": min(turn_counts) if turn_counts else 0,
        "turn_count_max": max(turn_counts) if turn_counts else 0,
        "turn_count_mean": (sum(turn_counts) / len(turn_counts)) if turn_counts else 0,
        "total_assistant_chars": asst_chars,
        "conversations_by_model": dict(models),
        "conversations_by_status": dict(statuses),
        "token_usage": dict(token_usage),
        "total_tool_calls": tool_calls,
        "traces_with_tools": traces_with_tools,
        "traces_with_diffs": traces_with_diffs,
        "rows_by_variant": dict(rows_by_variant),
        "total_rows": sum(rows_by_variant.values()),
    }
