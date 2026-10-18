"""Corpus statistics for exported TraceSmith datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def corpus_stats(in_dir: Path) -> dict:
    sources: Counter[str] = Counter()
    models: Counter[str] = Counter()
    token_usage: Counter[str] = Counter()
    model_metrics: dict[str, Counter[str]] = {}
    turn_counts: list[int] = []
    seen_traces: set[str] = set()
    total_rows = 0
    assistant_chars = 0
    assistant_messages = 0
    tool_calls = 0
    tool_results = 0
    cached_input_anomalies = 0
    traces_with_tools = 0
    traces_with_diffs = 0

    for path in sorted(in_dir.glob("*.jsonl")):
        with path.open() as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                total_rows += 1
                row = json.loads(line)
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    source = str(row.get("source") or path.stem)
                    sources[source] += 1
                    messages = row.get("messages") or []
                    turn_counts.append(len(messages))
                    assistant_rows = [
                        message
                        for message in messages
                        if isinstance(message, dict)
                        and message.get("role") == "assistant"
                    ]
                    assistant_chars += sum(
                        len(str(message.get("content") or ""))
                        for message in assistant_rows
                    )
                    assistant_messages += len(assistant_rows)
                    continue
                trace_id = str(metadata.get("trace_id") or f"legacy:{path}:{row_index}")
                if trace_id in seen_traces:
                    continue
                seen_traces.add(trace_id)
                source = str(metadata.get("source") or "unknown")
                sources[source] += 1
                counts = metadata.get("counts") or {}
                message_count = int(counts.get("messages") or 0)
                turn_counts.append(message_count)
                assistant_chars += int(counts.get("assistant_chars") or 0)
                assistant_messages += int(counts.get("roles", {}).get("assistant") or 0)
                tool_calls += int(counts.get("tool_calls") or 0)
                tool_results += int(counts.get("tool_results") or 0)
                traces_with_tools += int(int(counts.get("tool_calls") or 0) > 0)
                traces_with_diffs += int(int(counts.get("diffs") or 0) > 0)
                trace_models = list(metadata.get("models") or ["unknown"])
                models.update(trace_models)
                for model_name in trace_models:
                    metrics = model_metrics.setdefault(model_name, Counter())
                    metrics["conversations"] += 1
                    metrics["assistant_chars"] += int(
                        counts.get("assistant_chars") or 0
                    )
                    metrics["assistant_messages"] += int(
                        counts.get("roles", {}).get("assistant") or 0
                    )
                    metrics["tool_calls"] += int(counts.get("tool_calls") or 0)
                    metrics["zero_tool_conversations"] += int(
                        int(counts.get("tool_calls") or 0) == 0
                    )
                usage = metadata.get("token_usage") or {}
                token_usage.update(usage)
                cached = usage.get("cached_input")
                prompt = usage.get("input")
                if (
                    isinstance(cached, (int, float))
                    and not isinstance(cached, bool)
                    and isinstance(prompt, (int, float))
                    and not isinstance(prompt, bool)
                    and cached > prompt
                ):
                    cached_input_anomalies += 1

    return {
        "conversations_by_source": dict(sources),
        "conversations_by_model": dict(models),
        "model_metrics": {
            model_name: {
                "conversations": metrics["conversations"],
                "assistant_chars": metrics["assistant_chars"],
                "assistant_messages": metrics["assistant_messages"],
                "tool_calls": metrics["tool_calls"],
                "tool_calls_per_assistant_message": (
                    metrics["tool_calls"] / metrics["assistant_messages"]
                    if metrics["assistant_messages"]
                    else 0
                ),
                "zero_tool_conversations": metrics["zero_tool_conversations"],
            }
            for model_name, metrics in model_metrics.items()
        },
        "total_conversations": sum(sources.values()),
        "total_rows": total_rows,
        "turn_count_min": min(turn_counts) if turn_counts else 0,
        "turn_count_max": max(turn_counts) if turn_counts else 0,
        "turn_count_mean": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
        "total_assistant_chars": assistant_chars,
        "total_assistant_messages": assistant_messages,
        "total_tool_calls": tool_calls,
        "total_tool_results": tool_results,
        "tool_calls_per_assistant_message": (
            tool_calls / assistant_messages if assistant_messages else 0
        ),
        "zero_tool_conversations": sum(sources.values()) - traces_with_tools,
        "traces_with_tools": traces_with_tools,
        "traces_with_diffs": traces_with_diffs,
        "token_usage": dict(token_usage),
        "reported_cache_hit_ratio": (
            token_usage["cached_input"] / token_usage["input"]
            if token_usage.get("cached_input") is not None and token_usage.get("input")
            else None
        ),
        "token_usage_anomalies": {
            "cached_input_exceeds_input": cached_input_anomalies,
        },
    }
