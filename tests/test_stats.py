from __future__ import annotations

from pathlib import Path

from tests.conftest import make_conversation, make_message, write_jsonl
from tracesmith.stats import corpus_stats


def test_stats_happy_path(tmp_path: Path):
    in_dir = tmp_path / "export"
    write_jsonl(
        in_dir / "claude_code.jsonl",
        [
            make_conversation(
                messages=[
                    make_message("user", "hi"),
                    make_message("assistant", "hello there"),
                ],
                source="claude_code",
            ),
            make_conversation(
                messages=[
                    make_message("user", "again"),
                    make_message("assistant", "yes"),
                    make_message("user", "ok"),
                    make_message("assistant", "done"),
                ],
                source="claude_code",
            ),
        ],
    )
    write_jsonl(
        in_dir / "codex.jsonl",
        [
            make_conversation(
                messages=[make_message("user", "x"), make_message("assistant", "y")],
                source="codex",
            ),
        ],
    )

    report = corpus_stats(in_dir)

    assert report["conversations_by_source"] == {"claude_code": 2, "codex": 1}
    assert report["total_conversations"] == 3
    assert report["turn_count_min"] == 2
    assert report["turn_count_max"] == 4
    assert report["turn_count_mean"] == (2 + 4 + 2) / 3
    assert report["total_assistant_chars"] == len("hello there") + len("yes") + len(
        "done"
    ) + len("y")
    assert report["total_assistant_messages"] == 4
    assert report["tool_calls_per_assistant_message"] == 0
    assert report["zero_tool_conversations"] == 3


def test_stats_empty_dir(tmp_path: Path):
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    report = corpus_stats(in_dir)
    assert report["total_conversations"] == 0
    assert report["turn_count_min"] == 0
    assert report["turn_count_mean"] == 0
    assert report["total_assistant_chars"] == 0
    assert report["total_tool_calls"] == 0
    assert report["token_usage_anomalies"] == {
        "cached_input_exceeds_input": 0,
    }


def test_stats_deduplicates_the_same_trace_across_export_variants(tmp_path: Path):
    in_dir = tmp_path / "export"
    metadata = {
        "trace_id": "ts_same",
        "source": "codex",
        "models": ["model-a"],
        "counts": {
            "messages": 2,
            "roles": {"assistant": 1},
            "assistant_chars": 4,
            "tool_calls": 1,
            "tool_results": 1,
            "diffs": 1,
        },
        "token_usage": {"input": 10, "output": 5},
    }
    write_jsonl(in_dir / "messages.jsonl", [{"messages": [], "metadata": metadata}])
    write_jsonl(
        in_dir / "sharegpt.jsonl",
        [
            {
                "conversations": [],
                "metadata": {**metadata, "pair": {"index": 0, "count": 1}},
            },
        ],
    )

    report = corpus_stats(in_dir)

    assert report["total_rows"] == 2
    assert report["total_conversations"] == 1
    assert report["conversations_by_source"] == {"codex": 1}
    assert report["conversations_by_model"] == {"model-a": 1}
    assert report["model_metrics"]["model-a"] == {
        "conversations": 1,
        "assistant_chars": 4,
        "assistant_messages": 1,
        "tool_calls": 1,
        "tool_calls_per_assistant_message": 1,
        "zero_tool_conversations": 0,
    }
    assert report["traces_with_tools"] == 1
    assert report["traces_with_diffs"] == 1
    assert report["total_tool_calls"] == 1
    assert report["total_tool_results"] == 1
    assert report["total_assistant_messages"] == 1
    assert report["tool_calls_per_assistant_message"] == 1
    assert report["token_usage"] == {"input": 10, "output": 5}


def test_stats_reports_cached_input_anomalies(tmp_path: Path):
    in_dir = tmp_path / "export"
    metadata = {
        "trace_id": "ts_anomaly",
        "source": "codex",
        "counts": {"messages": 2, "roles": {"assistant": 1}},
        "token_usage": {"input": 10, "cached_input": 12, "output": 2},
    }
    write_jsonl(in_dir / "messages.jsonl", [{"messages": [], "metadata": metadata}])

    report = corpus_stats(in_dir)

    assert report["reported_cache_hit_ratio"] == 1.2
    assert report["token_usage_anomalies"] == {
        "cached_input_exceeds_input": 1,
    }
