from __future__ import annotations

from pathlib import Path

from tracesmith.stats import corpus_stats
from tracesmith.config import ExportConfig
from tracesmith.export.messages import export_messages
from tracesmith.export.sharegpt import export_sharegpt
from tests.conftest import make_conversation, make_message, write_jsonl


def test_stats_happy_path(tmp_path: Path):
    in_dir = tmp_path / "export"
    write_jsonl(in_dir / "claude_code.jsonl", [
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
    ])
    write_jsonl(in_dir / "codex.jsonl", [
        make_conversation(
            messages=[make_message("user", "x"), make_message("assistant", "y")],
            source="codex",
        ),
    ])

    report = corpus_stats(in_dir)

    assert report["conversations_by_source"] == {"claude_code": 2, "codex": 1}
    assert report["total_conversations"] == 3
    assert report["turn_count_min"] == 2
    assert report["turn_count_max"] == 4
    assert report["turn_count_mean"] == (2 + 4 + 2) / 3
    assert report["total_assistant_chars"] == len("hello there") + len("yes") + len("done") + len("y")


def test_stats_empty_dir(tmp_path: Path):
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    report = corpus_stats(in_dir)
    assert report["total_conversations"] == 0
    assert report["turn_count_min"] == 0
    assert report["turn_count_mean"] == 0
    assert report["total_assistant_chars"] == 0


def test_stats_uses_exported_metadata_for_source_model_and_tools(tmp_path: Path):
    redacted = tmp_path / "redacted"
    exported = tmp_path / "export"
    write_jsonl(redacted / "cursor.jsonl", [
        make_conversation(
            messages=[
                make_message("user", "fix"),
                make_message(
                    "assistant",
                    "done",
                    model="gpt-5.5",
                    tool_use={"name": "edit", "input": {"path": "x.py"}},
                ),
            ],
            source="cursor-chat",
        )
    ])
    export_messages(redacted, exported / "messages.jsonl", ExportConfig())

    report = corpus_stats(exported)

    assert report["conversations_by_source"] == {"cursor": 1}
    assert report["conversations_by_model"] == {"gpt-5.5": 1}
    assert report["total_tool_calls"] == 1
    assert report["traces_with_tools"] == 1


def test_stats_deduplicates_same_trace_across_export_variants(tmp_path: Path):
    redacted = tmp_path / "redacted"
    exported = tmp_path / "export"
    write_jsonl(redacted / "claude_code.jsonl", [
        make_conversation(
            messages=[
                make_message("user", "one"),
                make_message("assistant", "two"),
                make_message("user", "three"),
                make_message("assistant", "four"),
            ],
            source="claude_code",
            session_id="shared-trace",
        )
    ])
    export_messages(redacted, exported / "messages.jsonl", ExportConfig())
    export_sharegpt(redacted, exported / "sharegpt.jsonl", ExportConfig())

    report = corpus_stats(exported)

    assert report["rows_by_variant"] == {"messages": 1, "sharegpt": 2}
    assert report["total_rows"] == 3
    assert report["total_conversations"] == 1
    assert report["conversations_by_source"] == {"claude_code": 1}


def test_stats_counts_legacy_sharegpt_rows_without_collapsing_them(tmp_path: Path):
    legacy = tmp_path / "legacy"
    write_jsonl(legacy / "sharegpt.jsonl", [
        {"conversations": [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
        ]},
        {"conversations": [
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ]},
    ])

    report = corpus_stats(legacy)

    assert report["total_conversations"] == 2
    assert report["conversations_by_source"] == {"sharegpt": 2}
    assert report["turn_count_min"] == 2
    assert report["total_assistant_chars"] == 4
