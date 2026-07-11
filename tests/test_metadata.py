"""Acceptance tests for the canonical, privacy-safe trace metadata envelope."""
from __future__ import annotations

import json

import pytest

from tracesmith.export.metadata import (
    build_message_metadata,
    build_metadata,
    source_family,
    trace_id,
)


@pytest.mark.parametrize(
    ("source", "family"),
    [
        ("cursor", "cursor"),
        ("cursor-chat", "cursor"),
        ("cursor-global-composer", "cursor"),
        ("cursor-workspace-composer", "cursor"),
        ("claude_code", "claude_code"),
        ("opencode-cli", "opencode"),
    ],
)
def test_source_family_normalizes_source_variants(source: str, family: str) -> None:
    assert source_family(source) == family


def test_trace_id_is_deterministic_opaque_and_namespaced_by_source() -> None:
    cursor = {"source": "cursor-chat", "session_id": "session-secret-123"}
    same_cursor = {"source": "cursor-chat", "session_id": "session-secret-123"}
    codex = {"source": "codex", "session_id": "session-secret-123"}

    cursor_id = trace_id(cursor)

    assert cursor_id == trace_id(same_cursor)
    assert cursor_id != trace_id(codex)
    assert "session-secret-123" not in cursor_id
    assert cursor_id
    assert cursor_id != trace_id({"source": "cursor-chat", "session_id": "different"})


def test_build_metadata_preserves_filterable_session_and_project_fields() -> None:
    conv = {
        "source": "cursor-global-composer",
        "session_id": "sess-1",
        "name": "Implement metadata export",
        "status": "completed",
        "version": "2.3.4",
        "project_name": "tracesmith",
        "project_path": "/work/tracesmith",
        "created_at": 100,
        "updated_at": 400,
        "started_at": 120,
        "last_active_at": 390,
        "messages": [
            {"role": "user", "content": "Please inspect this", "model": "ignored-user-model"},
            {"role": "assistant", "content": "Working", "model": "model-b"},
            {"role": "assistant", "content": "Done", "model": "model-a"},
            {"role": "assistant", "content": "Verified", "model": "model-b"},
        ],
    }

    metadata = build_metadata(conv)

    assert metadata["schema_version"] == "1.0"
    assert metadata["trace_id"] == trace_id(conv)
    assert metadata["source"] == {
        "family": "cursor",
        "variant": "cursor-global-composer",
    }
    assert metadata["session"] == {
        "status": "completed",
        "version": "2.3.4",
        "created_at": 100,
        "updated_at": 400,
        "started_at": 120,
        "last_active_at": 390,
    }
    assert metadata["project"]["id"].startswith("tp_")
    assert "tracesmith" not in metadata["project"]["id"]
    # Models are ordered by first use and only assistant generations count.
    assert metadata["models"] == ["model-b", "model-a"]


def test_build_metadata_counts_trace_structure_and_sets_flags() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "change it", "code_context": [{"path": "a.py"}]},
        {
            "role": "assistant",
            "content": "editing",
            "tool_use": {"name": "read", "input": {"path": "a.py"}},
            "tool_uses": [
                {"name": "edit", "input": {"path": "a.py"}},
                {"name": "test", "input": {"cmd": "pytest"}},
            ],
            "suggested_diffs": [{"diff": "secret diff"}],
        },
        {
            "role": "tool",
            "content": "tool output",
            "tool_results": [
                {"tool": "edit", "output": "changed"},
                {"tool": "test", "output": "passed"},
            ],
            "code_context": [{"path": "b.py"}, {"path": "c.py"}],
            "diff_histories": [{"diff": "one"}, {"diff": "two"}],
        },
        {"role": "assistant", "content": "done"},
    ]

    metadata = build_metadata(
        {"source": "codex", "session_id": "counts", "status": "completed", "messages": messages}
    )

    assert metadata["counts"] == {
        "messages": 5,
        "roles": {"system": 1, "user": 1, "assistant": 2, "tool": 1},
        "assistant_chars": len("editing") + len("done"),
        "total_chars": sum(len(message["content"]) for message in messages),
        "tool_calls": 3,
        "tool_results": 2,
        "code_contexts": 3,
        "diffs": 3,
    }
    assert metadata["flags"] == {
        "has_tools": True,
        "has_code_context": True,
        "has_diffs": True,
        "complete": True,
    }


def test_source_metadata_keeps_approved_scalars_and_drops_sensitive_locations() -> None:
    conv = {
        "source": "cursor-chat",
        "session_id": "scalar-fields",
        "messages": [],
        "storage_type": "inline",
        "originator": "ide",
        "is_sidechain": False,
        "request_count": 7,
        "source_file": "/private/raw.jsonl",
        "installation": "/Applications/Cursor.app",
        "session_file": "/private/session.json",
        "cwd": "/private/repository",
        "workspace": "/private/workspace",
        "directory": "/private/directory",
        "summary": {"nested": "not an approved scalar"},
    }

    metadata = build_metadata(conv)

    assert metadata["source_metadata"] == {
        "storage_type": "inline",
        "originator": "ide",
        "is_sidechain": False,
        "request_count": 7,
    }
    encoded = json.dumps(metadata, sort_keys=True)
    for private_value in (
        "/private/raw.jsonl",
        "/Applications/Cursor.app",
        "/private/session.json",
        "/private/repository",
        "/private/workspace",
        "/private/directory",
    ):
        assert private_value not in encoded


def test_public_metadata_pseudonymizes_projects_and_coarsens_timestamps() -> None:
    metadata = build_metadata({
        "source": "cursor-chat",
        "session_id": "private-session-id",
        "title": "Discuss Bluebird merger with Carol",
        "project_name": "StealthAcquisitionBluebird",
        "project_id": "fedcba9876543210",
        "workspace_id": "workspace-customer-0199",
        "created_at": "2026-07-10T12:34:56.123456Z",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
    })

    encoded = json.dumps(metadata, sort_keys=True)
    for private_value in (
        "private-session-id",
        "Discuss Bluebird merger with Carol",
        "StealthAcquisitionBluebird",
        "fedcba9876543210",
        "workspace-customer-0199",
        "2026-07-10T12:34:56.123456Z",
    ):
        assert private_value not in encoded
    assert metadata["session"]["created_at"] == "2026-07-10"
    assert metadata["project"]["id"].startswith("tp_")


def test_message_metadata_keeps_safe_structure_without_payloads() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "RAW_MESSAGE_SECRET",
            "timestamp": "2026-07-10T12:00:00Z",
            "model": "gpt-test",
            "status": "completed",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 5,
                "account_id": 123456,
                "account": {"id": "PRIVATE_ACCOUNT"},
            },
            "tokens": {"input": 20, "output": 5, "details": {"secret": "PRIVATE_TOKEN_DETAIL"}},
            "tool_use": {
                "id": "call-1",
                "name": "shell",
                "status": "completed",
                "input": {"cmd": "RAW_TOOL_INPUT"},
                "output": "RAW_TOOL_OUTPUT",
            },
            "tool_results": [
                {
                    "tool_call_id": "call-1",
                    "tool": "shell",
                    "status": "completed",
                    "output": "RAW_RESULT_OUTPUT",
                }
            ],
        }
    ]

    metadata = build_message_metadata(messages)

    assert metadata == [
        {
            "index": 0,
            "role": "assistant",
            "timestamp": "2026-07-10",
            "model": "gpt-test",
            "status": "completed",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 5},
            "tokens": {"input": 20, "output": 5},
            "tools": [
                {"kind": "call", "name": "shell", "status": "completed", "call_id": "call-1"},
                {"kind": "result", "name": "shell", "status": "completed", "call_id": "call-1"},
            ],
        }
    ]

    encoded = json.dumps(metadata, sort_keys=True)
    for raw_payload in (
        "RAW_MESSAGE_SECRET",
        "RAW_TOOL_INPUT",
        "RAW_TOOL_OUTPUT",
        "RAW_RESULT_OUTPUT",
        "PRIVATE_ACCOUNT",
        "account_id",
        "PRIVATE_TOKEN_DETAIL",
    ):
        assert raw_payload not in encoded


def test_message_ids_are_opaque_but_linkable() -> None:
    metadata = build_message_metadata([
        {"id": "provider-user-id", "role": "user", "content": "q"},
        {
            "id": "provider-assistant-id",
            "parent_id": "provider-user-id",
            "role": "assistant",
            "content": "a",
        },
    ])

    assert metadata[0]["message_id"].startswith("tm_")
    assert metadata[1]["message_id"].startswith("tm_")
    assert metadata[1]["parent_message_id"] == metadata[0]["message_id"]
    assert "provider-user-id" not in json.dumps(metadata)


def test_build_metadata_never_copies_message_or_tool_payloads() -> None:
    conv = {
        "source": "opencode-cli",
        "session_id": "safe-envelope",
        "messages": [
            {"role": "user", "content": "RAW_USER_CONTENT"},
            {
                "role": "assistant",
                "content": "RAW_ASSISTANT_CONTENT",
                "tool_calls": [
                    {"id": "call-9", "name": "write", "input": {"text": "RAW_TOOL_INPUT"}}
                ],
                "tool_results": [
                    {"tool_call_id": "call-9", "tool": "write", "output": "RAW_TOOL_OUTPUT"}
                ],
            },
        ],
    }

    encoded = json.dumps(build_metadata(conv), sort_keys=True)

    assert "RAW_USER_CONTENT" not in encoded
    assert "RAW_ASSISTANT_CONTENT" not in encoded
    assert "RAW_TOOL_INPUT" not in encoded
    assert "RAW_TOOL_OUTPUT" not in encoded


def test_build_metadata_normalizes_token_usage_and_performance() -> None:
    metadata = build_metadata({
        "source": "claude_code",
        "total_turn_duration_ms": 2500,
        "min_time_to_first_token_ms": 125,
        "max_time_to_first_token_ms": 300,
        "messages": [
            {
                "role": "assistant",
                "content": "a",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 6,
                },
            },
            {
                "role": "assistant",
                "content": "b",
                "tokens": {"input": 3, "output": 2, "thoughts": 1},
            },
        ],
    })

    assert metadata["token_usage"] == {
        "input": 13,
        "output": 6,
        "cached": 6,
        "reasoning": 1,
    }
    assert metadata["performance"] == {
        "total_turn_duration_ms": 2500,
        "min_time_to_first_token_ms": 125,
        "max_time_to_first_token_ms": 300,
    }


def test_build_metadata_summarizes_conversation_level_tool_events() -> None:
    metadata = build_metadata({
        "source": "codex",
        "messages": [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "done"},
        ],
        "tool_results": [
            {
                "type": "function_call",
                "tool": "shell",
                "call_id": "call-1",
                "input": {"cmd": "RAW_COMMAND"},
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "RAW_OUTPUT",
            },
            {"type": "diff", "file": "secret.py", "diff": "RAW_DIFF"},
        ],
    })

    assert metadata["counts"]["tool_calls"] == 1
    assert metadata["counts"]["tool_results"] == 1
    assert metadata["counts"]["diffs"] == 1
    assert metadata["tools"] == [
        {"kind": "call", "name": "shell", "call_id": "call-1"},
        {"kind": "result", "call_id": "call-1"},
    ]
    encoded = json.dumps(metadata)
    assert "RAW_COMMAND" not in encoded
    assert "RAW_OUTPUT" not in encoded
    assert "RAW_DIFF" not in encoded


def test_explicit_diff_count_sets_filterable_diff_flag() -> None:
    metadata = build_metadata({
        "source": "codex",
        "diff_count": 3,
        "files_changed_count": 3,
        "messages": [
            {"role": "user", "content": "change it"},
            {"role": "assistant", "content": "done"},
        ],
    })

    assert metadata["counts"]["diffs"] == 3
    assert metadata["flags"]["has_diffs"] is True
    assert metadata["source_metadata"]["files_changed_count"] == 3
