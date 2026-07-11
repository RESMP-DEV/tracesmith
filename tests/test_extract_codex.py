"""Tests for the Codex extractor."""
from __future__ import annotations

import json
from pathlib import Path

from tracesmith.extract.codex import CodexExtractor


def write_rollout(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_extract_basic_user_assistant_turns(tmp_path):
    install = tmp_path / ".codex"
    session = install / "sessions" / "2024" / "01" / "01" / "rollout-abc.jsonl"
    write_rollout(session, [
        {"type": "session_meta", "payload": {"id": "abc", "cwd": "/tmp/proj",
                                              "timestamp": "2024-01-01T00:00:00Z"}},
        {"type": "event_msg", "timestamp": "t1",
         "payload": {"type": "user_message", "message": "hello"}},
        {"type": "event_msg", "timestamp": "t2",
         "payload": {"type": "agent_message", "message": "hi there",
                     "model": "gpt-5-codex"}},
    ])

    convs = list(CodexExtractor().extract(install))

    assert len(convs) == 1
    conv = convs[0]
    assert conv["source"] == "codex"
    assert conv["session_id"] == "abc"
    assert conv["cwd"] == "/tmp/proj"
    assert conv["timestamp"] == "2024-01-01T00:00:00Z"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0] == {"role": "user", "content": "hello", "timestamp": "t1"}
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["content"] == "hi there"
    assert conv["messages"][1]["model"] == "gpt-5-codex"


def test_extract_tool_use_and_tool_results_collected(tmp_path):
    install = tmp_path / ".codex"
    session = install / "sessions" / "rollout-r.jsonl"
    write_rollout(session, [
        {"type": "session_meta", "payload": {"id": "r"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "do it"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "ok"}},
        {"type": "event_msg", "payload": {"type": "tool_use", "tool": "shell",
                                          "input": {"cmd": "ls"}}},
        {"type": "event_msg", "payload": {"type": "tool_result", "tool": "shell",
                                          "output": "file.txt"}},
        {"type": "event_msg", "payload": {"type": "diff", "file": "f.py",
                                          "diff": "-old\n+new"}},
    ])

    convs = list(CodexExtractor().extract(install))
    conv = convs[0]
    # Codex collects tool_use/tool_result/diff into a top-level tool_results list
    assert len(conv["tool_results"]) == 3
    assert conv["tool_results"][0]["type"] == "tool_use"
    assert conv["tool_results"][0]["tool"] == "shell"
    assert conv["tool_results"][1]["type"] == "tool_result"
    assert conv["tool_results"][2]["type"] == "diff"
    assert conv["tool_results"][2]["file"] == "f.py"


def test_extract_current_rollout_metadata_and_response_items(tmp_path):
    install = tmp_path / ".codex"
    session = install / "sessions" / "rollout-current.jsonl"
    write_rollout(session, [
        {
            "type": "session_meta",
            "timestamp": "2026-07-10T12:00:00Z",
            "payload": {
                "id": "current",
                "cwd": "/work/repo",
                "timestamp": "2026-07-10T12:00:00Z",
                "cli_version": "1.2.3",
                "model_provider": "openai",
                "originator": "codex_cli",
                "agent_nickname": "worker",
                "parent_thread_id": "parent",
            },
        },
        {
            "type": "turn_context",
            "payload": {
                "model": "gpt-5.5-codex",
                "effort": "high",
                "approval_policy": "never",
                "collaboration_mode": {"mode": "default"},
                "permission_profile": {"type": "danger-full-access"},
                "sandbox_policy": {"type": "unrestricted"},
            },
        },
        {
            "type": "event_msg",
            "timestamp": "2026-07-10T12:00:01Z",
            "payload": {"type": "user_message", "message": "implement it"},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "started_at": "2026-07-10T12:00:01Z",
                "model_context_window": 200000,
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-10T12:00:02Z",
            "payload": {
                "type": "function_call",
                "id": "call-1",
                "name": "shell",
                "arguments": {"cmd": "uv run pytest"},
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-10T12:00:03Z",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "passed",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-10T12:00:04Z",
            "payload": {
                "type": "message",
                "id": "assistant-1",
                "role": "assistant",
                "phase": "final",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": 200000,
                    "total_token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "cached_input_tokens": 50,
                    },
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "duration_ms": 3000,
                "time_to_first_token_ms": 250,
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "patch_apply_end",
                "success": True,
                "status": "completed",
                "changes": {"a.py": {"type": "update"}, "b.py": {"type": "add"}},
            },
        },
    ])

    conv = list(CodexExtractor().extract(install))[0]

    assert conv["messages"][-1] == {
        "role": "assistant",
        "content": "done",
        "timestamp": "2026-07-10T12:00:04Z",
        "id": "assistant-1",
        "phase": "final",
        "model": "gpt-5.5-codex",
    }
    assert conv["version"] == "1.2.3"
    assert conv["project_path"] == "/work/repo"
    assert conv["model_provider"] == "openai"
    assert conv["reasoning_effort"] == "high"
    assert conv["context_window"] == 200000
    assert conv["status"] == "completed"
    assert conv["turn_count"] == 1
    assert conv["total_turn_duration_ms"] == 3000
    assert conv["min_time_to_first_token_ms"] == 250
    assert conv["max_time_to_first_token_ms"] == 250
    assert conv["token_usage"]["cached_input_tokens"] == 50
    assert conv["diff_count"] == 2
    assert conv["files_changed_count"] == 2
    assert [item["type"] for item in conv["tool_results"]] == [
        "function_call",
        "function_call_output",
    ]


def test_repeated_messages_survive_while_cross_representation_duplicates_do_not(tmp_path):
    install = tmp_path / ".codex"
    session = install / "sessions" / "rollout-repeated.jsonl"
    write_rollout(session, [
        {"type": "session_meta", "payload": {"id": "repeated"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "continue"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "continue"}},
        {
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "done", "phase": "final"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "provider-message-id",
                "role": "assistant",
                "phase": "final",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
    ])

    conv = list(CodexExtractor().extract(install))[0]

    assert [message["content"] for message in conv["messages"]] == [
        "continue",
        "continue",
        "done",
    ]
    assert conv["messages"][-1]["id"] == "provider-message-id"
    assert conv["messages"][-1]["phase"] == "final"


def test_extract_empty_install(tmp_path):
    install = tmp_path / ".codex"
    install.mkdir(parents=True)
    assert list(CodexExtractor().extract(install)) == []


def test_find_installations_discovers_codex(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".codex").mkdir()
    installs = CodexExtractor().find_installations()
    assert tmp_path / ".codex" in installs
