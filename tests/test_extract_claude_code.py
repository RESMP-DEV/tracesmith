"""Tests for the Claude Code extractor."""
from __future__ import annotations

import json
from pathlib import Path

from tracesmith.extract.claude_code import ClaudeCodeExtractor


def write_session_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_extract_basic_user_assistant_turns(tmp_path):
    # Arrange: fake a .claude/projects/myproj/<session>.jsonl
    install = tmp_path / ".claude"
    proj = install / "projects" / "myproj"
    session = proj / "abc123.jsonl"
    write_session_jsonl(session, [
        {"type": "user", "timestamp": "t1", "cwd": "/tmp/proj",
         "message": {"content": "hello"}},
        {"type": "assistant", "timestamp": "t2",
         "message": {"content": [{"type": "text", "text": "hi there"}],
                     "model": "claude-sonnet-4-5"}},
    ])

    ext = ClaudeCodeExtractor()
    convs = list(ext.extract(install))

    assert len(convs) == 1
    conv = convs[0]
    assert conv["source"] == "claude_code"
    assert conv["session_id"] == "abc123"
    assert conv["project_path"] == "/tmp/proj"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0] == {"role": "user", "content": "hello", "timestamp": "t1"}
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["content"] == "hi there"
    assert conv["messages"][1]["model"] == "claude-sonnet-4-5"


def test_extract_tool_use_and_tool_result_attached(tmp_path):
    install = tmp_path / ".claude"
    proj = install / "projects" / "p"
    session = proj / "s.jsonl"
    write_session_jsonl(session, [
        {"type": "user", "message": {"content": "do the thing"}},
        {"type": "assistant",
         "message": {"content": [
             {"type": "text", "text": "ok"},
             {"type": "tool_use", "name": "edit", "input": {"path": "f.py"}},
         ], "model": "m"}},
        {"type": "tool_result", "toolResult": {"type": "text", "text": "done"}},
    ])

    convs = list(ClaudeCodeExtractor().extract(install))
    asst = convs[0]["messages"][1]
    assert asst["role"] == "assistant"
    assert asst["content"] == "ok"
    assert asst["tool_uses"][0]["name"] == "edit"
    assert asst["tool_results"][0]["text"] == "done"


def test_extract_preserves_session_and_message_metadata(tmp_path):
    install = tmp_path / ".claude"
    session = install / "projects" / "p" / "metadata.jsonl"
    write_session_jsonl(session, [
        {
            "type": "user",
            "uuid": "user-1",
            "timestamp": "2026-07-10T12:00:00Z",
            "version": "2.1.0",
            "entrypoint": "cli",
            "userType": "external",
            "isSidechain": False,
            "requestId": "request-1",
            "message": {"content": "hello"},
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "timestamp": "2026-07-10T12:00:01Z",
            "requestId": "request-1",
            "message": {
                "id": "provider-message-1",
                "model": "claude-test",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ])

    conv = list(ClaudeCodeExtractor().extract(install))[0]

    assert conv["version"] == "2.1.0"
    assert conv["entrypoint"] == "cli"
    assert conv["user_type"] == "external"
    assert conv["is_sidechain"] is False
    assert conv["request_count"] == 1
    assert conv["created_at"] == "2026-07-10T12:00:00Z"
    assert conv["updated_at"] == "2026-07-10T12:00:01Z"
    assert conv["messages"][0]["id"] == "user-1"
    assert conv["messages"][1]["id"] == "provider-message-1"
    assert conv["messages"][1]["parent_id"] == "user-1"
    assert conv["messages"][1]["stop_reason"] == "end_turn"
    assert conv["messages"][1]["usage"] == {"input_tokens": 3, "output_tokens": 1}


def test_extract_current_user_wrapped_tool_results(tmp_path):
    install = tmp_path / ".claude"
    session = install / "projects" / "p" / "tool-results.jsonl"
    write_session_jsonl(session, [
        {"type": "user", "message": {"content": "run tests"}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-test",
                "content": [{
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "shell",
                    "input": {"command": "uv run pytest"},
                }],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "is_error": False,
                    "content": [{"type": "text", "text": "135 passed"}],
                }],
            },
            "toolUseResult": {"stdout": "135 passed", "exitCode": 0},
        },
        {"type": "assistant", "message": {"content": "done", "model": "claude-test"}},
    ])

    conv = list(ClaudeCodeExtractor().extract(install))[0]
    tool_message = conv["messages"][2]

    assert tool_message["role"] == "tool"
    assert tool_message["content"] == ""
    assert tool_message["tool_results"] == [{
        "tool_call_id": "tool-1",
        "status": "completed",
        "output": "135 passed",
    }]


def test_extract_skips_empty_session(tmp_path):
    install = tmp_path / ".claude"
    proj = install / "projects" / "p"
    write_session_jsonl(proj / "empty.jsonl", [])
    convs = list(ClaudeCodeExtractor().extract(install))
    assert convs == []


def test_find_installations_discovers_dotclaude(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()

    ext = ClaudeCodeExtractor()
    installs = ext.find_installations()
    assert tmp_path / ".claude" in installs
