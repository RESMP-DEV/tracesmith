"""Shared types for extractors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol, TypedDict


class Message(TypedDict, total=False):
    id: str | None
    parent_id: str | None
    role: str            # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: str | None
    model: str | None
    phase: str | None
    status: str | None
    stop_reason: str | None
    usage: dict[str, Any]
    tokens: dict[str, Any]
    code_context: list[dict[str, Any]]
    suggested_diffs: list[dict[str, Any]]
    tool_use: dict[str, Any]
    tool_uses: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_result: dict[str, Any]
    tool_results: list[dict[str, Any]]
    diff_histories: list[dict[str, Any]]


class Conversation(TypedDict, total=False):
    messages: list[Message]
    source: str
    name: str | None
    title: str | None
    status: str | None
    version: str | None
    model: str | None
    session_id: str | None
    parent_session_id: str | None
    project_path: str | None
    project_name: str | None
    project_id: str | None
    project_hash: str | None
    source_file: str | None
    created_at: int | str | None
    updated_at: int | str | None
    started_at: int | str | None
    last_active_at: int | str | None


class Extractor(Protocol):
    """A source-specific extractor.

    Implementations discover local installations and yield normalized
    Conversation dicts. Source name is the canonical short key
    (e.g. "claude_code", "cursor").
    """

    source: str

    def find_installations(self) -> list[Path]: ...

    def extract(self, install_dir: Path) -> Iterator[Conversation]: ...
