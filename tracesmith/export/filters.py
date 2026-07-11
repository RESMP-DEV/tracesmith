"""Conversation-level filters applied before export."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from fnmatch import fnmatchcase

from tracesmith.config import ExportConfig
from tracesmith.export.metadata import build_metadata


def _matches(value: str, patterns: list[str]) -> bool:
    normalized = value.casefold()
    return any(fnmatchcase(normalized, pattern.casefold()) for pattern in patterns)


def _timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000 if number > 10_000_000_000 else number
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return _timestamp(float(text))
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def filter_reason(conv: dict, config: ExportConfig) -> str | None:
    """Return the first reason a conversation is excluded, or ``None``."""
    metadata = build_metadata(conv, metadata_key=config.metadata_key)
    counts = metadata["counts"]
    source = metadata["source"]
    session = metadata["session"]
    flags = metadata["flags"]

    if config.include_sources and not any(
        _matches(source[key], config.include_sources) for key in ("family", "variant")
    ):
        return "source_not_included"
    if config.drop_sources and any(
        _matches(source[key], config.drop_sources) for key in ("family", "variant")
    ):
        return "source_dropped"

    project_values = [
        str(conv[key])
        for key in (
            "project_name",
            "project_path",
            "project_id",
            "project_hash",
            "workspace_id",
        )
        if conv.get(key) is not None
    ]
    if config.projects and not any(
        _matches(value, config.projects) for value in project_values
    ):
        return "project_not_included"
    if config.models and not any(_matches(model, config.models) for model in metadata["models"]):
        return "model_not_included"
    normalized_status = session.get("status")
    if not normalized_status:
        normalized_status = "completed" if flags["complete"] else "unknown"
    if config.statuses and not _matches(str(normalized_status), config.statuses):
        return "status_not_included"

    started = _timestamp(session.get("started_at") or session.get("created_at"))
    if config.since:
        since = _timestamp(config.since)
        if since is None:
            raise ValueError(f"invalid --since timestamp: {config.since!r}")
        if started is None or started < since:
            return "before_since"
    if config.until:
        until = _timestamp(config.until)
        if until is None:
            raise ValueError(f"invalid --until timestamp: {config.until!r}")
        if started is None or started > until:
            return "after_until"

    if config.min_turns is not None and counts["messages"] < config.min_turns:
        return "below_min_turns"
    if config.max_turns is not None and counts["messages"] > config.max_turns:
        return "above_max_turns"
    if (
        config.min_assistant_chars is not None
        and counts["assistant_chars"] < config.min_assistant_chars
    ):
        return "below_min_assistant_chars"
    if config.require_tools and not flags["has_tools"]:
        return "tools_required"
    if config.require_diffs and not flags["has_diffs"]:
        return "diffs_required"
    return None


def passes(conv: dict, config: ExportConfig) -> bool:
    return filter_reason(conv, config) is None


def dedup_key(conv: dict) -> str:
    """Stable source-namespaced identity for exact deduplication."""
    source = str(conv.get("source") or "unknown")
    if conv.get("session_id"):
        return f"sid:{source}:{conv['session_id']}"
    blob = json.dumps(
        conv.get("messages", []), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return "h:" + hashlib.sha256(blob).hexdigest()
