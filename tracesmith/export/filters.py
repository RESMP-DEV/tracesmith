"""Conversation-level filters applied before export."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from fnmatch import fnmatchcase

from tracesmith.config import ExportConfig
from tracesmith.export.metadata import build_metadata


def _matches(value: object, patterns: list[str]) -> bool:
    if not isinstance(value, (str, int)):
        return False
    normalized = str(value).casefold()
    return any(fnmatchcase(normalized, pattern.casefold()) for pattern in patterns)


def _timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 1000 if value > 10_000_000_000 else float(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def time_bounds(config: ExportConfig) -> tuple[float | None, float | None]:
    """Parse and validate configured time bounds once."""
    bounds: list[float | None] = []
    for name, raw in (("since", config.since), ("until", config.until)):
        if not raw:
            bounds.append(None)
            continue
        parsed = _timestamp(raw)
        if parsed is None:
            raise ValueError(f"invalid --{name} timestamp: {raw!r}")
        bounds.append(parsed)
    return bounds[0], bounds[1]


def filter_reason(
    conv: dict,
    config: ExportConfig,
    *,
    metadata: dict | None = None,
    bounds: tuple[float | None, float | None] | None = None,
) -> str | None:
    """Return the first exclusion reason for a normalized conversation."""
    since, until = bounds if bounds is not None else time_bounds(config)
    if metadata is None:
        metadata = build_metadata(conv, metadata_key=config.metadata_key)
    counts = metadata["counts"]
    source = metadata["source"]

    if config.include_sources and not _matches(source, config.include_sources):
        return "source_not_included"
    if config.drop_sources and _matches(source, config.drop_sources):
        return "source_dropped"

    project_values = [
        conv.get(key)
        for key in ("project_id", "project_hash", "project_path", "workspace_id")
    ]
    if config.projects and not any(
        _matches(value, config.projects) for value in project_values
    ):
        return "project_not_included"
    if config.models and not any(
        _matches(model, config.models) for model in metadata["models"]
    ):
        return "model_not_included"
    if config.statuses and not _matches(conv.get("status") or "unknown", config.statuses):
        return "status_not_included"

    created_at = _timestamp(conv.get("created_at"))
    if since is not None:
        if created_at is None or created_at < since:
            return "before_since"
    if until is not None:
        if created_at is None or created_at > until:
            return "after_until"

    if config.min_turns is not None and counts["messages"] < config.min_turns:
        return "below_min_messages"
    if config.max_turns is not None and counts["messages"] > config.max_turns:
        return "above_max_messages"
    if (
        config.min_assistant_chars is not None
        and counts["assistant_chars"] < config.min_assistant_chars
    ):
        return "below_min_assistant_chars"
    if config.require_tools and counts["tool_calls"] == 0:
        return "tools_required"
    if config.require_diffs and counts["diffs"] == 0:
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
        [source, conv.get("messages", [])], sort_keys=True, separators=(",", ":")
    ).encode()
    return "h:" + hashlib.sha256(blob).hexdigest()
