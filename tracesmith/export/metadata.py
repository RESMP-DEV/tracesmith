"""Canonical, privacy-safe metadata for exported trace rows."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
_PROCESS_KEY = (
    os.environ.get("TRACESMITH_METADATA_KEY") or secrets.token_hex(32)
).encode("utf-8")

_SOURCE_FAMILIES = {
    "claude": "claude_code",
    "claude-code": "claude_code",
    "claude_code": "claude_code",
    "gemini-cli": "gemini",
    "opencode-cli": "opencode",
    "opencode-desktop": "opencode",
    "windsurf-agent": "windsurf",
    "windsurf-chat": "windsurf",
}

_SOURCE_METADATA_ALLOWED = {
    "agent_name",
    "approval_policy",
    "app_version",
    "cli_version",
    "collaboration_mode",
    "context_window",
    "entrypoint",
    "error_count",
    "files_changed_count",
    "has_code_context",
    "is_agentic",
    "is_plan_execution",
    "is_sidechain",
    "mode",
    "model_provider",
    "originator",
    "permission_profile",
    "reasoning_effort",
    "request_count",
    "sandbox_profile",
    "storage_type",
    "tool_call_count",
    "turn_count",
    "user_type",
}

_TOOL_COLLECTIONS = (
    ("tool_use", "call"),
    ("tool_uses", "call"),
    ("tool_calls", "call"),
    ("tool_result", "result"),
    ("tool_results", "result"),
)

_TOKEN_KEYS = {
    "input": "input",
    "input_tokens": "input",
    "prompt_tokens": "input",
    "output": "output",
    "output_tokens": "output",
    "completion_tokens": "output",
    "reasoning": "reasoning",
    "reasoning_tokens": "reasoning",
    "reasoning_output_tokens": "reasoning",
    "thoughts": "reasoning",
    "cached": "cached",
    "cached_input_tokens": "cached",
    "cache_read_input_tokens": "cached",
    "cache_creation_input_tokens": "cache_write",
    "tool": "tool",
    "total": "total",
    "total_tokens": "total",
}

_PERFORMANCE_FIELDS = {
    "total_turn_duration_ms",
    "min_time_to_first_token_ms",
    "max_time_to_first_token_ms",
    "tokens_per_second",
    "peak_rss_bytes",
    "retry_count",
    "compaction_count",
    "cancellation_count",
    "error_count",
}


def source_family(source: str | None) -> str:
    """Return the stable family for a source-specific trace variant."""
    normalized = str(source or "unknown").strip().lower()
    if normalized in _SOURCE_FAMILIES:
        return _SOURCE_FAMILIES[normalized]
    for prefix in ("cursor", "opencode", "windsurf", "gemini"):
        if normalized == prefix or normalized.startswith(prefix + "-"):
            return prefix
    return normalized or "unknown"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _key_bytes(metadata_key: str | bytes | None) -> bytes:
    if isinstance(metadata_key, bytes):
        return metadata_key
    if isinstance(metadata_key, str):
        return metadata_key.encode("utf-8")
    return _PROCESS_KEY


def _opaque_id(kind: str, value: Any, metadata_key: str | bytes | None) -> str:
    digest = hmac.new(
        _key_bytes(metadata_key),
        (f"tracesmith:{kind}:v1:" + _canonical_json(value)).encode("utf-8", "ignore"),
        hashlib.sha256,
    ).hexdigest()
    prefix = {"trace": "ts", "project": "tp"}.get(kind, "to")
    return f"{prefix}_{digest[:24]}"


def trace_id(
    conv: dict[str, Any], metadata_key: str | bytes | None = None
) -> str:
    """Build a keyed opaque ID from source identity or redacted content."""
    source = str(conv.get("source") or "unknown")
    session_id = conv.get("session_id")
    if _is_scalar(session_id) and session_id != "":
        identity: dict[str, Any] = {"source": source, "session_id": session_id}
    else:
        identity = {
            "source": source,
            "messages": [
                {"role": message.get("role"), "content": message.get("content")}
                for message in conv.get("messages", [])
                if isinstance(message, dict)
            ],
            "created_at": _first(conv, "created_at", "start_time", "timestamp"),
        }
    return _opaque_id("trace", identity, metadata_key)


def _coarse_timestamp(value: Any) -> Any:
    """Reduce real timestamps to UTC day precision for public metadata."""
    parsed: datetime | None = None
    if isinstance(value, (int, float)) and value > 10_000_000:
        seconds = float(value) / 1000 if value > 10_000_000_000 else float(value)
        try:
            parsed = datetime.fromtimestamp(seconds, timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if parsed is None:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def _message_id(index: int, namespace: str | None = None) -> str:
    digest = hashlib.sha256(
        (f"tracesmith:message:v1:{namespace or ''}:" + str(index)).encode(
            "utf-8", "ignore"
        )
    ).hexdigest()
    return "tm_" + digest[:24]


def _first(conv: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = conv.get(key)
        if value is not None and value != "":
            return value
    return None


def _count_items(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def _safe_numeric_mapping(value: Any) -> dict[str, int | float | bool]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key).casefold() in _TOKEN_KEYS
        and isinstance(item, (int, float, bool))
    }


def _tool_items(message: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for field, kind in _TOOL_COLLECTIONS:
        value = message.get(field)
        if not value:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                yield kind, item


def _tool_summary(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"kind": kind}
    name = _first(item, "name", "tool")
    status = item.get("status")
    call_id = _first(item, "id", "call_id", "tool_call_id", "toolCallID")
    if _is_scalar(name):
        summary["name"] = str(name)
    if _is_scalar(status):
        summary["status"] = str(status)
    if _is_scalar(call_id):
        summary["call_id"] = str(call_id)
    for key in ("duration_ms", "exit_code", "truncated", "background"):
        value = item.get(key)
        if isinstance(value, (int, float, bool)):
            summary[key] = value
    return summary


def build_message_metadata(
    messages: list[dict[str, Any]], namespace: str | None = None
) -> list[dict[str, Any]]:
    """Describe message structure without copying content or tool payloads."""
    result: list[dict[str, Any]] = []
    provider_ids = {
        str(message["id"]): _message_id(index, namespace)
        for index, message in enumerate(messages)
        if _is_scalar(message.get("id")) and message.get("id") != ""
    }
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        if not _is_scalar(role):
            role = "unknown"
        metadata: dict[str, Any] = {
            "index": index,
            "role": str(role),
        }
        if _is_scalar(message.get("id")) and message.get("id") != "":
            metadata["message_id"] = _message_id(index, namespace)
        parent_id = message.get("parent_id")
        if _is_scalar(parent_id) and str(parent_id) in provider_ids:
            metadata["parent_message_id"] = provider_ids[str(parent_id)]
        for key in ("timestamp", "model", "status", "stop_reason", "phase"):
            value = message.get(key)
            if _is_scalar(value) and value != "":
                metadata[key] = _coarse_timestamp(value) if key == "timestamp" else value
        usage = _safe_numeric_mapping(message.get("usage"))
        if usage:
            metadata["usage"] = usage
        tokens = _safe_numeric_mapping(message.get("tokens"))
        if tokens:
            metadata["tokens"] = tokens
        cost = message.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            metadata["cost"] = cost
        tools = [_tool_summary(kind, item) for kind, item in _tool_items(message)]
        if tools:
            metadata["tools"] = tools
        result.append(metadata)
    return result


def _ordered_models(conv: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    candidates: list[Any] = [conv.get("model")]
    candidates.extend(
        message.get("model")
        for message in messages
        if message.get("role") == "assistant"
    )
    for value in candidates:
        if not _is_scalar(value) or value == "":
            continue
        model = str(value)
        if model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _source_metadata(conv: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        str(key): value
        for key, value in conv.items()
        if key in _SOURCE_METADATA_ALLOWED
        and isinstance(value, (str, int, float, bool))
    }


def _token_usage(
    conv: dict[str, Any], messages: list[dict[str, Any]]
) -> dict[str, int | float]:
    totals: Counter[str] = Counter()
    conversation_usage = conv.get("token_usage")
    usage_sources: list[Any]
    if isinstance(conversation_usage, dict):
        usage_sources = [conversation_usage]
    else:
        usage_sources = [
            message.get("usage") or message.get("tokens") for message in messages
        ]
    for values in usage_sources:
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            canonical = _TOKEN_KEYS.get(str(key).casefold())
            if canonical and isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[canonical] += value
    return dict(totals)


def _performance(conv: dict[str, Any]) -> dict[str, int | float]:
    return {
        key: value
        for key, value in conv.items()
        if key in _PERFORMANCE_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def build_metadata(
    conv: dict[str, Any], metadata_key: str | bytes | None = None
) -> dict[str, Any]:
    """Build the canonical metadata envelope for one conversation."""
    messages = [message for message in conv.get("messages", []) if isinstance(message, dict)]
    source = str(conv.get("source") or "unknown")

    role_counts = Counter(
        str(message.get("role", "user"))
        if _is_scalar(message.get("role", "user"))
        else "unknown"
        for message in messages
    )
    assistant_chars = sum(
        len(str(message.get("content", "")))
        for message in messages
        if message.get("role") == "assistant"
    )
    total_chars = sum(len(str(message.get("content", ""))) for message in messages)
    tool_calls = sum(
        _count_items(message.get(field))
        for message in messages
        for field in ("tool_use", "tool_uses", "tool_calls")
    )
    tool_results = sum(
        _count_items(message.get(field))
        for message in messages
        for field in ("tool_result", "tool_results")
    )
    code_contexts = sum(
        _count_items(message.get(field))
        for message in messages
        for field in ("code_context", "context_items")
    )
    diffs = sum(
        _count_items(message.get(field))
        for message in messages
        for field in (
            "suggested_diffs",
            "diff_histories",
            "diffs",
            "edits",
            "suggested_code_blocks",
        )
    )
    explicit_diff_count = conv.get("diff_count")
    if isinstance(explicit_diff_count, int) and explicit_diff_count > 0:
        diffs += explicit_diff_count

    conversation_tool_events: list[dict[str, Any]] = []
    for item in conv.get("tool_results", []):
        if not isinstance(item, dict):
            continue
        event_type = str(item.get("type") or "").casefold()
        if event_type in {"diff", "patch"}:
            diffs += 1
            continue
        kind = "call" if event_type in {"tool_use", "function_call", "custom_tool_call"} else "result"
        conversation_tool_events.append(_tool_summary(kind, item))
        if kind == "call":
            tool_calls += 1
        else:
            tool_results += 1

    session: dict[str, Any] = {}
    session_fields = {
        "status": conv.get("status"),
        "version": conv.get("version"),
        "created_at": _coarse_timestamp(conv.get("created_at")),
        "updated_at": _coarse_timestamp(conv.get("updated_at")),
        "started_at": _coarse_timestamp(
            _first(conv, "started_at", "start_time", "timestamp")
        ),
        "last_active_at": _coarse_timestamp(
            _first(conv, "last_active_at", "last_updated")
        ),
    }
    session.update(
        (key, value)
        for key, value in session_fields.items()
        if _is_scalar(value) and value != ""
    )

    project: dict[str, Any] = {}
    project_identity = _first(
        conv,
        "project_id",
        "project_hash",
        "workspace_id",
        "project_path",
        "project_name",
    )
    if _is_scalar(project_identity) and project_identity != "":
        project["id"] = _opaque_id("project", project_identity, metadata_key)

    status = str(conv.get("status") or "").casefold()
    successful_statuses = {"complete", "completed", "done", "success", "succeeded"}
    failed_statuses = {"aborted", "cancelled", "canceled", "error", "failed"}
    complete = status in successful_statuses or (
        status not in failed_statuses
        and bool(messages and messages[-1].get("role") == "assistant")
    )

    opaque_trace_id = trace_id(conv, metadata_key=metadata_key)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": opaque_trace_id,
        "source": {"family": source_family(source), "variant": source},
        "session": session,
        "project": project,
        "models": _ordered_models(conv, messages),
        "counts": {
            "messages": len(messages),
            "roles": dict(role_counts),
            "assistant_chars": assistant_chars,
            "total_chars": total_chars,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "code_contexts": code_contexts,
            "diffs": diffs,
        },
        "flags": {
            "has_tools": bool(tool_calls or tool_results),
            "has_code_context": bool(code_contexts),
            "has_diffs": bool(diffs),
            "complete": complete,
        },
        "source_metadata": _source_metadata(conv),
        "messages": build_message_metadata(messages, namespace=opaque_trace_id),
    }

    if conversation_tool_events:
        metadata["tools"] = conversation_tool_events

    token_usage = _token_usage(conv, messages)
    if token_usage:
        metadata["token_usage"] = token_usage
    performance = _performance(conv)
    if performance:
        metadata["performance"] = performance

    parent_id = _first(conv, "parent_session_id", "forked_from_id")
    if parent_id:
        metadata["lineage"] = {"has_parent": True}
    return metadata
