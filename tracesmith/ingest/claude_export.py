"""Import externally published Claude Code JSONL traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracesmith import __version__
from tracesmith.extract.base import Conversation, Message

_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") == "text":
        return str(value.get("text") or "")
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _tool_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_call_id": item.get("tool_use_id"),
        "status": "error" if item.get("is_error") else "completed",
        "output": _text(item.get("content")),
    }


def _user(record: dict[str, Any]) -> list[Message]:
    message = record.get("message") or {}
    if not isinstance(message, dict):
        raise TypeError("message is not an object")
    content = message.get("content")
    if isinstance(content, str):
        text, blocks = content, []
    else:
        blocks = [item for item in content or [] if isinstance(item, dict)]
        text = "\n".join(
            str(item.get("text") or "") for item in blocks if item.get("type") == "text"
        )
        results = [
            _tool_result(item) for item in blocks if item.get("type") == "tool_result"
        ]

    messages: list[Message] = []
    if text and not text.startswith("[init] session start"):
        user: Message = {"role": "user", "content": text}
        if isinstance(record.get("timestamp"), str):
            user["timestamp"] = record["timestamp"]
        if record.get("uuid"):
            user["id"] = str(record["uuid"])
        if record.get("parentUuid"):
            user["parent_id"] = str(record["parentUuid"])
        messages.append(user)
    if blocks and any(item.get("type") == "tool_result" for item in blocks):
        tool: Message = {"role": "tool", "content": "", "tool_results": results}
        if isinstance(record.get("timestamp"), str):
            tool["timestamp"] = record["timestamp"]
        messages.append(tool)
    return messages


def _assistant(record: dict[str, Any]) -> Message | None:
    message = record.get("message") or {}
    if not isinstance(message, dict):
        raise TypeError("message is not an object")
    content = message.get("content")
    if isinstance(content, str):
        parts = [content]
        blocks = []
    else:
        blocks = [item for item in content or [] if isinstance(item, dict)]
        parts = []
        for item in blocks:
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif item.get("type") == "thinking":
                thinking = item.get("thinking") or ""
                parts.append(f"<reasoning_summary>{thinking}</reasoning_summary>")
    tool_uses = [item for item in blocks if item.get("type") == "tool_use"]
    content = "\n".join(part for part in parts if part)
    if not content and not tool_uses:
        return None

    result: Message = {"role": "assistant", "content": content}
    if message.get("model"):
        result["model"] = str(message["model"])
    if message.get("id") or record.get("uuid"):
        result["id"] = str(message.get("id") or record.get("uuid"))
    if tool_uses:
        result["tool_uses"] = tool_uses
    if isinstance(record.get("timestamp"), str):
        result["timestamp"] = record["timestamp"]
    if record.get("parentUuid"):
        result["parent_id"] = str(record["parentUuid"])
    return result


def convert_file(
    path: Path, relative: Path, source: str
) -> tuple[Conversation | None, int]:
    """Convert one Claude Code export file without loading sidechain records."""
    messages: list[Message] = []
    timestamps: list[str] = []
    project_path: str | None = None
    token_usage: dict[str, int | float] = {}
    bad_lines = 0
    name = path.stem
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(record, dict):
                continue
            if record.get("isSidechain"):
                continue
            if isinstance(record.get("timestamp"), str):
                timestamps.append(record["timestamp"])
            if isinstance(record.get("cwd"), str) and record["cwd"]:
                project_path = record["cwd"]
            kind = record.get("type")
            if kind == "custom-title" and record.get("customTitle"):
                name = str(record["customTitle"])
            elif kind == "user":
                messages.extend(_user(record))
            elif kind == "assistant":
                assistant = _assistant(record)
                if assistant is not None:
                    messages.append(assistant)
                    raw_message = record.get("message")
                    usage = (
                        raw_message.get("usage")
                        if isinstance(raw_message, dict)
                        else None
                    )
                    if isinstance(usage, dict):
                        for source_key, target_key in (
                            ("input_tokens", "input"),
                            ("output_tokens", "output"),
                            ("cache_read_input_tokens", "cached_input"),
                            ("cache_creation_input_tokens", "cache_write"),
                        ):
                            value = usage.get(source_key)
                            if isinstance(value, (int, float)):
                                token_usage[target_key] = (
                                    token_usage.get(target_key, 0) + value
                                )

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 2 or messages[0]["role"] != "user":
        return None, bad_lines

    session_id = relative.with_suffix("").as_posix()
    conversation: Conversation = {
        "messages": messages,
        "source": source,
        "name": name,
        "session_id": session_id,
        "source_file": session_id + ".jsonl",
    }
    if project_path:
        conversation["project_path"] = project_path
    if any(token_usage.values()):
        conversation["token_usage"] = token_usage
    if timestamps:
        conversation["created_at"] = min(timestamps)
        conversation["updated_at"] = max(timestamps)
    return conversation, bad_lines


def ingest_claude_export(
    source_dir: Path, out_file: Path, source: str
) -> dict[str, Any]:
    """Stream normalized conversations and write a source receipt."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    if not _SOURCE_ID.fullmatch(source) or source in {".", ".."}:
        raise ValueError(
            "source must contain only letters, digits, dot, underscore, or hyphen"
        )

    files = sorted(path for path in source_dir.rglob("*.jsonl") if path.is_file())
    if not files:
        raise ValueError(f"no JSONL files found under: {source_dir}")
    receipt_path = out_file.parent.parent / f"{source}.IMPORT_RECEIPT.json"
    if out_file.exists() or receipt_path.exists():
        raise FileExistsError(f"import output already exists: {out_file}")

    failed: list[dict[str, str]] = []
    manifest_digest = hashlib.sha256()
    skipped = 0
    bad_lines = 0
    conversations = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out_file.name}.", dir=out_file.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for path in files:
                relative = path.relative_to(source_dir)
                manifest_digest.update(
                    f"{relative.as_posix()}:{_sha256(path)}\n".encode()
                )
                try:
                    conversation, file_bad_lines = convert_file(path, relative, source)
                except Exception as exc:  # noqa: BLE001 - keep one bad export isolated
                    failed.append({"path": relative.as_posix(), "error": str(exc)})
                    continue
                bad_lines += file_bad_lines
                if conversation is None:
                    skipped += 1
                else:
                    conversations += 1
                    handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        if conversations == 0:
            raise ValueError("no usable conversations found")
        os.replace(temporary_name, out_file)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    receipt = {
        "schema_version": 1,
        "tracesmith_version": __version__,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_manifest_sha256": manifest_digest.hexdigest(),
        "files_scanned": len(files),
        "files_failed": len(failed),
        "files_skipped": skipped,
        "lines_failed": bad_lines,
        "failures": failed,
        "conversations": conversations,
        "output": str(out_file),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
