"""Codex extractor.

Vendored from 0xSero/ai-data-extraction/extract_codex.py (MIT).
Refactored to the Extractor protocol; discovery + parsing logic preserved.
"""
from __future__ import annotations

import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from tracesmith.extract.base import Conversation


def find_codex_installations() -> list[Path]:
    """Find all Codex installation directories."""
    system = platform.system()
    home = Path.home()

    locations: list[Path] = []

    # Search patterns for Codex directories
    codex_patterns = [
        'codex', 'codex-local', '.codex', '.codex-local'
    ]

    if system == "Darwin":  # macOS
        base_dirs = [
            home / "Library/Application Support",
            home / ".config",
            home
        ]
    elif system == "Linux":
        base_dirs = [
            home / ".config",
            home / ".local/share",
            home
        ]
    elif system == "Windows":
        base_dirs = [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')),
            Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local')),
            home
        ]
    else:
        base_dirs = [home / ".config", home]

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        for pattern in codex_patterns:
            codex_dir = base_dir / pattern
            if codex_dir.exists():
                locations.append(codex_dir)

    return list({p.resolve() for p in locations})


def extract_codex_session(session_file: Path) -> Conversation | None:
    """Extract conversation from a Codex rollout file with full context."""
    messages: list[dict] = []
    session_meta: dict = {}
    model: str | None = None
    status: str | None = None
    updated_at: str | None = None
    tool_call_count = 0
    tool_result_count = 0
    diff_count = 0
    tool_events: list[dict] = []
    token_usage: dict[str, int | float] = {}
    message_origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    message_indexes: dict[tuple[str, str], int] = {}
    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    seen_diffs: set[str] = set()

    def first_event(seen: set[str], payload: dict) -> bool:
        identity = next(
            (
                payload.get(key)
                for key in ("call_id", "id", "tool_use_id")
                if payload.get(key) not in (None, "")
            ),
            None,
        )
        if identity is None:
            return True
        normalized = str(identity)
        if normalized in seen:
            return False
        seen.add(normalized)
        return True

    def append_message(
        role: str, content: object, timestamp: object, *, origin: str,
        message_model: object = None,
    ) -> None:
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
                and item.get("type") in {"input_text", "output_text", "text"}
            )
        text = str(content or "").strip()
        if not text:
            return
        key = (role, text)
        index = message_indexes.get(key)
        if (
            index is not None
            and index == len(messages) - 1
            and origin not in message_origins[key]
        ):
            message_origins[key].add(origin)
            existing = messages[index]
            if message_model and not existing.get("model"):
                existing["model"] = message_model
            return
        message_origins[key] = {origin}
        message = {"role": role, "content": text, "timestamp": timestamp}
        if message_model:
            message["model"] = message_model
        message_indexes[key] = len(messages)
        messages.append(message)

    with open(session_file, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                event_type = obj.get('type')
                if isinstance(obj.get('timestamp'), str):
                    updated_at = obj['timestamp']

                if event_type == 'session_meta':
                    session_meta = obj.get('payload', {})

                elif event_type == 'turn_context':
                    payload = obj.get('payload', {})
                    if isinstance(payload.get('model'), str):
                        model = payload['model']

                elif event_type == 'event_msg':
                    payload = obj.get('payload', {})
                    payload_type = payload.get('type')

                    if payload_type == 'user_message':
                        append_message(
                            'user', payload.get('message'), obj.get('timestamp'),
                            origin='event_msg',
                        )

                    elif payload_type == 'agent_message':
                        append_message(
                            'assistant', payload.get('message'), obj.get('timestamp'),
                            origin='event_msg',
                            message_model=payload.get('model') or model,
                        )

                    elif payload_type == 'tool_use':
                        if first_event(seen_tool_calls, payload):
                            tool_call_count += 1
                        tool_events.append({
                            'type': 'tool_use',
                            'tool': payload.get('tool'),
                            'input': payload.get('input'),
                            'timestamp': obj.get('timestamp'),
                        })

                    elif payload_type == 'tool_result':
                        if first_event(seen_tool_results, payload):
                            tool_result_count += 1
                        tool_events.append({
                            'type': 'tool_result',
                            'tool': payload.get('tool'),
                            'output': payload.get('output'),
                            'timestamp': obj.get('timestamp'),
                        })

                    elif payload_type == 'diff':
                        if first_event(seen_diffs, payload):
                            diff_count += 1
                        tool_events.append({
                            'type': 'diff',
                            'file': payload.get('file'),
                            'diff': payload.get('diff'),
                            'timestamp': obj.get('timestamp'),
                        })

                    elif payload_type == 'patch_apply_end':
                        changes = payload.get('changes')
                        if (
                            isinstance(changes, (dict, list))
                            and changes
                            and first_event(seen_diffs, payload)
                        ):
                            diff_count += 1

                    elif payload_type == 'task_complete':
                        status = 'completed'

                    elif payload_type == 'turn_aborted':
                        status = 'aborted'

                    elif payload_type == 'token_count':
                        info = payload.get('info')
                        totals = info.get('total_token_usage') if isinstance(info, dict) else None
                        if isinstance(totals, dict):
                            for source_key, target_key in (
                                ('input_tokens', 'input'),
                                ('cached_input_tokens', 'cached_input'),
                                ('output_tokens', 'output'),
                                ('reasoning_output_tokens', 'reasoning'),
                                ('total_tokens', 'total'),
                            ):
                                value = totals.get(source_key)
                                if isinstance(value, (int, float)):
                                    token_usage[target_key] = value

                elif event_type == 'response_item':
                    payload = obj.get('payload', {})
                    payload_type = payload.get('type')
                    if payload_type == 'message' and payload.get('role') in {'user', 'assistant'}:
                        role = payload['role']
                        append_message(
                            role, payload.get('content'), obj.get('timestamp'),
                            origin='response_item',
                            message_model=model if role == 'assistant' else None,
                        )
                    elif payload_type in {'function_call', 'custom_tool_call'}:
                        if first_event(seen_tool_calls, payload):
                            tool_call_count += 1
                    elif payload_type in {'function_call_output', 'custom_tool_call_output'}:
                        if first_event(seen_tool_results, payload):
                            tool_result_count += 1

            except json.JSONDecodeError:
                continue

    if messages:
        conv: Conversation = {
            'messages': messages,
            'session_id': session_meta.get('id'),
            'cwd': session_meta.get('cwd'),
            'project_path': session_meta.get('cwd'),
            'source': 'codex',
            'session_file': str(session_file),
            'timestamp': session_meta.get('timestamp'),
            'created_at': session_meta.get('timestamp'),
            'updated_at': updated_at,
            'version': session_meta.get('cli_version'),
        }

        if model:
            conv['model'] = model
        if status:
            conv['status'] = status
        if token_usage:
            conv['token_usage'] = token_usage
        if tool_call_count:
            conv['tool_call_count'] = tool_call_count
        if tool_result_count:
            conv['tool_result_count'] = tool_result_count
        if diff_count:
            conv['diff_count'] = diff_count
        if tool_events:
            conv['tool_results'] = tool_events

        return conv

    return None


def find_all_codex_sessions(installation: Path) -> list[Path]:
    """Find all Codex session files in an installation."""
    session_files: list[Path] = []

    # Check for sessions directory
    sessions_dir = installation / 'sessions'
    if sessions_dir.exists():
        # Sessions are organized by date: YYYY/MM/DD/rollout-*.jsonl
        session_files.extend(list(sessions_dir.rglob('rollout-*.jsonl')))

    # Also check for project-based structure
    projects_dir = installation / 'projects'
    if projects_dir.exists():
        session_files.extend(list(projects_dir.rglob('*.jsonl')))

    return session_files


def extract_codex_conversations(installation: Path) -> list[Conversation]:
    """Extract all conversations from a Codex installation."""
    conversations: list[Conversation] = []
    for session_file in find_all_codex_sessions(installation):
        conv = extract_codex_session(session_file)
        if conv:
            conv['installation'] = str(installation)
            conversations.append(conv)
    return conversations


class CodexExtractor:
    source = "codex"

    def find_installations(self) -> list[Path]:
        return find_codex_installations()

    def extract(self, install_dir: Path) -> Iterator[Conversation]:
        for conv in extract_codex_conversations(install_dir):
            yield conv
