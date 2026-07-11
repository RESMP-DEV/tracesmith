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
    tool_results: list[dict] = []
    turn_metadata: dict = {}
    performance: dict = {}
    token_usage: dict = {}
    diff_count = 0
    message_origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    message_indexes: dict[tuple[str, str], int] = {}

    def append_message(
        role: str, content: object, timestamp: object, *, origin: str, **metadata
    ) -> None:
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in {"input_text", "output_text", "text"}
            )
        text = str(content or "").strip()
        key = (role, text)
        if not text:
            return
        if message_origins[key] and origin not in message_origins[key]:
            message_origins[key].add(origin)
            existing = messages[message_indexes[key]]
            existing.update(
                (name, value)
                for name, value in metadata.items()
                if value is not None and value != "" and not existing.get(name)
            )
            return
        message_origins[key].add(origin)
        message = {"role": role, "content": text, "timestamp": timestamp}
        message.update(
            (name, value) for name, value in metadata.items()
            if value is not None and value != ""
        )
        message_indexes[key] = len(messages)
        messages.append(message)

    with open(session_file, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                event_type = obj.get('type')

                if event_type == 'session_meta':
                    session_meta = obj.get('payload', {})

                elif event_type == 'turn_context':
                    payload = obj.get('payload', {})
                    turn_metadata.update({
                        'model': payload.get('model'),
                        'reasoning_effort': payload.get('effort'),
                        'approval_policy': payload.get('approval_policy'),
                        'collaboration_mode': (
                            payload.get('collaboration_mode', {}).get('mode')
                            if isinstance(payload.get('collaboration_mode'), dict)
                            else payload.get('collaboration_mode')
                        ),
                        'permission_profile': (
                            payload.get('permission_profile', {}).get('type')
                            if isinstance(payload.get('permission_profile'), dict)
                            else payload.get('permission_profile')
                        ),
                        'sandbox_profile': (
                            payload.get('sandbox_policy', {}).get('type')
                            if isinstance(payload.get('sandbox_policy'), dict)
                            else payload.get('sandbox_policy')
                        ),
                    })

                elif event_type == 'event_msg':
                    payload = obj.get('payload', {})
                    payload_type = payload.get('type')

                    if payload_type == 'user_message':
                        append_message(
                            'user', payload.get('message'), obj.get('timestamp'),
                            origin='event_msg',
                            context=payload.get('context'),
                            id=payload.get('id'),
                        )

                    elif payload_type == 'agent_message':
                        append_message(
                            'assistant', payload.get('message'), obj.get('timestamp'),
                            origin='event_msg',
                            model=payload.get('model') or turn_metadata.get('model'),
                            id=payload.get('id'),
                            phase=payload.get('phase'),
                        )

                    elif payload_type == 'task_started':
                        if payload.get('started_at') is not None:
                            performance.setdefault('started_at', payload['started_at'])
                        turn_metadata['turn_count'] = turn_metadata.get('turn_count', 0) + 1
                        if payload.get('model_context_window') is not None:
                            turn_metadata['context_window'] = payload['model_context_window']

                    elif payload_type == 'task_complete':
                        duration_ms = payload.get('duration_ms')
                        if isinstance(duration_ms, (int, float)):
                            performance['total_turn_duration_ms'] = (
                                performance.get('total_turn_duration_ms', 0) + duration_ms
                            )
                        ttft = payload.get('time_to_first_token_ms')
                        if isinstance(ttft, (int, float)):
                            performance['min_time_to_first_token_ms'] = min(
                                performance.get('min_time_to_first_token_ms', ttft), ttft
                            )
                            performance['max_time_to_first_token_ms'] = max(
                                performance.get('max_time_to_first_token_ms', ttft), ttft
                            )
                        turn_metadata['status'] = 'completed'

                    elif payload_type == 'turn_aborted':
                        duration_ms = payload.get('duration_ms')
                        if isinstance(duration_ms, (int, float)):
                            performance['total_turn_duration_ms'] = (
                                performance.get('total_turn_duration_ms', 0) + duration_ms
                            )
                        turn_metadata['status'] = 'aborted'

                    elif payload_type == 'token_count':
                        info = payload.get('info') or {}
                        token_usage = info.get('total_token_usage') or token_usage
                        if info.get('model_context_window') is not None:
                            turn_metadata['context_window'] = info['model_context_window']

                    elif payload_type == 'tool_use':
                        # Code execution, file edits, etc.
                        tool_use = {
                            'type': 'tool_use',
                            'tool': payload.get('tool'),
                            'input': payload.get('input'),
                            'timestamp': obj.get('timestamp')
                        }
                        tool_results.append(tool_use)

                    elif payload_type == 'tool_result':
                        # Results from tool execution (diffs, outputs, etc.)
                        tool_result = {
                            'type': 'tool_result',
                            'tool': payload.get('tool'),
                            'output': payload.get('output'),
                            'timestamp': obj.get('timestamp')
                        }
                        tool_results.append(tool_result)

                    elif payload_type == 'diff':
                        # Code diffs
                        diff = {
                            'type': 'diff',
                            'file': payload.get('file'),
                            'diff': payload.get('diff'),
                            'timestamp': obj.get('timestamp')
                        }
                        tool_results.append(diff)

                    elif payload_type == 'patch_apply_end':
                        changes = payload.get('changes')
                        if isinstance(changes, (dict, list)):
                            diff_count += len(changes)
                        elif payload.get('success'):
                            diff_count += 1

                elif event_type == 'response_item':
                    payload = obj.get('payload', {})
                    payload_type = payload.get('type')
                    if payload_type in {'message', 'agent_message'}:
                        role = payload.get('role', 'assistant')
                        if role in {'assistant', 'user'}:
                            append_message(
                                role,
                                payload.get('content'),
                                obj.get('timestamp'),
                                origin='response_item',
                                id=payload.get('id'),
                                phase=payload.get('phase'),
                                model=turn_metadata.get('model') if role == 'assistant' else None,
                            )
                    elif payload_type in {'function_call', 'custom_tool_call'}:
                        tool_results.append({
                            'type': payload_type,
                            'tool': payload.get('name'),
                            'call_id': payload.get('call_id') or payload.get('id'),
                            'status': payload.get('status'),
                            'input': payload.get('arguments', payload.get('input')),
                            'timestamp': obj.get('timestamp'),
                        })
                    elif payload_type in {'function_call_output', 'custom_tool_call_output'}:
                        tool_results.append({
                            'type': payload_type,
                            'call_id': payload.get('call_id') or payload.get('id'),
                            'output': payload.get('output'),
                            'timestamp': obj.get('timestamp'),
                        })

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
            'version': session_meta.get('cli_version'),
            'model_provider': session_meta.get('model_provider'),
            'originator': session_meta.get('originator'),
            'agent_name': session_meta.get('agent_nickname'),
            'parent_session_id': (
                session_meta.get('parent_thread_id') or session_meta.get('forked_from_id')
            ),
        }

        for key, value in turn_metadata.items():
            if value is not None and value != '':
                conv[key] = value
        for key, value in performance.items():
            if value is not None and value != '':
                conv[key] = value
        if token_usage:
            conv['token_usage'] = token_usage
        if diff_count:
            conv['diff_count'] = diff_count
            conv['files_changed_count'] = diff_count

        if tool_results:
            conv['tool_results'] = tool_results

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
