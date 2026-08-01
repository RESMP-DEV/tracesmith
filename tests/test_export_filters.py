from __future__ import annotations

import pytest

from tracesmith.config import ExportConfig
from tracesmith.export.filters import dedup_key, filter_reason
from tests.conftest import make_conversation, make_message


def rich_conversation() -> dict:
    return make_conversation(
        source="opencode-cli",
        project_path="/work/alpha",
        status="completed",
        created_at="2026-07-10T12:00:00Z",
        messages=[
            make_message("user", "do it"),
            make_message(
                "assistant",
                "done",
                model="model-a",
                tool_calls=[{"name": "edit"}],
                diffs=[{"file": "a.py"}],
            ),
        ],
    )


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (ExportConfig(include_sources=["codex"]), "source_not_included"),
        (ExportConfig(drop_sources=["opencode-*"]), "source_dropped"),
        (ExportConfig(projects=["*/beta"]), "project_not_included"),
        (ExportConfig(models=["model-b"]), "model_not_included"),
        (ExportConfig(statuses=["aborted"]), "status_not_included"),
        (ExportConfig(since="2026-07-11T00:00:00Z"), "before_since"),
        (ExportConfig(until="2026-07-09T00:00:00Z"), "after_until"),
        (ExportConfig(min_turns=3), "below_min_messages"),
        (ExportConfig(max_turns=1), "above_max_messages"),
    ],
)
def test_filters_report_auditable_reasons(config: ExportConfig, reason: str) -> None:
    assert filter_reason(rich_conversation(), config) == reason


def test_tool_and_diff_filters_use_normalized_message_fields() -> None:
    assert filter_reason(rich_conversation(), ExportConfig(require_tools=True)) is None
    assert filter_reason(rich_conversation(), ExportConfig(require_diffs=True)) is None
    plain = make_conversation()
    assert filter_reason(plain, ExportConfig(require_tools=True)) == "tools_required"
    assert filter_reason(plain, ExportConfig(require_diffs=True)) == "diffs_required"


def test_invalid_time_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid --since"):
        filter_reason(rich_conversation(), ExportConfig(since="not-a-time"))


def test_invalid_time_filter_is_rejected_before_other_filters() -> None:
    with pytest.raises(ValueError, match="invalid --until"):
        filter_reason(
            rich_conversation(),
            ExportConfig(include_sources=["codex"], until="not-a-time"),
        )


def test_no_session_dedup_key_is_source_namespaced() -> None:
    messages = [make_message("user", "same"), make_message("assistant", "same")]
    claude = make_conversation(source="claude_code", session_id=None, messages=messages)
    codex = make_conversation(source="codex", session_id=None, messages=messages)

    assert dedup_key(claude) != dedup_key(codex)
