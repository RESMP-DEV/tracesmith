from __future__ import annotations

import pytest

from tracesmith.config import ExportConfig
from tracesmith.export.filters import dedup_key, filter_reason, passes


def conversation(**overrides):
    value = {
        "source": "cursor-chat",
        "session_id": "session-1",
        "project_name": "tracesmith",
        "project_path": "/work/tracesmith",
        "status": "completed",
        "start_time": "2026-07-10T12:00:00Z",
        "messages": [
            {"role": "user", "content": "please fix"},
            {
                "role": "assistant",
                "content": "fixed",
                "model": "gpt-5.5",
                "tool_use": {"name": "edit", "input": {"path": "x.py"}},
                "suggested_diffs": [{"diff": "private"}],
            },
        ],
    }
    value.update(overrides)
    return value


def test_drop_sources_matches_canonical_family() -> None:
    assert filter_reason(
        conversation(), ExportConfig(drop_sources=["cursor"])
    ) == "source_dropped"


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (ExportConfig(include_sources=["codex"]), "source_not_included"),
        (ExportConfig(projects=["other*"]), "project_not_included"),
        (ExportConfig(models=["claude-*"]), "model_not_included"),
        (ExportConfig(statuses=["failed"]), "status_not_included"),
        (ExportConfig(since="2026-07-11"), "before_since"),
        (ExportConfig(until="2026-07-09"), "after_until"),
        (ExportConfig(min_turns=3), "below_min_turns"),
        (ExportConfig(max_turns=1), "above_max_turns"),
        (ExportConfig(min_assistant_chars=10), "below_min_assistant_chars"),
    ],
)
def test_filter_reasons_are_auditable(config: ExportConfig, reason: str) -> None:
    assert filter_reason(conversation(), config) == reason


def test_metadata_backed_filters_pass_matching_trace() -> None:
    config = ExportConfig(
        include_sources=["cursor"],
        projects=["*tracesmith*"],
        models=["gpt-*"],
        statuses=["complete*"],
        since="2026-07-10",
        until="2026-07-11",
        require_tools=True,
        require_diffs=True,
    )
    assert passes(conversation(), config)


def test_tool_and_diff_requirements_report_separately() -> None:
    no_tools = conversation(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ])
    assert filter_reason(no_tools, ExportConfig(require_tools=True)) == "tools_required"
    assert filter_reason(no_tools, ExportConfig(require_diffs=True)) == "diffs_required"


def test_completed_status_filter_uses_derived_lifecycle_when_source_omits_status() -> None:
    conv = conversation()
    conv.pop("status")
    assert passes(conv, ExportConfig(statuses=["completed"]))


def test_dedup_session_ids_are_namespaced_by_source() -> None:
    cursor = conversation(source="cursor-chat", session_id="same")
    codex = conversation(source="codex", session_id="same")
    assert dedup_key(cursor) != dedup_key(codex)
