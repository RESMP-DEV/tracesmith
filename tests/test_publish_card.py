from __future__ import annotations

from tracesmith.publish.dataset_card import render_card


def test_card_has_yaml_frontmatter():
    manifest = {
        "sources": {"claude_code": {
            "raw_records": 7,
            "exported_messages": 6,
            "exported_sharegpt_pairs": 12,
        }},
        "export_summary": {"messages_rows": 10, "sharegpt_pairs": 22},
        "metadata_schema_version": "1.0",
        "config": {"gitleaks": True},
    }
    card = render_card(manifest, repo_id="user/my-traces")
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert "user/my-traces" in card
    assert "claude_code" in card
    assert "| claude_code | 7 | 6 | 12 |" in card
    assert "metadata schema version" in card
    assert "`1.0`" in card
    # Totals come from export_summary, not summed per-source fields.
    assert "10 conversations" in card
    assert "22 instruction pairs" in card


def test_card_shows_per_source_attribution_and_global_totals():
    manifest = {
        "sources": {
            "claude_code": {
                "raw_records": 5,
                "exported_messages": 4,
                "exported_sharegpt_pairs": 7,
            },
            "cursor": {
                "raw_records": 2,
                "exported_messages": 1,
                "exported_sharegpt_pairs": 1,
            },
        },
        "export_summary": {
            "messages_rows": 9,
            "messages_dropped_no_assistant": 1,
            "sharegpt_pairs": 8,
            "sharegpt_dropped_trailing_user": 0,
        },
        "config": {},
    }
    card = render_card(manifest, repo_id="user/traces")
    assert "9 conversations" in card
    assert "8 instruction pairs" in card
    assert "| claude_code | 5 | 4 | 7 |" in card
    assert "| cursor | 2 | 1 | 1 |" in card


def test_card_repo_url_is_resmp_dev():
    manifest = {"sources": {}, "export_summary": {}, "config": {}}
    card = render_card(manifest, repo_id="user/x")
    assert "https://github.com/resmp-dev/tracesmith" in card
    assert "your-org" not in card
    assert "kearm/agent-trace-share" not in card
