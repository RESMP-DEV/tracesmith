from __future__ import annotations

from agent_trace_share.publish.dataset_card import render_card


def test_card_has_yaml_frontmatter():
    manifest = {"sources": {"claude_code": {"exported_messages": 10, "exported_sharegpt_pairs": 22}},
                "config": {"gitleaks": True}}
    card = render_card(manifest, repo_id="user/my-traces")
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert "user/my-traces" in card
    assert "claude_code" in card
    assert "| 10 |" in card
    assert "| 22 |" in card
