from __future__ import annotations

from click.testing import CliRunner

from agent_trace_share.cli import cli
from tests.conftest import write_jsonl, make_conversation, make_message


def test_run_chains_all_three_stages(tmp_path, monkeypatch):
    # Pre-stage raw_extracted/ with one harmless conversation.
    # (We bypass real extraction by faking HOME so claude_code finds nothing,
    # then manually drop in a raw file.)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    out = tmp_path / "out"
    raw_dir = out / "raw_extracted"
    write_jsonl(raw_dir / "claude_code.jsonl", [
        make_conversation(messages=[make_message("user", "x"), make_message("assistant", "y")]),
    ])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "redact", "--in", str(raw_dir), "--out", str(out / "redacted"),
        "--user", "tester", "--home", str(fake_home),
    ])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, [
        "export", "--in", str(out / "redacted"), "--out", str(out / "export"),
        "--variant", "both",
    ])
    assert result.exit_code == 0, result.output

    assert (out / "export" / "messages.jsonl").exists()
    assert (out / "export" / "sharegpt.jsonl").exists()
