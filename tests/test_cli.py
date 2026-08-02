from __future__ import annotations

import json
from dataclasses import fields

from click.testing import CliRunner

from tests.conftest import make_conversation, make_message, write_jsonl
from tracesmith.cli import cli
from tracesmith.config import ExportConfig


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


def test_run_command_writes_manifest(tmp_path, monkeypatch):
    # Fake HOME so every extractor finds nothing.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    out = tmp_path / "out"

    # run_extract would scan a (fake) HOME and find nothing, so stub it to
    # write our staged raw file and report the count the manifest will record.
    def fake_run_extract(sources, root, out_dir):
        from pathlib import Path
        raw = Path(out_dir) / "raw_extracted"
        write_jsonl(raw / "claude_code.jsonl", [
            make_conversation(messages=[
                make_message("user", "hello"),
                make_message("assistant", "hi there"),
            ]),
        ])
        return {"claude_code": 1}

    monkeypatch.setattr(
        "tracesmith.extract.discovery.run_extract", fake_run_extract
    )

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--root", str(fake_home), "--out", str(out),
        "--variant", "both", "--user", "tester", "--home", str(fake_home),
    ])
    assert result.exit_code == 0, result.output

    manifest_path = out / "MANIFEST.json"
    assert manifest_path.exists(), result.output
    manifest = json.loads(manifest_path.read_text())

    assert manifest["tool_version"]
    assert "claude_code" in manifest["sources"]
    assert manifest["sources"]["claude_code"]["raw_records"] == 1
    # Per-source export attribution was removed (export counts are global);
    # the per-source block must carry only raw_records + redaction_counts.
    assert "exported_messages" not in manifest["sources"]["claude_code"]
    assert manifest["export_summary"]["messages_rows"] >= 1
    assert manifest["export_summary"]["sharegpt_pairs"] >= 0
    assert "export/messages.jsonl" in manifest["files"]
    assert "export/sharegpt.jsonl" in manifest["files"]
    assert manifest["config"]["variant"] == "both"
    expected_export_keys = {
        field.name for field in fields(ExportConfig)
    } - {"metadata_key", "variant"}
    assert set(manifest["config"]["export"]) == expected_export_keys
    assert "metadata_key" not in manifest["config"]["export"]
