from __future__ import annotations

import json
from pathlib import Path

from tracesmith.extract.discovery import EXTRACTORS, run_extract


def test_extractors_registry_has_claude_code():
    assert "claude_code" in EXTRACTORS
    assert EXTRACTORS["claude_code"].source == "claude_code"


def test_run_extract_writes_jsonl_one_per_source(tmp_path, monkeypatch):
    # Fake a .claude install with one session
    fake_home = tmp_path / "home"
    fake_claude = fake_home / ".claude"
    proj = fake_claude / "projects" / "p"
    proj.mkdir(parents=True)
    (proj / "s.jsonl").write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")

    monkeypatch.setenv("HOME", str(fake_home))

    out = tmp_path / "out"
    counts = run_extract(sources=["claude_code"], root=fake_home, out_dir=out)

    assert counts == {"claude_code": 1}
    out_file = out / "raw_extracted" / "claude_code.jsonl"
    assert out_file.exists()
    lines = out_file.read_text().strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["source"] == "claude_code"
    assert rec["messages"][0]["content"] == "hi"
