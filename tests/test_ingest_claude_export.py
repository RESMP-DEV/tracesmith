import json

from click.testing import CliRunner

from tracesmith.cli import cli
from tracesmith.config import ExportConfig, RedactorConfig
from tracesmith.export.messages import export_messages
from tracesmith.ingest.claude_export import ingest_claude_export
from tracesmith.redact.pipeline import run_redact


def write_trace(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_ingest_claude_export_preserves_training_fields(tmp_path):
    source = tmp_path / "source"
    records = [
        {"type": "custom-title", "customTitle": "linear decode"},
        {
            "type": "user",
            "timestamp": "2026-10-18T01:00:00Z",
            "message": {"content": "[init] session start"},
        },
        {
            "type": "user",
            "timestamp": "2026-10-18T01:00:01Z",
            "uuid": "u1",
            "parentUuid": "u0",
            "cwd": "/home/alice/project",
            "message": {
                "content": [
                    {"type": "text", "text": "run the kernel"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "3 ms"}],
                    },
                ]
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-18T01:00:02Z",
            "uuid": "a1",
            "message": {
                "id": "m1",
                "model": "claude-opus-5",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                },
                "content": [
                    {"type": "thinking", "thinking": "measure first"},
                    {"type": "text", "text": "use tiled stores"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"cmd": "nvbandwidth"},
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-18T05:00:00Z",
            "isSidechain": True,
            "message": {"content": [{"type": "text", "text": "hidden"}]},
        },
    ]
    write_trace(source / "b200" / "trace.jsonl", records)
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        [
            "import-claude",
            "--source-dir",
            str(source),
            "--out",
            str(out),
            "--source-name",
            "infatoshi_mega",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (out / "raw_extracted/infatoshi_mega.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 1
    conversation = rows[0]
    assert conversation["source"] == "infatoshi_mega"
    assert conversation["session_id"] == "b200/trace"
    assert conversation["name"] == "linear decode"
    assert conversation["created_at"] == "2026-10-18T01:00:00Z"
    assert conversation["project_path"] == "/home/alice/project"
    assert conversation["token_usage"] == {
        "input": 10,
        "output": 5,
        "cached_input": 3,
        "cache_write": 2,
    }
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assert "run the kernel" in conversation["messages"][0]["content"]
    assert (
        '<tool_result id="t1">3 ms</tool_result>'
        in conversation["messages"][0]["content"]
    )
    assert conversation["messages"][0]["id"] == "u1"
    assert conversation["messages"][0]["parent_id"] == "u0"
    assistant = conversation["messages"][1]
    assert (
        "<reasoning_summary>measure first</reasoning_summary>" in assistant["content"]
    )
    assert assistant["model"] == "claude-opus-5"
    assert assistant["tool_uses"][0]["name"] == "Bash"
    assert assistant["timestamp"] == "2026-10-18T01:00:02Z"

    receipt = json.loads((out / "infatoshi_mega.IMPORT_RECEIPT.json").read_text())
    assert receipt["files_scanned"] == 1
    assert receipt["files_failed"] == 0
    assert receipt["conversations"] == 1
    assert len(receipt["source_manifest_sha256"]) == 64


def test_ingest_claude_export_records_bad_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    good = [
        {"type": "user", "message": {"content": "hello"}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "hi"}],
            },
        },
    ]
    write_trace(source / "good.jsonl", good)
    with (source / "good.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{truncated\n")
    (source / "bad.jsonl").write_text("{not-json\n", encoding="utf-8")
    write_trace(source / "semantic.jsonl", [{"type": "user", "message": ["bad"]}])
    write_trace(
        source / "lone-assistant.jsonl",
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hi"}]},
            }
        ],
    )
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        [
            "import-claude",
            "--source-dir",
            str(source),
            "--out",
            str(out),
            "--source-name",
            "public",
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads((out / "public.IMPORT_RECEIPT.json").read_text())
    assert receipt["files_scanned"] == 4
    assert receipt["files_failed"] == 1
    assert receipt["files_skipped"] == 2
    assert receipt["lines_failed"] == 2
    assert receipt["failures"][0]["path"] == "semantic.jsonl"
    assert receipt["conversations"] == 1

    rerun = CliRunner().invoke(
        cli,
        [
            "import-claude",
            "--source-dir",
            str(source),
            "--out",
            str(out),
            "--source-name",
            "public",
        ],
    )
    assert rerun.exit_code != 0
    assert "already exists" in rerun.output


def test_imported_claude_export_feeds_redact_and_export(tmp_path):
    source = tmp_path / "source"
    records = [
        {"type": "user", "message": {"content": "optimize the kernel"}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-5",
                "content": [
                    {"type": "thinking", "thinking": "profile first"},
                    {"type": "text", "text": "measure bandwidth"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"cmd": "nvprof"},
                    },
                ],
            },
        },
    ]
    write_trace(source / "trace.jsonl", records)
    out = tmp_path / "pipeline"
    ingest_claude_export(source, out / "raw_extracted/public.jsonl", "public")

    redact_report = run_redact(
        out / "raw_extracted", out / "redacted", RedactorConfig()
    )
    export_summary = export_messages(
        out / "redacted",
        out / "export/messages.jsonl",
        ExportConfig(variant="messages"),
    )
    row = json.loads((out / "export/messages.jsonl").read_text())

    assert "public" in redact_report["per_source"]
    assert export_summary["rows"] == 1
    assert row["metadata"]["source"] == "public"
    assert row["metadata"]["models"] == ["claude-opus-5"]
    flattened = "\n".join(message["content"] for message in row["messages"])
    assert "<reasoning_summary>profile first</reasoning_summary>" in flattened
    assert '<tool_use name="Bash">' in flattened


def test_import_claude_rejects_unsafe_source_name(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        [
            "import-claude",
            "--source-dir",
            str(source),
            "--out",
            str(out),
            "--source-name",
            "../public",
        ],
    )

    assert result.exit_code != 0
    assert not (out / "raw_extracted").exists()


def test_import_claude_accepts_tool_first_session(tmp_path):
    source = tmp_path / "source"
    records = [
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "4 ms"}
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "tune tiles"}],
            },
        },
    ]
    write_trace(source / "trace.jsonl", records)
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        [
            "import-claude",
            "--source-dir",
            str(source),
            "--out",
            str(out),
            "--source-name",
            "public",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (out / "raw_extracted/public.jsonl").read_text().splitlines()
    ]
    assert [message["role"] for message in rows[0]["messages"]] == ["user", "assistant"]
    assert (
        '<tool_result id="t1">4 ms</tool_result>' in rows[0]["messages"][0]["content"]
    )
