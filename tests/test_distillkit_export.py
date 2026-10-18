from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.conftest import write_jsonl
from tracesmith.cli import cli
from tracesmith.distillkit import build_distillkit_dataset, verify_distillkit_dataset


def test_builds_deterministic_train_eval_dataset(tmp_path: Path) -> None:
    """Splitting keeps every row of a conversation in one dataset split."""
    pytest.importorskip("datasets")
    from datasets import load_from_disk

    source = tmp_path / "export"
    rows = [
        {
            "conversations": [{"from": "user", "value": "prompt a-0"}],
            "metadata": {"trace_id": 0},
        },
        {
            "conversations": [{"from": "user", "value": "prompt a-1"}],
            "metadata": {"trace_id": 0},
        },
        {
            "conversations": [{"from": "user", "value": "prompt b"}],
            "metadata": {"trace_id": "trace-b"},
        },
        {
            "conversations": [{"from": "user", "value": "prompt c"}],
            "metadata": {"trace_id": "trace-c"},
        },
    ]
    write_jsonl(source / "sharegpt.jsonl", rows)
    output = tmp_path / "dataset"

    receipt = build_distillkit_dataset(
        source,
        output,
        variant="sharegpt",
        eval_fraction=0.2,
        seed=17,
    )

    dataset = load_from_disk(str(output))
    verified = verify_distillkit_dataset(output)
    assert set(dataset) == {"train", "eval"}
    assert len(dataset["train"]) + len(dataset["eval"]) == 4
    assert receipt["train_conversations"] == 2
    assert receipt["eval_conversations"] == 1
    assert verified["source_sha256"] == receipt["source_sha256"]
    source_path = source / "sharegpt.jsonl"
    assert (
        receipt["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    assert dataset["train"].column_names == ["conversations"]
    split_groups = {
        split: {
            row["conversations"][0]["value"].split(" ")[1].split("-")[0]
            for row in dataset[split]
        }
        for split in ("train", "eval")
    }
    assert not split_groups["train"] & split_groups["eval"]


def test_rejects_missing_variant_field(tmp_path: Path) -> None:
    """Reject exports that do not match the requested variant schema."""
    source = tmp_path / "export"
    write_jsonl(source / "sharegpt.jsonl", [{"messages": []}])

    with pytest.raises(ValueError, match="missing required field"):
        build_distillkit_dataset(
            source,
            tmp_path / "dataset",
            variant="sharegpt",
            eval_fraction=0.2,
            seed=17,
        )


def test_rejects_missing_trace_id(tmp_path: Path) -> None:
    """Reject data that cannot preserve conversation-level split boundaries."""
    export_dir = tmp_path / "export"
    write_jsonl(
        export_dir / "sharegpt.jsonl",
        [{"conversations": [{"from": "user", "value": "prompt"}]}],
    )

    with pytest.raises(ValueError, match="metadata.trace_id"):
        build_distillkit_dataset(
            export_dir,
            tmp_path / "dataset",
            variant="sharegpt",
            eval_fraction=0.2,
            seed=17,
        )


def test_rejects_malformed_json_row_type(tmp_path: Path) -> None:
    """Reject malformed JSON before it reaches the Arrow serializer."""
    source = tmp_path / "export"
    source.mkdir()
    (source / "sharegpt.jsonl").write_text('"conversations"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="sharegpt.jsonl:1"):
        build_distillkit_dataset(
            source,
            tmp_path / "dataset",
            variant="sharegpt",
            eval_fraction=0.2,
            seed=17,
        )


def test_eval_fraction_floors_small_group_counts(tmp_path: Path) -> None:
    """Avoid banker's rounding when the corpus has only three conversations."""
    pytest.importorskip("datasets")

    source = tmp_path / "export"
    rows = [
        {
            "conversations": [{"from": "user", "value": f"prompt {index}"}],
            "metadata": {"trace_id": f"trace-{index}"},
        }
        for index in range(3)
    ]
    write_jsonl(source / "sharegpt.jsonl", rows)

    receipt = build_distillkit_dataset(
        source,
        tmp_path / "dataset",
        variant="sharegpt",
        eval_fraction=0.5,
        seed=17,
    )

    assert receipt["eval_conversations"] == 1
    assert receipt["train_conversations"] == 2


def test_missing_source_has_actionable_error(tmp_path: Path) -> None:
    """A missing export produces an actionable error before dataset loading."""
    with pytest.raises(FileNotFoundError, match="tracesmith export"):
        build_distillkit_dataset(
            tmp_path,
            tmp_path / "dataset",
            variant="sharegpt",
            eval_fraction=0.2,
            seed=17,
        )


def test_rejects_existing_output_before_writing(tmp_path: Path) -> None:
    """A failed build cannot overwrite a previously accepted dataset."""
    export_dir = tmp_path / "export"
    write_jsonl(
        export_dir / "sharegpt.jsonl",
        [
            {
                "conversations": [{"from": "user", "value": "prompt"}],
                "metadata": {"trace_id": "trace-a"},
            },
            {
                "conversations": [{"from": "assistant", "value": "reply"}],
                "metadata": {"trace_id": "trace-b"},
            },
        ],
    )
    output = tmp_path / "dataset"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output already exists"):
        build_distillkit_dataset(
            export_dir,
            output,
            variant="sharegpt",
            eval_fraction=0.2,
            seed=17,
        )


def test_distillkit_cli_reports_input_errors(tmp_path: Path) -> None:
    """Expected builder failures produce Click errors rather than tracebacks."""
    result = CliRunner().invoke(
        cli,
        [
            "distillkit",
            "--in",
            str(tmp_path),
            "--out",
            str(tmp_path / "dataset"),
        ],
    )

    assert result.exit_code != 0
    assert "Expected export file not found" in result.output
