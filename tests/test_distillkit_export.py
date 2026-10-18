from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.conftest import write_jsonl
from tracesmith.distillkit import build_distillkit_dataset


def test_builds_deterministic_train_eval_dataset(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    from datasets import load_from_disk

    source = tmp_path / "export"
    rows = [
        {
            "conversations": [{"from": "user", "value": "prompt a-0"}],
            "metadata": {"trace_id": "trace-a"},
        },
        {
            "conversations": [{"from": "user", "value": "prompt a-1"}],
            "metadata": {"trace_id": "trace-a"},
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
    assert set(dataset) == {"train", "eval"}
    assert len(dataset["train"]) + len(dataset["eval"]) == 4
    assert receipt["train_conversations"] == 2
    assert receipt["eval_conversations"] == 1
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
