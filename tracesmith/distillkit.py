"""Create local Hugging Face datasets for DistillKit training."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_rows(path: Path, variant: str) -> list[dict[str, Any]]:
    """Read export rows and retain metadata needed for grouped splits."""
    required = "messages" if variant == "messages" else "conversations"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get(required), list):
                # ValueError keeps malformed user input in the CLI error path.
                raise ValueError(  # noqa: TRY004
                    f"{path}:{line_number}: row is missing required field {required!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no usable rows")
    return rows


def build_distillkit_dataset(
    in_dir: Path,
    out_dir: Path,
    *,
    variant: str,
    eval_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Write a deterministic conversation-grouped DatasetDict for DistillKit."""
    if variant not in {"messages", "sharegpt"}:
        raise ValueError("variant must be 'messages' or 'sharegpt'")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")

    source = in_dir / f"{variant}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(
            f"Expected export file not found: {source}. "
            "Run 'tracesmith export' first or check --in directory."
        )
    rows = _read_rows(source, variant)

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "Install TraceSmith with the distillkit extra: pip install '.[distillkit]'"
        ) from exc

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not metadata.get("trace_id"):
            raise ValueError(
                f"row {index} is missing metadata.trace_id; "
                "cannot preserve conversation-level splits"
            )
        trace_id = str(metadata["trace_id"])
        groups[trace_id].append(index)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    eval_group_count = max(1, int(len(group_ids) * eval_fraction))
    if eval_group_count >= len(group_ids):
        raise ValueError("dataset is too small for a non-empty train split")
    eval_groups = set(group_ids[:eval_group_count])
    eval_indexes = {index for group_id in eval_groups for index in groups[group_id]}
    train_rows = [
        _public_row(rows[index], variant)
        for index in range(len(rows))
        if index not in eval_indexes
    ]
    eval_rows = [
        _public_row(rows[index], variant)
        for index in range(len(rows))
        if index in eval_indexes
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    ).save_to_disk(str(out_dir))

    receipt = {
        "schema_version": 1,
        "variant": variant,
        "source": str(source),
        "source_sha256": _sha256(source),
        "seed": seed,
        "eval_fraction": eval_fraction,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_conversations": len(group_ids) - len(eval_groups),
        "eval_conversations": len(eval_groups),
        "output": str(out_dir),
    }
    (out_dir / "DISTILLKIT_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def _public_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return only the training field expected by DistillKit."""
    required = "messages" if variant == "messages" else "conversations"
    return {required: row[required]}


def _sha256(path: Path) -> str:
    """Hash a potentially large export without reading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
