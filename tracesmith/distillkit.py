"""Create local Hugging Face datasets for DistillKit training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from tracesmith import __version__


def _required_field(variant: str) -> str:
    """Return the export field used as the DistillKit training payload."""
    return "messages" if variant == "messages" else "conversations"


def _iter_rows(path: Path, variant: str) -> Iterator[tuple[int, str, dict[str, Any]]]:
    """Yield validated export rows with their source trace IDs."""
    required = _required_field(variant)
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
            if not row[required]:
                raise ValueError(
                    f"{path}:{line_number}: row has an empty {required!r} field"
                )
            metadata = row.get("metadata")
            trace_id = (
                str(metadata.get("trace_id")) if isinstance(metadata, dict) else ""
            )
            if (
                not isinstance(metadata, dict)
                or "trace_id" not in metadata
                or not trace_id
            ):
                raise ValueError(
                    f"{path}:{line_number}: row is missing metadata.trace_id; "
                    "cannot preserve conversation-level splits"
                )
            yield line_number, trace_id, row


def _scan_groups(path: Path, variant: str) -> dict[str, int]:
    """Validate an export once and index rows by source trace ID."""
    groups: dict[str, int] = defaultdict(int)
    found = False
    for row_number, trace_id, _row in _iter_rows(path, variant):
        groups[trace_id] += 1
        found = True
    if not found:
        raise ValueError(f"{path} contains no usable rows")
    return groups


def _eval_groups(group_ids: list[str], eval_fraction: float, seed: int) -> set[str]:
    """Select eval groups with stable SHA-256 ordering across Python versions."""
    eval_count = max(1, int(len(group_ids) * eval_fraction))
    if eval_count >= len(group_ids):
        raise ValueError("dataset is too small for a non-empty train split")
    ranked = sorted(
        group_ids, key=lambda group_id: (stable_rank(group_id, seed), group_id)
    )
    return set(ranked[:eval_count])


def stable_rank(group_id: str, seed: int) -> bytes:
    """Derive a split ranking that does not depend on Python's RNG."""
    return hashlib.sha256(f"{seed}\0{group_id}".encode()).digest()


def _public_rows(
    path: Path,
    variant: str,
    eval_groups: set[str],
    is_eval: bool,
) -> Iterator[dict[str, Any]]:
    """Stream the requested split without retaining the full export in memory."""
    for _line_number, trace_id, row in _iter_rows(path, variant):
        if (trace_id in eval_groups) is is_eval:
            yield _public_row(row, variant)


def build_distillkit_dataset(
    in_dir: Path,
    out_dir: Path,
    *,
    variant: str,
    eval_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Write a deterministic, conversation-grouped DatasetDict for DistillKit."""
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

    groups = _scan_groups(source, variant)
    group_ids = sorted(groups)
    eval_groups = _eval_groups(group_ids, eval_fraction, seed)
    source_sha256 = _sha256(source)

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "Install TraceSmith with the distillkit extra: pip install '.[distillkit]'"
        ) from exc

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent)
    )
    try:
        dataset = DatasetDict(
            {
                "train": Dataset.from_generator(
                    partial(
                        _public_rows,
                        source,
                        variant,
                        eval_groups,
                        False,
                    )
                ),
                "eval": Dataset.from_generator(
                    partial(
                        _public_rows,
                        source,
                        variant,
                        eval_groups,
                        True,
                    )
                ),
            }
        )
        dataset.save_to_disk(str(temporary_dir))
        if _sha256(source) != source_sha256:
            raise RuntimeError("source export changed while building the dataset")

        receipt = {
            "schema_version": 1,
            "tracesmith_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "variant": variant,
            "source": str(source),
            "source_sha256": source_sha256,
            "seed": seed,
            "eval_fraction": eval_fraction,
            "train_rows": int(dataset["train"].num_rows),
            "eval_rows": int(dataset["eval"].num_rows),
            "train_conversations": len(group_ids) - len(eval_groups),
            "eval_conversations": len(eval_groups),
            "output": str(out_dir),
        }
        (temporary_dir / "DISTILLKIT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_dir, out_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return receipt


def verify_distillkit_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Verify a saved dataset against its DistillKit receipt."""
    receipt_path = dataset_dir / "DISTILLKIT_RECEIPT.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"receipt not found: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    variant = str(receipt.get("variant"))
    if variant not in {"messages", "sharegpt"}:
        raise ValueError(f"unsupported receipt variant: {variant!r}")

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Install TraceSmith with the distillkit extra: pip install '.[distillkit]'"
        ) from exc

    dataset = load_from_disk(str(dataset_dir))
    for split in ("train", "eval"):
        if split not in dataset:
            raise ValueError(f"dataset split is missing: {split}")
        expected_rows = int(receipt.get(f"{split}_rows", -1))
        if int(dataset[split].num_rows) != expected_rows:
            raise ValueError(f"{split} row count does not match receipt")
        if dataset[split].column_names != [_required_field(variant)]:
            raise ValueError(f"{split} columns do not match variant {variant!r}")

    source = Path(str(receipt.get("source")))
    source_hash = receipt.get("source_sha256")
    if (
        source.is_file()
        and isinstance(source_hash, str)
        and _sha256(source) != source_hash
    ):
        raise ValueError("source export hash does not match receipt")
    return receipt


def _public_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return only the training field expected by DistillKit."""
    return {_required_field(variant): row[_required_field(variant)]}


def _sha256(path: Path) -> str:
    """Hash a potentially large export without reading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
