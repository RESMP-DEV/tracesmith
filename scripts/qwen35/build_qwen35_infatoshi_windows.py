#!/usr/bin/env python3
"""Build Qwen3.5 training windows from exported Infatoshi conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

TOOL_MARKERS = ("<tool_call", "<tool_result", "<tool_use", "<reasoning_summary")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def encode(text: str, tokenizer: Any) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def score_target(text: str) -> int:
    score = 1
    if "<tool_use" in text or "<tool_result" in text:
        score += 4
    if "```" in text:
        score += 3
    if "<reasoning_summary>" in text:
        score += 1
    return score


def conversation_windows(
    messages: list[dict],
    metadata: dict,
    source: str,
    tokenizer: Any,
    seed: int,
    target_tokens: int,
    context_tokens: int,
    max_per_conversation: int,
) -> list[dict]:
    candidates: list[dict] = []
    previous_user: list[dict] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            if message.get("role") == "user":
                previous_user.append(message)
            continue

        context_messages = previous_user[-3:]
        if context_messages:
            context_text = tokenizer.apply_chat_template(
                context_messages, tokenize=False, add_generation_prompt=False
            )
            context_ids = encode(context_text, tokenizer)[-context_tokens:]
            context = "<previous_context>\n" + tokenizer.decode(context_ids)
        else:
            context = "<previous_context>conversation start</previous_context>"

        target = str(message.get("content") or "")
        target_ids = encode(target, tokenizer)
        for chunk_index, start in enumerate(range(0, len(target_ids), target_tokens)):
            chunk = target_ids[start : start + target_tokens]
            if len(chunk) < 50 and chunk_index:
                continue
            target_text = tokenizer.decode(chunk)
            key = f"{metadata.get('trace_id', index)}:{index}:{chunk_index}"
            candidates.append(
                {
                    "messages": [
                        {"role": "user", "content": context},
                        {"role": "assistant", "content": target_text},
                    ],
                    "source": source,
                    "model": str((metadata.get("models") or [""])[0]),
                    "session_id": key,
                    "n_turns": 2,
                    "assistant_chars": len(target_text),
                    "total_chars": len(context) + len(target_text),
                    "has_tool": any(marker in target_text for marker in TOOL_MARKERS),
                    "_score": score_target(target_text),
                    "_rank": stable_rank(key, seed),
                }
            )
        previous_user = []

    candidates.sort(key=lambda row: (-row["_score"], row["_rank"]))
    return candidates[:max_per_conversation]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/home/kearm/training/data/qwen35")
    parser.add_argument(
        "--tokenizer", default="/home/kearm/training/models/Qwen3.5-0.8B"
    )
    parser.add_argument("--export", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--target-tokens", type=int, default=768)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--max-per-conversation", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    base = load_from_disk(args.base)
    queues: list[list[dict]] = []
    stats = {
        "export_conversations": 0,
        "assistant_windows_before_cap": 0,
        "selected_windows": 0,
    }
    source_counts: dict[str, int] = {}
    export_hashes: dict[str, str] = {}
    for export_name in args.export:
        path = Path(export_name)
        export_hashes[path.parent.parent.name + "/" + path.name] = sha256(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                export = json.loads(line)
                metadata = (
                    export.get("metadata")
                    if isinstance(export.get("metadata"), dict)
                    else {}
                )
                source = str(metadata.get("source") or path.parent.parent.name)
                windows = conversation_windows(
                    export.get("messages", []),
                    metadata,
                    source,
                    tokenizer,
                    args.seed,
                    args.target_tokens,
                    args.context_tokens,
                    args.max_per_conversation,
                )
                stats["export_conversations"] += 1
                stats["assistant_windows_before_cap"] += len(windows)
                if windows:
                    queues.append(sorted(windows, key=lambda row: row["_rank"]))

    queues.sort(key=lambda queue: stable_rank(queue[0]["session_id"], args.seed))
    extras: list[dict] = []
    while queues and len(extras) < args.max_windows:
        for queue in list(queues):
            if len(extras) >= args.max_windows:
                break
            row = queue.pop(0)
            row.pop("_score")
            row.pop("_rank")
            extras.append(row)
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            if not queue:
                queues.remove(queue)
    stats["selected_windows"] = len(extras)
    if not extras:
        raise RuntimeError("no assistant-bearing windows selected")

    base_train = list(base["train"])
    random.Random(args.seed).shuffle(base_train)
    fill = max(0, args.max_train - len(extras))
    if len(extras) > args.max_train:
        raise RuntimeError("selected windows exceed max-train")
    train_rows = extras + base_train[:fill]
    random.Random(args.seed).shuffle(train_rows)
    for row in base_train[:fill]:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    DatasetDict(
        {"train": Dataset.from_list(train_rows), "eval": base["eval"]}
    ).save_to_disk(str(output))
    base_receipt = Path(args.base) / "BUILD_RECEIPT.json"
    receipt = {
        "schema_version": 2,
        "seed": args.seed,
        "sequence_budget": args.target_tokens + args.context_tokens,
        "target_tokens": args.target_tokens,
        "context_tokens": args.context_tokens,
        "max_per_conversation": args.max_per_conversation,
        "base_dataset": str(args.base),
        "base_receipt_sha256": sha256(base_receipt),
        "export_sha256": export_hashes,
        "stats": stats,
        "extra_rows": len(extras),
        "base_fill_rows": fill,
        "train_rows": len(train_rows),
        "eval_rows": base["eval"].num_rows,
        "train_with_tool": sum(bool(row["has_tool"]) for row in train_rows),
        "per_source": dict(sorted(source_counts.items(), key=lambda item: -item[1])),
        "output": str(output),
        "non_claims": [
            "Windowing preserves assistant targets but not every token of every source conversation.",
            "Training-loss movement alone is not model acceptance.",
            "The fixed evaluation rows do not prove broad coding-agent quality.",
        ],
    }
    (output / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
