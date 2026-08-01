# ruff: noqa: I001
"""Regenerate deterministic, redacted example exports."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracesmith.config import ExportConfig, RedactorConfig
from tracesmith.export.messages import export_messages
from tracesmith.export.sharegpt import export_sharegpt
from tracesmith.redact.pipeline import RedactionPipeline


EXAMPLE_METADATA_KEY = "tracesmith-public-example-v1"


def render_examples() -> dict[str, str]:
    raw = json.loads((ROOT / "examples" / "sample_input.jsonl").read_text())
    redactor = RedactionPipeline(
        RedactorConfig(user_name="fixture-user", home_dir="/Users/fixture-user")
    )
    redacted = redactor.redact_record(raw)

    with tempfile.TemporaryDirectory(prefix="tracesmith-examples-") as temp:
        temp_root = Path(temp)
        input_dir = temp_root / "input"
        input_dir.mkdir()
        (input_dir / "claude_code.jsonl").write_text(
            json.dumps(redacted, ensure_ascii=False) + "\n"
        )
        config = ExportConfig(metadata_key=EXAMPLE_METADATA_KEY)
        messages = temp_root / "messages.jsonl"
        sharegpt = temp_root / "sharegpt.jsonl"
        export_messages(input_dir, messages, config)
        export_sharegpt(input_dir, sharegpt, config)
        return {
            "messages_expected.jsonl": messages.read_text(),
            "sharegpt_expected.jsonl": sharegpt.read_text(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if tracked examples are stale"
    )
    args = parser.parse_args()
    rendered = render_examples()
    stale = [
        name
        for name, content in rendered.items()
        if not (ROOT / "examples" / name).exists()
        or (ROOT / "examples" / name).read_text() != content
    ]
    if args.check:
        if stale:
            print("stale example exports: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    for name, content in rendered.items():
        (ROOT / "examples" / name).write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
