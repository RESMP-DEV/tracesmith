from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_tracked_examples_are_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(  # noqa: S603,RUF100 - repository-controlled interpreter and script
        [sys.executable, str(root / "examples" / "regenerate.py"), "--check"],
        cwd=root,
        check=True,
    )
