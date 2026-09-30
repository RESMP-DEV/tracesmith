"""Independent leftover scanner. Reuses redact.rules patterns for detection only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tracesmith.redact.rules import secrets, identifiers, paths, urls
from tracesmith.redact.rules import RuleContext


# Matches that consist of a KEY=<placeholder> shape (or contain any redaction
# placeholder) are already-redacted text: the rule regexes still match the
# surrounding shape, but no original value survives. Skip those.
_PLACEHOLDER_MARKERS = ("[SECRET", "[PII", "[PRIVATE", "[OMITTED", "[REDACTED")


@dataclass
class Finding:
    path: Path
    line_no: int
    category: str
    snippet: str


@dataclass
class ScanReport:
    leftovers: list[Finding]
    files_scanned: int

    @property
    def ok(self) -> bool:
        return not self.leftovers


def _detection_patterns() -> list[tuple[str, re.Pattern]]:
    """Detection-only patterns (no replacement). Reuses rules module patterns."""
    ctx = RuleContext(user_name="__never_match__", home_dir="__never_match__")
    patterns: list[tuple[str, re.Pattern]] = []
    for family in (secrets, identifiers, paths, urls):
        for category, pat, _rep in family.build(ctx):
            patterns.append((category, pat))
    return patterns


def _iter_strings(obj: object, path: str = "$") -> Iterator[tuple[str, str]]:
    """Yield (field_path, text) for every string in a decoded JSON value."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_strings(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _iter_strings(value, f"{path}[{idx}]")


def _snippet(text: str, match: re.Match) -> str:
    start = max(0, match.start() - 60)
    end = min(len(text), match.end() + 60)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}".replace("\n", "\\n")[:200]


def scan(root: Path) -> ScanReport:
    """Scan decoded JSONL string fields for patterns that survived redaction.

    The redactor transforms decoded string values, so detection must run on the
    same decoded form: scanning the raw serialized line manufactures false
    positives across JSON escapes (e.g. `\\n` turning `@LEAN-CTX.md` into an
    email-shaped `n@LEAN-CTX.md`).
    """
    patterns = _detection_patterns()
    findings: list[Finding] = []
    files = 0
    for path in root.rglob("*.jsonl"):
        files += 1
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = line.rstrip("\n")
                seen: set[str] = set()
                for field, text in _iter_strings(record):
                    for category, pat in patterns:
                        if category in seen:
                            continue
                        for match in pat.finditer(text):
                            if any(marker in match.group(0) for marker in _PLACEHOLDER_MARKERS):
                                continue
                            findings.append(
                                Finding(path, line_no, category, f"{field}: {_snippet(text, match)}")
                            )
                            seen.add(category)
                            break
    return ScanReport(findings, files)
