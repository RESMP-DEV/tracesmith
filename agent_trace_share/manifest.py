"""MANIFEST.json: provenance, counts, hashes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_trace_share import __version__


def file_hashes(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "rows": sum(1 for _ in path.open())}


def write_manifest(
    out_root: Path,
    extract_counts: dict[str, int],
    redact_report: dict[str, Any],
    export_summaries: dict[str, dict],
    config_snapshot: dict[str, Any],
) -> Path:
    sources: dict[str, Any] = {}
    for src, n in extract_counts.items():
        sources[src] = {
            "raw_records": n,
            "redaction_counts": redact_report.get("per_source", {}).get(src, {}),
            "exported_messages": export_summaries.get("messages", {}).get("rows", 0) if src else 0,
        }
    files: dict[str, Any] = {}
    export_dir = out_root / "export"
    for name in ("messages.jsonl", "sharegpt.jsonl"):
        p = export_dir / name
        if p.exists():
            files[f"export/{name}"] = file_hashes(p)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": __version__,
        "config": config_snapshot,
        "sources": sources,
        "files": files,
    }
    path = out_root / "MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
