"""CLI entrypoint: ats."""
from __future__ import annotations

from pathlib import Path

import click


@click.group()
def cli() -> None:
    """agent-trace-share: extract, redact, export AI coding-agent traces."""


@cli.command("extract")
@click.option("--sources", default=None, help="Comma-separated source names. Default: all available.")
@click.option("--root", default="~", help="Home root to scan.")
@click.option("--out", default="./ats_output", type=click.Path(), help="Output directory.")
def extract_cmd(sources: str | None, root: str, out: str) -> None:
    """Extract raw conversations from installed agents."""
    from agent_trace_share.extract.discovery import run_extract
    src_list = sources.split(",") if sources else None
    counts = run_extract(src_list, Path(root).expanduser(), Path(out))
    for k, v in counts.items():
        click.echo(f"{k}: {v} records")


@cli.command("redact")
@click.option("--in", "in_dir", default="./ats_output/raw_extracted", type=click.Path())
@click.option("--out", "out_dir", default="./ats_output/redacted", type=click.Path())
@click.option("--allow-public-urls", is_flag=True)
@click.option("--allow-domain", "allowed_domains", multiple=True)
@click.option("--private-term", "private_terms", multiple=True)
@click.option("--private-domain", "private_domains", multiple=True)
@click.option("--user", "user_name", default=None)
@click.option("--home", "home_dir", default=None)
@click.option("--privacy-filter", is_flag=True,
              help="Enable the transformers-based privacy filter (requires torch+transformers).")
@click.option("--privacy-filter-device", "privacy_filter_device", default="auto",
              help="Device for the privacy filter: auto, cpu, cuda, cuda:N.")
@click.option("--gitleaks", is_flag=True,
              help="Run a final gitleaks scan over the redacted tree (requires gitleaks).")
@click.option("--gitleaks-fix", is_flag=True,
              help="Iteratively redact gitleaks findings in place before the final scan.")
@click.option("--llm-residue", "llm_residue_url", default=None,
              help="OpenAI-compatible base URL for the LLM residue pass, e.g. http://localhost:8000.")
def redact_cmd(in_dir: str, out_dir: str, allow_public_urls: bool,
               allowed_domains: tuple, private_terms: tuple, private_domains: tuple,
               user_name: str | None, home_dir: str | None,
               privacy_filter: bool, privacy_filter_device: str,
               gitleaks: bool, gitleaks_fix: bool,
               llm_residue_url: str | None) -> None:
    """Redact PII/secrets from extracted conversations."""
    from agent_trace_share.config import RedactorConfig
    from agent_trace_share.redact.pipeline import run_redact
    config = RedactorConfig(
        user_name=user_name, home_dir=home_dir,
        allow_public_urls=allow_public_urls,
        allowed_domains=list(allowed_domains),
        private_terms=list(private_terms),
        private_domains=list(private_domains),
        privacy_filter=privacy_filter,
        privacy_filter_device=privacy_filter_device,
        gitleaks=gitleaks,
        gitleaks_fix=gitleaks_fix,
        llm_residue_url=llm_residue_url,
    )
    report = run_redact(Path(in_dir), Path(out_dir), config)
    click.echo(f"Redacted. Counts: {dict(report['counts'])}")


@cli.command("export")
@click.option("--in", "in_dir", default="./ats_output/redacted", type=click.Path())
@click.option("--out", "out_dir", default="./ats_output/export", type=click.Path())
@click.option("--variant", default="both", type=click.Choice(["messages", "sharegpt", "both"]))
@click.option("--min-turns", type=int, default=None)
@click.option("--max-turns", type=int, default=None)
@click.option("--min-assistant-chars", type=int, default=None)
@click.option("--drop-sources", default=None, help="Comma-separated source names to drop")
@click.option("--dedup", is_flag=True)
def export_cmd(in_dir: str, out_dir: str, variant: str, min_turns: int | None,
               max_turns: int | None, min_assistant_chars: int | None,
               drop_sources: str | None, dedup: bool) -> None:
    """Export redacted conversations to DistillKit-ready formats."""
    from agent_trace_share.config import ExportConfig
    from agent_trace_share.export.messages import export_messages
    from agent_trace_share.export.sharegpt import export_sharegpt
    config = ExportConfig(
        variant=variant, min_turns=min_turns, max_turns=max_turns,
        min_assistant_chars=min_assistant_chars,
        drop_sources=drop_sources.split(",") if drop_sources else [],
        dedup=dedup,
    )
    if variant in ("messages", "both"):
        s = export_messages(Path(in_dir), Path(out_dir) / "messages.jsonl", config)
        click.echo(f"messages.jsonl: {s}")
    if variant in ("sharegpt", "both"):
        s = export_sharegpt(Path(in_dir), Path(out_dir) / "sharegpt.jsonl", config)
        click.echo(f"sharegpt.jsonl: {s}")


@cli.command("verify")
@click.option("--in", "in_dir", default="./ats_output/redacted", type=click.Path())
def verify_cmd(in_dir: str) -> None:
    """Scan for leftover PII/secrets that survived redaction."""
    from agent_trace_share.verify.scanner import scan
    report = scan(Path(in_dir))
    if report.ok:
        click.echo(f"OK: {report.files_scanned} files scanned, no leftovers.")
    else:
        click.echo(f"FAIL: {len(report.leftovers)} leftover(s) found:")
        for f in report.leftovers[:20]:
            click.echo(f"  {f.path}:{f.line_no} [{f.category}] {f.snippet}")
        raise click.exceptions.Exit(1)


@cli.command("publish")
@click.option("--repo", "repo_id", required=True,
              help="HF Hub dataset repo id, e.g. user/my-traces.")
@click.option("--in", "in_dir", default="./ats_output/export", type=click.Path())
@click.option("--variant", default="both", type=click.Choice(["messages", "sharegpt", "both"]))
@click.option("--private", is_flag=True, help="Create the dataset as private.")
def publish_cmd(repo_id: str, in_dir: str, variant: str, private: bool) -> None:
    """Upload exported variants + an auto-generated dataset card to HuggingFace Hub."""
    from agent_trace_share.publish.huggingface import upload_variant
    url = upload_variant(Path(in_dir), repo_id, variant=variant, private=private)
    click.echo(f"Published: {url}")


@cli.command("stats")
@click.option("--in", "in_dir", default="./ats_output/export", type=click.Path())
def stats_cmd(in_dir: str) -> None:
    """Report corpus metrics across the exported *.jsonl files."""
    import json as _json
    from agent_trace_share.stats import corpus_stats
    report = corpus_stats(Path(in_dir))
    click.echo(_json.dumps(report, indent=2))


@cli.command("sample")
@click.option("--in", "in_dir", default="./ats_output/export", type=click.Path())
@click.option("--n", "n", default=10, type=int, help="Number of conversations to sample.")
@click.option("--seed", default=0, type=int, help="RNG seed for reproducibility.")
@click.option("--out", "out_dir", default=None, type=click.Path(),
              help="Optional output directory; writes sample.jsonl when provided.")
def sample_cmd(in_dir: str, n: int, seed: int, out_dir: str | None) -> None:
    """Sample N random conversations for pre-publish inspection."""
    from agent_trace_share.sample import sample_conversations
    out_path = Path(out_dir) if out_dir else None
    rows = sample_conversations(Path(in_dir), n, seed=seed, out_dir=out_path)
    click.echo(f"Sampled {len(rows)} conversation(s).")
    if out_path is not None:
        click.echo(f"Wrote {out_path / 'sample.jsonl'}")


@cli.command("run")
@click.option("--sources", default=None)
@click.option("--root", default="~")
@click.option("--out", default="./ats_output", type=click.Path())
@click.option("--variant", default="both", type=click.Choice(["messages", "sharegpt", "both"]))
@click.pass_context
def run_cmd(ctx, sources: str | None, root: str, out: str, variant: str) -> None:
    """Chain extract -> redact -> export."""
    out_path = Path(out)
    ctx.invoke(extract_cmd, sources=sources, root=root, out=out)
    ctx.invoke(redact_cmd,
               in_dir=str(out_path / "raw_extracted"),
               out_dir=str(out_path / "redacted"))
    ctx.invoke(export_cmd,
               in_dir=str(out_path / "redacted"),
               out_dir=str(out_path / "export"),
               variant=variant, min_turns=None, max_turns=None,
               min_assistant_chars=None, drop_sources=None, dedup=False)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
