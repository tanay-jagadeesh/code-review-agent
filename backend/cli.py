"""cr — AI code reviewer CLI."""

import json as json_mod
import click
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from rich import box

console = Console()

SEVERITY_STYLES = {
    "critical": ("bold red", "CRIT"),
    "warning": ("bold yellow", "WARN"),
    "suggestion": ("bold blue", "SUGG"),
    "nitpick": ("dim", "NIT "),
}

CATEGORY_STYLES = {
    "security": "red",
    "bug": "yellow",
    "performance": "magenta",
    "style": "dim",
    "logic": "cyan",
    "convention": "blue",
}

# Severities that should cause --exit-code to fail
FAIL_SEVERITIES = {"critical", "warning"}


def _comments_to_json(comments: list) -> str:
    """Serialize comments to JSON for CI output."""
    out = []
    for c in comments:
        out.append({
            "file": c.file if hasattr(c, "file") else c["file"],
            "line": c.line if hasattr(c, "line") else c["line"],
            "severity": c.severity if hasattr(c, "severity") else c["severity"],
            "category": c.category if hasattr(c, "category") else c["category"],
            "comment": c.comment if hasattr(c, "comment") else c["comment"],
            "suggestion": c.suggestion if hasattr(c, "suggestion") else c["suggestion"],
            "reproduction": (c.reproduction if hasattr(c, "reproduction") else c.get("reproduction", None)) if hasattr(c, "reproduction") or isinstance(c, dict) else None,
        })
    return json_mod.dumps(out, indent=2)


def _render_comments(comments: list) -> None:
    """Render review comments to the terminal with rich formatting."""
    if not comments:
        console.print()
        console.print("[green]  No issues found. Code looks clean.[/green]")
        return

    by_file: dict[str, list] = {}
    for c in comments:
        f = c.file if hasattr(c, "file") else c["file"]
        by_file.setdefault(f, []).append(c)

    counts: dict[str, int] = {}
    for c in comments:
        sev = c.severity if hasattr(c, "severity") else c["severity"]
        counts[sev] = counts.get(sev, 0) + 1

    parts = []
    for sev in ["critical", "warning", "suggestion", "nitpick"]:
        if sev in counts:
            style, label = SEVERITY_STYLES[sev]
            parts.append(f"[{style}]{counts[sev]} {sev}[/{style}]")
    summary = "  ".join(parts)
    console.print()
    console.print(f"  Found {len(comments)} issue{'s' if len(comments) != 1 else ''}:  {summary}")
    console.print()

    for filename in sorted(by_file.keys()):
        file_comments = sorted(by_file[filename], key=lambda c: c.line if hasattr(c, "line") else c["line"])
        console.print(Rule(f"[bold]{filename}[/bold]", style="dim"))
        console.print()

        for c in file_comments:
            line = c.line if hasattr(c, "line") else c["line"]
            sev = c.severity if hasattr(c, "severity") else c["severity"]
            cat = c.category if hasattr(c, "category") else c["category"]
            comment = c.comment if hasattr(c, "comment") else c["comment"]
            suggestion = c.suggestion if hasattr(c, "suggestion") else c["suggestion"]
            reproduction = (c.reproduction if hasattr(c, "reproduction") else c.get("reproduction")) if hasattr(c, "reproduction") or isinstance(c, dict) else None

            sev_style, sev_label = SEVERITY_STYLES.get(sev, ("dim", sev.upper()[:4]))
            cat_style = CATEGORY_STYLES.get(cat, "dim")

            header = Text()
            header.append(f"  {sev_label}", style=sev_style)
            header.append(f"  {cat}", style=cat_style)
            header.append(f"  L{line}", style="dim")
            console.print(header)
            console.print(f"    {comment}")
            if reproduction:
                console.print(f"    [dim italic]Why: {reproduction}[/dim italic]")
            if suggestion:
                console.print(f"    [green]Fix: {suggestion}[/green]")
            console.print()


def _has_failures(comments: list) -> bool:
    """Check if any comments are critical or warning severity."""
    for c in comments:
        sev = c.severity if hasattr(c, "severity") else c["severity"]
        if sev in FAIL_SEVERITIES:
            return True
    return False


@click.group()
@click.version_option(version="0.1.0", prog_name="cr")
def main():
    """AI code reviewer — catch issues before human review."""
    pass


@main.command()
@click.argument("pr_url")
@click.option("--post/--no-post", default=False, help="Post comments to GitHub (default: just print)")
@click.option("--model", default=None, help="Override model (e.g. gpt-4o, claude-sonnet-4-20250514)")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--exit-code", is_flag=True, help="Exit 1 if critical/warning issues found (for CI)")
def review(pr_url: str, post: bool, model: str | None, fmt: str, exit_code: bool):
    """Review a GitHub pull request."""
    from backend.config import load_config
    from backend.services.github_service import parse_pr_url, fetch_diff
    from backend.services.diff_parser import FileDiff
    from backend.services.llm import review_file
    from backend.services.pattern_service import load_patterns

    import time as _time
    from backend.services.llm import start_metrics

    config = load_config()
    owner, repo, pr_num = parse_pr_url(pr_url)

    if fmt == "text":
        console.print(f"  [bold]Reviewing[/bold] {owner}/{repo}#{pr_num}")
        console.print()

    ignore_paths = config.get("ignore_paths", [])

    with console.status("[dim]Fetching diff from GitHub...[/dim]") if fmt == "text" else _nullcontext():
        raw_diff = fetch_diff(owner, repo, pr_num)
        file_diffs, total_files, filtered_files = FileDiff.parse_diff(raw_diff, extra_ignore=ignore_paths)

    if not file_diffs:
        if fmt == "json":
            click.echo("[]")
        else:
            console.print("[yellow]  No reviewable files in this PR.[/yellow]")
        return

    if fmt == "text":
        skipped = f", {filtered_files} skipped" if filtered_files else ""
        console.print(f"  [dim]{len(file_diffs)} file{'s' if len(file_diffs) != 1 else ''} to review{skipped}[/dim]")
        console.print()

    use_model = model or config.get("model", "gpt-4o")
    repo_patterns = load_patterns(owner, repo)

    metrics = start_metrics()
    metrics.total_files = total_files
    metrics.filtered_files = filtered_files
    t0 = _time.monotonic()

    all_comments = []
    for i, fd in enumerate(file_diffs, 1):
        if fmt == "text":
            console.print(f"  [dim][{i}/{len(file_diffs)}][/dim] {fd.filename}")
        try:
            comments = review_file(fd, model=use_model, repo_patterns=repo_patterns, owner=owner, repo=repo)
            all_comments.extend(comments)
        except RuntimeError as e:
            if fmt == "json":
                click.echo(json_mod.dumps({"error": str(e)}))
            else:
                console.print(f"\n  [red]{e}[/red]")
            raise SystemExit(1)

    metrics.elapsed_seconds = _time.monotonic() - t0

    if fmt == "json":
        click.echo(_comments_to_json(all_comments))
    else:
        _render_comments(all_comments)
        if metrics.total_tokens > 0:
            console.print(Rule("[dim]metrics[/dim]", style="dim"))
            for line in metrics.summary().splitlines():
                console.print(f"  [dim]{line}[/dim]")
            console.print()

    if post and all_comments:
        from backend.services.github_service import post_review as gh_post
        with console.status("[dim]Posting review to GitHub...[/dim]") if fmt == "text" else _nullcontext():
            gh_post(owner, repo, pr_num, all_comments)
        if fmt == "text":
            console.print("[green]  Review posted to GitHub.[/green]")

    if exit_code and _has_failures(all_comments):
        raise SystemExit(1)


@main.command()
@click.option("--staged", is_flag=True, help="Only review staged changes")
@click.option("--model", default=None, help="Override model")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--exit-code", is_flag=True, help="Exit 1 if critical/warning issues found (for CI)")
def diff(staged: bool, model: str | None, fmt: str, exit_code: bool):
    """Review local uncommitted changes."""
    import subprocess
    from backend.config import load_config
    from backend.services.diff_parser import FileDiff
    from backend.services.llm import review_file

    config = load_config()

    cmd = ["git", "diff", "--staged"] if staged else ["git", "diff", "HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if fmt == "json":
            click.echo(json_mod.dumps({"error": "Not a git repository"}))
        else:
            console.print("[red]  Not a git repository or git not available.[/red]")
        raise SystemExit(1)

    raw_diff = result.stdout
    if not raw_diff.strip():
        if fmt == "json":
            click.echo("[]")
        else:
            label = "staged" if staged else "uncommitted"
            console.print(f"[yellow]  No {label} changes to review.[/yellow]")
        return

    ignore_paths = config.get("ignore_paths", [])
    file_diffs, _, _ = FileDiff.parse_diff(raw_diff, extra_ignore=ignore_paths)
    if not file_diffs:
        if fmt == "json":
            click.echo("[]")
        else:
            console.print("[yellow]  No reviewable files changed.[/yellow]")
        return

    if fmt == "text":
        console.print(f"  [bold]Reviewing local {'staged' if staged else 'uncommitted'} changes[/bold]")
        console.print(f"  [dim]{len(file_diffs)} file{'s' if len(file_diffs) != 1 else ''}[/dim]")
        console.print()

    use_model = model or config.get("model", "gpt-4o")

    all_comments = []
    for i, fd in enumerate(file_diffs, 1):
        if fmt == "text":
            console.print(f"  [dim][{i}/{len(file_diffs)}][/dim] {fd.filename}")
        try:
            comments = review_file(fd, model=use_model)
            all_comments.extend(comments)
        except RuntimeError as e:
            if fmt == "json":
                click.echo(json_mod.dumps({"error": str(e)}))
            else:
                console.print(f"\n  [red]{e}[/red]")
            raise SystemExit(1)

    if fmt == "json":
        click.echo(_comments_to_json(all_comments))
    else:
        _render_comments(all_comments)

    if exit_code and _has_failures(all_comments):
        raise SystemExit(1)


@main.command()
@click.argument("action", required=False, default="show")
@click.argument("key", required=False)
@click.argument("value", required=False)
def config(action: str, key: str | None, value: str | None):
    """Show or update configuration.

    \b
    cr config              Show current config
    cr config set model gpt-4o
    cr config set github_token ghp_xxx
    """
    from backend.config import load_config, save_config, CONFIG_PATH

    cfg = load_config()

    if action == "show" or (action != "set"):
        console.print()
        console.print(f"  [dim]Config:[/dim] {CONFIG_PATH}")
        console.print()
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for k, v in cfg.items():
            display = "••••••••" if "key" in k.lower() or "token" in k.lower() else str(v)
            table.add_row(k, display)
        console.print(Padding(table, (0, 2)))
        return

    if action == "set" and key and value:
        cfg[key] = value
        save_config(cfg)
        display = "••••••••" if "key" in key.lower() or "token" in key.lower() else value
        console.print(f"  [green]Set {key} = {display}[/green]")
    else:
        console.print("[red]  Usage: cr config set <key> <value>[/red]")


@main.command()
def init():
    """Initialize cr-agent in the current project.

    Creates a .cr.toml config file and optionally sets up a pre-commit hook.
    """
    import os
    from pathlib import Path

    cr_toml = Path(".cr.toml")

    if cr_toml.exists():
        console.print("[yellow]  .cr.toml already exists.[/yellow]")
        return

    # Detect language from common files
    language = "generic"
    if Path("package.json").exists():
        language = "javascript"
    elif Path("pyproject.toml").exists() or Path("setup.py").exists() or Path("requirements.txt").exists():
        language = "python"
    elif Path("go.mod").exists():
        language = "go"
    elif Path("Cargo.toml").exists():
        language = "rust"
    elif Path("pom.xml").exists() or Path("build.gradle").exists():
        language = "java"

    # Write .cr.toml
    cr_toml.write_text(f'''# cr-agent config — shared across the team via version control
# https://github.com/tanayj/code-review-agent

[review]
language = "{language}"
model = "gpt-4o"

# Fail CI on these severities (used with --exit-code)
fail_on = ["critical", "warning"]

# Custom review rules (appended to the system prompt)
# rules = """
# Always flag SQL queries without parameterized inputs
# Require error handling on all network calls
# """

[ignore]
# Glob patterns for files to skip
paths = [
    "*.lock",
    "*.min.js",
    "*.min.css",
    "*.generated.*",
    "dist/**",
    "build/**",
    "node_modules/**",
    "__pycache__/**",
]
''')

    console.print(f"  [green]Created .cr.toml[/green] (detected: {language})")

    # Offer to set up pre-commit hook
    git_dir = Path(".git")
    if git_dir.exists() and click.confirm("  Set up a pre-commit hook to review staged changes?", default=False):
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook_path = hooks_dir / "pre-commit"

        hook_content = """#!/bin/sh
# cr-agent pre-commit hook — review staged changes before commit
# Remove this file or run `cr hooks remove` to disable

cr diff --staged --exit-code
"""
        hook_path.write_text(hook_content)
        os.chmod(hook_path, 0o755)
        console.print("  [green]Installed pre-commit hook[/green] (.git/hooks/pre-commit)")
    else:
        console.print("  [dim]Tip: run `cr init` inside a git repo to set up a pre-commit hook[/dim]")

    console.print()
    console.print("  [dim]Next steps:[/dim]")
    console.print("    1. cr config set openai_api_key sk-...")
    console.print("    2. cr diff --staged")
    console.print("    3. Commit .cr.toml so your team shares the same review rules")


# ── Helpers ─────────────────────────────────────────────────────

class _nullcontext:
    """Minimal no-op context manager for json mode."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


if __name__ == "__main__":
    main()
