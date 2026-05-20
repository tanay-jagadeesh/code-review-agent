"""Unified LLM service — supports OpenAI and Anthropic models with tool use."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel
from backend.services.diff_parser import FileDiff
from backend.config import load_config

MAX_TOOL_ROUNDS = 5
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class ReviewComment(BaseModel):
    file: str
    line: int
    severity: str
    category: str
    comment: str
    suggestion: str
    reproduction: str | None = None


@dataclass
class ReviewMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    files_reviewed: int = 0
    total_files: int = 0
    filtered_files: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        total_input = self.input_tokens + self.cache_read_tokens
        if total_input == 0:
            return 0.0
        return self.cache_read_tokens / total_input

    @property
    def noise_filter_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.filtered_files / self.total_files

    @property
    def estimated_cost_saved_usd(self) -> float:
        # Cache reads cost 10% of normal input tokens (sonnet-4: $3/1M input, $0.30/1M cache read)
        normal_cost_per_1m = 3.0
        cache_cost_per_1m = 0.30
        saved = (self.cache_read_tokens / 1_000_000) * (normal_cost_per_1m - cache_cost_per_1m)
        return round(saved, 4)

    def summary(self) -> str:
        lines = [
            f"Files in PR:         {self.total_files}  (filtered: {self.filtered_files}  reviewed: {self.files_reviewed})",
        ]
        if self.total_files:
            lines.append(f"Noise filter rate:   {self.noise_filter_rate:.0%}")
        lines.append(
            f"Total tokens:        {self.total_tokens:,}  (in: {self.input_tokens:,}  out: {self.output_tokens:,})"
        )
        if self.cache_read_tokens or self.cache_write_tokens:
            lines += [
                f"Cache write tokens:  {self.cache_write_tokens:,}",
                f"Cache read tokens:   {self.cache_read_tokens:,}",
                f"Cache hit rate:      {self.cache_hit_rate:.0%}",
                f"Est. cost saved:     ${self.estimated_cost_saved_usd:.4f}",
            ]
        lines.append(f"Elapsed:             {self.elapsed_seconds:.1f}s")
        return "\n".join(lines)


# Global metrics accumulator for the current review run
_current_metrics: ReviewMetrics | None = None


def start_metrics() -> ReviewMetrics:
    global _current_metrics
    _current_metrics = ReviewMetrics()
    return _current_metrics


def get_metrics() -> ReviewMetrics | None:
    return _current_metrics


# ── Prompt loading ──────────────────────────────────────────────

def _load_system_prompt() -> str:
    candidates = [
        Path("prompts/system.md"),
        Path(__file__).parent.parent.parent / "prompts" / "system.md",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text()
    raise FileNotFoundError("Could not find prompts/system.md")


def _build_user_message(
    file_diff: FileDiff,
    context: dict | None = None,
    repo_patterns: list[dict] | None = None,
    custom_rules: str | None = None,
) -> str:
    msg = f"""Review this diff for {file_diff.filename}:

Changed lines: {file_diff.changed_lines}
Change type: {file_diff.change_type}

Diff:
{file_diff.diff_content}"""

    # Inject gathered context (full file, test file)
    if context:
        if context.get("full_file"):
            msg += f"""

Full file content for surrounding context:
{context['full_file']}"""
        if context.get("test_file"):
            msg += f"""

Corresponding test file:
{context['test_file']}"""
        if context.get("high_risk"):
            msg += "\n\nThis file is HIGH RISK (auth/security/payment related). Review with extra scrutiny."

    if repo_patterns:
        patterns_text = "\n".join(
            f"- [{p['category']}] {p['pattern']}" for p in repo_patterns
        )
        msg += f"""

This repository has the following established conventions. Flag violations of these patterns:
{patterns_text}"""

    if custom_rules:
        msg += f"""

Additional rules to enforce:
{custom_rules}"""

    msg += """

You have tools available to fetch more context if needed:
- fetch_file_content: Get the full content of any file in the repo
- fetch_test_file: Find and fetch the test file for a source file
- search_codebase: Search the repo for function callers, imports, usages

Use these tools if the diff alone isn't enough to determine whether something is a real issue.
When you're done reviewing, respond with the JSON array of comments."""

    return msg


# ── Tool definitions ────────────────────────────────────────────

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "fetch_file_content",
            "description": "Fetch the full content of a file from the repository. Use this to see surrounding context, imports, or related code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "ref": {"type": "string", "description": "Git ref (default: main)", "default": "main"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_test_file",
            "description": "Find and fetch the corresponding test file for a source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string", "description": "Path to the source file"},
                },
                "required": ["source_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search the repository for code matching a query. Find callers, usages, or related imports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — function name, class, import, etc."},
                },
                "required": ["query"],
            },
        },
    },
]

TOOLS_ANTHROPIC = [
    {
        "name": "fetch_file_content",
        "description": "Fetch the full content of a file from the repository. Use this to see surrounding context, imports, or related code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "ref": {"type": "string", "description": "Git ref (default: main)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "fetch_test_file",
        "description": "Find and fetch the corresponding test file for a source file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "description": "Path to the source file"},
            },
            "required": ["source_path"],
        },
    },
    {
        "name": "search_codebase",
        "description": "Search the repository for code matching a query. Find callers, usages, or related imports.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query — function name, class, import, etc."},
            },
            "required": ["query"],
        },
    },
]


# ── Tool execution ──────────────────────────────────────────────

def _execute_tool(name: str, args: dict, owner: str | None, repo: str | None) -> str:
    """Execute a tool call and return the result as a string."""
    from backend.tools.fetch_file import fetch_file_content
    from backend.tools.fetch_tests import fetch_test_file
    from backend.tools.search_code import search_codebase

    if name == "fetch_file_content":
        if not owner or not repo:
            return "Cannot fetch files: no repository context (local diff mode)"
        return fetch_file_content(owner, repo, args["path"], args.get("ref", "main"))

    elif name == "fetch_test_file":
        if not owner or not repo:
            return "Cannot fetch test files: no repository context (local diff mode)"
        return fetch_test_file(owner, repo, args["source_path"])

    elif name == "search_codebase":
        if not owner or not repo:
            return "Cannot search: no repository context (local diff mode)"
        results = search_codebase(owner, repo, args["query"])
        return json.dumps(results, indent=2)

    return f"Unknown tool: {name}"


# ── Response parsing ────────────────────────────────────────────

def _parse_response(raw: str) -> list[ReviewComment]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    # Try to extract JSON array even if there's surrounding text
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    comments = json.loads(raw)
    return [ReviewComment(**c) for c in comments]


def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


# ── OpenAI with tool-use loop ───────────────────────────────────

def _call_openai(
    model: str, system_prompt: str, user_message: str, api_key: str,
    owner: str | None = None, repo: str | None = None,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    use_tools = owner is not None and repo is not None

    def _record_usage(usage) -> None:
        if _current_metrics is None or usage is None:
            return
        _current_metrics.input_tokens += getattr(usage, "prompt_tokens", 0)
        _current_metrics.output_tokens += getattr(usage, "completion_tokens", 0)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            **({"tools": TOOLS_OPENAI} if use_tools else {}),
        )
        _record_usage(response.usage)

        choice = response.choices[0]

        if not choice.message.tool_calls:
            return choice.message.content

        messages.append(choice.message)
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, args, owner, repo)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Exhausted tool rounds — get final answer without tools
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0,
    )
    _record_usage(response.usage)
    return response.choices[0].message.content


# ── Anthropic with tool-use loop ────────────────────────────────

def _call_anthropic(
    model: str, system_prompt: str, user_message: str, api_key: str,
    owner: str | None = None, repo: str | None = None,
) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # Cache the system prompt — it's identical across all files in a PR
    system_with_cache = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    messages = [{"role": "user", "content": user_message}]
    use_tools = owner is not None and repo is not None

    def _record_usage(usage) -> None:
        if _current_metrics is None:
            return
        _current_metrics.input_tokens += getattr(usage, "input_tokens", 0)
        _current_metrics.output_tokens += getattr(usage, "output_tokens", 0)
        _current_metrics.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0)
        _current_metrics.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_with_cache,
            messages=messages,
            temperature=0,
            **({"tools": TOOLS_ANTHROPIC} if use_tools else {}),
        )
        _record_usage(response.usage)

        # Check if the model wants to use tools
        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "[]"

        # Process tool calls
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                result = _execute_tool(block.name, block.input, owner, repo)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    # Exhausted tool rounds — get final answer without tools
    response = client.messages.create(
        model=model, max_tokens=4096, system=system_with_cache,
        messages=messages, temperature=0,
    )
    _record_usage(response.usage)
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "[]"


# ── Main entry point ────────────────────────────────────────────

def review_file(
    file_diff: FileDiff,
    model: str | None = None,
    repo_patterns: list[dict] | None = None,
    custom_rules: str | None = None,
    owner: str | None = None,
    repo: str | None = None,
    context: dict | None = None,
) -> list[ReviewComment]:
    """Review a single file diff using the configured LLM with tool use."""
    config = load_config()
    model = model or config.get("model", "gpt-4o")
    custom_rules = custom_rules or config.get("custom_rules") or None

    system_prompt = _load_system_prompt()
    user_message = _build_user_message(file_diff, context, repo_patterns, custom_rules)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if _is_anthropic_model(model):
                api_key = config.get("anthropic_api_key", "")
                if not api_key:
                    raise RuntimeError("No Anthropic API key. Run: cr config set anthropic_api_key <key>")
                raw = _call_anthropic(model, system_prompt, user_message, api_key, owner, repo)
            else:
                api_key = config.get("openai_api_key", "")
                if not api_key:
                    raise RuntimeError("No OpenAI API key. Run: cr config set openai_api_key <key>")
                raw = _call_openai(model, system_prompt, user_message, api_key, owner, repo)

            result = _parse_response(raw)
            if _current_metrics is not None:
                _current_metrics.files_reviewed += 1
            return result

        except RuntimeError:
            raise
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

    raise RuntimeError(f"Review failed after {MAX_RETRIES} attempts: {last_error}")
