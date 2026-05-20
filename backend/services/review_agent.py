# Core review pipeline — orchestrates diff → context → LLM review → post
import time
from backend.services.diff_parser import FileDiff
from backend.services.llm import review_file, ReviewComment, ReviewMetrics, start_metrics
from backend.services.github_service import parse_pr_url, fetch_diff, post_review
from backend.services.pattern_service import load_patterns, learn_patterns
from backend.services.context_service import gather_context, is_high_risk


def run_review(
    pr_url: str,
    model: str | None = None,
    post_to_github: bool = False,
    on_file_start: callable | None = None,
) -> tuple[list[ReviewComment], ReviewMetrics]:
    metrics = start_metrics()
    t0 = time.monotonic()

    owner, repo, pr_num = parse_pr_url(pr_url)
    raw_diff = fetch_diff(owner, repo, pr_num)
    file_diffs, total_files, filtered_files = FileDiff.parse_diff(raw_diff)

    metrics.total_files = total_files
    metrics.filtered_files = filtered_files

    repo_patterns = load_patterns(owner, repo)

    all_comments = []
    for i, file_diff in enumerate(file_diffs):
        if on_file_start:
            on_file_start(i + 1, len(file_diffs), file_diff.filename)

        context = None
        if is_high_risk(file_diff.filename):
            context = gather_context(owner, repo, file_diff)

        comments = review_file(
            file_diff,
            model=model,
            repo_patterns=repo_patterns,
            owner=owner,
            repo=repo,
            context=context,
        )
        all_comments.extend(comments)

    learn_patterns(owner, repo, all_comments)

    if post_to_github and all_comments:
        post_review(owner, repo, pr_num, all_comments)

    metrics.elapsed_seconds = time.monotonic() - t0
    return all_comments, metrics
