# Parse unified diffs — chunk by file, extract line numbers, filter noise
from pydantic import BaseModel
import fnmatch
import re

DEFAULT_IGNORE = [
    "package-lock.json", "poetry.lock", "yarn.lock", "pnpm-lock.yaml",
    "*.min.js", "*.min.css", "*.png", "*.jpg", "*.ico", "*.svg",
    "*.pb.go", "*.generated.*", "*.map",
]


class FileDiff(BaseModel):
    filename: str
    change_type: str
    diff_content: str
    changed_lines: list[int]

    @staticmethod
    def parse_diff(
        raw_diff: str,
        extra_ignore: list[str] | None = None,
    ) -> tuple[list["FileDiff"], int, int]:
        """Returns (file_diffs, total_files, filtered_files)."""
        ignore_patterns = DEFAULT_IGNORE + (extra_ignore or [])

        chunks = raw_diff.split('diff --git')
        chunks = [chunk for chunk in chunks if chunk.strip()]
        results = []
        total = 0
        filtered = 0

        for chunk in chunks:
            lines = chunk.split('\n')
            filename = ""
            change_type = "modified"
            changed_lines = []

            for line in lines:
                if line.startswith('+++ '):
                    if line == '+++ /dev/null':
                        change_type = "deleted"
                    else:
                        filename = line.removeprefix('+++ b/')
                elif line.startswith('--- '):
                    if line == '--- /dev/null':
                        change_type = "added"
                elif line.startswith('@@'):
                    changed_lines += parse_hunk(line)

            if not filename:
                continue

            total += 1
            if any(fnmatch.fnmatch(filename, pat) for pat in ignore_patterns):
                filtered += 1
                continue

            results.append(FileDiff(
                filename=filename,
                change_type=change_type,
                diff_content=chunk,
                changed_lines=changed_lines,
            ))

        return results, total, filtered


def parse_hunk(line):
    pattern = r'\+(\d+),(\d+)'
    match = re.search(pattern, line)

    if match:
        start = int(match.group(1))
        count = int(match.group(2))
        return list(range(start, start + count))

    return []
