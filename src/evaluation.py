"""Evaluation helpers and validator integration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils import PROJECT_ROOT


def run_validator(csv_path: Path) -> tuple[bool, list[str]]:
    validator = PROJECT_ROOT / "validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(csv_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, []
    errors = []
    for line in result.stdout.splitlines():
        if line.startswith("- "):
            errors.append(line[2:])
    return False, errors


def honeypot_rate_in_top(
    ranked_ids: list[str],
    honeypot_ids: set[str],
) -> float:
    if not ranked_ids:
        return 0.0
    hits = sum(1 for cid in ranked_ids if cid in honeypot_ids)
    return hits / len(ranked_ids)


def summarize_submission(ranked: list[Any]) -> dict[str, Any]:
    return {
        "count": len(ranked),
        "top_10_ids": [r.candidate_id for r in ranked[:10]],
        "score_range": (ranked[0].final_score, ranked[-1].final_score) if ranked else (0, 0),
    }
