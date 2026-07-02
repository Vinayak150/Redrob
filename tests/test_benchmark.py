"""Benchmark ranking runtime against compute constraints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.evaluation import honeypot_rate_in_top, run_validator
from src.ranking_engine import RankingEngine
from src.semantic_matching import SemanticMatcher
from src.utils import DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR, load_config, load_candidates


@pytest.mark.benchmark
def test_full_ranking_under_five_minutes():
    candidates = load_candidates(DATA_RAW / "candidates.jsonl")
    jd_config = load_config("jd_profile.yaml")
    weights_config = load_config("ranking_weights.yaml")

    matcher = SemanticMatcher()
    matcher.load_artifacts()

    start = time.time()
    engine = RankingEngine(jd_config, weights_config, matcher)
    ranked = engine.rank(candidates, top_k=100)
    elapsed = time.time() - start

    out = OUTPUTS_DIR / "submission_benchmark.csv"
    engine.write_submission(ranked, out)
    ok, errors = run_validator(out)
    assert ok, errors
    assert elapsed < 300, f"Ranking took {elapsed:.1f}s, exceeds 5-minute limit"

    honeypots = set(json.loads((DATA_PROCESSED / "honeypot_ids.json").read_text()))
    rate = honeypot_rate_in_top([r.candidate_id for r in ranked], honeypots)
    assert rate <= 0.10, f"Honeypot rate {rate:.1%} exceeds 10% threshold"

    report = {
        "elapsed_seconds": round(elapsed, 2),
        "top_10": [r.candidate_id for r in ranked[:10]],
        "honeypot_rate_top_100": rate,
        "validation_passed": ok,
    }
    (OUTPUTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))
