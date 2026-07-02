#!/usr/bin/env python3
"""CLI entrypoint for the Redrob Candidate Discovery Engine."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src.preprocessing import build_canonical_text, build_career_text, detect_honeypots, extract_candidate_features
from src.ranking_engine import RankingEngine
from src.semantic_matching import SemanticMatcher
from src.evaluation import run_validator, summarize_submission
from src.utils import DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR, load_config, load_candidates, setup_logging

logger = logging.getLogger(__name__)


def cmd_precompute(args: argparse.Namespace) -> None:
    jd_config = load_config("jd_profile.yaml")
    candidates_path = Path(args.candidates)
    candidates = load_candidates(candidates_path)

    canonical = [build_canonical_text(c) for c in candidates]
    career = [build_career_text(c) for c in candidates]

    matcher = SemanticMatcher(model_name=args.model)
    matcher.precompute(candidates, jd_config["jd_text"], career, canonical)

    honeypots = [c["candidate_id"] for c in candidates if detect_honeypots(c)[0]]
    with open(DATA_PROCESSED / "honeypot_ids.json", "w", encoding="utf-8") as f:
        json.dump(honeypots, f)
    logger.info("Identified %d honeypot candidates", len(honeypots))


def cmd_rank(args: argparse.Namespace) -> None:
    start = time.time()
    jd_config = load_config("jd_profile.yaml")
    weights_config = load_config("ranking_weights.yaml")
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)

    candidates = load_candidates(candidates_path)
    matcher = SemanticMatcher(model_name=args.model)
    matcher.load_artifacts()

    engine = RankingEngine(jd_config, weights_config, matcher)
    ranked = engine.rank(candidates, top_k=100)
    engine.write_submission(ranked, out_path)

    elapsed = time.time() - start
    summary = summarize_submission(ranked)
    logger.info("Ranking completed in %.1fs", elapsed)
    logger.info("Summary: %s", summary)

    if args.validate:
        ok, errors = run_validator(out_path)
        if ok:
            logger.info("Submission validation passed.")
        else:
            for e in errors:
                logger.error("Validation: %s", e)
            raise SystemExit(1)


def cmd_eda(args: argparse.Namespace) -> None:
    from scripts.run_eda import run_eda

    run_eda(Path(args.candidates), Path(args.out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Redrob Candidate Discovery Engine")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("precompute", help="Precompute embeddings and BM25 index")
    p_pre.add_argument("--candidates", default=str(DATA_RAW / "candidates.jsonl"))
    p_pre.set_defaults(func=cmd_precompute)

    p_rank = sub.add_parser("rank", help="Rank candidates and write submission CSV")
    p_rank.add_argument("--candidates", default=str(DATA_RAW / "candidates.jsonl"))
    p_rank.add_argument("--out", default=str(OUTPUTS_DIR / "submission.csv"))
    p_rank.add_argument("--validate", action="store_true")
    p_rank.set_defaults(func=cmd_rank)

    p_eda = sub.add_parser("eda", help="Run dataset audit")
    p_eda.add_argument("--candidates", default=str(DATA_RAW / "candidates.jsonl"))
    p_eda.add_argument("--out", default=str(OUTPUTS_DIR / "eda_report.json"))
    p_eda.set_defaults(func=cmd_eda)

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
