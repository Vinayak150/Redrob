"""Hybrid ranking engine combining semantic, structured, and behavioral signals."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.behavioral_scoring import behavioral_modifier
from src.feature_engineering import compute_structured_features
from src.preprocessing import CandidateFeatures, extract_candidate_features
from src.semantic_matching import SemanticMatcher
from src.utils import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    candidate_id: str
    raw_score: float
    final_score: float = 0.0
    rank: int = 0
    confidence: float = 0.0
    reasoning: str = ""
    breakdown: dict[str, Any] = field(default_factory=dict)


def generate_reasoning(
    cf: CandidateFeatures,
    features: dict[str, Any],
    behavioral: dict[str, float],
    rank: int,
) -> str:
    title = cf.profile.get("current_title", "Unknown")
    yoe = cf.profile.get("years_of_experience", 0)
    location = cf.profile.get("location", "")
    must = features.get("matched_must_skills", [])
    pref = features.get("matched_pref_skills", [])
    response = behavioral.get("response_rate", 0)

    positives: list[str] = []
    if features.get("title_score", 0) >= 0.7:
        positives.append(f"{title} aligns with Senior AI Engineer scope")
    if must:
        positives.append(f"trusted skills include {', '.join(must[:3])}")
    if features.get("career_narrative", 0) >= 0.5:
        positives.append("career history shows ranking/retrieval production work")
    if location:
        positives.append(f"based in {location}")

    concerns: list[str] = []
    if cf.is_keyword_stuffer:
        concerns.append("title-skill mismatch suggests keyword stuffing")
    if cf.is_consulting_only:
        concerns.append("consulting-only career per JD disqualifier")
    if cf.is_title_chaser:
        concerns.append("short average tenures")
    notice = cf.redrob_signals.get("notice_period_days", 0)
    if notice > 90:
        concerns.append(f"notice period {notice} days")
    if response < 0.3:
        concerns.append(f"low recruiter response rate ({response:.2f})")
    if cf.is_honeypot:
        concerns.append("profile timeline inconsistencies")

    pos_text = "; ".join(positives[:2]) if positives else f"{title} with {yoe:.1f} yrs experience"
    if rank <= 20:
        tone = f"Strong fit: {pos_text}."
    elif rank <= 50:
        tone = f"Solid fit: {pos_text}."
    else:
        tone = f"Partial fit: {pos_text}."

    if concerns:
        tone += f" Concerns: {'; '.join(concerns[:2])}."
    else:
        tone += f" Response rate {response:.2f}."

    if pref and rank <= 30:
        tone += f" Preferred skills: {', '.join(pref[:2])}."

    return tone[:280]


class RankingEngine:
    def __init__(
        self,
        jd_config: dict[str, Any],
        weights_config: dict[str, Any],
        matcher: SemanticMatcher,
    ) -> None:
        self.jd_config = jd_config
        self.weights = weights_config["weights"]
        self.coarse = weights_config["coarse_filter"]
        self.score_range = weights_config["score_range"]
        self.matcher = matcher
        self.ai_keywords = {
            normalize_text(s)
            for s in jd_config.get("must_have_skills", []) + jd_config.get("preferred_skills", [])
        }
        self.positive_titles = jd_config.get("positive_titles", [])
        self.query_terms = normalize_text(jd_config.get("jd_text", "")).split()

    def score_candidate(
        self,
        cf: CandidateFeatures,
        semantic: float,
        bm25: float,
    ) -> tuple[float, dict[str, Any]]:
        structured = compute_structured_features(cf, self.jd_config)
        behavioral_mult, behavioral_breakdown = behavioral_modifier(cf.redrob_signals, self.jd_config)

        w = self.weights
        base = (
            w["semantic"] * semantic
            + w["title_career"] * structured["title_career"]
            + w["trusted_skills"] * structured["trusted_skills"]
            + w["yoe_edu_loc"] * structured["yoe_edu_loc"]
            + w["career_quality"] * structured["career_quality"]
            + 0.05 * bm25
        )
        raw = base * behavioral_mult - structured["penalties"]

        breakdown = {
            "semantic": semantic,
            "bm25": bm25,
            "behavioral_multiplier": behavioral_mult,
            "behavioral": behavioral_breakdown,
            **structured,
        }
        return raw, breakdown

    def _coarse_filter_from_arrays(
        self,
        candidates: list[dict[str, Any]],
        features_map: dict[str, CandidateFeatures],
        semantic: np.ndarray,
        bm25: np.ndarray,
    ) -> list[str]:
        career_keywords = self.jd_config.get("career_keywords", [])
        pool: list[tuple[str, float]] = []
        for i, cand in enumerate(candidates):
            cid = cand["candidate_id"]
            cf = features_map[cid]
            if cf.is_honeypot:
                continue
            sem = float(semantic[i])
            bm = float(bm25[i])
            title = cf.title_score
            career = bm * 0.6 + (
                1.0 if any(k in cf.career_text for k in career_keywords) else 0.0
            ) * 0.4

            passes = (
                title >= self.coarse["min_title_score"]
                or sem >= self.coarse["min_semantic"]
                or career >= self.coarse["min_career_narrative"]
            )
            if not passes:
                continue
            if cf.is_keyword_stuffer and title < 0.3:
                continue

            coarse_score = 0.4 * sem + 0.3 * title + 0.3 * career
            pool.append((cid, coarse_score))

        pool.sort(key=lambda x: (-x[1], x[0]))
        max_pool = self.coarse.get("max_pool_size", 15000)
        return [cid for cid, _ in pool[:max_pool]]

    def rank(
        self,
        candidates: list[dict[str, Any]],
        top_k: int = 100,
    ) -> list[RankedCandidate]:
        logger.info("Extracting features for %d candidates...", len(candidates))
        features_map = {
            c["candidate_id"]: extract_candidate_features(c, self.ai_keywords, self.positive_titles)
            for c in candidates
        }

        n = len(candidates)
        all_semantic = self.matcher.semantic_similarity_batch(np.arange(n))
        all_bm25 = self.matcher.bm25_scores_batch(self.query_terms)

        pool_ids = self._coarse_filter_from_arrays(candidates, features_map, all_semantic, all_bm25)
        logger.info("Coarse filter retained %d candidates", len(pool_ids))

        scored: list[RankedCandidate] = []
        for cid in pool_ids:
            cf = features_map[cid]
            idx = self.matcher.id_to_idx[cid]
            raw, breakdown = self.score_candidate(
                cf, float(all_semantic[idx]), float(all_bm25[idx])
            )
            scored.append(RankedCandidate(candidate_id=cid, raw_score=raw, breakdown=breakdown))

        scored.sort(key=lambda r: (-r.raw_score, r.candidate_id))

        if len(scored) < top_k:
            logger.warning("Only %d candidates in pool; padding from full set", len(scored))
            in_pool = {s.candidate_id for s in scored}
            extras: list[RankedCandidate] = []
            for i, cand in enumerate(candidates):
                cid = cand["candidate_id"]
                if cid in in_pool:
                    continue
                cf = features_map[cid]
                if cf.is_honeypot:
                    continue
                raw, breakdown = self.score_candidate(
                    cf, float(all_semantic[i]), float(all_bm25[i])
                )
                extras.append(RankedCandidate(candidate_id=cid, raw_score=raw, breakdown=breakdown))
            extras.sort(key=lambda r: (-r.raw_score, r.candidate_id))
            scored.extend(extras[: top_k - len(scored)])
            scored.sort(key=lambda r: (-r.raw_score, r.candidate_id))

        top = scored[:top_k]
        cutoff = scored[top_k].raw_score if len(scored) > top_k else 0.0
        max_score = self.score_range["max"]
        min_score = self.score_range["min"]
        step = (max_score - min_score) / max(top_k - 1, 1)

        for i, rc in enumerate(top):
            rc.rank = i + 1
            rc.final_score = round(max_score - i * step, 4)
            cf = features_map[rc.candidate_id]
            margin = rc.raw_score - cutoff
            rc.confidence = round(min(max(margin / max(abs(cutoff), 0.01), 0.0), 1.0), 3)
            rc.reasoning = generate_reasoning(
                cf, rc.breakdown, rc.breakdown.get("behavioral", {}), rc.rank
            )

        return top

    def write_submission(self, ranked: list[RankedCandidate], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rc in ranked:
                writer.writerow([rc.candidate_id, rc.rank, rc.final_score, rc.reasoning])
        logger.info("Wrote submission to %s", out_path)
