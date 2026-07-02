"""Structured feature engineering for candidate ranking."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from src.preprocessing import PROFICIENCY_WEIGHT, CandidateFeatures
from src.utils import normalize_text

RESEARCH_ONLY_PATTERN = re.compile(
    r"\b(research scientist|phd|postdoc|academic|university lab)\b", re.I
)
CV_ONLY_PATTERN = re.compile(
    r"\b(computer vision|speech recognition|robotics)\b", re.I
)
NLP_IR_PATTERN = re.compile(
    r"\b(nlp|natural language|retrieval|ranking|embeddings|llm|rag)\b", re.I
)


def gaussian_yoe_score(yoe: float, target: float = 7.0, sigma: float = 2.0) -> float:
    return math.exp(-0.5 * ((yoe - target) / sigma) ** 2)


def trusted_skill_score(
    skills: list[dict[str, Any]],
    must_have: set[str],
    preferred: set[str],
) -> tuple[float, list[str], list[str]]:
    matched_must: list[str] = []
    matched_pref: list[str] = []
    must_score = 0.0
    pref_score = 0.0

    for skill in skills:
        name = normalize_text(skill.get("name", ""))
        prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.25)
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months") or 0
        trust = prof * math.log1p(endorsements) * min(duration / 12.0, 5.0) / 5.0
        if trust <= 0:
            trust = prof * 0.1

        for kw in must_have:
            if kw == name or kw in name:
                must_score += trust
                matched_must.append(skill.get("name", name))
                break
        else:
            for kw in preferred:
                if kw == name or kw in name:
                    pref_score += trust * 0.5
                    matched_pref.append(skill.get("name", name))
                    break

    must_norm = min(must_score / 4.0, 1.0)
    pref_norm = min(pref_score / 3.0, 1.0)
    return 0.7 * must_norm + 0.3 * pref_norm, matched_must, matched_pref


def career_narrative_score(career_text: str, keywords: list[str]) -> float:
    if not career_text:
        return 0.0
    hits = sum(1 for kw in keywords if kw in career_text)
    production = 1.0 if any(
        p in career_text
        for p in ("production", "deployed", "shipped", "scaled", "users")
    ) else 0.0
    return min(hits / max(len(keywords) * 0.25, 1), 1.0) * 0.7 + production * 0.3


def education_score(education: list[dict[str, Any]]) -> float:
    if not education:
        return 0.3
    tier_weights = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}
    best = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = normalize_text(edu.get("field_of_study", ""))
        base = tier_weights.get(tier, 0.4)
        if any(f in field for f in ("computer", "machine learning", "ai", "data science")):
            base += 0.1
        best = max(best, min(base, 1.0))
    return best


def location_score(profile: dict[str, Any], signals: dict[str, Any], config: dict[str, Any]) -> float:
    location = normalize_text(profile.get("location", ""))
    country = profile.get("country", "")
    preferred = [normalize_text(x) for x in config.get("locations_preferred", [])]
    score = 0.0
    if country == config.get("country_preferred", "India"):
        score += 0.5
    if any(c in location for c in preferred):
        score += 0.4
    if signals.get("willing_to_relocate"):
        score += 0.2
    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in ("hybrid", "flexible", "remote"):
        score += 0.1
    return min(score, 1.0)


def career_quality_score(cf: CandidateFeatures) -> float:
    score = 0.5
    career = cf.career_text
    title = normalize_text(cf.profile.get("current_title", ""))

    if cf.is_consulting_only:
        score -= 0.4
    if cf.is_title_chaser:
        score -= 0.3
    if RESEARCH_ONLY_PATTERN.search(title + " " + career):
        if "production" not in career and "deployed" not in career:
            score -= 0.35
    if CV_ONLY_PATTERN.search(career) and not NLP_IR_PATTERN.search(career):
        score -= 0.25

    product_signals = sum(
        1 for j in cf.career_history
        if any(w in normalize_text(j.get("description", "")) for w in ("product", "users", "shipped"))
    )
    score += min(product_signals * 0.15, 0.45)
    return max(min(score, 1.0), 0.0)


def assessment_boost(signals: dict[str, Any], must_have: set[str]) -> float:
    scores = signals.get("skill_assessment_scores") or {}
    if not scores:
        return 0.0
    relevant = []
    for skill, val in scores.items():
        sname = normalize_text(skill)
        if any(kw in sname for kw in must_have):
            relevant.append(val / 100.0)
    return sum(relevant) / len(relevant) if relevant else 0.0


def compute_structured_features(
    cf: CandidateFeatures,
    jd_config: dict[str, Any],
) -> dict[str, Any]:
    must_have = {normalize_text(s) for s in jd_config.get("must_have_skills", [])}
    preferred = {normalize_text(s) for s in jd_config.get("preferred_skills", [])}
    career_keywords = [normalize_text(k) for k in jd_config.get("career_keywords", [])]

    yoe = cf.profile.get("years_of_experience", 0.0)
    yoe_score = gaussian_yoe_score(float(yoe))
    skill_score, matched_must, matched_pref = trusted_skill_score(cf.skills, must_have, preferred)
    career_score = career_narrative_score(cf.career_text, career_keywords)
    edu_score = education_score(cf.education)
    loc_score = location_score(cf.profile, cf.redrob_signals, jd_config)
    quality_score = career_quality_score(cf)
    assess_score = assessment_boost(cf.redrob_signals, must_have)

    yoe_edu_loc = 0.4 * yoe_score + 0.3 * edu_score + 0.3 * loc_score
    title_career = 0.55 * cf.title_score + 0.45 * career_score

    penalties = 0.0
    if cf.is_honeypot:
        penalties += 1.0
    if cf.is_keyword_stuffer:
        penalties += 0.6

    return {
        "title_career": title_career,
        "trusted_skills": min(skill_score + assess_score * 0.2, 1.0),
        "yoe_edu_loc": yoe_edu_loc,
        "career_quality": quality_score,
        "matched_must_skills": matched_must[:5],
        "matched_pref_skills": matched_pref[:3],
        "yoe": yoe,
        "penalties": penalties,
        "title_score": cf.title_score,
        "career_narrative": career_score,
    }
