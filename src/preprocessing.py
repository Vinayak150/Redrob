"""Candidate preprocessing, canonical text, and trap detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.utils import normalize_text

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra",
}

AI_TITLE_PATTERN = re.compile(
    r"\b(ai|ml|machine learning|data sci|nlp|llm|deep learning|"
    r"recommendation|search engineer|applied ml)\b",
    re.I,
)

NON_AI_TITLE_PATTERN = re.compile(
    r"\b(hr manager|marketing manager|accountant|mechanical engineer|"
    r"civil engineer|graphic designer|content writer|sales executive|"
    r"customer support|operations manager)\b",
    re.I,
)

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}


@dataclass
class CandidateFeatures:
    candidate_id: str
    profile: dict[str, Any]
    career_history: list[dict[str, Any]]
    education: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    redrob_signals: dict[str, Any]
    canonical_text: str = ""
    career_text: str = ""
    skill_text: str = ""
    is_honeypot: bool = False
    honeypot_reasons: list[str] = field(default_factory=list)
    is_keyword_stuffer: bool = False
    is_consulting_only: bool = False
    is_title_chaser: bool = False
    title_score: float = 0.0
    ai_skill_hits: int = 0


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def build_canonical_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend([job.get("title", ""), job.get("company", ""), job.get("description", "")])
    for edu in candidate.get("education", []):
        parts.extend([edu.get("degree", ""), edu.get("field_of_study", ""), edu.get("institution", "")])
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    return normalize_text(" ".join(p for p in parts if p))


def build_career_text(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    for job in candidate.get("career_history", []):
        parts.extend([job.get("title", ""), job.get("description", "")])
    return normalize_text(" ".join(p for p in parts if p))


def build_skill_text(candidate: dict[str, Any]) -> str:
    return " ".join(s.get("name", "") for s in candidate.get("skills", []))


def detect_honeypots(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    skills = candidate.get("skills", [])
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if expert_zero >= 3:
        reasons.append(f"expert_skills_zero_duration:{expert_zero}")

    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    total_months = 0
    for job in candidate.get("career_history", []):
        dm = job.get("duration_months") or 0
        total_months += dm
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) if job.get("end_date") else datetime(2026, 7, 2)
        if start and end and start > end:
            reasons.append("start_after_end")
        if start and end:
            computed = (end.year - start.year) * 12 + (end.month - start.month)
            if dm > 0 and abs(computed - dm) > 6:
                reasons.append("duration_mismatch")

    if yoe and total_months > 0:
        implied_yoe = total_months / 12.0
        if implied_yoe + 2 < yoe * 0.5:
            reasons.append("yoe_exceeds_career_months")

    endorsements = sum(s.get("endorsements", 0) for s in skills)
    if len(skills) >= 15 and endorsements == 0 and expert_zero >= 2:
        reasons.append("many_skills_no_endorsements")

    return bool(reasons), reasons


def detect_keyword_stuffer(candidate: dict[str, Any], ai_keywords: set[str]) -> bool:
    title = candidate.get("profile", {}).get("current_title", "")
    if AI_TITLE_PATTERN.search(title):
        return False
    skill_names = {normalize_text(s.get("name", "")) for s in candidate.get("skills", [])}
    hits = sum(1 for kw in ai_keywords if kw in skill_names or any(kw in n for n in skill_names))
    return hits >= 6 and bool(NON_AI_TITLE_PATTERN.search(title) or not AI_TITLE_PATTERN.search(title))


def detect_consulting_only(candidate: dict[str, Any]) -> bool:
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    consulting = 0
    for job in jobs:
        company = normalize_text(job.get("company", ""))
        if any(firm in company for firm in CONSULTING_FIRMS):
            consulting += 1
    return consulting == len(jobs) and len(jobs) >= 2


def detect_title_chaser(candidate: dict[str, Any], min_avg_months: int = 18) -> bool:
    jobs = candidate.get("career_history", [])
    if len(jobs) < 3:
        return False
    durations = [j.get("duration_months") or 0 for j in jobs if not j.get("is_current")]
    if not durations:
        return False
    return (sum(durations) / len(durations)) < min_avg_months


def score_title(candidate: dict[str, Any], positive_titles: list[str]) -> float:
    title = normalize_text(candidate.get("profile", {}).get("current_title", ""))
    headline = normalize_text(candidate.get("profile", {}).get("headline", ""))
    positive = {normalize_text(t) for t in positive_titles}

    if title in positive:
        return 1.0
    if AI_TITLE_PATTERN.search(title) or AI_TITLE_PATTERN.search(headline):
        return 0.85
    if any(t in title for t in ("engineer", "scientist", "developer")):
        return 0.5
    if NON_AI_TITLE_PATTERN.search(title):
        return 0.05
    return 0.2


def count_ai_skill_hits(candidate: dict[str, Any], ai_keywords: set[str]) -> int:
    skill_names = [normalize_text(s.get("name", "")) for s in candidate.get("skills", [])]
    hits = 0
    for kw in ai_keywords:
        if any(kw == n or kw in n for n in skill_names):
            hits += 1
    return hits


def extract_candidate_features(
    candidate: dict[str, Any],
    ai_keywords: set[str],
    positive_titles: list[str],
) -> CandidateFeatures:
    honeypot, reasons = detect_honeypots(candidate)
    cf = CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        profile=candidate.get("profile", {}),
        career_history=candidate.get("career_history", []),
        education=candidate.get("education", []),
        skills=candidate.get("skills", []),
        redrob_signals=candidate.get("redrob_signals", {}),
        canonical_text=build_canonical_text(candidate),
        career_text=build_career_text(candidate),
        skill_text=build_skill_text(candidate),
        is_honeypot=honeypot,
        honeypot_reasons=reasons,
        is_keyword_stuffer=detect_keyword_stuffer(candidate, ai_keywords),
        is_consulting_only=detect_consulting_only(candidate),
        is_title_chaser=detect_title_chaser(candidate),
        title_score=score_title(candidate, positive_titles),
        ai_skill_hits=count_ai_skill_hits(candidate, ai_keywords),
    )
    return cf
