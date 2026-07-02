"""Behavioral signal scoring from Redrob platform data."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def days_since(date_str: str, reference: datetime | None = None) -> float:
    if not date_str:
        return 9999.0
    ref = reference or datetime(2026, 7, 2)
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return max((ref - dt).days, 0)


def behavioral_modifier(signals: dict[str, Any], config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    beh = config.get("behavioral", {})
    notice_pref = beh.get("notice_period_preferred_days", 30)
    notice_ok = beh.get("notice_period_acceptable_days", 90)
    inactive_threshold = beh.get("inactive_days_penalty_threshold", 180)

    response_rate = signals.get("recruiter_response_rate", 0.0)
    interview_rate = signals.get("interview_completion_rate", 0.0)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.6
    notice = signals.get("notice_period_days", 90)
    inactive_days = days_since(signals.get("last_active_date", ""))

    notice_score = 1.0
    if notice <= notice_pref:
        notice_score = 1.0
    elif notice <= notice_ok:
        notice_score = 0.75
    else:
        notice_score = 0.45

    activity_score = 1.0
    if inactive_days > inactive_threshold:
        activity_score = 0.4
    elif inactive_days > 90:
        activity_score = 0.7

    github = signals.get("github_activity_score", -1)
    github_score = github / 100.0 if github >= 0 else 0.5

    saved = min(signals.get("saved_by_recruiters_30d", 0) / 20.0, 1.0)
    verified = sum([
        0.15 if signals.get("verified_email") else 0.0,
        0.15 if signals.get("verified_phone") else 0.0,
        0.1 if signals.get("linkedin_connected") else 0.0,
    ])

    completeness = signals.get("profile_completeness_score", 50) / 100.0

    modifier = (
        0.25 * response_rate
        + 0.15 * interview_rate
        + 0.15 * open_to_work
        + 0.15 * notice_score
        + 0.10 * activity_score
        + 0.05 * github_score
        + 0.05 * saved
        + 0.05 * completeness
        + verified
    )
    modifier = max(min(modifier, 1.2), 0.35)

    breakdown = {
        "response_rate": response_rate,
        "interview_rate": interview_rate,
        "open_to_work": open_to_work,
        "notice_score": notice_score,
        "activity_score": activity_score,
        "github_score": github_score,
    }
    return modifier, breakdown
