"""Dataset exploratory analysis script."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.preprocessing import detect_honeypots, detect_keyword_stuffer, extract_candidate_features
from src.utils import load_config, iter_candidates, normalize_text


def run_eda(candidates_path: Path, out_path: Path) -> dict[str, Any]:
    jd_config = load_config("jd_profile.yaml")
    ai_keywords = {
        normalize_text(s)
        for s in jd_config.get("must_have_skills", []) + jd_config.get("preferred_skills", [])
    }
    positive_titles = jd_config.get("positive_titles", [])

    n = 0
    yoe_vals: list[float] = []
    skill_counts: list[int] = []
    honeypots: list[str] = []
    keyword_stuffers: list[str] = []
    titles = Counter()
    countries = Counter()
    open_to_work = Counter()
    missing_certs = 0

    for cand in iter_candidates(candidates_path):
        n += 1
        prof = cand.get("profile", {})
        yoe_vals.append(prof.get("years_of_experience", 0))
        skill_counts.append(len(cand.get("skills", [])))
        titles[prof.get("current_title", "")] += 1
        countries[prof.get("country", "")] += 1
        open_to_work[cand["redrob_signals"].get("open_to_work_flag")] += 1
        if not cand.get("certifications"):
            missing_certs += 1

        hp, _ = detect_honeypots(cand)
        if hp:
            honeypots.append(cand["candidate_id"])
        if detect_keyword_stuffer(cand, ai_keywords):
            keyword_stuffers.append(cand["candidate_id"])

    report: dict[str, Any] = {
        "row_count": n,
        "years_of_experience": {
            "min": min(yoe_vals),
            "max": max(yoe_vals),
            "mean": round(statistics.mean(yoe_vals), 2),
            "median": round(statistics.median(yoe_vals), 2),
        },
        "skills_per_candidate": {
            "min": min(skill_counts),
            "max": max(skill_counts),
            "mean": round(statistics.mean(skill_counts), 2),
        },
        "top_titles": titles.most_common(15),
        "top_countries": countries.most_common(10),
        "open_to_work": dict(open_to_work),
        "missing_certifications_pct": round(100 * missing_certs / n, 1),
        "honeypot_count": len(honeypots),
        "honeypot_sample": honeypots[:20],
        "keyword_stuffer_count": len(keyword_stuffers),
        "keyword_stuffer_sample": keyword_stuffers[:20],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
