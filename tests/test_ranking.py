"""Tests for ranking engine components."""

from src.preprocessing import score_title, build_canonical_text
from src.feature_engineering import trusted_skill_score, gaussian_yoe_score


def test_title_score_ml_engineer():
    cand = {"profile": {"current_title": "ML Engineer", "headline": ""}}
    positive = ["ML Engineer", "AI Engineer"]
    assert score_title(cand, positive) >= 0.85


def test_title_score_hr_manager():
    cand = {"profile": {"current_title": "HR Manager", "headline": ""}}
    assert score_title(cand, []) <= 0.1


def test_yoe_gaussian_peak():
    assert gaussian_yoe_score(7.0) > gaussian_yoe_score(2.0)
    assert gaussian_yoe_score(7.0) > gaussian_yoe_score(15.0)


def test_trusted_skill_score():
    skills = [
        {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
        {"name": "Milvus", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
    ]
    must = {"python", "milvus"}
    pref = set()
    score, matched, _ = trusted_skill_score(skills, must, pref)
    assert score > 0
    assert "Python" in matched
