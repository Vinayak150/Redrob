"""Tests for honeypot detection."""

from src.preprocessing import detect_honeypots, detect_keyword_stuffer
from src.utils import normalize_text


def _base_candidate():
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Engineer",
            "summary": "Summary",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "Acme",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "Acme",
            "title": "ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 72,
            "is_current": True,
            "industry": "Software",
            "company_size": "51-200",
            "description": "Built ranking systems in production.",
        }],
        "education": [{
            "institution": "IIT",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2014,
            "end_year": 2018,
            "tier": "tier_1",
        }],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
            {"name": "Milvus", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
        ],
        "redrob_signals": {
            "profile_completeness_score": 80,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4,
            "skill_assessment_scores": {"Python": 90},
            "connection_count": 100,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def test_honeypot_expert_zero_duration():
    cand = _base_candidate()
    cand["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for i in range(4)
    ]
    is_hp, reasons = detect_honeypots(cand)
    assert is_hp
    assert any("expert_skills_zero_duration" in r for r in reasons)


def test_keyword_stuffer():
    cand = _base_candidate()
    cand["profile"]["current_title"] = "HR Manager"
    cand["skills"] = [
        {"name": n, "proficiency": "expert", "endorsements": 50, "duration_months": 60}
        for n in ["Python", "Milvus", "Embeddings", "FAISS", "NDCG", "LoRA", "RAG"]
    ]
    ai_kw = {normalize_text(x) for x in ["python", "milvus", "embeddings", "faiss", "ndcg", "lora", "rag"]}
    assert detect_keyword_stuffer(cand, ai_kw)
