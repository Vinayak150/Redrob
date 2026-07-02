# Technical Report — Intelligent Candidate Discovery Engine

**Redrob Data & AI Challenge | Final Submission**  
**Date:** July 2026  
**Repository:** `/Users/vinayakmahindrakar/Sites/Redrob`

---

## 1. Problem Statement

The challenge requires ranking the **top 100 candidates** from a pool of **100,000** anonymized profiles against a **Senior AI Engineer — Founding Team** job description at Redrob AI.

### Evaluation Objective

Hidden ground truth scoring:

```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**Top-10 precision dominates.** The system must optimize for high-quality picks at ranks 1–10, not broad recall across all 100 positions.

### Hard Constraints

| Constraint | Limit |
|------------|-------|
| Runtime (ranking step) | ≤ 5 minutes |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only |
| Network | Off during ranking |
| Disk (intermediate) | ≤ 5 GB |
| Honeypot gate (Stage 3) | ≤ 10% honeypots in top 100 |

### Dataset Traps

The JD explicitly warns against naive keyword matching. The dataset contains:

- **Keyword stuffers** — unrelated titles (HR Manager, Accountant) with many AI skills listed
- **Honeypots** (~80) — subtly impossible profiles (expert skills with 0 duration, timeline inconsistencies)
- **Plain-language strong fits** — candidates whose career narratives demonstrate ranking/retrieval experience without buzzword-heavy skill lists
- **Behavioral traps** — perfect-on-paper profiles with low recruiter response rates or long inactivity

---

## 2. Dataset Summary

| Metric | Value |
|--------|-------|
| Total candidates | 100,000 |
| Unique IDs | 100,000 (0 duplicates) |
| File size | 465 MB JSONL |
| Years of experience | 1.0 – 16.9 (mean 7.2, median 6.8) |
| Skills per candidate | 5 – 23 (mean 9.6) |
| India-based | 75,113 (75%) |
| Open to work | 35,339 (35%) |
| AI/ML-titled candidates | ~994 |
| Detected honeypots (heuristic) | 60 |
| Keyword stuffers (heuristic) | 4,862 |

**Key insight:** The candidate pool is overwhelmingly non-AI by title. A naive "count AI keywords" approach ranks HR Managers and Accountants highly — exactly the anti-pattern shown in `sample_submission.csv`.

Top titles: Business Analyst (5,833), HR Manager (5,830), Mechanical Engineer (5,791). Relevant pool: ML Engineer (167), AI Engineer (21), Senior AI Engineer (4), Search Engineer, Data Engineer, etc.

Full EDA: `outputs/eda_report.json`

---

## 3. Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE PRECOMPUTE                        │
│  candidates.jsonl → Sentence Transformers → embeddings.npy  │
│                   → BM25 index over career descriptions      │
│                   → honeypot ID detection                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    RANKING STEP (≤5 min)                     │
│                                                              │
│  Stage A: Coarse Filter (~15,000 candidates)                │
│    • Exclude honeypots                                       │
│    • Title OR semantic OR career narrative threshold         │
│    • Reject keyword stuffers with low title score            │
│                                                              │
│  Stage B: Hybrid Scoring (filtered pool)                     │
│    base = Σ(weight_i × feature_i) + 0.05 × BM25             │
│    raw = base × behavioral_modifier − penalties              │
│                                                              │
│  Stage C: Rank, assign monotonic scores, generate reasoning  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    submission.csv
```

### Design Principles

1. **Title + career narrative > skill keyword count** (per JD guidance)
2. **Trusted skills** — proficiency × endorsements × duration, not raw skill list length
3. **Behavioral availability as multiplier** — inactive candidates down-weighted
4. **Explicit trap guards** — honeypots excluded; keyword stuffers penalized
5. **Explainability** — reasoning generated from factual feature breakdowns, not LLM calls

---

## 4. Feature Engineering

### 4.1 Semantic Match (weight: 0.35)

Cosine similarity between precomputed candidate embedding and JD embedding.

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-efficient)
- Candidate text: headline + summary + title + career descriptions + education + skills
- Captures plain-language fits whose skill lists omit buzzwords

### 4.2 Title & Career Fit (weight: 0.25)

**Title score:**
- 1.0 for positive titles (Senior AI Engineer, ML Engineer, Search Engineer, etc.)
- 0.85 for AI/ML regex match in title or headline
- 0.05 for explicit negative titles (HR Manager, Accountant, etc.)

**Career narrative score:**
- Keyword hits in `career_history.description` for: ranking, retrieval, recommendation, embeddings, production deployment
- Production signal boost for "shipped", "deployed", "scaled", "users"

### 4.3 Trusted Skill Match (weight: 0.20)

For each skill matching JD must-have or preferred lists:

```
trust = proficiency_weight × log(1 + endorsements) × min(duration_months / 12, 5) / 5
```

Must-have skills weighted 70%, preferred 30%. Platform assessment scores (`skill_assessment_scores`) provide additional boost when present.

**Why:** Penalizes "expert Python" with 0 endorsements and 0 months — a honeypot pattern.

### 4.4 YoE / Education / Location (weight: 0.10)

- **YoE:** Gaussian peak at 7 years (JD target 5–9)
- **Education:** Tier boost (tier_1 > tier_2 > tier_3) + CS/ML field bonus
- **Location:** India + preferred cities (Pune, Noida, Bangalore, etc.) + willing_to_relocate + hybrid/flexible work mode

### 4.5 Career Quality (weight: 0.10)

Penalties for:
- Consulting-only career (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
- Title-chaser pattern (avg tenure < 18 months across 3+ jobs)
- Pure research without production signals
- CV/speech/robotics without NLP/IR exposure

Boosts for product-company signals in career descriptions.

### 4.6 Penalty Terms

| Penalty | Trigger | Magnitude |
|---------|---------|-----------|
| Honeypot | Timeline/skill impossibilities | Excluded from pool + 1.0 penalty |
| Keyword stuffer | 6+ AI skills + non-AI title | 0.6 penalty |

---

## 5. Semantic Matching Pipeline

### Precompute (offline)

```python
# 1. Build canonical text per candidate
# 2. Embed all 100K texts with all-MiniLM-L6-v2 (batch_size=256)
# 3. L2-normalize embeddings → candidate_embeddings.npy
# 4. Embed JD text → jd_embedding.npy
# 5. Tokenize career descriptions → BM25Okapi index
```

**Artifacts:** ~146 MB embeddings, BM25 pickle, honeypot ID list.

### Inference (ranking step)

```python
semantic_scores = embeddings @ jd_embedding          # O(n) batch dot product
bm25_scores = bm25.get_scores(jd_query_terms)          # O(n) single pass
```

Both computed once for all 100K candidates, then indexed for filtered pool scoring. No per-candidate model inference at ranking time.

---

## 6. Behavioral Signal Scoring

Redrob platform signals act as a **multiplier** (range 0.35 – 1.2), not a primary rank driver:

| Signal | Weight in Modifier |
|--------|-------------------|
| Recruiter response rate | 0.25 |
| Interview completion rate | 0.15 |
| Open to work flag | 0.15 |
| Notice period (≤30 days preferred) | 0.15 |
| Last active recency | 0.10 |
| GitHub activity score | 0.05 |
| Saved by recruiters (30d) | 0.05 |
| Profile completeness | 0.05 |
| Verified email/phone/LinkedIn | up to 0.40 bonus |

**Rationale:** A perfect skill match who hasn't logged in for 6 months and has 5% response rate is not hireable — per JD and `redrob_signals_doc`.

---

## 7. Hybrid Ranking Formula

```
base = 0.35 × semantic
     + 0.25 × title_career
     + 0.20 × trusted_skills
     + 0.10 × yoe_edu_loc
     + 0.10 × career_quality
     + 0.05 × bm25

raw_score = base × behavioral_modifier − penalties

final_rank = sort(raw_score DESC, candidate_id ASC)
final_score = linear_map(rank, 0.99 → 0.20)  # monotonic, validator-compliant
```

All weights configurable in `config/ranking_weights.yaml` without code changes.

---

## 8. Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Precomputed embeddings | Eliminates 100K model forward passes at ranking time |
| Batch dot product for semantic scores | Single O(n) matrix operation vs per-candidate calls |
| Single BM25 pass over full corpus | Avoids repeated `get_scores()` in loop |
| Two-stage coarse filter | Scores full hybrid formula on ~15K, not 100K |
| L2-normalized float64 matmul | Prevents overflow warnings in similarity computation |

### Runtime Progression

| Version | Ranking Time | Issue |
|---------|-------------|-------|
| v1 (per-candidate BM25) | >6 min (aborted) | O(pool × 100K) BM25 calls |
| v2 (batch scoring) | **19–28s** | Production-ready |

---

## 9. Runtime Analysis

Measured on MacBook Pro, Python 3.9, CPU-only, 16 GB RAM:

| Phase | Time | Memory |
|-------|------|--------|
| Precompute (100K embeddings) | ~198s | ~2 GB peak |
| Feature extraction (100K) | ~17s | ~1 GB |
| Batch semantic + BM25 | ~3s | ~200 MB |
| Hybrid scoring (15K pool) | ~2s | minimal |
| Reasoning + CSV write | <1s | minimal |
| **Total ranking step** | **~28s** | **< 4 GB** |

Well within 5-minute / 16 GB constraints.

Benchmark artifact: `outputs/benchmark.json`

---

## 10. Validation Results

### Official Validator

```
python validate_submission.py outputs/submission.csv
→ Submission is valid.
```

### Submission Audit

| Check | Result |
|-------|--------|
| Data rows | 100 ✓ |
| Header | `candidate_id,rank,score,reasoning` ✓ |
| Column order | Exact match ✓ |
| Unique candidate IDs | 100/100 ✓ |
| Ranks 1–100 each once | ✓ |
| Scores non-increasing | ✓ |
| Tie-break (ID ascending) | ✓ |
| All IDs in candidates.jsonl | ✓ |
| Empty reasoning | 0 ✓ |
| Unique reasoning strings | 100/100 ✓ |
| UTF-8 encoding | ✓ |

### Quality Checks

| Check | Result |
|-------|--------|
| Honeypots in top 100 | **0%** (gate: ≤10%) |
| Top-10 titles | All AI/ML roles ✓ |
| Top-1 | Senior ML Engineer, 7.2 yrs, Noida |
| vs sample_submission anti-pattern | No HR Managers in top 10 ✓ |

### Test Suite

```
8 passed (including full 100K benchmark test)
```

---

## 11. Why This Methodology Is Robust

1. **Aligns with JD intent, not keywords alone.** Title gating and career narrative scoring implement the JD's explicit warning against keyword counting.

2. **Captures plain-language fits.** Semantic embeddings over full profile text find candidates whose career stories match even without "Pinecone" in their skills list.

3. **Resists traps.** Honeypot exclusion, trusted-skill duration checks, and keyword-stuffer penalties address all documented trap types.

4. **Behaviorally grounded.** Availability multiplier ensures inactive/low-response candidates cannot dominate despite perfect skill lists.

5. **Explainable and auditable.** Every reasoning string cites specific profile facts (title, skills, response rate, concerns) — designed for Stage 4 manual review.

6. **Reproducible and constraint-compliant.** Precompute/rank separation, config-driven weights, single CLI command, validator integration, and benchmark test ensure Stage 3 reproduction.

7. **Optimized for NDCG@10.** Coarse filter and weight distribution prioritize title + semantic + career quality — the signals most likely to identify tier 3+ candidates at the very top.

---

## 12. Limitations

1. **No labeled training data.** Weights are hand-tuned from JD analysis and EDA, not learned from ground truth. Hidden labels could favor patterns we under-weight.

2. **Honeypot detection is heuristic.** We detect 60 candidates vs ~80 documented; some subtle honeypots may pass filters (though 0% in current top 100).

3. **Cross-encoder reranking not implemented.** Config references `cross_encoder_top_k: 500` but reranking is not active — reserved for future improvement.

4. **English-only model.** `all-MiniLM-L6-v2` may under-represent candidates with minimal English profile text.

5. **No online feedback loop.** Static ranking with no A/B test calibration against recruiter engagement.

6. **Geographic bias toward India.** Location scoring boosts India/preferred cities, which aligns with JD but may miss strong international candidates.

---

## 13. Future Improvements

1. **Cross-encoder reranking** on top 500 candidates for higher NDCG@10 precision
2. **Learning-to-rank** if partial labels or proxy relevance sets become available
3. **FAISS/ANN index** for sub-linear semantic retrieval at larger pool sizes
4. **Company quality features** — product vs consulting classification beyond firm name matching
5. **Behavioral twin detection** — pair-wise comparison of near-identical profiles with divergent signals
6. **Calibrated confidence scores** — map raw scores to estimated hire probability
7. **Sandbox deployment** — HuggingFace Space for Stage 3 sanity check

---

## Appendix: Top 10 Submission

| Rank | ID | Title | YoE | Location |
|------|-----|-------|-----|----------|
| 1 | CAND_0018499 | Senior ML Engineer | 7.2 | Noida |
| 2 | CAND_0071974 | Senior AI Engineer | 7.8 | Vizag |
| 3 | CAND_0046064 | Senior NLP Engineer | 8.9 | Coimbatore |
| 4 | CAND_0077337 | Staff ML Engineer | 7.0 | Kochi |
| 5 | CAND_0079387 | AI Engineer | 6.9 | Trivandrum |
| 6 | CAND_0002025 | Senior AI Engineer | 5.9 | Trivandrum |
| 7 | CAND_0055905 | Senior ML Engineer | 8.1 | London |
| 8 | CAND_0005538 | Senior AI Engineer | 5.9 | Kolkata |
| 9 | CAND_0030031 | AI Engineer | 5.7 | Trivandrum |
| 10 | CAND_0088025 | Staff ML Engineer | 8.6 | Jaipur |
