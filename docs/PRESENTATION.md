# Presentation — Intelligent Candidate Discovery Engine

10-slide deck content for the Redrob Data & AI Challenge final submission.

---

## Slide 1: Title

**Title:** Intelligent Candidate Discovery Engine  
**Subtitle:** Hybrid Ranking for Senior AI Engineer Discovery at Scale

**Bullet points:**
- Redrob Data & AI Challenge — Final Submission
- 100,000 candidates → Top 100 ranked with explainable reasoning
- Production-grade, CPU-only, constraint-compliant pipeline

**Speaker notes:**
Open by framing this as a real recruiting systems problem, not a Kaggle-style tabular ML task. We built a ranker that a Series A talent intelligence company could actually deploy — respecting latency, cost, and interpretability constraints. The target role is Senior AI Engineer on Redrob's founding AI team.

**Suggested visual:** Title slide with Redrob logo placeholder, pipeline icon (JSONL → Ranker → CSV), and key stat badges: "100K candidates", "28s runtime", "0% honeypots".

---

## Slide 2: Problem Statement

**Title:** What We're Solving

**Bullet points:**
- Rank top 100 from 100,000 candidates for a Senior AI Engineer JD
- Scored on NDCG@10 (50%), NDCG@50 (30%), MAP (15%), P@10 (5%)
- Hard constraints: ≤5 min, 16 GB RAM, CPU-only, no network at ranking time
- Dataset contains deliberate traps: keyword stuffers, honeypots, inactive profiles
- Stage 3 disqualification if >10% honeypots in top 100

**Speaker notes:**
Emphasize that top-10 quality dominates the composite score — this shaped our entire architecture. Also highlight that the JD explicitly says "don't count AI keywords" and the sample submission deliberately ranks HR Managers first as an anti-pattern. Our system had to reason about what the JD means, not just what it says.

**Suggested visual:** Scoring formula diagram + constraint table. Red warning box for trap types.

---

## Slide 3: Dataset Overview

**Title:** Know Your Pool

**Bullet points:**
- 100,000 unique candidates, 465 MB JSONL, 0 parse errors
- 75% India-based; YoE mean 7.2 years (JD target: 5–9)
- Only ~994 AI/ML-titled candidates in entire pool
- Top titles: Business Analyst, HR Manager, Mechanical Engineer
- 4,862 keyword stuffers; ~80 honeypots; 76% lack skill assessments

**Speaker notes:**
This is a needle-in-haystack problem. The pool looks nothing like a typical ML job board. If you rank by skill keyword count, you get HR Managers with 9 "AI core skills" — exactly what the sample submission shows at ranks 1–2. Our EDA informed every design decision.

**Suggested visual:** Bar chart of top-15 titles (mostly non-AI). Callout box: "~1% of pool has AI titles."

---

## Slide 4: Architecture

**Title:** Two-Stage Hybrid Ranker

**Bullet points:**
- **Offline precompute:** Sentence-transformer embeddings + BM25 index
- **Stage A — Coarse filter:** ~15K candidates by title, semantic, or career match
- **Stage B — Hybrid scoring:** Weighted features × behavioral modifier − penalties
- **Stage C — Output:** Monotonic scores + factual reasoning per candidate
- Config-driven weights — no code changes needed to tune

**Speaker notes:**
Walk through the three phases. Key insight: separate expensive embedding work (offline) from the 5-minute ranking step. At ranking time we only do batch dot products and structured feature computation — no model inference, no API calls.

**Suggested visual:** Architecture diagram (from TECHNICAL_REPORT Section 3). Color-code offline vs online phases.

---

## Slide 5: Data Pipeline

**Title:** From JSONL to Submission

**Bullet points:**
- Load 100K candidates → extract canonical text per profile
- Trap detection: honeypots, keyword stuffers, consulting-only, title-chasers
- Embed profiles with `all-MiniLM-L6-v2` (384-dim, L2-normalized)
- Build BM25 index over career history descriptions
- Batch score all candidates → filter → rank top 100 → write CSV

**Speaker notes:**
Highlight the trap detection layer — this is what separates a production ranker from a homework embedding cosine similarity script. We detect timeline impossibilities and expert-skills-with-zero-duration patterns. Honeypots are excluded before scoring, not just penalized.

**Suggested visual:** Flowchart: JSONL → Preprocessing → Embeddings/BM25 → Filter → Score → CSV. Show artifact files in `data/processed/`.

---

## Slide 6: Feature Engineering

**Title:** Beyond Keyword Matching

**Bullet points:**
- **Semantic match (0.35):** Full-profile embedding similarity to JD
- **Title + career (0.25):** Role relevance + production narrative keywords
- **Trusted skills (0.20):** proficiency × endorsements × duration — not raw count
- **YoE / education / location (0.10):** Gaussian YoE peak at 7 yrs, tier boost
- **Career quality (0.10):** Product company signals; consulting/title-chaser penalties

**Speaker notes:**
The trusted skill formula is our anti-keyword-stuffer weapon. An HR Manager with "expert" in 9 AI skills but 0 months duration gets near-zero trust score AND a title penalty AND a keyword-stuffer flag. Meanwhile, a Backend Engineer whose career description mentions "ranking system shipped to production" scores high on career narrative even without buzzwords in skills.

**Suggested visual:** Feature weight pie chart. Side-by-side: keyword stuffer vs genuine ML engineer feature breakdown.

---

## Slide 7: Hybrid Ranking Engine

**Title:** The Scoring Formula

**Bullet points:**
```
base = 0.35·semantic + 0.25·title_career + 0.20·skills
     + 0.10·yoe_edu_loc + 0.10·career_quality + 0.05·BM25
raw  = base × behavioral_modifier − penalties
```
- Behavioral modifier: response rate, activity, notice period, open-to-work
- Penalties: honeypot exclusion (1.0), keyword stuffer (0.6)
- Reasoning: factual templates from feature breakdown — no LLM at ranking time
- Tie-break: candidate_id ascending (validator requirement)

**Speaker notes:**
Explain why behavioral is a multiplier, not a primary signal. The JD says a perfect-on-paper candidate who hasn't logged in for 6 months isn't hireable. But we don't want behavioral signals to override a fundamentally wrong title match — hence multiplier, not additive weight.

**Suggested visual:** Formula with color-coded components. Example reasoning string for rank #1 candidate.

---

## Slide 8: Results

**Title:** Validation & Performance

**Bullet points:**
- **Validator:** Passed all format checks
- **Runtime:** 19–28 seconds on 100K candidates (limit: 300s)
- **Honeypots in top 100:** 0% (gate: ≤10%)
- **Top 10:** All AI/ML titles — Senior AI Engineer, NLP Engineer, Search Engineer
- **Reasoning:** 100 unique, factual strings (avg 162 chars)
- **Tests:** 8/8 passing including full-scale benchmark

**Speaker notes:**
Contrast with sample submission: ranks 1–2 are HR Managers. Our rank #1 is a Senior ML Engineer in Noida with trusted Weaviate/Pinecone skills and 0.61 response rate. Every top-10 candidate has an AI/ML title, relevant skills with duration backing, and reasonable behavioral signals.

**Suggested visual:** Results table (top 10). Runtime bar chart (28s vs 300s limit). Green checkmarks for validation checklist.

---

## Slide 9: Challenges & Future Work

**Title:** What We Learned

**Bullet points:**
- **Challenge:** 99% of pool is non-AI — precision at top matters more than recall
- **Challenge:** Keyword stuffers look strong on paper — title gating is decisive
- **Challenge:** First implementation took >6 min — fixed with batch scoring
- **Future:** Cross-encoder reranking on top 500
- **Future:** Learning-to-rank with proxy labels; FAISS for larger pools
- **Future:** Hosted sandbox (HuggingFace Space) for Stage 3 reproduction

**Speaker notes:**
Be honest about limitations. Weights are hand-tuned without ground truth labels. Cross-encoder reranking is configured but not yet implemented. The system is robust against documented traps but may miss subtle honeypots our heuristics don't catch. These are natural v2 improvements.

**Suggested visual:** Timeline: v1 (slow) → v2 (batch, 28s) → v3 (cross-encoder, planned). Limitations as muted callout boxes.

---

## Slide 10: Conclusion

**Title:** Production-Ready Candidate Discovery

**Bullet points:**
- Hybrid ranker combining semantic, structured, and behavioral signals
- Explicitly designed against dataset traps and JD guidance
- Constraint-compliant: 28s / CPU / no network / validator-passing
- Fully reproducible: one command, config-driven, tested at scale
- Explainable output ready for Stage 4 manual review

**Speaker notes:**
Close by tying back to Redrob's actual product need: ranking and retrieval for recruiter-candidate matching. This system demonstrates the exact skills the JD asks for — embeddings, hybrid search, ranking evaluation thinking, and shipping under production constraints. Happy to walk through any component in the defend-your-work interview.

**Suggested visual:** Summary badge grid: "100K → 100", "28s", "0% honeypots", "8/8 tests", "MIT licensed". QR code to GitHub repo.

---

## Appendix: Suggested Diagram Assets

| Slide | Diagram Type | Tool Suggestion |
|-------|-------------|-----------------|
| 4 | Architecture flowchart | Mermaid / draw.io |
| 5 | Data pipeline | Left-to-right flow with file icons |
| 6 | Feature weights | Pie chart or stacked bar |
| 7 | Scoring formula | Annotated equation with color |
| 8 | Results table | Simple table with green highlights |
| 9 | Version timeline | Horizontal timeline |

All Mermaid diagrams from `README.md` and `TECHNICAL_REPORT.md` can be exported directly for slides.
