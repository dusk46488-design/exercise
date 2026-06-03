# Data Collection & Generator Design

## 1. Data Collection Method

### 1.1 Platform Architecture

A single **computer-based experimental platform** (PsychoPy / Gorilla.js) serves all four studies via a standardized pipeline:

```
Participant → Browser → Platform Server → CSV Export
                             ├── Task Engine (trials, timing, randomization)
                             ├── Survey Engine (Likert, open-text, JOL)
                             └── Session Manager (scheduling, counterbalancing)
```

### 1.2 Per-Study Data Collection

**RS1 — Note-Formatting Methods** (between-subjects, N=120)

| Phase | Instruments | Key Metric |
|-------|------------|------------|
| Screening | Demographics, GPA, note-taking experience | — |
| Review (Day 3) | Self-paced timer: `review_start`, `review_end`, `review_duration_s` | — |
| Test (Day 3) | 20 MCQ + 6 short-answer + 2 essay, tagged by `question_type` | `test_total_score` (0–100) |
| Survey | NASA-TLX (5 items), ease-of-review (2 items), satisfaction (2 items) | `review_efficiency` = score ÷ review_min |

**RS2 — Study-Planning Tools** (between-subjects, N=90, 14-day)

| Phase | Instruments | Key Metric |
|-------|------------|------------|
| Screening | SSRQ (31 items), baseline habits (6 items) | Self-regulation score |
| Daily log (×14) | Planned/completed tasks, start times, study mins, planning mins, interruptions, productivity (1–7) | `mean_productivity_index` |
| Midpoint (Day 7) | Adherence, usability (5 items) | — |
| Exit (Day 14) | Satisfaction, behavioral change (9 items) + digital-tool export (Group A) | — |

**RS3 — Spaced Repetition vs. Daily Review** (2×4 mixed, N=80, 97-day)

| Phase | Instruments | Key Metric |
|-------|------------|------------|
| Baseline | Digit span (fwd + bwd), vocabulary pre-test (40 items) | Prior-knowledge screen |
| Learning (Day 1) | Per-trial flashcard log: `item_id`, `self_rated_correct`, `trial_rt_ms`, `pass_number` | Trials-to-90%-criterion |
| Review (Days 2–26) | Same flashcard log; spaced=4 sessions, daily=6 sessions | `total_review_time_s` (covariate) |
| Test ×4 (Days 7/14/37/97) | 40-item cued-recall + JOL (0–100 prediction) | `retention_score`, `forgetting_slope` |
| Relearning (Day 97) | Same flashcard log | `relearning_savings` |

**RS4 — Background Noise & Focus** (within-subjects, N=45)

| Phase | Instruments | Key Metric |
|-------|------------|------------|
| Screening | Pure-tone audiometry (5 freq × 2 ears), ASRS-v1.1 (6 items) | Hearing/ADHD screen |
| CPT ×3 | 600 trials/session, 20% targets; per-trial: `stimulus`, `is_target`, `response_rt_ms`, `hit/miss/FA/CR` | `d_prime`, `rt_variability_coefficient` |
| Post-session ×3 | NASA-TLX (6 items), focus (1 item), distraction (1 item) | Frustration, perceived focus |
| Exit | Preference ranking (1–3), open-ended rationale | — |

### 1.3 Cross-Study Quality Controls

| Rule | Applies To |
|------|-----------|
| Exclude on screening failure (prior knowledge, hearing, ADHD) | All |
| Flag if > 3 missed logs/sessions | RS2, RS3, RS4 |
| Attention catch-trials on long-gap tests | RS3 (Days 37, 97) |
| CPT validity: `hit_rate` ≥ 0.50, `false_alarm_rate` ≤ 0.40 | RS4 |
| Sound-level calibration ±3 dB of target | RS4 |
| IRR κ ≥ 0.80 for manual ratings (notes fidelity, essays) | RS1 |

---

## 2. Generator Design

### 2.1 Subspace Selection

The generator targets **RS3 only** as proof-of-concept. RS3 spans the richest data-type coverage (binary, continuous RT, longitudinal, Likert, between-subjects factor), has the highest structural complexity (2×4 mixed design), and is tagged High Priority. Validating on RS3 validates the schema for all other studies.

**Scope:** 80 participants (40 spaced, 40 daily), each with a 90%-criterion learning session, condition-dependent review sessions (4 or 6), and 4 cued-recall tests with JOLs.

### 2.2 Statistical Distributions

| Variable | Distribution | Parameters | Rationale |
|----------|-------------|------------|-----------|
| Trial RT (ms) | Ex-Gaussian(μ, σ, τ) | μ=300–600, σ=50–150, τ=100–400 | Wagenmakers & Brown (2007): RTs are right-skewed, SD ∝ mean |
| Recall accuracy | Bernoulli(logit⁻¹(η)) | β₀=2.2, β₁=−0.4, β₂=−1.2, β₃=+0.8, σ_u=0.3 | Logistic mixed-effects model with crossover interaction (H1b) |
| JOL (0–100) | TruncatedNormal(score + bias, σ=10) | bias_daily=+5, bias_spaced=0 | Overconfidence under high-familiarity (daily) condition |
| Trials-to-criterion | Zero-Truncated NB(μ, φ=1.5) | μ varies by session type | Count data with overdispersion, minimum 1 |
| Digit span | Normal(μ, σ), rounded, clipped | See table below | Standard population parameters |

**Participant covariates:**

| Covariate | Distribution | Range |
|-----------|-------------|-------|
| Digit span forward | Normal(7, 1.5) → round → clip | 3–9 |
| Digit span backward | Normal(5.5, 1.5) → round → clip | 2–8 |
| Random intercept uᵢ | Normal(0, 0.3) | — |

The logistic parameters produce the crossover interaction predicted by H1b: the daily group outperforms at Day 7 (c. 90% vs. 86%), but the spaced group's slower decay (β₃ = +0.8) causes them to overtake by Day 37.

### 2.3 CSV Output Format

Four tidy-format tables (Wickham, 2014) joined by `participant_id`:

| File | Granularity | Key Columns |
|------|------------|-------------|
| `rs3_participants.csv` | 1 row/participant | `participant_id`, `condition`, `digit_span_*`, `random_intercept` |
| `rs3_trials.csv` | 1 row/flashcard trial | +`day`, `session_type`, `item_id`, `pass_number`, `self_rated_correct`, `trial_rt_ms`, `is_bogus` |
| `rs3_sessions.csv` | 1 row/session | +`total_trials`, `total_time_s`, `passes_to_criterion`, `final_accuracy` |
| `rs3_tests.csv` | 1 row/test occasion | +`test_occasion`, `days_since_learning`, `retention_score`, `jol_prediction`, `is_missing`, `is_bogus_jol` |

Missing values use blank/empty cells consistently. A companion `rs3_data_dictionary.csv` documents all columns.

### 2.4 Bogus Data Injection

Following Meade & Craig's (2012) taxonomy of careless responding:

| Type | Rate | Implementation |
|------|------|---------------|
| Random careless JOL | 5% of JOL responses | Replace with Uniform(0, 100) |
| Straight-lining JOL | 3% of participants | Set all JOL at one random occasion to 50 |
| Implausible-fast RT | 0.2% of trials | Replace RT with Uniform(50, 150) ms |
| Implausible-slow RT | 0.5% of trials | Replace RT with Uniform(10000, 30000) ms |

All injected bogus data carry ground-truth flags (`is_bogus`, `is_bogus_jol`) for detection-method evaluation.

### 2.5 Missing Data Injection

Under Rubin's (1976) three-mechanism taxonomy (Little & Rubin, 2019):

| Mechanism | Target | Rate | Logic |
|-----------|--------|------|-------|
| **MCAR** | Random trial rows | 1% deleted | `Uniform(0,1) < 0.01` |
| **MCAR** | Random test scores | 2% → NA | `Uniform(0,1) < 0.02` |
| **MAR** | Day 97 scores | ~15–25% | P(missing) = logit⁻¹(2.0 − 0.5 × digit_span_combined) |
| **MNAR** | Day 37 scores | ~10–15% | P(missing) = logit⁻¹(0.5 − 0.08 × true_score) |
| **Attrition** | Monotone dropout | ~25% total | Once missing, all later occasions missing; higher risk for daily-condition + low-span participants |

Missing data are injected post-generation: the complete dataset is the ground truth, and missingness is applied on top — enabling imputation-method benchmarking against known true values.

---

## 3. Implementation

The generator is implemented as a single-file Python CLI at `rs3_generator.py` following a backend→model→pipeline architecture. Usage:

```bash
python rs3_generator.py --seed 2026 --n 40 -o ./output
```

Generated output (verified):

| File | Rows | Notes |
|------|------|-------|
| `rs3_participants.csv` | 80 | 40 spaced + 40 daily |
| `rs3_trials.csv` | ~42,000 | ~294 bogus RTs; ~1% MCAR-deleted |
| `rs3_sessions.csv` | 480 | 80 learning + 400 review |
| `rs3_tests.csv` | 320 | ~33 missing scores; ~12 bogus JOLs |
| `rs3_data_dictionary.csv` | 34 columns | Complete codebook |

---

## 4. Literature

| # | Core Concept | Reference |
|---|-------------|-----------|
| 1 | Proof-of-concept / pilot rationale | Leon, A. C., Davis, L. L., & Kraemer, H. C. (2011). The role and interpretation of pilot studies in clinical research. *Journal of Psychiatric Research*, 45(5), 626–629. [doi:10.1016/j.jpsychires.2010.10.008](https://doi.org/10.1016/j.jpsychires.2010.10.008) |
| 2 | Ex-Gaussian RT distribution | Wagenmakers, E.-J., & Brown, S. (2007). On the linear relation between the mean and the standard deviation of a response time distribution. *Psychological Review*, 114(3), 830–841. [doi:10.1037/0033-295X.114.3.830](https://doi.org/10.1037/0033-295X.114.3.830) |
| 3 | Tidy data principles | Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1–23. [doi:10.18637/jss.v059.i10](https://doi.org/10.18637/jss.v059.i10) |
| 4 | Careless responding | Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. *Psychological Methods*, 17(3), 437–455. [doi:10.1037/a0028085](https://doi.org/10.1037/a0028085) |
| 5 | Missing data mechanisms | Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley. [doi:10.1002/9781119482260](https://doi.org/10.1002/9781119482260) |
