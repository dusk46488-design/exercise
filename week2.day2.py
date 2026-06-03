#!/usr/bin/env python3
"""Minimalist RS3 Data Generator — Spaced Repetition vs. Daily Review.

Architecture: Backend → Model → Pipeline
    Backend:   Statistical distributions and random-number machinery.
    Model:     Data structures (Participant, Trial, Session, Test).
    Pipeline:  Generate → Inject bogus → Inject missing → Export CSV.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: BACKEND — Statistical distributions
# ═══════════════════════════════════════════════════════════════════════════════

class Backend:
    """Stateless collection of distribution samplers driven by a shared RNG."""

    def __init__(self, seed: int = 2026):
        self.rng = np.random.default_rng(seed)

    def exgaussian(self, mu: float, sigma: float, tau: float, size: int = 1) -> np.ndarray:
        """Ex-Gaussian: Normal(mu, sigma) + Exponential(tau)."""
        g = self.rng.normal(mu, sigma, size=size)
        e = self.rng.exponential(tau, size=size)
        return g + e

    def bernoulli_logistic(self, logit: np.ndarray) -> np.ndarray:
        """Bernoulli trial with logistic probability."""
        p = 1.0 / (1.0 + np.exp(-logit))
        return self.rng.random(size=len(p)) < p

    def logistic(self, logit: np.ndarray) -> np.ndarray:
        """Return probability from logit."""
        return 1.0 / (1.0 + np.exp(-logit))

    def truncated_normal(self, mean: float, sd: float, low: float, high: float, size: int = 1) -> np.ndarray:
        """Sample from Normal(mean, sd), clamped to [low, high]."""
        samples = self.rng.normal(mean, sd, size=size)
        return np.clip(samples, low, high)

    def negbinom_zt(self, mean: float, phi: float, size: int = 1) -> np.ndarray:
        """Zero-truncated negative binomial.

        Uses standard NB parameterization: n = phi, p = phi/(phi + mean).
        Rejection-samples until all values >= 1.
        """
        n_val = phi
        p_val = phi / (phi + mean)
        out = np.empty(size, dtype=int)
        remaining = np.ones(size, dtype=bool)
        while remaining.any():
            k = remaining.sum()
            draws = self.rng.negative_binomial(n_val, p_val, size=k)
            filled = draws >= 1
            ri = np.where(remaining)[0]        # indices of not-yet-filled slots
            out[ri[filled]] = draws[filled]    # store accepted draws
            remaining[ri[filled]] = False       # mark as done
        return out

    def uniform(self, low=0.0, high=1.0, size=1):
        return self.rng.uniform(low, high, size=size)

    def choice(self, a, size=1, p=None):
        return self.rng.choice(a, size=size, p=p)

    def normal(self, mean=0.0, sd=1.0, size=1):
        return self.rng.normal(mean, sd, size=size)

    def integers(self, low, high=None, size=1):
        return self.rng.integers(low, high, size=size)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: MODEL — Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Participant:
    participant_id: str
    condition: str          # "spaced" | "daily"
    digit_span_forward: int
    digit_span_backward: int
    random_intercept: float  # u_i — individual ability deviation

    @property
    def condition_code(self) -> int:
        return 1 if self.condition == "spaced" else 0

    @property
    def digit_span_combined(self) -> int:
        return self.digit_span_forward + self.digit_span_backward


@dataclass
class Trial:
    participant_id: str
    condition: str
    day: int
    session_type: str       # "learning" | "review" | "relearning"
    item_id: int
    trial_number: int
    pass_number: int
    self_rated_correct: int  # 0/1
    trial_rt_ms: int
    is_bogus: bool = False


@dataclass
class Session:
    participant_id: str
    condition: str
    day: int
    session_type: str
    total_trials: int
    total_time_s: float
    passes_to_criterion: int
    final_accuracy: float


@dataclass
class TestOccasion:
    participant_id: str
    condition: str
    test_occasion: int          # 1-4
    days_since_learning: int
    retention_score: float       # 0-100
    retention_score_true: float  # before missing-data injection
    jol_prediction: float
    jol_prediction_true: float
    is_missing: bool = False
    is_bogus_jol: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: PIPELINE — Generate → Inject → Export
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """Orchestrates data generation, error injection, and CSV export."""

    # ── Fixed study parameters ──
    N_PER_GROUP = 40
    N_ITEMS = 40

    # Review schedule: {condition: [review days]}
    REVIEW_SCHEDULE = {
        "spaced": [2, 5, 12, 26],
        "daily":  [2, 3, 4, 5, 6, 7],
    }

    # Test occasions: (occasion_number, study_day, days_since_learning)
    TEST_SCHEDULE = [
        (1, 7, 7),
        (2, 14, 14),
        (3, 37, 37),
        (4, 97, 97),
    ]

    # Logistic regression parameters for recall probability
    BETA_0  = 2.2    # intercept: daily group at Day 7
    BETA_1  = -0.4   # spaced shift at intercept
    BETA_2  = -1.2   # log(days) slope for daily
    BETA_3  = 0.8    # spaced × log(days) interaction

    SIGMA_PARTICIPANT = 0.3  # SD of random intercepts
    SIGMA_JOL = 10.0         # SD of JOL noise
    PHI_TRIALS = 1.5         # NB overdispersion

    # Bogus-data rates
    P_CARELESS_RANDOM = 0.05
    P_STRAIGHTLINE = 0.03
    P_IMPLAUSIBLE_FAST = 0.002
    P_IMPLAUSIBLE_SLOW = 0.005

    # Missing-data rates
    P_MCAR_TRIAL = 0.01
    P_MCAR_TEST = 0.02

    def __init__(self, seed: int = 2026, output_dir: str = "."):
        self.be = Backend(seed)
        self.output_dir = output_dir
        self.participants: List[Participant] = []
        self.trials: List[Trial] = []
        self.sessions: List[Session] = []
        self.tests: List[TestOccasion] = []

    # ── Generation ────────────────────────────────────────────────────────

    def run(self):
        print("=== RS3 Data Generator ===")
        self._generate_participants()
        self._generate_learning()
        self._generate_reviews()
        self._generate_tests()
        print("  [generate] done — trials=%d, sessions=%d, tests=%d" % (
            len(self.trials), len(self.sessions), len(self.tests)))

        self._inject_bogus()
        print("  [inject]   bogus data injected")

        self._inject_missing()
        print("  [inject]   missing data injected")

        self._export()
        print("  [export]   CSV files written to %s" % os.path.abspath(self.output_dir))
        print("=== Done ===")

    def _generate_participants(self):
        ids = ["P%03d" % i for i in range(1, self.N_PER_GROUP * 2 + 1)]
        conditions = (["spaced"] * self.N_PER_GROUP) + (["daily"] * self.N_PER_GROUP)
        ds_fwd = self.be.normal(7, 1.5, size=self.N_PER_GROUP * 2).astype(int)
        ds_fwd = np.clip(ds_fwd, 3, 9)
        ds_bwd = self.be.normal(5.5, 1.5, size=self.N_PER_GROUP * 2).astype(int)
        ds_bwd = np.clip(ds_bwd, 2, 8)
        u_i = self.be.normal(0, self.SIGMA_PARTICIPANT, size=self.N_PER_GROUP * 2)

        for i in range(self.N_PER_GROUP * 2):
            self.participants.append(Participant(
                participant_id=ids[i],
                condition=conditions[i],
                digit_span_forward=int(ds_fwd[i]),
                digit_span_backward=int(ds_bwd[i]),
                random_intercept=float(u_i[i]),
            ))

    def _get_participant(self, pid: str) -> Participant:
        for p in self.participants:
            if p.participant_id == pid:
                return p
        raise KeyError(pid)

    def _generate_learning(self):
        """Day 1 — initial learning to 90% criterion for all participants."""
        for p in self.participants:
            mu_trials = 180.0 + p.random_intercept * 20
            passes = max(1, int(self.be.negbinom_zt(mu_trials / 40.0, self.PHI_TRIALS, size=1)[0]))
            total_trials = passes * self.N_ITEMS
            trial_num = 0
            for pas in range(1, passes + 1):
                for item in range(1, self.N_ITEMS + 1):
                    trial_num += 1
                    # Accuracy improves across passes
                    acc_p = 0.5 + 0.45 * (pas / max(passes, 1))
                    correct = int(self.be.uniform(size=1)[0] < acc_p)
                    rt = int(self.be.exgaussian(600, 120, 200, size=1)[0])
                    rt = max(150, rt)  # physiological floor
                    self.trials.append(Trial(
                        participant_id=p.participant_id, condition=p.condition,
                        day=1, session_type="learning", item_id=item,
                        trial_number=trial_num, pass_number=pas,
                        self_rated_correct=correct, trial_rt_ms=rt))

            final_acc = 0.90 + self.be.uniform(-0.03, 0.05, size=1)[0]
            total_time = total_trials * 3.5 + self.be.normal(0, 30, size=1)[0]
            self.sessions.append(Session(
                participant_id=p.participant_id, condition=p.condition,
                day=1, session_type="learning", total_trials=total_trials,
                total_time_s=round(max(60, total_time), 1),
                passes_to_criterion=passes,
                final_accuracy=round(min(1.0, max(0.8, final_acc)), 3)))

    def _generate_reviews(self):
        """Review sessions per condition schedule."""
        for p in self.participants:
            review_days = self.REVIEW_SCHEDULE[p.condition]
            for day in review_days:
                base_trials_per_pass = self.N_ITEMS * 0.8  # fewer trials than learning
                passes = max(1, int(self.be.negbinom_zt(base_trials_per_pass / 40.0, self.PHI_TRIALS, size=1)[0]))
                total_trials = int(passes * self.N_ITEMS)
                trial_num = 0
                for pas in range(1, passes + 1):
                    for item in range(1, self.N_ITEMS + 1):
                        trial_num += 1
                        acc_p = 0.6 + 0.37 * (pas / max(passes, 1))
                        correct = int(self.be.uniform(size=1)[0] < acc_p)
                        rt = int(self.be.exgaussian(500, 100, 150, size=1)[0])
                        rt = max(150, rt)
                        self.trials.append(Trial(
                            participant_id=p.participant_id, condition=p.condition,
                            day=day, session_type="review", item_id=item,
                            trial_number=trial_num, pass_number=pas,
                            self_rated_correct=correct, trial_rt_ms=rt))

                final_acc = 0.90 + self.be.uniform(-0.02, 0.04, size=1)[0]
                total_time = total_trials * 2.8 + self.be.normal(0, 20, size=1)[0]
                self.sessions.append(Session(
                    participant_id=p.participant_id, condition=p.condition,
                    day=day, session_type="review", total_trials=total_trials,
                    total_time_s=round(max(30, total_time), 1),
                    passes_to_criterion=passes,
                    final_accuracy=round(min(1.0, max(0.8, final_acc)), 3)))

    def _generate_tests(self):
        """Four cued-recall test occasions with JOLs."""
        for p in self.participants:
            for occ, day, days_since in self.TEST_SCHEDULE:
                log_days = math.log(max(1, days_since))
                cond = p.condition_code
                logit = (self.BETA_0
                         + self.BETA_1 * cond
                         + self.BETA_2 * log_days
                         + self.BETA_3 * cond * log_days
                         + p.random_intercept)
                prob = float(self.be.logistic(np.array([logit]))[0])
                prob_correct_per_item = min(0.99, max(0.01, prob))
                score = prob_correct_per_item * 100.0 + self.be.normal(0, 3, size=1)[0]
                score = round(min(100.0, max(0.0, score)), 1)

                # JOL with condition-specific bias
                bias = 5.0 if p.condition == "daily" else 0.0
                jol = self.be.truncated_normal(score + bias, self.SIGMA_JOL, 0, 100, size=1)[0]
                jol = round(float(jol), 1)

                self.tests.append(TestOccasion(
                    participant_id=p.participant_id, condition=p.condition,
                    test_occasion=occ, days_since_learning=days_since,
                    retention_score=score, retention_score_true=score,
                    jol_prediction=jol, jol_prediction_true=jol))

    # ── Bogus-data injection ──────────────────────────────────────────────

    def _inject_bogus(self):
        self._inject_implausible_rts()
        self._inject_careless_jol()

    def _inject_implausible_rts(self):
        """Inject physiologically implausible RTs in a small fraction of trials."""
        for t in self.trials:
            r = self.be.uniform(size=1)[0]
            if r < self.P_IMPLAUSIBLE_FAST:
                t.trial_rt_ms = int(self.be.uniform(50, 150, size=1)[0])
                t.is_bogus = True
            elif r < self.P_IMPLAUSIBLE_FAST + self.P_IMPLAUSIBLE_SLOW:
                t.trial_rt_ms = int(self.be.uniform(10000, 30000, size=1)[0])
                t.is_bogus = True

    def _inject_careless_jol(self):
        """Inject random and straight-lining JOL responses."""
        for test in self.tests:
            if self.be.uniform(size=1)[0] < self.P_CARELESS_RANDOM:
                test.jol_prediction = round(float(self.be.uniform(0, 100, size=1)[0]), 1)
                test.is_bogus_jol = True

        # Straight-lining: ~3% of participants get a flat JOL=50 at one random occasion
        straightliners = set()
        for p in self.participants:
            if self.be.uniform(size=1)[0] < self.P_STRAIGHTLINE:
                straightliners.add(p.participant_id)
        for pid in straightliners:
            occ = int(self.be.integers(1, 5, size=1)[0])
            for test in self.tests:
                if test.participant_id == pid and test.test_occasion == occ:
                    test.jol_prediction = 50.0
                    test.is_bogus_jol = True

    # ── Missing-data injection ────────────────────────────────────────────

    def _inject_missing(self):
        self._inject_mcar_trials()
        self._inject_mcar_tests()
        self._inject_mar_day97()
        self._inject_mnar_day37()
        self._inject_attrition()

    def _inject_mcar_trials(self):
        """MCAR: randomly delete ~1% of trial rows."""
        keep = []
        for t in self.trials:
            if self.be.uniform(size=1)[0] >= self.P_MCAR_TRIAL:
                keep.append(t)
        self.trials = keep

    def _inject_mcar_tests(self):
        """MCAR: randomly set ~2% of test scores to NaN."""
        for test in self.tests:
            if self.be.uniform(size=1)[0] < self.P_MCAR_TEST:
                test.retention_score = float("nan")
                test.is_missing = True

    def _inject_mar_day97(self):
        """MAR: Day-97 missingness depends on observed digit span.

        P(missing) = logit⁻¹(2.0 − 0.5 × digit_span_combined)
        """
        for test in self.tests:
            if test.test_occasion != 4:
                continue
            p = self._get_participant(test.participant_id)
            logit_mar = 2.0 - 0.5 * p.digit_span_combined
            prob_miss = float(self.be.logistic(np.array([logit_mar]))[0])
            if not math.isnan(test.retention_score) and self.be.uniform(size=1)[0] < prob_miss:
                test.retention_score = float("nan")
                test.is_missing = True

    def _inject_mnar_day37(self):
        """MNAR: Day-37 missingness depends on the (unobserved) true score.

        P(missing) = logit⁻¹(0.5 − 0.08 × true_retention_score)
        """
        for test in self.tests:
            if test.test_occasion != 3:
                continue
            logit_mnar = 0.5 - 0.08 * test.retention_score_true
            prob_miss = float(self.be.logistic(np.array([logit_mnar]))[0])
            if (not math.isnan(test.retention_score)
                    and self.be.uniform(size=1)[0] < prob_miss):
                test.retention_score = float("nan")
                test.is_missing = True

    def _inject_attrition(self):
        """Monotone dropout: once a participant misses a test, all later tests
        are also missing. Higher dropout for daily-condition and low-span
        participants."""
        for p in self.participants:
            # Base dropout probability per participant
            p_drop_base = 0.05
            if p.condition == "daily":
                p_drop_base += 0.08
            if p.digit_span_combined <= 10:
                p_drop_base += 0.05

            drop_occasion: Optional[int] = None
            for occ in [1, 2, 3, 4]:
                if drop_occasion is not None:
                    # Already dropped: make all later tests missing
                    for test in self.tests:
                        if (test.participant_id == p.participant_id
                                and test.test_occasion == occ
                                and not math.isnan(test.retention_score)):
                            test.retention_score = float("nan")
                            test.is_missing = True
                else:
                    r = self.be.uniform(size=1)[0]
                    if r < p_drop_base / 4.0:  # per-occasion risk
                        drop_occasion = occ
                        for test in self.tests:
                            if (test.participant_id == p.participant_id
                                    and test.test_occasion == occ
                                    and not math.isnan(test.retention_score)):
                                test.retention_score = float("nan")
                                test.is_missing = True

    # ── CSV Export ────────────────────────────────────────────────────────

    def _export(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_participants()
        self._write_trials()
        self._write_sessions()
        self._write_tests()
        self._write_data_dictionary()

    def _write_csv(self, filename: str, rows: list[dict]):
        import csv
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_participants(self):
        rows = [{
            "participant_id": p.participant_id,
            "condition": p.condition,
            "digit_span_forward": p.digit_span_forward,
            "digit_span_backward": p.digit_span_backward,
            "random_intercept": round(p.random_intercept, 4),
        } for p in self.participants]
        self._write_csv("rs3_participants.csv", rows)

    def _write_trials(self):
        rows = [{
            "participant_id": t.participant_id,
            "condition": t.condition,
            "day": t.day,
            "session_type": t.session_type,
            "item_id": t.item_id,
            "trial_number": t.trial_number,
            "pass_number": t.pass_number,
            "self_rated_correct": t.self_rated_correct,
            "trial_rt_ms": t.trial_rt_ms,
            "is_bogus": t.is_bogus,
        } for t in self.trials]
        self._write_csv("rs3_trials.csv", rows)

    def _write_sessions(self):
        rows = [{
            "participant_id": s.participant_id,
            "condition": s.condition,
            "day": s.day,
            "session_type": s.session_type,
            "total_trials": s.total_trials,
            "total_time_s": s.total_time_s,
            "passes_to_criterion": s.passes_to_criterion,
            "final_accuracy": s.final_accuracy,
        } for s in self.sessions]
        self._write_csv("rs3_sessions.csv", rows)

    def _write_tests(self):
        rows = []
        for t in self.tests:
            score = "" if (isinstance(t.retention_score, float) and math.isnan(t.retention_score)) else t.retention_score
            rows.append({
                "participant_id": t.participant_id,
                "condition": t.condition,
                "test_occasion": t.test_occasion,
                "days_since_learning": t.days_since_learning,
                "retention_score": score,
                "jol_prediction": t.jol_prediction,
                "is_missing": t.is_missing,
                "is_bogus_jol": t.is_bogus_jol,
            })
        self._write_csv("rs3_tests.csv", rows)

    def _write_data_dictionary(self):
        lines = [
            "table,column,type,description,missing_code",
            "rs3_participants,participant_id,string,Unique participant identifier (P001-P080),",
            "rs3_participants,condition,enum,Experimental condition: spaced|daily,",
            "rs3_participants,digit_span_forward,int,Forward digit span (3-9),",
            "rs3_participants,digit_span_backward,int,Backward digit span (2-8),",
            "rs3_participants,random_intercept,float,Per-participant ability deviation (u_i),",
            "rs3_trials,participant_id,string,Unique participant identifier,",
            "rs3_trials,condition,enum,Experimental condition: spaced|daily,",
            "rs3_trials,day,int,Study day (1-97),",
            "rs3_trials,session_type,enum,Session type: learning|review|relearning,",
            "rs3_trials,item_id,int,Swahili-English pair identifier (1-40),",
            "rs3_trials,trial_number,int,Sequential trial number within session,",
            "rs3_trials,pass_number,int,Pass number through the full item set,",
            "rs3_trials,self_rated_correct,int,Self-rated recall accuracy: 0=incorrect 1=correct,",
            "rs3_trials,trial_rt_ms,int,Reaction time in milliseconds (Ex-Gaussian),",
            "rs3_trials,is_bogus,bool,Ground-truth flag for implausible RT injection,",
            "rs3_sessions,participant_id,string,Unique participant identifier,",
            "rs3_sessions,condition,enum,Experimental condition,",
            "rs3_sessions,day,int,Study day,",
            "rs3_sessions,session_type,enum,Session type,",
            "rs3_sessions,total_trials,int,Total trials to reach criterion,",
            "rs3_sessions,total_time_s,float,Total session duration in seconds,",
            "rs3_sessions,passes_to_criterion,int,Number of passes to reach 90% criterion,",
            "rs3_sessions,final_accuracy,float,Proportion correct on final pass (0-1),",
            "rs3_tests,participant_id,string,Unique participant identifier,",
            "rs3_tests,condition,enum,Experimental condition,",
            "rs3_tests,test_occasion,int,Testing occasion (1=Day7 2=Day14 3=Day37 4=Day97),",
            "rs3_tests,days_since_learning,int,Days elapsed since initial learning (Day 1),",
            "rs3_tests,retention_score,float,Cued-recall test score (0-100) — empty = missing,",
            "rs3_tests,jol_prediction,float,Judgment of Learning predicted score (0-100),",
            "rs3_tests,is_missing,bool,True if retention_score was artificially removed,",
            "rs3_tests,is_bogus_jol,bool,True if JOL was replaced by careless/straight-line response,",
        ]
        path = os.path.join(self.output_dir, "rs3_data_dictionary.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RS3 Minimalist Data Generator — Spaced Repetition vs. Daily Review",
        epilog="Outputs: rs3_participants.csv, rs3_trials.csv, rs3_sessions.csv, "
               "rs3_tests.csv, rs3_data_dictionary.csv",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed for reproducible generation (default: 2026).",
    )
    parser.add_argument(
        "--n", type=int, default=40,
        help="Participants per condition (default: 40, total N=80).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=".",
        help="Output directory for CSV files (default: current directory).",
    )
    args = parser.parse_args()

    pipeline = Pipeline(seed=args.seed, output_dir=args.output)
    pipeline.N_PER_GROUP = args.n
    pipeline.run()


if __name__ == "__main__":
    main()
