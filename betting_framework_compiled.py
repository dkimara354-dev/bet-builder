#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DIXON-COLES SPORTS BETTING FRAMEWORK  —  COMPILED SINGLE FILE      ║
║                              Production v2.0                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BLOCK MAP  (Ctrl+F / search these tags to jump to any section)             ║
║                                                                              ║
║  [BLOCK 1]  IMPORTS & LOGGING SETUP                                          ║
║  [BLOCK 2]  DIXON-COLES PRIMITIVES  (tau, Poisson PMF)                      ║
║  [BLOCK 3]  TIME-DECAY WEIGHTS                                               ║
║  [BLOCK 4]  MLE OBJECTIVE  (_DCObjective)                                    ║
║  [BLOCK 5]  FALLBACK OPTIMISER  (coordinate descent)                        ║
║  [BLOCK 6]  PREDICTOR MODEL  (PredictorModel class)                         ║
║  [BLOCK 7]  DATA INGESTOR  (DataIngestor class)                              ║
║  [BLOCK 8]  PARLAY MATH HELPERS  (vig gate)                                  ║
║  [BLOCK 9]  BET MANAGER  (BetManager class)                                  ║
║  [BLOCK 10] BACKTESTER  (Backtester class)                                   ║
║  [BLOCK 11] DATABASE  (BettingDatabase class)                                ║
║  [BLOCK 12] ORCHESTRATOR  (BettingFramework class)                           ║
║  [BLOCK 13] CLI ENTRY POINT  (main / argparse)                               ║
║                                                                              ║
║  QUICK START                                                                 ║
║    python betting_framework_compiled.py --mode demo                          ║
║    python betting_framework_compiled.py --mode live --api-key KEY            ║
║    python betting_framework_compiled.py --mode backtest --league EPL         ║
║    python betting_framework_compiled.py --mode update_clv --api-key KEY      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 1]  IMPORTS & LOGGING SETUP                                         │
# └─────────────────────────────────────────────────────────────────────────────┘

import argparse
import csv
import io
import itertools
import json
import logging
import math
import random
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("betting_framework.log", mode="a"),
    ],
)

log_ingestor  = logging.getLogger("betting_framework.ingestor")
log_predictor = logging.getLogger("betting_framework.predictor")
log_bet_mgr   = logging.getLogger("betting_framework.bet_manager")
log_backtest  = logging.getLogger("betting_framework.backtester")
log_database  = logging.getLogger("betting_framework.database")
log_main      = logging.getLogger("betting_framework.main")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 2]  DIXON-COLES PRIMITIVES                                          │
# │             tau correction + Poisson PMF                                    │
# └─────────────────────────────────────────────────────────────────────────────┘

def _tau(hg: int, ag: int, mu_h: float, mu_a: float, rho: float) -> float:
    """
    Dixon-Coles τ correction for the four low-score cells.
    Adjusts the raw Poisson probability to fix the well-known draw-bias
    in basic Poisson football models.

    Scoreline corrections:
        (0,0)  →  1 − μₕ·μₐ·ρ
        (1,0)  →  1 + μₐ·ρ
        (0,1)  →  1 + μₕ·ρ
        (1,1)  →  1 − ρ
        else   →  1.0  (no adjustment)
    """
    if hg == 0 and ag == 0:
        return 1.0 - mu_h * mu_a * rho
    elif hg == 1 and ag == 0:
        return 1.0 + mu_a * rho
    elif hg == 0 and ag == 1:
        return 1.0 + mu_h * rho
    elif hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(lam: float, k: int) -> float:
    """
    Numerically stable Poisson PMF computed in log-space.
    P(X=k | λ) = exp(−λ + k·ln(λ) − ln(k!))
    Returns 0 on any overflow or domain error.
    """
    if lam <= 0.0 or k < 0:
        return 0.0
    try:
        log_p = -lam + k * math.log(lam) - math.lgamma(k + 1)
        return math.exp(log_p)
    except (OverflowError, ValueError):
        return 0.0


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 3]  TIME-DECAY WEIGHTS                                              │
# │             Exponential half-life weighting for match recency               │
# └─────────────────────────────────────────────────────────────────────────────┘

# Supported date formats across understat, FDCO, and manual inputs
_DATE_FMTS = [
    "%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y", "%Y/%m/%d",
    "%m/%d/%y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
]


def _compute_time_weights(
    matches: list[dict],
    half_life_days: float,
    reference_date: Optional[datetime] = None,
) -> np.ndarray:
    """
    Compute exponential time-decay weights for a list of matches.

    Formula:  weight_i = exp(−λ · age_i)   where λ = ln(2) / half_life_days

    A match exactly half_life_days old receives weight 0.5.
    Weights are normalised so Σw = len(matches) (preserves LL magnitude).
    If half_life_days = inf, all weights = 1.0 (no decay).

    Matches with unparseable dates receive a fallback age of 180 days.
    """
    if math.isinf(half_life_days):
        return np.ones(len(matches), dtype=np.float64)

    ref = reference_date or datetime.utcnow()
    lam = math.log(2.0) / half_life_days
    weights = []

    for m in matches:
        raw = str(m.get("date", "") or m.get("datetime", ""))
        raw = raw.split(" ")[0].split("T")[0].strip()
        age = 180.0  # fallback if date is unparseable
        for fmt in _DATE_FMTS:
            try:
                dt = datetime.strptime(raw, fmt)
                age = max(0.0, (ref - dt).days)
                break
            except ValueError:
                continue
        weights.append(math.exp(-lam * age))

    arr = np.array(weights, dtype=np.float64)
    total = arr.sum()
    if total > 0:
        arr *= len(arr) / total   # normalise
    return arr


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 4]  MLE OBJECTIVE  (_DCObjective)                                   │
# │             scipy-compatible callable for joint parameter estimation        │
# └─────────────────────────────────────────────────────────────────────────────┘

class _DCObjective:
    """
    Weighted negative log-likelihood for the full Dixon-Coles model.

    Parameter vector x layout:
        x[0 : N]        attack[0..N-1]     (log-scale)
        x[N : 2N]       defence[0..N-1]    (log-scale)
        x[2N]           home_advantage      (log-scale, bounded ≥ 0)
        x[2N + 1]       rho                 (real, bounded ∈ [−0.4, 0.4])

    Regularisation applied:
        L2 on attack + defence      — prevents parameter blow-up on sparse data
        Sum-to-zero penalty on atk  — soft identifiability constraint
    """

    RHO_MIN,  RHO_MAX  = -0.40,  0.40
    HADV_MIN, HADV_MAX =  0.00,  1.00
    L2_LAMBDA          = 1e-3
    ID_PENALTY         = 10.0

    def __init__(
        self,
        N: int,
        hg: np.ndarray, ag: np.ndarray,
        hi: np.ndarray, ai: np.ndarray,
        weights: np.ndarray,
        home_xg: Optional[np.ndarray] = None,
        away_xg: Optional[np.ndarray] = None,
        xg_blend: float = 0.0,
    ):
        self.N  = N
        self.hg = hg.astype(np.int32)
        self.ag = ag.astype(np.int32)
        self.hi = hi.astype(np.int32)
        self.ai = ai.astype(np.int32)
        self.w  = weights
        self.home_xg  = home_xg
        self.away_xg  = away_xg
        self.xg_blend = xg_blend
        self.use_xg   = xg_blend > 0 and home_xg is not None
        # Precompute log-factorials for goals 0..15
        self._lgam = np.array([math.lgamma(k + 1) for k in range(16)], dtype=np.float64)

    def _lgam_safe(self, k_arr: np.ndarray) -> np.ndarray:
        return np.array([
            self._lgam[k] if k < len(self._lgam) else math.lgamma(k + 1)
            for k in k_arr
        ])

    def __call__(self, x: np.ndarray) -> float:
        N    = self.N
        atk  = x[:N]
        dfc  = x[N:2*N]
        hadv = float(np.clip(x[2*N],     self.HADV_MIN, self.HADV_MAX))
        rho  = float(np.clip(x[2*N + 1], self.RHO_MIN,  self.RHO_MAX))

        mu_h = np.exp(atk[self.hi] + dfc[self.ai] + hadv)
        mu_a = np.exp(atk[self.ai] + dfc[self.hi])

        if self.use_xg:
            mu_h = (1 - self.xg_blend) * mu_h + self.xg_blend * self.home_xg
            mu_a = (1 - self.xg_blend) * mu_a + self.xg_blend * self.away_xg
            mu_h = np.clip(mu_h, 1e-8, None)
            mu_a = np.clip(mu_a, 1e-8, None)

        mu_h_c = np.maximum(mu_h, 1e-10)
        mu_a_c = np.maximum(mu_a, 1e-10)
        log_ph = -mu_h_c + self.hg * np.log(mu_h_c) - self._lgam_safe(self.hg)
        log_pa = -mu_a_c + self.ag * np.log(mu_a_c) - self._lgam_safe(self.ag)

        log_tau = np.empty(len(self.hg), dtype=np.float64)
        for i in range(len(self.hg)):
            tv = _tau(self.hg[i], self.ag[i], mu_h[i], mu_a[i], rho)
            log_tau[i] = math.log(max(abs(tv), 1e-12))

        nll  = -float(np.dot(self.w, log_ph + log_pa + log_tau))
        nll += self.L2_LAMBDA * (float(np.dot(atk, atk)) + float(np.dot(dfc, dfc)))
        nll += self.ID_PENALTY * float(np.sum(atk)) ** 2
        return nll


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 5]  FALLBACK OPTIMISER                                              │
# │             Coordinate descent — used when scipy is unavailable             │
# └─────────────────────────────────────────────────────────────────────────────┘

def _coord_descent(fn, x0: np.ndarray, max_iter: int = 400, tol: float = 1e-7) -> np.ndarray:
    """
    Gradient-free coordinate-descent minimiser.

    Cycles through each parameter, testing a ±step perturbation and
    accepting moves that reduce the objective.  Step size is halved
    whenever a full cycle produces no improvement.  Stops when step < 1e-8.

    Used as fallback when scipy.optimize is not installed or fails.
    """
    x      = x0.copy()
    best_f = fn(x)
    step   = 0.05

    for _ in range(max_iter):
        improved = False
        for i in range(len(x)):
            for sign in (1.0, -1.0):
                xt = x.copy()
                xt[i] += sign * step
                ft = fn(xt)
                if ft < best_f - tol:
                    x, best_f, improved = xt, ft, True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    return x


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 6]  PREDICTOR MODEL                                                 │
# │             Dixon-Coles joint MLE with time-decay and dynamic rho           │
# └─────────────────────────────────────────────────────────────────────────────┘

class PredictorModel:
    """
    Dixon-Coles Poisson model — Production MLE Edition.

    All four parameters (attack[], defence[], home_advantage, rho) are estimated
    simultaneously by maximising the time-decay-weighted Dixon-Coles log-likelihood
    using scipy's L-BFGS-B solver (with coordinate-descent fallback).

    Key features:
        ● Joint MLE   — rho is NOT grid-searched; it is solved with the other params
        ● Time-decay  — recent matches weighted by exp(−λ·age)
        ● xG blend    — optionally blend actual goals with xG for smoother μ
        ● Approx SEs  — finite-difference Hessian diagonal for rho & home_advantage
        ● scipy-free fallback — runs everywhere, just slower
    """

    MIN_MATCHES_PER_TEAM = 5
    GOALS_CAP            = 10

    def __init__(
        self,
        half_life_days: float = 90.0,
        xg_blend: float = 0.0,
        use_xg: bool = False,
        l2_lambda: float = 1e-3,
    ):
        """
        Parameters
        ──────────
        half_life_days : float
            Decay half-life in days.  float('inf') disables decay entirely.
        xg_blend : float [0, 1]
            Fraction of xG blended into μ (0 = goals only, 1 = xG only).
        use_xg : bool
            Activate xG blending; requires 'home_xg'/'away_xg' in matches.
        l2_lambda : float
            L2 regularisation strength on attack/defence.
        """
        self.half_life_days = half_life_days
        self.xg_blend       = float(np.clip(xg_blend, 0.0, 1.0))
        self.use_xg         = use_xg
        self.l2_lambda      = l2_lambda

        # Populated by fit()
        self.attack:         dict[str, float] = {}
        self.defence:        dict[str, float] = {}
        self.home_advantage: float = 0.25
        self.fitted_rho:     float = -0.13
        self.teams:          list[str] = []
        self.is_fitted:      bool = False
        self._match_counts:  dict[str, int] = {}
        self._fit_nll:       float = float("inf")
        self._fit_stats:     dict = {}
        self._ref_date:      Optional[datetime] = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        matches: list[dict],
        reference_date: Optional[datetime] = None,
    ) -> "PredictorModel":
        """
        Fit model via joint MLE on all parameters simultaneously.

        matches : list[dict] with keys:
            home_team, away_team, home_goals, away_goals, date
            (optional: home_xg, away_xg when use_xg=True)

        reference_date : datetime, optional
            Anchor for time-decay age computation.  Pass the date of the
            last match in training data when backtesting to avoid
            look-ahead bias.  Defaults to utcnow() for live use.
        """
        if not matches:
            raise ValueError("No matches provided.")

        self._ref_date = reference_date or datetime.utcnow()

        # ── Team census ───────────────────────────────────────────────────────
        counts: dict[str, int] = defaultdict(int)
        for m in matches:
            counts[m["home_team"]] += 1
            counts[m["away_team"]] += 1
        self._match_counts = dict(counts)

        valid_teams = sorted(t for t, c in counts.items() if c >= self.MIN_MATCHES_PER_TEAM)
        if len(valid_teams) < 2:
            raise ValueError(
                f"Need ≥2 teams with ≥{self.MIN_MATCHES_PER_TEAM} matches. "
                f"Got {len(valid_teams)} qualifying teams."
            )
        self.teams = valid_teams
        N    = len(self.teams)
        tidx = {t: i for i, t in enumerate(self.teams)}

        valid = [m for m in matches if m["home_team"] in tidx and m["away_team"] in tidx]
        if not valid:
            raise ValueError("No matches remain after filtering valid teams.")

        log_predictor.info(
            f"Fitting Dixon-Coles (joint MLE) | {len(valid)} matches | {N} teams | "
            f"half-life={self.half_life_days}d | xG-blend={self.xg_blend}"
        )

        # ── Numpy arrays ──────────────────────────────────────────────────────
        hg = np.array([min(int(m["home_goals"]), self.GOALS_CAP) for m in valid])
        ag = np.array([min(int(m["away_goals"]), self.GOALS_CAP) for m in valid])
        hi = np.array([tidx[m["home_team"]] for m in valid], dtype=np.int32)
        ai = np.array([tidx[m["away_team"]] for m in valid], dtype=np.int32)

        home_xg = away_xg = None
        if self.use_xg:
            home_xg = np.array([float(m.get("home_xg") or 0) for m in valid])
            away_xg = np.array([float(m.get("away_xg") or 0) for m in valid])
            home_xg = np.where(home_xg > 0, home_xg, hg.astype(float))
            away_xg = np.where(away_xg > 0, away_xg, ag.astype(float))

        weights = _compute_time_weights(valid, self.half_life_days, self._ref_date)

        # ── Build objective ───────────────────────────────────────────────────
        obj = _DCObjective(
            N=N, hg=hg, ag=ag, hi=hi, ai=ai, weights=weights,
            home_xg=home_xg, away_xg=away_xg, xg_blend=self.xg_blend,
        )
        obj.L2_LAMBDA = self.l2_lambda

        # ── Warm-start initial vector ─────────────────────────────────────────
        avg_goals = max(float(np.mean(np.concatenate([hg, ag]))), 0.5)
        mean_log  = math.log(avg_goals) * 0.5
        x0 = np.zeros(2 * N + 2)
        x0[:N]      = mean_log
        x0[2*N]     = 0.25
        x0[2*N + 1] = -0.10

        # ── scipy L-BFGS-B ───────────────────────────────────────────────────
        bounds = (
            [(-3.0, 3.0)] * N +   # attack
            [(-3.0, 3.0)] * N +   # defence
            [(0.00, 1.00)]    +   # home_advantage
            [(-0.40, 0.40)]       # rho
        )

        opt_x, opt_nll = None, float("inf")

        try:
            from scipy.optimize import minimize as sp_min
            res = sp_min(
                fun=obj, x0=x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-8},
            )
            if res.success or res.fun < obj(x0):
                opt_x, opt_nll = res.x, float(res.fun)
                log_predictor.info(
                    f"scipy L-BFGS-B: {res.nit} iters | -LL={opt_nll:.4f} | {res.message}"
                )
            else:
                log_predictor.warning(f"scipy did not fully converge: {res.message}")
                if res.fun < obj(x0):
                    opt_x, opt_nll = res.x, float(res.fun)
        except ImportError:
            log_predictor.warning("scipy not installed — using coordinate descent fallback")
        except Exception as exc:
            log_predictor.warning(f"scipy error ({exc}) — using coordinate descent fallback")

        # ── Fallback ──────────────────────────────────────────────────────────
        if opt_x is None:
            log_predictor.info("Running coordinate-descent optimiser...")
            opt_x   = _coord_descent(obj, x0)
            opt_nll = float(obj(opt_x))
            log_predictor.info(f"Coord-descent: -LL={opt_nll:.4f}")

        # ── Store parameters ──────────────────────────────────────────────────
        for i, team in enumerate(self.teams):
            self.attack[team]  = float(opt_x[i])
            self.defence[team] = float(opt_x[N + i])
        self.home_advantage = float(np.clip(opt_x[2*N],     0.00,  1.00))
        self.fitted_rho     = float(np.clip(opt_x[2*N + 1], -0.40, 0.40))
        self._fit_nll       = opt_nll
        self.is_fitted      = True

        # ── Approx standard errors ────────────────────────────────────────────
        ha_se = rho_se = None
        try:
            eps = 1e-4
            f0  = opt_nll
            for idx_p, name in [(2*N, "ha"), (2*N+1, "rho")]:
                xp, xm = opt_x.copy(), opt_x.copy()
                xp[idx_p] += eps; xm[idx_p] -= eps
                h_ii = (obj(xp) - 2*f0 + obj(xm)) / eps**2
                se   = 1.0 / math.sqrt(max(h_ii, 1e-12))
                if name == "ha":
                    ha_se = se
                else:
                    rho_se = se
        except Exception:
            pass

        self._fit_stats = {
            "n_matches":         len(valid),
            "n_teams":           N,
            "final_nll":         round(opt_nll, 4),
            "home_advantage_x":  round(math.exp(self.home_advantage), 4),
            "home_advantage_se": round(ha_se,  4) if ha_se  else None,
            "rho":               round(self.fitted_rho, 4),
            "rho_se":            round(rho_se, 4) if rho_se else None,
            "half_life_days":    self.half_life_days,
            "xg_blend":          self.xg_blend,
        }

        ha_str  = f"Home adv={math.exp(self.home_advantage):.3f}x"
        ha_str += f" (±{ha_se:.3f})" if ha_se else ""
        rho_str  = f"rho={self.fitted_rho:.4f}"
        rho_str += f" (±{rho_se:.4f})" if rho_se else ""
        log_predictor.info(f"Model fitted ✓ | {ha_str} | {rho_str}")
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10,
    ) -> dict:
        """
        Return score-probability matrix and 1X2 outcome probabilities.

        Returns empty dict {} with a logged reason when:
          - Either team was not in the training data
          - A team has fewer than MIN_MATCHES_PER_TEAM appearances
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

        for team, label in [(home_team, "Home"), (away_team, "Away")]:
            if team not in self.teams:
                log_predictor.warning(
                    f"Skipping prediction | {label}='{team}' not in model. "
                    "Reason: Insufficient xG Data or team absent from training set."
                )
                return {}
            cnt = self._match_counts.get(team, 0)
            if cnt < self.MIN_MATCHES_PER_TEAM:
                log_predictor.warning(
                    f"Insufficient xG Data for '{team}' "
                    f"({cnt} matches < {self.MIN_MATCHES_PER_TEAM})"
                )
                return {}

        mu_h = math.exp(
            self.attack[home_team] + self.defence[away_team] + self.home_advantage
        )
        mu_a = math.exp(
            self.attack[away_team] + self.defence[home_team]
        )

        # Score-probability matrix with τ correction
        matrix: list[list[float]] = []
        for hg in range(max_goals + 1):
            row = []
            for ag in range(max_goals + 1):
                p = (_poisson_pmf(mu_h, hg) * _poisson_pmf(mu_a, ag)
                     * _tau(hg, ag, mu_h, mu_a, self.fitted_rho))
                row.append(max(float(p), 0.0))
            matrix.append(row)

        # Normalise (τ correction slightly perturbs total probability)
        total = sum(p for row in matrix for p in row)
        if total > 0:
            matrix = [[p / total for p in row] for row in matrix]

        home_win = sum(
            matrix[hg][ag]
            for hg in range(max_goals + 1)
            for ag in range(max_goals + 1)
            if hg > ag
        )
        draw     = sum(matrix[g][g] for g in range(max_goals + 1))
        away_win = max(0.0, 1.0 - home_win - draw)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "mu_home":   round(mu_h, 4),
            "mu_away":   round(mu_a, 4),
            "home_win":  round(home_win, 6),
            "draw":      round(draw, 6),
            "away_win":  round(away_win, 6),
            "matrix":    matrix,
            "rho":       self.fitted_rho,
            "fit_stats": self._fit_stats,
        }

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def get_implied_prob(decimal_odds: float) -> float:
        """Decimal odds → raw implied probability (includes bookmaker margin)."""
        return 1.0 / decimal_odds if decimal_odds > 1.0 else 1.0

    @staticmethod
    def remove_vig(
        home_odd: float, draw_odd: float, away_odd: float
    ) -> dict[str, float]:
        """Proportional vig removal → fair (no-margin) implied probabilities."""
        raw = {
            "home": 1.0 / home_odd if home_odd > 1 else 0.0,
            "draw": 1.0 / draw_odd if draw_odd > 1 else 0.0,
            "away": 1.0 / away_odd if away_odd > 1 else 0.0,
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()} if total > 0 else raw

    @staticmethod
    def overround(home_odd: float, draw_odd: float, away_odd: float) -> float:
        """
        Bookmaker margin as a decimal, e.g. 0.05 = 5%.
        overround = Σ(1/odds) − 1
        """
        if min(home_odd, draw_odd, away_odd) <= 1.0:
            return 0.0
        return max(0.0, 1/home_odd + 1/draw_odd + 1/away_odd - 1.0)

    @staticmethod
    def calculate_ev(model_prob: float, decimal_odds: float) -> float:
        """EV = p·(odds−1) − (1−p).  Positive EV = edge over the bookmaker."""
        if decimal_odds <= 1.0 or model_prob <= 0.0:
            return -1.0
        return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)

    @staticmethod
    def kelly_fraction(
        model_prob: float, decimal_odds: float, fraction: float = 0.22
    ) -> float:
        """Fractional Kelly stake as a proportion of bankroll (0 if no edge)."""
        b = decimal_odds - 1.0
        if b <= 0.0 or model_prob <= 0.0:
            return 0.0
        full_k = (b * model_prob - (1.0 - model_prob)) / b
        return max(0.0, full_k * fraction)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def top_teams(self, n: int = 5) -> list[dict]:
        """Top-N teams by attack rating (descending)."""
        if not self.is_fitted:
            return []
        ranked = sorted(self.teams, key=lambda t: self.attack.get(t, 0), reverse=True)
        return [{"team": t, "attack": round(self.attack[t], 4),
                 "defence": round(self.defence[t], 4)} for t in ranked[:n]]

    def fit_summary(self) -> dict:
        return dict(self._fit_stats)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 7]  DATA INGESTOR                                                   │
# │             understat xG  |  The Odds API  |  football-data.co.uk CSVs     │
# └─────────────────────────────────────────────────────────────────────────────┘

class DataIngestor:
    """
    Zero-cost data stack pulling from three free sources:

        1. understat.com       — xG, shots, goals for Top 5 EU leagues
        2. The Odds API        — real-time H2H odds (free tier: 500 req/month)
        3. football-data.co.uk — historical CSVs for backtesting
    """

    LEAGUE_URLS = {
        "EPL":        "https://understat.com/league/EPL",
        "La_liga":    "https://understat.com/league/La_liga",
        "Bundesliga": "https://understat.com/league/Bundesliga",
        "Serie_A":    "https://understat.com/league/Serie_A",
        "Ligue_1":    "https://understat.com/league/Ligue_1",
    }

    FDCO_BASE = "https://www.football-data.co.uk/mmz4281"

    FDCO_LEAGUE_CODES = {
        "EPL":        ("E0",  "E"),
        "La_liga":    ("SP1", "SP"),
        "Bundesliga": ("D1",  "D"),
        "Serie_A":    ("I1",  "I"),
        "Ligue_1":    ("F1",  "F"),
    }

    ODDS_API_BASE = "https://api.the-odds-api.com/v4"

    ODDS_API_SPORT_KEYS = {
        "EPL":        "soccer_epl",
        "La_liga":    "soccer_spain_la_liga",
        "Bundesliga": "soccer_germany_bundesliga",
        "Serie_A":    "soccer_italy_serie_a",
        "Ligue_1":    "soccer_france_ligue_one",
    }

    def __init__(self, odds_api_key: str = ""):
        self.odds_api_key    = odds_api_key
        self._request_cache: dict = {}

    # ── understat xG scraper ──────────────────────────────────────────────────

    def fetch_understat_xg(self, league: str, season: int = 2024) -> list[dict]:
        """Scrape understat.com for xG, shots, goals per match."""
        if league not in self.LEAGUE_URLS:
            raise ValueError(f"Unknown league '{league}'. Options: {list(self.LEAGUE_URLS)}")

        url = f"{self.LEAGUE_URLS[league]}/{season}"
        log_ingestor.info(f"Fetching understat xG: {url}")

        try:
            html = self._http_get(url)
        except Exception as e:
            log_ingestor.error(f"understat fetch failed for {league}/{season}: {e}")
            return []

        matches = self._parse_understat_json(html)
        log_ingestor.info(f"Parsed {len(matches)} xG matches ({league} {season})")
        return matches

    def _parse_understat_json(self, html: str) -> list[dict]:
        """Extract the embedded datesData JSON from understat HTML."""
        pattern = r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)"
        m = re.search(pattern, html)
        if not m:
            log_ingestor.warning("datesData not found in understat HTML")
            return []

        raw = m.group(1).replace("\\'", "'").replace("\\\\", "\\")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log_ingestor.error(f"understat JSON decode error: {e}")
            return []

        matches = []
        for match in data:
            try:
                parsed = {
                    "id":         match.get("id"),
                    "date":       match.get("datetime", ""),
                    "home_team":  match.get("h", {}).get("title", ""),
                    "away_team":  match.get("a", {}).get("title", ""),
                    "home_goals": int(match.get("goals", {}).get("h", 0) or 0),
                    "away_goals": int(match.get("goals", {}).get("a", 0) or 0),
                    "home_xg":    float(match.get("xG",   {}).get("h", 0) or 0),
                    "away_xg":    float(match.get("xG",   {}).get("a", 0) or 0),
                    "home_shots": int(match.get("shots", {}).get("h", 0) or 0),
                    "away_shots": int(match.get("shots", {}).get("a", 0) or 0),
                    "status":     match.get("isResult", False),
                }
                if parsed["home_xg"] > 0 or parsed["away_xg"] > 0:
                    matches.append(parsed)
                else:
                    log_ingestor.debug(f"Skipping match {parsed['id']}: No xG data")
            except (KeyError, ValueError, TypeError) as e:
                log_ingestor.debug(f"Skipping malformed match: {e}")
        return matches

    # ── The Odds API ──────────────────────────────────────────────────────────

    def fetch_odds(self, league: str, regions: str = "eu", markets: str = "h2h") -> list[dict]:
        """Fetch real-time H2H odds from The Odds API (free tier)."""
        if not self.odds_api_key:
            log_ingestor.warning("No Odds API key provided — skipping odds fetch")
            return []

        sport_key = self.ODDS_API_SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league for Odds API: '{league}'")

        params = urllib.parse.urlencode({
            "apiKey": self.odds_api_key, "regions": regions,
            "markets": markets, "oddsFormat": "decimal",
        })
        url = f"{self.ODDS_API_BASE}/sports/{sport_key}/odds?{params}"
        log_ingestor.info(f"Fetching odds: {league}")

        try:
            events = json.loads(self._http_get(url))
        except Exception as e:
            log_ingestor.error(f"Odds API fetch failed: {e}")
            return []

        parsed = []
        for event in events:
            best = self._extract_best_odds(event)
            if best:
                parsed.append({
                    "event_id":   event.get("id"),
                    "home_team":  event.get("home_team"),
                    "away_team":  event.get("away_team"),
                    "commence":   event.get("commence_time"),
                    "best_home":  best["home"],
                    "best_draw":  best["draw"],
                    "best_away":  best["away"],
                    "bookmakers": best["bookmakers"],
                })
        log_ingestor.info(f"Fetched {len(parsed)} events with odds ({league})")
        return parsed

    def _extract_best_odds(self, event: dict) -> Optional[dict]:
        """Find the highest odds across all bookmakers for each outcome."""
        best = {"home": 0.0, "draw": 0.0, "away": 0.0, "bookmakers": []}
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes  = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                best["home"] = max(best["home"], outcomes.get(event.get("home_team", ""), 0))
                best["draw"] = max(best["draw"], outcomes.get("Draw", 0))
                best["away"] = max(best["away"], outcomes.get(event.get("away_team", ""), 0))
                best["bookmakers"].append(bm["title"])
        return best if best["home"] > 0 else None

    # ── football-data.co.uk CSV ingest ────────────────────────────────────────

    def fetch_historical_csv(self, league: str, season_code: str = "2324") -> list[dict]:
        """Download and parse a historical match CSV from football-data.co.uk."""
        code_tuple = self.FDCO_LEAGUE_CODES.get(league)
        if not code_tuple:
            raise ValueError(f"Unknown league: '{league}'")

        url = f"{self.FDCO_BASE}/{season_code}/{code_tuple[0]}.csv"
        log_ingestor.info(f"Fetching historical CSV: {url}")
        try:
            raw = self._http_get(url)
        except Exception as e:
            log_ingestor.error(f"CSV fetch failed ({league} {season_code}): {e}")
            return []
        return self._parse_fdco_csv(raw, league, season_code)

    def ingest_local_csv(self, filepath: str, league: str = "unknown") -> list[dict]:
        """Load a locally saved football-data.co.uk CSV file."""
        log_ingestor.info(f"Loading local CSV: {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return self._parse_fdco_csv(raw, league, "local")

    def _parse_fdco_csv(self, raw: str, league: str, season: str) -> list[dict]:
        """Parse football-data.co.uk CSV into standardised match dicts."""
        reader  = csv.DictReader(io.StringIO(raw))
        matches = []
        skipped = 0

        for row in reader:
            try:
                home_goals = int(row.get("FTHG") or row.get("HG") or 0)
                away_goals = int(row.get("FTAG") or row.get("AG") or 0)

                def _odd(keys):
                    for k in keys:
                        v = row.get(k, "")
                        if v:
                            try:
                                return float(v)
                            except ValueError:
                                pass
                    return None

                home_odd = _odd(["B365H", "BbMxH", "WHH", "IWH"])
                draw_odd = _odd(["B365D", "BbMxD", "WHD", "IWD"])
                away_odd = _odd(["B365A", "BbMxA", "WHA", "IWA"])

                if not all([home_odd, draw_odd, away_odd]):
                    skipped += 1
                    continue

                matches.append({
                    "league":     league, "season":    season,
                    "date":       row.get("Date", ""),
                    "home_team":  row.get("HomeTeam", row.get("Home", "")),
                    "away_team":  row.get("AwayTeam", row.get("Away", "")),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result":     row.get("FTR", row.get("Res", "")),
                    "home_shots": int(row.get("HS", 0) or 0),
                    "away_shots": int(row.get("AS", 0) or 0),
                    "home_xg":    float(row.get("xG_home", 0) or 0),
                    "away_xg":    float(row.get("xG_away", 0) or 0),
                    "odds_home":  home_odd, "odds_draw": draw_odd, "odds_away": away_odd,
                })
            except (ValueError, KeyError) as e:
                skipped += 1
                log_ingestor.debug(f"Skipping CSV row: {e}")

        log_ingestor.info(f"Parsed {len(matches)} CSV matches ({skipped} skipped)")
        return matches

    # ── HTTP utility ──────────────────────────────────────────────────────────

    def _http_get(self, url: str, timeout: int = 15) -> str:
        """Simple HTTP GET with in-memory caching and courtesy rate-limit delay."""
        if url in self._request_cache:
            log_ingestor.debug(f"Cache hit: {url}")
            return self._request_cache[url]

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; BettingFramework/1.0)",
            "Accept": "text/html,application/json",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")

        self._request_cache[url] = content
        time.sleep(0.5)   # courtesy delay — avoid hammering free endpoints
        return content


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 8]  PARLAY MATH HELPERS                                             │
# │             Compounded vig computation and parlay EV gate                   │
# └─────────────────────────────────────────────────────────────────────────────┘

def _compounded_vig(legs: list[dict]) -> float:
    """
    Total dead-weight cost of a parlay from N legs.

    Each bookmaker keeps a fractional margin m_i on their market.
    Across N legs this compounds multiplicatively:

        V_compound = ∏(1 + m_i) − 1

    Example: three legs each at 5% margin →
        (1.05)³ − 1 = 0.1576  →  15.76% compounded vig
    """
    product = 1.0
    for leg in legs:
        product *= 1.0 + leg.get("bookmaker_margin", 0.0)
    return product - 1.0


def _parlay_ev_net(legs: list[dict]) -> float:
    """
    Net EV of a parlay after subtracting the compounded vig cost.

        net_EV = Σ EV_i − V_compound

    This is a conservative additive lower bound.  A parlay is only
    worthwhile if net_EV > 0 — otherwise the combined bookmaker margin
    exceeds the modelled edge on each individual leg.
    """
    return sum(leg["ev"] for leg in legs) - _compounded_vig(legs)


def _parlay_passes_vig_gate(legs: list[dict], min_net_ev: float = 0.0) -> tuple[bool, str]:
    """
    Return (True, detail) if the parlay passes the vig gate, else (False, reason).

    Rejection condition:  net_EV ≤ min_net_ev
    """
    if not legs:
        return False, "Empty leg list"

    vig     = _compounded_vig(legs)
    net_ev  = _parlay_ev_net(legs)
    margins = [leg.get("bookmaker_margin", 0.0) for leg in legs]
    detail  = (
        f"Legs={len(legs)} | ΣEV={sum(l['ev'] for l in legs):.4f} | "
        f"margins={[round(m,3) for m in margins]} | "
        f"compound_vig={vig:.4f} | net_EV={net_ev:.4f}"
    )
    if net_ev <= min_net_ev:
        return False, f"Parlay rejected — net EV={net_ev:.4f} ≤ {min_net_ev}. {detail}"
    return True, detail


def make_value_bet(
    home_team: str, away_team: str, selection: str,
    model_prob: float, decimal_odds: float, ev: float, kelly_stake: float,
    bookmaker_margin: float = 0.0, league: str = "",
    match_date: str = "", match_id: str = "",
) -> dict:
    """Construct a standardised value-bet dict."""
    return {
        "home_team":        home_team,
        "away_team":        away_team,
        "selection":        selection,
        "model_prob":       round(model_prob, 4),
        "decimal_odds":     round(decimal_odds, 3),
        "ev":               round(ev, 4),
        "kelly_stake":      round(kelly_stake, 4),
        "bookmaker_margin": round(bookmaker_margin, 4),
        "league":           league,
        "match_date":       match_date,
        "match_id":         match_id,
        "timestamp":        datetime.utcnow().isoformat(),
    }


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 9]  BET MANAGER                                                     │
# │             EV filter  |  Kelly staking  |  Betslip generation              │
# └─────────────────────────────────────────────────────────────────────────────┘

class BetManager:
    """
    Full betting workflow with strict parlay vig gate.

    Pipeline:
        1. evaluate_match()     — compare model probs vs bookmaker odds
        2. collect_value_bets() — accumulate EV-positive selections
        3. generate_betslips()  — package into Singles / Trixie / Accumulator
                                  (only if combined EV beats compounded vig)
        4. Fractional Kelly staking with 5% hard bankroll cap per slip
        5. Every skipped bet is logged with an explicit reason
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.22,
        max_stake_pct: float = 0.05,
        min_odds: float = 1.50,
        max_odds: float = 10.0,
        parlay_min_net_ev: float = 0.0,
    ):
        self.bankroll          = bankroll
        self.ev_threshold      = ev_threshold
        self.kelly_fraction    = kelly_fraction
        self.max_stake_pct     = max_stake_pct
        self.min_odds          = min_odds
        self.max_odds          = max_odds
        self.parlay_min_net_ev = parlay_min_net_ev

        self._value_bets:  list[dict] = []
        self._skipped_log: list[dict] = []

    # ── bet identification ────────────────────────────────────────────────────

    def evaluate_match(
        self,
        prediction: dict,
        odds: dict,
        league: str = "",
        match_date: str = "",
        match_id: str = "",
    ) -> list[dict]:
        """
        Find value bets for a single match by comparing model probs to market odds.

        prediction : output of PredictorModel.predict()
        odds       : dict with keys best_home, best_draw, best_away
        """
        if not prediction:
            self._log_skip(odds, "Empty prediction — team not in model")
            return []

        home_team = prediction["home_team"]
        away_team = prediction["away_team"]

        # Compute bookmaker overround on this specific market
        bm_margin = 0.0
        try:
            bm_margin = PredictorModel.overround(
                odds.get("best_home", 0),
                odds.get("best_draw", 0),
                odds.get("best_away", 0),
            )
        except Exception:
            pass

        found = []
        for selection, model_prob, decimal_odds in [
            ("home", prediction["home_win"], odds.get("best_home", 0)),
            ("draw", prediction["draw"],     odds.get("best_draw", 0)),
            ("away", prediction["away_win"], odds.get("best_away", 0)),
        ]:
            reason = self._pre_filter(model_prob, decimal_odds, home_team, away_team, selection)
            if reason:
                self._log_skip(
                    {"home_team": home_team, "away_team": away_team, "selection": selection},
                    reason,
                )
                continue

            ev = PredictorModel.calculate_ev(model_prob, decimal_odds)
            if ev < self.ev_threshold:
                self._log_skip(
                    {"home_team": home_team, "away_team": away_team, "selection": selection},
                    f"Edge below threshold: EV={ev:.4f} < {self.ev_threshold}",
                )
                continue

            kelly = PredictorModel.kelly_fraction(model_prob, decimal_odds, self.kelly_fraction)
            stake = min(kelly * self.bankroll, self.max_stake_pct * self.bankroll)

            bet = make_value_bet(
                home_team=home_team, away_team=away_team, selection=selection,
                model_prob=model_prob, decimal_odds=decimal_odds,
                ev=ev, kelly_stake=stake, bookmaker_margin=bm_margin,
                league=league, match_date=match_date, match_id=match_id,
            )
            found.append(bet)
            log_bet_mgr.info(
                f"✅ VALUE BET: {home_team} vs {away_team} | {selection.upper()} "
                f"@ {decimal_odds:.2f} | EV={ev:.4f} | margin={bm_margin:.3f} | "
                f"Stake=£{stake:.2f}"
            )
        return found

    def _pre_filter(
        self, model_prob: float, decimal_odds: float,
        home_team: str, away_team: str, selection: str,
    ) -> Optional[str]:
        """Return a skip reason string, or None if the bet passes pre-filters."""
        if model_prob <= 0:
            return "Zero model probability"
        if decimal_odds < self.min_odds:
            return f"Odds too short: {decimal_odds:.2f} < {self.min_odds}"
        if decimal_odds > self.max_odds:
            return f"Odds too high (low liquidity): {decimal_odds:.2f} > {self.max_odds}"
        if model_prob < 0.10:
            return f"Insufficient model confidence: {model_prob:.4f}"
        return None

    def collect_value_bets(self, bets: list[dict]):
        self._value_bets.extend(bets)
        self._value_bets.sort(key=lambda b: b["ev"], reverse=True)

    def clear_bets(self):
        self._value_bets = []
        self._skipped_log = []

    # ── betslip generation ────────────────────────────────────────────────────

    def generate_betslips(
        self,
        value_bets: Optional[list[dict]] = None,
        games_per_slip: int = 3,
        slip_type: str = "auto",
        top_n: int = 10,
    ) -> list[dict]:
        """
        Package value bets into betslips with strict compounded-vig gating.

        slip_type : "single" | "trixie" | "accumulator" | "auto"
            auto → trixie if games_per_slip==3, else accumulator

        Trixie and Accumulator combos are REJECTED if:
            Σ EV_i  ≤  ∏(1 + margin_i) − 1   (net EV ≤ compounded vig)
        """
        bets = value_bets if value_bets is not None else self._value_bets
        if not bets:
            log_bet_mgr.warning("No value bets available to generate betslips")
            return []

        # Deduplicate: one selection per match pair
        seen, unique = set(), []
        for b in sorted(bets, key=lambda x: x["ev"], reverse=True):
            key = f"{b['home_team']}|{b['away_team']}"
            if key not in seen:
                seen.add(key)
                unique.append(b)
            if len(unique) >= top_n:
                break

        if slip_type == "auto":
            slip_type = "trixie" if games_per_slip == 3 else "accumulator"

        log_bet_mgr.info(
            f"Generating {slip_type} betslips | "
            f"{len(unique)} candidates | {games_per_slip} legs"
        )

        if slip_type == "single":
            return [self._make_single(b) for b in unique]

        elif slip_type == "trixie":
            if len(unique) < 3:
                log_bet_mgr.warning("Need ≥3 bets for Trixie — reverting to singles")
                return [self._make_single(b) for b in unique]

            slips, rejected = [], 0
            for combo in itertools.combinations(unique, 3):
                legs = list(combo)
                ok, detail = _parlay_passes_vig_gate(legs, self.parlay_min_net_ev)
                if not ok:
                    self._log_skip(
                        {"home_team": "Trixie", "away_team": "", "selection": "combo"},
                        detail,
                    )
                    rejected += 1
                    continue
                slips.append(self._make_trixie(legs))

            if rejected:
                log_bet_mgr.info(
                    f"Trixie vig gate: {rejected} combos rejected | {len(slips)} accepted"
                )
            return slips

        elif slip_type == "accumulator":
            selected = unique[:games_per_slip]
            if not selected:
                return []

            ok, detail = _parlay_passes_vig_gate(selected, self.parlay_min_net_ev)
            if not ok:
                log_bet_mgr.warning(f"Accumulator rejected by vig gate: {detail}")
                self._log_skip(
                    {"home_team": "Acca", "away_team": "", "selection": "combo"}, detail
                )
                log_bet_mgr.info("Falling back to singles")
                return [self._make_single(b) for b in selected]

            return [self._make_accumulator(selected)]

        else:
            raise ValueError(f"Unknown slip_type: '{slip_type}'")

    # ── slip constructors ─────────────────────────────────────────────────────

    def _make_single(self, bet: dict) -> dict:
        stake   = self._cap_stake(bet["kelly_stake"])
        net_ev  = bet["ev"] - bet.get("bookmaker_margin", 0.0)
        return {
            "slip_type":        "Single",
            "legs":             [bet],
            "stake":            round(stake, 2),
            "combined_odds":    round(bet["decimal_odds"], 3),
            "potential_return": round(stake * bet["decimal_odds"], 2),
            "combined_ev":      round(bet["ev"], 4),
            "net_ev":           round(net_ev, 4),
            "compounded_vig":   round(bet.get("bookmaker_margin", 0.0), 4),
        }

    def _make_trixie(self, legs: list[dict]) -> dict:
        """
        Trixie = 3 doubles (C(3,2)) + 1 treble = 4 unit bets.
        Stake per unit = kelly_min / 4 so total outlay = kelly_min.
        """
        assert len(legs) == 3
        base_stake       = self._cap_stake(min(b["kelly_stake"] for b in legs) / 4)
        vig              = _compounded_vig(legs)
        nev              = _parlay_ev_net(legs)
        components       = []
        total_potential  = 0.0

        for a, b in itertools.combinations(legs, 2):
            odds = a["decimal_odds"] * b["decimal_odds"]
            pot  = base_stake * odds
            components.append({
                "type":          "Double",
                "legs":          [f"{a['home_team']} ({a['selection']})",
                                  f"{b['home_team']} ({b['selection']})"],
                "combined_odds": round(odds, 3),
                "potential":     round(pot, 2),
            })
            total_potential += pot

        treble_odds = math.prod(b["decimal_odds"] for b in legs)
        pot         = base_stake * treble_odds
        components.append({
            "type":          "Treble",
            "legs":          [f"{b['home_team']} ({b['selection']})" for b in legs],
            "combined_odds": round(treble_odds, 3),
            "potential":     round(pot, 2),
        })
        total_potential += pot

        return {
            "slip_type":        "Trixie",
            "legs":             legs,
            "components":       components,
            "stake_per_unit":   round(base_stake, 2),
            "total_stake":      round(base_stake * 4, 2),
            "combined_odds":    round(treble_odds, 3),
            "potential_return": round(total_potential, 2),
            "combined_ev":      round(sum(b["ev"] for b in legs) / 3, 4),
            "net_ev":           round(nev, 4),
            "compounded_vig":   round(vig, 4),
            "vig_passed":       True,
        }

    def _make_accumulator(self, legs: list[dict]) -> dict:
        """N-fold accumulator.  Stake discounted by 0.5^(N-1) for variance."""
        combined_odds     = math.prod(b["decimal_odds"] for b in legs)
        vig               = _compounded_vig(legs)
        nev               = _parlay_ev_net(legs)
        variance_discount = 0.5 ** (len(legs) - 1)
        base_stake        = self._cap_stake(
            min(b["kelly_stake"] for b in legs) * variance_discount
        )
        return {
            "slip_type":        f"{len(legs)}-Fold Accumulator",
            "legs":             legs,
            "stake":            round(base_stake, 2),
            "combined_odds":    round(combined_odds, 3),
            "potential_return": round(base_stake * combined_odds, 2),
            "combined_ev":      round(sum(b["ev"] for b in legs) / len(legs), 4),
            "net_ev":           round(nev, 4),
            "compounded_vig":   round(vig, 4),
            "vig_passed":       True,
        }

    def _cap_stake(self, kelly_stake: float) -> float:
        """Hard cap: stake cannot exceed max_stake_pct × bankroll."""
        return min(float(kelly_stake), self.max_stake_pct * self.bankroll)

    # ── logging & reporting ───────────────────────────────────────────────────

    def _log_skip(self, bet_info: dict, reason: str):
        entry = {
            "match":     f"{bet_info.get('home_team','')} vs {bet_info.get('away_team','')}",
            "selection": bet_info.get("selection", ""),
            "reason":    reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._skipped_log.append(entry)
        log_bet_mgr.debug(f"⏭  SKIP — {entry['match']} [{entry['selection']}]: {reason}")

    def get_skip_log(self)    -> list[dict]: return list(self._skipped_log)
    def get_value_bets(self)  -> list[dict]: return list(self._value_bets)

    def summary(self) -> dict:
        vb = self._value_bets
        return {
            "value_bets_found":  len(vb),
            "bets_skipped":      len(self._skipped_log),
            "bankroll":          self.bankroll,
            "total_exposure":    round(sum(b["kelly_stake"] for b in vb), 2),
            "avg_ev":            round(sum(b["ev"] for b in vb) / len(vb), 4) if vb else 0.0,
            "avg_bookie_margin": round(
                sum(b.get("bookmaker_margin", 0) for b in vb) / len(vb), 4
            ) if vb else 0.0,
        }


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 10]  BACKTESTER                                                     │
# │              Walk-forward simulation  |  ROI / drawdown / Sharpe metrics   │
# └─────────────────────────────────────────────────────────────────────────────┘

class Backtester:
    """
    Walk-forward backtesting engine — no look-ahead bias.

    Strategy:
        1. Sort all historical matches by date
        2. Fit model on a rolling `train_window` of matches
        3. Evaluate bets on the next `test_window` matches
        4. Slide forward and refit every 3 test windows
        5. Track bankroll evolution and aggregate performance metrics

    Critical fix (v2): reference_date is set to the LAST DATE in each
    training window so time-decay weights are historically anchored, not
    anchored to today's date (which would make all history look maximally old).
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.22,
        max_stake_pct: float = 0.05,
        min_odds: float = 1.50,
        max_odds: float = 10.0,
        train_window: int = 200,
        test_window: int = 50,
    ):
        self.initial_bankroll = initial_bankroll
        self.ev_threshold     = ev_threshold
        self.kelly_fraction   = kelly_fraction
        self.max_stake_pct    = max_stake_pct
        self.min_odds         = min_odds
        self.max_odds         = max_odds
        self.train_window     = train_window
        self.test_window      = test_window

        self.results:        list[dict]  = []
        self.bankroll_curve: list[float] = []

    # ── main entry point ──────────────────────────────────────────────────────

    def run_backtest(self, historical_data: list[dict]) -> dict:
        """
        Simulate the full algorithm over historical_data.

        historical_data: list of match dicts (from DataIngestor.fetch_historical_csv)
            Required keys: home_team, away_team, home_goals, away_goals,
                           result (H/D/A), odds_home, odds_draw, odds_away

        Returns performance dict: ROI, win_rate, max_drawdown, Sharpe, etc.
        """
        if len(historical_data) < self.train_window + self.test_window:
            raise ValueError(
                f"Need ≥{self.train_window + self.test_window} matches. "
                f"Got {len(historical_data)}."
            )

        data     = self._sort_by_date(historical_data)
        bankroll = self.initial_bankroll
        self.results, self.bankroll_curve = [], [bankroll]
        bets_placed = bets_won = 0
        total_staked = total_profit = 0.0

        cursor        = self.train_window
        refit_counter = 0
        model         = PredictorModel(half_life_days=90.0)

        log_backtest.info(
            f"Starting walk-forward backtest | "
            f"Matches={len(data)} | Train={self.train_window} | Test={self.test_window}"
        )

        while cursor < len(data):
            if refit_counter == 0:
                train_data = data[max(0, cursor - self.train_window): cursor]
                try:
                    ref_date = Backtester._latest_date(train_data)
                    model.fit(train_data, reference_date=ref_date)
                    log_backtest.debug(f"Model refitted at cursor={cursor} | ref={ref_date}")
                except Exception as e:
                    log_backtest.warning(f"Fit failed at cursor={cursor}: {e}")
                    cursor += self.test_window
                    continue

            for match in data[cursor: cursor + self.test_window]:
                if bankroll <= 0:
                    log_backtest.warning("Bankroll exhausted — stopping")
                    break
                try:
                    for br in self._evaluate_historical_match(match, model, bankroll):
                        bankroll     += br["profit"]
                        total_staked += br["stake"]
                        total_profit += br["profit"]
                        bets_placed  += 1
                        bets_won     += int(br["won"])
                        self.results.append(br)
                        self.bankroll_curve.append(bankroll)
                except Exception as e:
                    log_backtest.debug(
                        f"Error on {match.get('home_team')} vs {match.get('away_team')}: {e}"
                    )

            cursor        += self.test_window
            refit_counter  = (refit_counter + 1) % 3

        metrics = self._calculate_metrics(bets_placed, bets_won, total_staked, total_profit, bankroll)
        log_backtest.info(
            f"Backtest complete | Bets={bets_placed} | ROI={metrics['roi_pct']:.2f}% | "
            f"WR={metrics['win_rate_pct']:.1f}% | MaxDD={metrics['max_drawdown_pct']:.1f}%"
        )
        return metrics

    # ── internal evaluation ───────────────────────────────────────────────────

    def _evaluate_historical_match(
        self, match: dict, model: PredictorModel, bankroll: float
    ) -> list[dict]:
        """Apply the betting strategy to one historical match — returns list of placed bets."""
        pred = model.predict(match["home_team"], match["away_team"])
        if not pred:
            return []

        placed = []
        for selection, model_prob, decimal_odds, result_code in [
            ("home", pred["home_win"], match.get("odds_home", 0), "H"),
            ("draw", pred["draw"],     match.get("odds_draw", 0), "D"),
            ("away", pred["away_win"], match.get("odds_away", 0), "A"),
        ]:
            if decimal_odds < self.min_odds or decimal_odds > self.max_odds:
                continue
            if model_prob <= 0:
                continue

            ev = PredictorModel.calculate_ev(model_prob, decimal_odds)
            if ev < self.ev_threshold:
                continue

            kelly  = PredictorModel.kelly_fraction(model_prob, decimal_odds, self.kelly_fraction)
            stake  = max(min(kelly * bankroll, self.max_stake_pct * bankroll), 0.01)
            actual = match.get("result", "")
            won    = (actual == result_code)
            profit = (stake * (decimal_odds - 1)) if won else -stake

            placed.append({
                "home_team":    match["home_team"],
                "away_team":    match["away_team"],
                "selection":    selection,
                "model_prob":   round(model_prob, 4),
                "decimal_odds": round(decimal_odds, 3),
                "ev":           round(ev, 4),
                "stake":        round(stake, 2),
                "profit":       round(profit, 2),
                "won":          won,
                "actual_result": actual,
                "date":         match.get("date", ""),
                "league":       match.get("league", ""),
            })
        return placed

    # ── metrics ───────────────────────────────────────────────────────────────

    def _calculate_metrics(
        self,
        bets_placed: int, bets_won: int,
        total_staked: float, total_profit: float, final_bankroll: float,
    ) -> dict:
        roi      = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        win_rate = (bets_won / bets_placed * 100)      if bets_placed  > 0 else 0.0
        avg_ev   = (sum(r["ev"] for r in self.results) / len(self.results)) if self.results else 0.0

        by_league: dict = defaultdict(
            lambda: {"bets": 0, "profit": 0.0, "staked": 0.0, "wins": 0}
        )
        for r in self.results:
            lg = r.get("league", "unknown")
            by_league[lg]["bets"]   += 1
            by_league[lg]["profit"] += r["profit"]
            by_league[lg]["staked"] += r["stake"]
            by_league[lg]["wins"]   += int(r["won"])

        league_breakdown = {
            lg: {
                "bets":     stats["bets"],
                "wins":     stats["wins"],
                "roi_pct":  round(stats["profit"] / stats["staked"] * 100
                                  if stats["staked"] > 0 else 0, 2),
                "win_rate": round(stats["wins"] / stats["bets"] * 100
                                  if stats["bets"] > 0 else 0, 1),
            }
            for lg, stats in by_league.items()
        }

        return {
            "bets_placed":          bets_placed,
            "bets_won":             bets_won,
            "total_staked":         round(total_staked, 2),
            "total_profit":         round(total_profit, 2),
            "roi_pct":              round(roi, 2),
            "win_rate_pct":         round(win_rate, 1),
            "max_drawdown_pct":     round(self._max_drawdown(), 2),
            "sharpe_ratio":         round(self._sharpe_ratio(), 3),
            "avg_ev":               round(avg_ev, 4),
            "final_bankroll":       round(final_bankroll, 2),
            "initial_bankroll":     self.initial_bankroll,
            "bankroll_growth_pct":  round(
                (final_bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 2
            ),
            "league_breakdown":     league_breakdown,
            "bankroll_curve":       self.bankroll_curve,
        }

    def _max_drawdown(self) -> float:
        """Maximum peak-to-trough bankroll decline as a percentage of peak."""
        if not self.bankroll_curve:
            return 0.0
        peak = self.bankroll_curve[0]
        max_dd = 0.0
        for value in self.bankroll_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _sharpe_ratio(self, risk_free: float = 0.0) -> float:
        """Sharpe ratio using profit-per-unit-staked as the return series."""
        if not self.results:
            return 0.0
        returns = [r["profit"] / r["stake"] for r in self.results if r["stake"] > 0]
        if len(returns) < 2:
            return 0.0
        mean_r   = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r    = math.sqrt(variance) if variance > 0 else 1e-10
        return (mean_r - risk_free) / std_r

    # ── utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _latest_date(data: list[dict]) -> datetime:
        """
        Return the datetime of the most recent match in a list.
        Used to anchor time-decay reference_date in walk-forward backtesting,
        preventing look-ahead bias from anchoring decay to today's date.
        Falls back to datetime.utcnow() if no dates are parseable.
        """
        best = None
        for m in data:
            raw = str(m.get("date", "") or "").split(" ")[0].split("T")[0].strip()
            for fmt in _DATE_FMTS:
                try:
                    dt = datetime.strptime(raw, fmt)
                    if best is None or dt > best:
                        best = dt
                    break
                except ValueError:
                    continue
        return best or datetime.utcnow()

    @staticmethod
    def _sort_by_date(data: list[dict]) -> list[dict]:
        """Sort match list chronologically (best-effort, falls back to original order)."""
        def date_key(m):
            raw = str(m.get("date", "") or "").split(" ")[0].split("T")[0].strip()
            for fmt in _DATE_FMTS:
                try:
                    return datetime.strptime(raw, fmt)
                except ValueError:
                    pass
            return raw
        try:
            return sorted(data, key=date_key)
        except Exception:
            return data

    def print_report(self, metrics: dict):
        """Pretty-print a full backtest report to the logger."""
        sep = "=" * 60
        log_backtest.info(sep)
        log_backtest.info("BACKTEST REPORT")
        log_backtest.info(sep)
        log_backtest.info(f"  Bets placed:    {metrics['bets_placed']}")
        log_backtest.info(f"  Win rate:       {metrics['win_rate_pct']:.1f}%")
        log_backtest.info(f"  ROI:            {metrics['roi_pct']:+.2f}%")
        log_backtest.info(f"  Max Drawdown:   {metrics['max_drawdown_pct']:.1f}%")
        log_backtest.info(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.3f}")
        log_backtest.info(f"  Avg EV/bet:     {metrics['avg_ev']:.4f}")
        log_backtest.info(f"  Total profit:   £{metrics['total_profit']:.2f}")
        log_backtest.info(
            f"  Final bankroll: £{metrics['final_bankroll']:.2f} "
            f"({metrics['bankroll_growth_pct']:+.2f}%)"
        )
        log_backtest.info("")
        log_backtest.info("  League Breakdown:")
        for lg, s in metrics.get("league_breakdown", {}).items():
            log_backtest.info(
                f"    {lg:15s}  Bets={s['bets']:4d}  "
                f"WR={s['win_rate']:5.1f}%  ROI={s['roi_pct']:+6.2f}%"
            )
        log_backtest.info(sep)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 11]  DATABASE                                                        │
# │              SQLite persistence  |  CLV (Closing Line Value) tracking       │
# └─────────────────────────────────────────────────────────────────────────────┘

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS betslips (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    slip_uuid        TEXT    NOT NULL UNIQUE,
    slip_type        TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    league           TEXT,
    total_stake      REAL    NOT NULL,
    combined_odds    REAL,
    potential_return REAL,
    combined_ev      REAL,
    net_ev           REAL,
    compounded_vig   REAL,
    status           TEXT    DEFAULT 'pending',
    profit           REAL,
    settled_at       TEXT
);

CREATE TABLE IF NOT EXISTS bet_legs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    slip_uuid        TEXT    NOT NULL REFERENCES betslips(slip_uuid),
    home_team        TEXT    NOT NULL,
    away_team        TEXT    NOT NULL,
    selection        TEXT    NOT NULL,
    model_prob       REAL,
    opening_odds     REAL    NOT NULL,
    closing_odds     REAL,
    clv_pct          REAL,
    bookmaker_margin REAL,
    ev               REAL,
    kelly_stake      REAL,
    match_date       TEXT,
    match_id         TEXT,
    league           TEXT,
    result           TEXT,
    leg_status       TEXT    DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS clv_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    leg_id      INTEGER NOT NULL REFERENCES bet_legs(id),
    snapshot_at TEXT    NOT NULL,
    odds        REAL    NOT NULL,
    source      TEXT
);

CREATE INDEX IF NOT EXISTS idx_legs_slip   ON bet_legs(slip_uuid);
CREATE INDEX IF NOT EXISTS idx_legs_match  ON bet_legs(match_id);
CREATE INDEX IF NOT EXISTS idx_legs_status ON bet_legs(leg_status);
"""


class BettingDatabase:
    """
    SQLite persistence layer for betslips, bet legs, and CLV tracking.

    CLV (Closing Line Value) formula:
        CLV% = (opening_odds / closing_odds − 1) × 100

    Positive CLV means you obtained better odds than the market closed at —
    the gold-standard indicator of long-run model edge.

    Tables:
        betslips      — one row per generated betslip
        bet_legs      — one row per leg within a slip
        clv_snapshots — timestamped odds snapshots for CLV history
    """

    def __init__(self, db_path: str = "betting_framework.db"):
        self.db_path = Path(db_path)
        self._init_db()
        log_database.info(f"Database initialised: {self.db_path.resolve()}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ── save betslips ─────────────────────────────────────────────────────────

    def save_betslips(self, betslips: list[dict]) -> list[str]:
        """
        Persist a list of betslip dicts to the database.
        Returns list of generated slip_uuids.
        """
        saved_uuids = []
        with self._conn() as conn:
            for slip in betslips:
                slip_uuid   = str(uuid.uuid4())
                now         = datetime.utcnow().isoformat()
                total_stake = slip.get("total_stake") or slip.get("stake") or 0.0

                conn.execute("""
                    INSERT INTO betslips
                        (slip_uuid, slip_type, created_at, league, total_stake,
                         combined_odds, potential_return, combined_ev, net_ev, compounded_vig)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    slip_uuid, slip.get("slip_type", "Unknown"), now,
                    slip.get("legs", [{}])[0].get("league", ""),
                    round(float(total_stake), 4),
                    slip.get("combined_odds"),    slip.get("potential_return"),
                    slip.get("combined_ev"),      slip.get("net_ev"),
                    slip.get("compounded_vig"),
                ))

                for leg in slip.get("legs", []):
                    conn.execute("""
                        INSERT INTO bet_legs
                            (slip_uuid, home_team, away_team, selection,
                             model_prob, opening_odds, bookmaker_margin,
                             ev, kelly_stake, match_date, match_id, league)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        slip_uuid,
                        leg.get("home_team", ""),   leg.get("away_team", ""),
                        leg.get("selection", ""),   leg.get("model_prob"),
                        leg.get("decimal_odds"),    leg.get("bookmaker_margin"),
                        leg.get("ev"),              leg.get("kelly_stake"),
                        leg.get("match_date", ""),  leg.get("match_id", ""),
                        leg.get("league", ""),
                    ))

                saved_uuids.append(slip_uuid)
                log_database.info(
                    f"Saved {slip.get('slip_type')} | "
                    f"uuid={slip_uuid[:8]}… | Stake=£{total_stake:.2f}"
                )
        return saved_uuids

    # ── CLV update ────────────────────────────────────────────────────────────

    def update_clv(
        self, odds_updates: list[dict], source: str = "the_odds_api"
    ) -> int:
        """
        Update closing odds and CLV% for pending bet legs.

        odds_updates: list of dicts with keys:
            match_id, selection ("home"/"draw"/"away"), closing_odds

        Returns number of legs updated.
        Run 15-30 minutes pre-kickoff when odds are most efficient.
        """
        updated = 0
        now     = datetime.utcnow().isoformat()

        with self._conn() as conn:
            for upd in odds_updates:
                mid     = upd.get("match_id", "")
                sel     = upd.get("selection", "")
                cl_odds = upd.get("closing_odds", 0.0)
                if not mid or not sel or cl_odds <= 1.0:
                    continue

                rows = conn.execute("""
                    SELECT id, opening_odds FROM bet_legs
                    WHERE match_id = ? AND selection = ? AND leg_status = 'pending'
                """, (mid, sel)).fetchall()

                for row in rows:
                    leg_id = row["id"]
                    op     = row["opening_odds"]
                    clv    = ((op / cl_odds) - 1.0) * 100 if cl_odds > 0 and op else None

                    conn.execute("""
                        UPDATE bet_legs SET closing_odds = ?, clv_pct = ? WHERE id = ?
                    """, (cl_odds, clv, leg_id))

                    conn.execute("""
                        INSERT INTO clv_snapshots (leg_id, snapshot_at, odds, source)
                        VALUES (?, ?, ?, ?)
                    """, (leg_id, now, cl_odds, source))

                    updated += 1
                    sign = "✅" if (clv or 0) >= 0 else "⚠️"
                    if clv is not None:
                        log_database.info(
                            f"{sign} CLV leg={leg_id} | {mid} | {sel} | "
                            f"open={op:.2f} → close={cl_odds:.2f} | CLV={clv:.2f}%"
                        )

        log_database.info(f"CLV update: {updated} legs updated")
        return updated

    def build_clv_updates_from_odds(self, live_events: list[dict]) -> list[dict]:
        """Convert a fetch_odds() result list into update_clv() input format."""
        updates = []
        for ev in live_events:
            event_id = ev.get("event_id", "")
            for sel, key in [("home", "best_home"), ("draw", "best_draw"), ("away", "best_away")]:
                closing = ev.get(key, 0.0)
                if closing > 1.0:
                    updates.append({
                        "match_id":     event_id,
                        "selection":    sel,
                        "closing_odds": closing,
                    })
        return updates

    # ── settling ──────────────────────────────────────────────────────────────

    def settle_bet(self, slip_uuid: str, results: dict[str, str]) -> dict:
        """
        Settle a betslip.  results = {match_id: "H"/"D"/"A"}.
        Marks each leg won/lost and computes P&L for the slip.
        """
        now = datetime.utcnow().isoformat()
        RESULT_MAP = {"home": "H", "draw": "D", "away": "A"}

        with self._conn() as conn:
            slip = conn.execute(
                "SELECT * FROM betslips WHERE slip_uuid = ?", (slip_uuid,)
            ).fetchone()
            if not slip:
                return {"error": f"Slip {slip_uuid} not found"}

            legs      = conn.execute(
                "SELECT * FROM bet_legs WHERE slip_uuid = ?", (slip_uuid,)
            ).fetchall()
            legs_won  = 0

            for leg in legs:
                actual   = results.get(leg["match_id"], "")
                expected = RESULT_MAP.get(leg["selection"], "?")
                won      = (actual == expected)
                conn.execute("""
                    UPDATE bet_legs SET result = ?, leg_status = ? WHERE id = ?
                """, (actual, "won" if won else "lost", leg["id"]))
                if won:
                    legs_won += 1

            slip_won = (legs_won == len(legs))
            stake    = slip["total_stake"]
            profit   = (round(stake * (slip["combined_odds"] or 1.0) - stake, 2)
                        if slip_won else -stake)

            conn.execute("""
                UPDATE betslips SET status = ?, profit = ?, settled_at = ?
                WHERE slip_uuid = ?
            """, ("won" if slip_won else "lost", profit, now, slip_uuid))

        return {
            "slip_uuid":  slip_uuid,
            "status":     "won" if slip_won else "lost",
            "legs_won":   legs_won,
            "total_legs": len(legs),
            "profit":     profit,
        }

    # ── reports ───────────────────────────────────────────────────────────────

    def clv_report(self) -> dict:
        """Aggregate CLV statistics across all legs that have closing odds."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT selection, league, clv_pct, opening_odds, closing_odds, leg_status
                FROM bet_legs WHERE clv_pct IS NOT NULL
            """).fetchall()

        if not rows:
            return {"message": "No CLV data yet — run --mode update_clv first"}

        clvs = [r["clv_pct"] for r in rows]
        pos  = [c for c in clvs if c > 0]
        by_league: dict[str, list[float]] = {}
        for r in rows:
            by_league.setdefault(r["league"] or "unknown", []).append(r["clv_pct"])

        return {
            "total_legs":        len(clvs),
            "avg_clv_pct":       round(sum(clvs) / len(clvs), 3),
            "pct_positive_clv":  round(len(pos) / len(clvs) * 100, 1),
            "max_clv_pct":       round(max(clvs), 3),
            "min_clv_pct":       round(min(clvs), 3),
            "by_league": {
                lg: {
                    "legs":    len(vs),
                    "avg_clv": round(sum(vs) / len(vs), 3),
                    "pct_pos": round(len([v for v in vs if v > 0]) / len(vs) * 100, 1),
                }
                for lg, vs in by_league.items()
            },
        }

    def pending_betslips(self) -> list[dict]:
        """Return all pending betslips with their legs."""
        with self._conn() as conn:
            slips = conn.execute(
                "SELECT * FROM betslips WHERE status='pending' ORDER BY created_at DESC"
            ).fetchall()
            return [
                {**dict(slip), "legs": [
                    dict(l) for l in conn.execute(
                        "SELECT * FROM bet_legs WHERE slip_uuid=?", (slip["slip_uuid"],)
                    ).fetchall()
                ]}
                for slip in slips
            ]

    def profit_summary(self) -> dict:
        """Overall P&L summary across all settled slips."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS total_slips,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) AS wins,
                       SUM(total_stake) AS total_staked,
                       SUM(COALESCE(profit, 0)) AS total_profit
                FROM betslips WHERE status IN ('won','lost')
            """).fetchone()

        if not row or not row["total_slips"]:
            return {"message": "No settled bets yet"}

        return {
            "total_slips":  row["total_slips"],
            "wins":         row["wins"],
            "win_rate_pct": round(row["wins"] / row["total_slips"] * 100, 1),
            "total_staked": round(row["total_staked"] or 0, 2),
            "total_profit": round(row["total_profit"] or 0, 2),
            "roi_pct":      round(
                (row["total_profit"] or 0) / (row["total_staked"] or 1) * 100, 2
            ),
        }


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 12]  ORCHESTRATOR                                                   │
# │              BettingFramework — wires all modules into runnable pipelines  │
# └─────────────────────────────────────────────────────────────────────────────┘

_SEASON_CODE_MAP = {
    2024: "2425", 2023: "2324", 2022: "2223",
    2021: "2122", 2020: "2021", 2019: "1920",
}


class BettingFramework:
    """
    Top-level orchestrator connecting all five modules:
        DataIngestor  →  PredictorModel  →  BetManager
                      ↘  Backtester
                      ↘  BettingDatabase

    Supported modes (mirrors the CLI):
        run_live()          — full live prediction + betslip pipeline
        run_update_clv()    — fetch closing odds and write CLV to DB
        run_backtest_pipeline() — walk-forward simulation over CSVs
        run_demo()          — fully synthetic, no external APIs needed
    """

    def __init__(
        self,
        odds_api_key: str = "",
        bankroll: float = 1000.0,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.22,
        max_stake_pct: float = 0.05,
        half_life_days: float = 90.0,
        xg_blend: float = 0.0,
        use_xg: bool = False,
        db_path: str = "betting_framework.db",
    ):
        self.odds_api_key = odds_api_key
        self.bankroll     = bankroll

        self.ingestor   = DataIngestor(odds_api_key=odds_api_key)
        self.model      = PredictorModel(
            half_life_days=half_life_days, xg_blend=xg_blend, use_xg=use_xg,
        )
        self.bet_mgr    = BetManager(
            bankroll=bankroll, ev_threshold=ev_threshold,
            kelly_fraction=kelly_fraction, max_stake_pct=max_stake_pct,
        )
        self.backtester = Backtester(
            initial_bankroll=bankroll, ev_threshold=ev_threshold,
            kelly_fraction=kelly_fraction, max_stake_pct=max_stake_pct,
        )
        self.db = BettingDatabase(db_path)

    # ── live pipeline ─────────────────────────────────────────────────────────

    def run_live(
        self,
        league: str = "EPL",
        season: int = 2024,
        cold_start_seasons: list[int] = None,
        games_per_slip: int = 3,
        slip_type: str = "auto",
        save_to_db: bool = True,
    ) -> dict:
        """
        Full live pipeline with multi-season cold-start training.

        Cold-start fix: pull historical seasons first so the model has
        ample data even at the start of a new campaign (5-10 rounds played).
        Historical seasons are fetched from understat with FDCO CSV fallback.
        """
        log_main.info("=" * 65)
        log_main.info(f"LIVE PIPELINE  {league} | Season {season}")
        if cold_start_seasons:
            log_main.info(f"Cold-start seasons: {cold_start_seasons}")
        log_main.info("=" * 65)

        all_xg = []
        for hist in (cold_start_seasons or []):
            log_main.info(f"  [Cold-start] understat {league} {hist}…")
            hm = self.ingestor.fetch_understat_xg(league, hist)
            if not hm:
                sc = _SEASON_CODE_MAP.get(hist)
                if sc:
                    log_main.info(f"  [Fallback] FDCO CSV {league} {sc}…")
                    hm = self.ingestor.fetch_historical_csv(league, sc)
            log_main.info(f"    → {len(hm)} matches from {hist}")
            all_xg.extend(hm)

        log_main.info(f"  [Current] understat {league} {season}…")
        current = self.ingestor.fetch_understat_xg(league, season)
        if not current:
            sc = _SEASON_CODE_MAP.get(season)
            if sc:
                current = self.ingestor.fetch_historical_csv(league, sc)
        log_main.info(f"  Current season: {len(current)} matches")
        all_xg.extend(current)

        if len(all_xg) < 30:
            log_main.warning(
                f"Only {len(all_xg)} training matches — model may be unstable."
            )

        log_main.info(f"Fitting on {len(all_xg)} total matches…")
        try:
            self.model.fit(all_xg)
        except Exception as exc:
            log_main.error(f"Model fit failed: {exc}")
            return {"error": str(exc)}
        log_main.info(f"Fit stats: {self.model.fit_summary()}")

        upcoming = self.ingestor.fetch_odds(league)
        if not upcoming:
            return {"error": "No odds data returned — check API key or quota"}

        self.bet_mgr.clear_bets()
        all_value_bets = []
        for event in upcoming:
            pred  = self.model.predict(event["home_team"], event["away_team"])
            vbets = self.bet_mgr.evaluate_match(
                prediction=pred, odds=event, league=league,
                match_date=event.get("commence", ""), match_id=event.get("event_id", ""),
            )
            all_value_bets.extend(vbets)

        self.bet_mgr.collect_value_bets(all_value_bets)
        betslips    = self.bet_mgr.generate_betslips(games_per_slip=games_per_slip, slip_type=slip_type)
        saved_uuids = self.db.save_betslips(betslips) if save_to_db and betslips else []

        summary = self.bet_mgr.summary()
        log_main.info(f"Summary: {summary}")
        return {
            "league": league, "season": season,
            "training_matches": len(all_xg), "matches_analysed": len(upcoming),
            "value_bets": all_value_bets, "betslips": betslips,
            "saved_uuids": saved_uuids, "skipped": self.bet_mgr.get_skip_log(),
            "summary": summary, "model_stats": self.model.fit_summary(),
        }

    # ── CLV update pipeline ───────────────────────────────────────────────────

    def run_update_clv(self, league: str = "EPL") -> dict:
        """
        Fetch current market odds and update CLV for all pending bet legs.
        Run 15-30 minutes before kick-off when odds are most efficient.
        """
        log_main.info("=" * 65)
        log_main.info(f"CLV UPDATE  {league}")
        log_main.info("=" * 65)

        if not self.odds_api_key:
            return {"error": "Odds API key required for CLV update"}

        live_events  = self.ingestor.fetch_odds(league)
        if not live_events:
            return {"warning": "No live events returned"}

        updates   = self.db.build_clv_updates_from_odds(live_events)
        n_updated = self.db.update_clv(updates)
        report    = self.db.clv_report()
        log_main.info(f"CLV report: {json.dumps(report, indent=2)}")
        return {"events_fetched": len(live_events), "legs_updated": n_updated, "clv_report": report}

    # ── backtest pipeline ─────────────────────────────────────────────────────

    def run_backtest_pipeline(
        self,
        league: str = "EPL",
        season_codes: list[str] = None,
        local_csv: str = None,
    ) -> dict:
        """Walk-forward backtest over one or more historical seasons."""
        log_main.info("=" * 65)
        log_main.info(f"BACKTEST PIPELINE  {league}")
        log_main.info("=" * 65)

        all_matches = []
        if local_csv:
            all_matches = self.ingestor.ingest_local_csv(local_csv, league)
        else:
            for sc in (season_codes or ["2223", "2324"]):
                all_matches.extend(self.ingestor.fetch_historical_csv(league, sc))

        if not all_matches:
            return {"error": "No historical data loaded"}
        log_main.info(f"Total historical matches: {len(all_matches)}")

        metrics = self.backtester.run_backtest(all_matches)
        self.backtester.print_report(metrics)
        return metrics

    # ── demo pipeline ─────────────────────────────────────────────────────────

    def run_demo(self) -> dict:
        """
        Fully self-contained demo with synthetic EPL data.
        No external API calls — runs offline in any environment.
        Demonstrates all pipeline stages including DB persistence and CLV.
        """
        log_main.info("DEMO MODE — generating 3 synthetic EPL seasons")
        random.seed(42)

        teams = [
            "Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd",
            "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
            "Brentford", "Fulham", "Wolves", "Everton", "Crystal Palace",
            "Nottm Forest", "Bournemouth", "Leicester", "Luton", "Sheffield Utd",
        ]
        strengths = {t: (random.uniform(0.8, 2.0), random.uniform(0.8, 2.0)) for t in teams}

        all_matches = []
        for year in [2022, 2023, 2024]:
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i == j:
                        continue
                    ha, hd = strengths[home]
                    aa, ad = strengths[away]
                    mu_h = max(ha / ad * 1.25, 0.1)
                    mu_a = max(aa / hd, 0.1)
                    hg   = min(int(random.expovariate(1 / mu_h)), 8)
                    ag   = min(int(random.expovariate(1 / mu_a)), 8)
                    result = "H" if hg > ag else ("D" if hg == ag else "A")
                    m = random.randint(8, 17)
                    all_matches.append({
                        "home_team": home,  "away_team": away,
                        "home_goals": hg,   "away_goals": ag,
                        "home_xg": round(mu_h + random.gauss(0, 0.15), 2),
                        "away_xg": round(mu_a + random.gauss(0, 0.15), 2),
                        "home_shots": int(mu_h * 5), "away_shots": int(mu_a * 5),
                        "result":  result,
                        "odds_home": max(1.10, round(
                            1 / (ha / (ha + aa) + 0.05) + random.uniform(-0.1, 0.1), 2)),
                        "odds_draw": max(1.10, round(3.2 + random.uniform(-0.3, 0.3), 2)),
                        "odds_away": max(1.10, round(
                            1 / (aa / (ha + aa) + 0.05) + random.uniform(-0.1, 0.1), 2)),
                        "date":   f"{random.randint(1,28):02d}/{((m-1)%12)+1:02d}/{year + (0 if m <= 12 else 1)}",
                        "league": "EPL", "season": str(year),
                    })

        log_main.info(f"Generated {len(all_matches)} synthetic matches across 3 seasons")
        self.model.fit(all_matches)
        log_main.info(f"Fit stats: {self.model.fit_summary()}")

        self.bet_mgr.clear_bets()
        upcoming_pairs = [(teams[i], teams[i+1]) for i in range(0, min(10, len(teams)-1), 2)]
        all_value_bets = []
        for home, away in upcoming_pairs:
            pred = self.model.predict(home, away)
            ho = round(random.uniform(1.8, 4.5), 2)
            do = round(random.uniform(2.8, 4.2), 2)
            ao = round(random.uniform(1.8, 4.5), 2)
            vbets = self.bet_mgr.evaluate_match(
                prediction=pred,
                odds={"best_home": ho, "best_draw": do, "best_away": ao},
                league="EPL", match_date="demo",
            )
            all_value_bets.extend(vbets)

        self.bet_mgr.collect_value_bets(all_value_bets)
        betslips    = self.bet_mgr.generate_betslips(games_per_slip=3, slip_type="trixie")
        saved_uuids = self.db.save_betslips(betslips) if betslips else []
        log_main.info(f"Saved {len(saved_uuids)} betslips to DB")

        for slip in betslips:
            log_main.info(
                f"  {slip['slip_type']:20s} | "
                f"netEV={slip.get('net_ev','N/A'):.4f} | "
                f"vig={slip.get('compounded_vig','N/A'):.4f} | "
                f"stake=£{slip.get('total_stake', slip.get('stake', 0)):.2f}"
            )

        self.backtester.train_window = 100
        self.backtester.test_window  = 30
        metrics = self.backtester.run_backtest(all_matches)
        self.backtester.print_report(metrics)

        return {
            "mode":              "demo",
            "training_matches":  len(all_matches),
            "value_bets":        all_value_bets,
            "betslips":          betslips,
            "saved_uuids":       saved_uuids,
            "backtest":          metrics,
            "summary":           self.bet_mgr.summary(),
            "model_stats":       self.model.fit_summary(),
        }


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  [BLOCK 13]  CLI ENTRY POINT                                                │
# │              argparse  |  mode dispatch  |  JSON output                    │
# └─────────────────────────────────────────────────────────────────────────────┘

def main():
    parser = argparse.ArgumentParser(
        description="Dixon-Coles Sports Betting Framework v2 — Single File Edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "demo", "update_clv", "clv_report", "settle"],
        default="demo",
        help="Pipeline mode to execute",
    )

    # ── League & Season ───────────────────────────────────────────────────────
    parser.add_argument("--league", default="EPL",
                        choices=["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"])
    parser.add_argument("--season", type=int, default=2024,
                        help="Current season (understat integer, e.g. 2024)")
    parser.add_argument("--cold-start-seasons", nargs="*", type=int,
                        default=[2023, 2022], dest="cold_start_seasons",
                        help="Historical seasons to supplement training (e.g. 2023 2022)")
    parser.add_argument("--seasons", nargs="+", default=["2223", "2324"],
                        help="FDCO season codes for backtest mode")
    parser.add_argument("--csv", default=None,
                        help="Path to a local football-data.co.uk CSV file")

    # ── API & Credentials ─────────────────────────────────────────────────────
    parser.add_argument("--api-key", default="", help="The Odds API key (free tier)")

    # ── Staking Parameters ────────────────────────────────────────────────────
    parser.add_argument("--bankroll",  type=float, default=1000.0)
    parser.add_argument("--ev",        type=float, default=0.05,
                        help="Minimum EV threshold for flagging a bet")
    parser.add_argument("--kelly",     type=float, default=0.22,
                        help="Fractional Kelly multiplier (0–1)")
    parser.add_argument("--half-life", type=float, default=90.0,  dest="half_life",
                        help="Time-decay half-life in days (inf = no decay)")
    parser.add_argument("--xg-blend",  type=float, default=0.0,   dest="xg_blend",
                        help="xG blend fraction 0–1 (0=goals only, 1=xG only)")
    parser.add_argument("--use-xg",    action="store_true", dest="use_xg",
                        help="Enable xG blending in the model")

    # ── Betslip Configuration ─────────────────────────────────────────────────
    parser.add_argument("--slip-size", type=int, default=3,
                        help="Number of legs per betslip")
    parser.add_argument("--slip-type", default="auto",
                        choices=["auto", "single", "trixie", "accumulator"])

    # ── Database & Output ─────────────────────────────────────────────────────
    parser.add_argument("--db",        default="betting_framework.db",
                        help="SQLite database file path")
    parser.add_argument("--no-save",   action="store_true", dest="no_save",
                        help="Do not persist betslips to database")
    parser.add_argument("--output",    default=None,
                        help="Write full results JSON to this file")

    # ── Settle helpers ────────────────────────────────────────────────────────
    parser.add_argument("--slip-uuid", default=None,
                        help="Betslip UUID for settle mode")
    parser.add_argument("--results",   default=None,
                        help='JSON: {match_id: "H"/"D"/"A"} for settle mode')

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument("--verbose", action="store_true",
                        help="Set logging level to DEBUG")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Build framework ───────────────────────────────────────────────────────
    framework = BettingFramework(
        odds_api_key=args.api_key,
        bankroll=args.bankroll,
        ev_threshold=args.ev,
        kelly_fraction=args.kelly,
        half_life_days=float(args.half_life),
        xg_blend=args.xg_blend,
        use_xg=args.use_xg,
        db_path=args.db,
    )

    # ── Mode dispatch ─────────────────────────────────────────────────────────
    results = {}

    if args.mode == "demo":
        results = framework.run_demo()

    elif args.mode == "live":
        if not args.api_key:
            log_main.error("--api-key is required for live mode")
            sys.exit(1)
        results = framework.run_live(
            league=args.league,
            season=args.season,
            cold_start_seasons=args.cold_start_seasons,
            games_per_slip=args.slip_size,
            slip_type=args.slip_type,
            save_to_db=not args.no_save,
        )

    elif args.mode == "backtest":
        results = framework.run_backtest_pipeline(
            league=args.league,
            season_codes=args.seasons,
            local_csv=args.csv,
        )

    elif args.mode == "update_clv":
        if not args.api_key:
            log_main.error("--api-key is required for update_clv mode")
            sys.exit(1)
        results = framework.run_update_clv(league=args.league)

    elif args.mode == "clv_report":
        results = framework.db.clv_report()
        results["profit_summary"] = framework.db.profit_summary()
        log_main.info(json.dumps(results, indent=2))

    elif args.mode == "settle":
        if not args.slip_uuid or not args.results:
            log_main.error("--slip-uuid and --results required for settle mode")
            sys.exit(1)
        results = framework.db.settle_bet(args.slip_uuid, json.loads(args.results))
        log_main.info(f"Settlement: {results}")

    # ── Optional JSON output ──────────────────────────────────────────────────
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log_main.info(f"Results written to {args.output}")

    return results


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Script entry                                                               │
# └─────────────────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":
    main()
