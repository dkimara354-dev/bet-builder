"""
predictor_model.py  —  Dixon-Coles Poisson Model  (v2 Production MLE)

Upgrades over v1
────────────────
1.  Joint MLE via scipy.optimize.minimize (L-BFGS-B):
    attack[], defence[], home_advantage, AND rho are estimated simultaneously
    in a single optimisation pass — no grid search, no sequential loops.

2.  Time-decay weighting:
    Each match is weighted by  w = exp(−λ · age_days),  λ = ln(2)/half_life.
    A match at exactly half_life_days old gets w = 0.5.  Weights are
    normalised so the total ≈ number of matches (preserves LL scale).

3.  xG blending:
    If use_xg=True, the expected goals μ used in the likelihood are:
        μ_blended = (1−α)·μ_model + α·xG_observed,   α = xg_blend ∈ [0,1]
    Defaults to 0 (pure goals model).

4.  Identifiability:
    The sum-to-zero constraint on attack is enforced via a soft quadratic
    penalty so the attack/defence parameters do not drift arbitrarily.

5.  Approximate standard errors:
    Finite-difference diagonal Hessian gives approximate SEs for
    home_advantage and rho after optimisation.

6.  Graceful fallback:
    If scipy is unavailable or optimisation fails, the original coordinate-
    descent engine is used with a WARNING logged.
"""

import math
import logging
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger("betting_framework.predictor")


# ─────────────────────────────────────────────────────────────────────────────
#  Dixon-Coles primitives
# ─────────────────────────────────────────────────────────────────────────────

def _tau(hg: int, ag: int, mu_h: float, mu_a: float, rho: float) -> float:
    """τ correction — adjusts the four low-score cells to fix draw-bias."""
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
    """Numerically stable Poisson PMF via log-space computation."""
    if lam <= 0.0 or k < 0:
        return 0.0
    try:
        log_p = -lam + k * math.log(lam) - math.lgamma(k + 1)
        return math.exp(log_p)
    except (OverflowError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Time-decay weights
# ─────────────────────────────────────────────────────────────────────────────

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
    Exponential time-decay weights.  Match at age d gets weight exp(−λd).
    If half_life_days is inf, all weights = 1.
    Weights are normalised so sum = len(matches) (preserves LL magnitude).
    """
    if math.isinf(half_life_days):
        return np.ones(len(matches), dtype=np.float64)

    ref = reference_date or datetime.utcnow()
    lam = math.log(2.0) / half_life_days
    weights = []

    for m in matches:
        raw = str(m.get("date", "") or m.get("datetime", "")).split(" ")[0].split("T")[0].strip()
        age = 180.0  # default fallback age (roughly mid-season)
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
        arr *= len(arr) / total       # normalise
    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  Joint Dixon-Coles Objective  (scipy-compatible callable)
# ─────────────────────────────────────────────────────────────────────────────

class _DCObjective:
    """
    Negative weighted log-likelihood for the full Dixon-Coles model.

    Parameter vector x layout:
        x[0 : N]          attack[0..N-1]        (log-scale)
        x[N : 2N]         defence[0..N-1]       (log-scale)
        x[2N]             home_advantage         (log-scale, clipped ≥ 0)
        x[2N + 1]         rho                    (real, clipped ∈ [−0.4, 0.4])

    Regularisation:
        L2 on attack + defence  →  prevents blow-up on sparse data
        Sum-to-zero penalty on attack  →  identifiability
    """

    RHO_MIN, RHO_MAX   = -0.40,  0.40
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

        # Precompute log-factorials for goal counts 0..15
        self._lgam = np.array([math.lgamma(k + 1) for k in range(16)], dtype=np.float64)

    def _lgam_safe(self, k_arr: np.ndarray) -> np.ndarray:
        return np.array([
            self._lgam[k] if k < len(self._lgam) else math.lgamma(k + 1)
            for k in k_arr
        ])

    def __call__(self, x: np.ndarray) -> float:
        N = self.N
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

        # Poisson log-likelihoods
        mu_h_c = np.maximum(mu_h, 1e-10)
        mu_a_c = np.maximum(mu_a, 1e-10)
        log_ph = -mu_h_c + self.hg * np.log(mu_h_c) - self._lgam_safe(self.hg)
        log_pa = -mu_a_c + self.ag * np.log(mu_a_c) - self._lgam_safe(self.ag)

        # Dixon-Coles τ correction (loop — small cost, exact)
        log_tau = np.empty(len(self.hg), dtype=np.float64)
        for i in range(len(self.hg)):
            tv = _tau(self.hg[i], self.ag[i], mu_h[i], mu_a[i], rho)
            log_tau[i] = math.log(max(abs(tv), 1e-12))

        nll = -float(np.dot(self.w, log_ph + log_pa + log_tau))

        # L2 regularisation
        nll += self.L2_LAMBDA * (float(np.dot(atk, atk)) + float(np.dot(dfc, dfc)))
        # Identifiability: sum(attack) == 0
        nll += self.ID_PENALTY * float(np.sum(atk)) ** 2

        return nll


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback coordinate descent (no scipy required)
# ─────────────────────────────────────────────────────────────────────────────

def _coord_descent(fn, x0: np.ndarray, max_iter: int = 400, tol: float = 1e-7) -> np.ndarray:
    x = x0.copy()
    best_f = fn(x)
    step = 0.05
    for _ in range(max_iter):
        improved = False
        for i in range(len(x)):
            for sign in (1.0, -1.0):
                xt = x.copy(); xt[i] += sign * step
                ft = fn(xt)
                if ft < best_f - tol:
                    x, best_f, improved = xt, ft, True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  PredictorModel
# ─────────────────────────────────────────────────────────────────────────────

class PredictorModel:
    """
    Dixon-Coles Poisson model — Production MLE Edition.

    All four parameters (attack, defence, home_advantage, rho) are estimated
    simultaneously by maximising the weighted Dixon-Coles log-likelihood using
    scipy's L-BFGS-B solver.  A time-decay kernel gives more recent matches
    exponentially higher influence on the parameter estimates.
    """

    MIN_MATCHES_PER_TEAM = 5
    GOALS_CAP = 10

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
            Exponential decay half-life in days.  float('inf') disables decay.
        xg_blend : float [0, 1]
            Blend fraction of xG into μ estimate (0 = goals only, 1 = xG only).
        use_xg : bool
            Activate xG blending — requires 'home_xg'/'away_xg' in match dicts.
        l2_lambda : float
            L2 regularisation strength.  Increase for small / early-season data.
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

    # ──────────────────────────────── fit ────────────────────────────────────

    def fit(
        self,
        matches: list[dict],
        reference_date: Optional[datetime] = None,
    ) -> "PredictorModel":
        """
        Fit the model via joint MLE on all parameters simultaneously.

        matches : list[dict]
            Required keys: home_team, away_team, home_goals, away_goals, date
            Optional:      home_xg, away_xg  (needed when use_xg=True)

        reference_date : datetime, optional
            Anchor for time-decay age computation.  Defaults to utcnow().

        Returns self  (chainable: model.fit(matches).predict(h, a))
        """
        if not matches:
            raise ValueError("No matches provided.")

        self._ref_date = reference_date or datetime.utcnow()

        # ── Team census & filtering ──────────────────────────────────────────
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
        N = len(self.teams)
        tidx = {t: i for i, t in enumerate(self.teams)}

        valid = [
            m for m in matches
            if m["home_team"] in tidx and m["away_team"] in tidx
        ]
        if not valid:
            raise ValueError("No matches remain after filtering valid teams.")

        logger.info(
            f"Fitting Dixon-Coles (joint MLE) | "
            f"{len(valid)} matches | {N} teams | "
            f"half-life={self.half_life_days}d | "
            f"xG-blend={self.xg_blend} | L2={self.l2_lambda}"
        )

        # ── Numpy arrays ─────────────────────────────────────────────────────
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

        # ── Initial parameter vector ──────────────────────────────────────────
        # Warm-start: league average log-goals as attack init
        avg_goals = max(float(np.mean(np.concatenate([hg, ag]))), 0.5)
        mean_log  = math.log(avg_goals) * 0.5
        x0 = np.zeros(2 * N + 2)
        x0[:N]      = mean_log    # attack
        x0[N:2*N]   = 0.0         # defence
        x0[2*N]     = 0.25        # home_advantage (ln-scale ≈ 1.28×)
        x0[2*N + 1] = -0.10       # rho

        # ── scipy L-BFGS-B ───────────────────────────────────────────────────
        atk_bds  = [(-3.0, 3.0)] * N
        dfc_bds  = [(-3.0, 3.0)] * N
        hadv_bds = [(0.00, 1.00)]
        rho_bds  = [(-0.40, 0.40)]
        bounds   = atk_bds + dfc_bds + hadv_bds + rho_bds

        opt_x   = None
        opt_nll = float("inf")

        try:
            from scipy.optimize import minimize as sp_min

            res = sp_min(
                fun=obj, x0=x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-8},
            )
            if res.success or res.fun < obj(x0):
                opt_x, opt_nll = res.x, float(res.fun)
                logger.info(
                    f"scipy L-BFGS-B: {res.nit} iters | "
                    f"-LL={opt_nll:.4f} | {res.message}"
                )
            else:
                logger.warning(f"scipy did not fully converge: {res.message}")
                # Still use result if it's better than x0
                if res.fun < obj(x0):
                    opt_x, opt_nll = res.x, float(res.fun)

        except ImportError:
            logger.warning("scipy not available — using coordinate descent fallback")
        except Exception as exc:
            logger.warning(f"scipy failed ({exc}) — using coordinate descent fallback")

        # ── Fallback ──────────────────────────────────────────────────────────
        if opt_x is None:
            logger.info("Running coordinate-descent optimiser...")
            opt_x   = _coord_descent(obj, x0)
            opt_nll = float(obj(opt_x))
            logger.info(f"Coord-descent: -LL={opt_nll:.4f}")

        # ── Store parameters ──────────────────────────────────────────────────
        for i, team in enumerate(self.teams):
            self.attack[team]  = float(opt_x[i])
            self.defence[team] = float(opt_x[N + i])
        self.home_advantage = float(np.clip(opt_x[2*N],     0.00, 1.00))
        self.fitted_rho     = float(np.clip(opt_x[2*N + 1], -0.40, 0.40))
        self._fit_nll       = opt_nll
        self.is_fitted      = True

        # ── Approx standard errors (finite-difference Hessian diagonal) ───────
        ha_se = rho_se = None
        try:
            eps = 1e-4
            f0  = opt_nll
            for idx_p, name in [(2*N, "home_advantage"), (2*N+1, "rho")]:
                xp, xm = opt_x.copy(), opt_x.copy()
                xp[idx_p] += eps; xm[idx_p] -= eps
                h_ii = (obj(xp) - 2*f0 + obj(xm)) / eps**2
                se   = 1.0 / math.sqrt(max(h_ii, 1e-12))
                if name == "home_advantage":
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
            "home_advantage_se": round(ha_se, 4) if ha_se else None,
            "rho":               round(self.fitted_rho, 4),
            "rho_se":            round(rho_se, 4) if rho_se else None,
            "half_life_days":    self.half_life_days,
            "xg_blend":          self.xg_blend,
        }

        ha_str  = f"Home adv={math.exp(self.home_advantage):.3f}x"
        ha_str += f" (±{ha_se:.3f})" if ha_se else ""
        rho_str  = f"rho={self.fitted_rho:.4f}"
        rho_str += f" (±{rho_se:.4f})" if rho_se else ""
        logger.info(f"Model fitted ✓ | {ha_str} | {rho_str}")

        return self

    # ──────────────────────────────── predict ─────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10,
    ) -> dict:
        """
        Dixon-Coles score matrix + 1X2 outcome probabilities.

        Returns {} with logged reason when a team is unknown or data-sparse.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

        for team, label in [(home_team, "Home"), (away_team, "Away")]:
            if team not in self.teams:
                logger.warning(
                    f"Skipping prediction | {label}='{team}' not in model. "
                    "Reason: Insufficient xG Data or team absent from training set."
                )
                return {}
            cnt = self._match_counts.get(team, 0)
            if cnt < self.MIN_MATCHES_PER_TEAM:
                logger.warning(
                    f"Insufficient xG Data for '{team}' "
                    f"({cnt} matches < {self.MIN_MATCHES_PER_TEAM} threshold)"
                )
                return {}

        mu_h = math.exp(self.attack[home_team] + self.defence[away_team] + self.home_advantage)
        mu_a = math.exp(self.attack[away_team] + self.defence[home_team])

        # Score-probability matrix with τ correction
        matrix: list[list[float]] = []
        for hg in range(max_goals + 1):
            row = []
            for ag in range(max_goals + 1):
                p = (_poisson_pmf(mu_h, hg) * _poisson_pmf(mu_a, ag)
                     * _tau(hg, ag, mu_h, mu_a, self.fitted_rho))
                row.append(max(float(p), 0.0))
            matrix.append(row)

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

    # ──────────────────────────── static helpers ──────────────────────────────

    @staticmethod
    def get_implied_prob(decimal_odds: float) -> float:
        return 1.0 / decimal_odds if decimal_odds > 1.0 else 1.0

    @staticmethod
    def remove_vig(
        home_odd: float, draw_odd: float, away_odd: float
    ) -> dict[str, float]:
        """Proportional vig removal → fair implied probabilities."""
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
        """EV = p·(b−1) − (1−p)  where b = decimal odds."""
        if decimal_odds <= 1.0 or model_prob <= 0.0:
            return -1.0
        return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)

    @staticmethod
    def kelly_fraction(
        model_prob: float, decimal_odds: float, fraction: float = 0.22
    ) -> float:
        """Fractional Kelly stake as a fraction of bankroll.  0 if no edge."""
        b = decimal_odds - 1.0
        if b <= 0.0 or model_prob <= 0.0:
            return 0.0
        full_k = (b * model_prob - (1.0 - model_prob)) / b
        return max(0.0, full_k * fraction)

    # ──────────────────────────── diagnostics ─────────────────────────────────

    def top_teams(self, n: int = 5) -> list[dict]:
        """Top-N teams by attack rating (descending)."""
        if not self.is_fitted:
            return []
        ranked = sorted(self.teams, key=lambda t: self.attack.get(t, 0), reverse=True)
        return [
            {"team": t,
             "attack": round(self.attack[t], 4),
             "defence": round(self.defence[t], 4)}
            for t in ranked[:n]
        ]

    def fit_summary(self) -> dict:
        return dict(self._fit_stats)
