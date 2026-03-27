"""
bet_manager.py  —  BetManager  (v2 — Strict Parlay Math)

Upgrades over v1
────────────────
1.  Compounded vig gating for parlays:
    Before accepting any Trixie or Accumulator combination, the module
    explicitly calculates the compounded bookmaker margin across all legs.
    A parlay is only accepted if:

        combined_EV_net > compounded_vig_cost

    i.e. the arithmetic sum of individual EVs must overcome the multiplicative
    erosion from each bookmaker's overround.

2.  Per-leg vig decomposition:
    Each value bet dict now carries `bookmaker_margin` (the single-market
    overround).  This lets the parlay math be exact rather than estimated.

3.  All existing features preserved:
    Fractional Kelly, 5% hard bankroll cap, single / trixie / accumulator
    slip types, EV threshold pre-filter, logged skip reasons.
"""

import logging
import math
import itertools
from datetime import datetime
from typing import Optional

logger = logging.getLogger("betting_framework.bet_manager")


# ─────────────────────────────────────────────────────────────────────────────
#  Value bet constructor
# ─────────────────────────────────────────────────────────────────────────────

def make_value_bet(
    home_team: str,
    away_team: str,
    selection: str,           # "home" | "draw" | "away"
    model_prob: float,
    decimal_odds: float,
    ev: float,
    kelly_stake: float,
    bookmaker_margin: float = 0.0,   # overround on this market
    league: str = "",
    match_date: str = "",
    match_id: str = "",
) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Parlay math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compounded_vig(legs: list[dict]) -> float:
    """
    Compounded bookmaker margin for a parlay of N legs.

    For each leg the bookmaker takes a fractional margin m_i so the
    "fair" version of the market would pay odds × (1 + m_i).
    The compounded vig across N legs is:

        V_compound = ∏(1 + m_i) − 1

    This represents the total dead-weight cost the parlay faces versus
    a hypothetical zero-margin book.
    """
    product = 1.0
    for leg in legs:
        product *= 1.0 + leg.get("bookmaker_margin", 0.0)
    return product - 1.0


def _parlay_ev_net(legs: list[dict]) -> float:
    """
    Net EV of a parlay after subtracting compounded vig.

    For a parlay to have positive net EV:
        Σ EV_i  >  compounded_vig_cost

    Note: this is a conservative, additive EV aggregation.  The true
    parlay EV is multiplicative, but the additive form provides a
    tractable lower bound that is *stricter* — exactly what we want for
    prudent parlay selection.
    """
    raw_sum_ev   = sum(leg["ev"] for leg in legs)
    compound_vig = _compounded_vig(legs)
    return raw_sum_ev - compound_vig


def _parlay_passes_vig_gate(legs: list[dict], min_net_ev: float = 0.0) -> tuple[bool, str]:
    """
    Return (True, "") if the parlay passes the strict vig gate,
    or (False, reason_string) if it is rejected.

    Gate rule:  net_EV = Σ EV_i − V_compound  must exceed min_net_ev.
    """
    if not legs:
        return False, "Empty leg list"

    margins  = [leg.get("bookmaker_margin", 0.0) for leg in legs]
    raw_ev   = [leg["ev"] for leg in legs]
    compound = _compounded_vig(legs)
    net_ev   = sum(raw_ev) - compound

    detail = (
        f"Legs={len(legs)} | "
        f"ΣEV={sum(raw_ev):.4f} | "
        f"margins={[round(m,3) for m in margins]} | "
        f"compound_vig={compound:.4f} | "
        f"net_EV={net_ev:.4f}"
    )

    if net_ev <= min_net_ev:
        return False, f"Parlay rejected — net EV after vig={net_ev:.4f} ≤ {min_net_ev}. {detail}"

    return True, detail


# ─────────────────────────────────────────────────────────────────────────────
#  BetManager
# ─────────────────────────────────────────────────────────────────────────────

class BetManager:
    """
    Full betting workflow with strict parlay vig gate.

    Workflow:
      1. evaluate_match()      — compare model probs vs bookmaker odds
      2. collect_value_bets()  — accumulate positively-EV bets
      3. generate_betslips()   — package into singles / trixies / accas
                                 (rejecting combos that fail the vig gate)
      4. Fractional Kelly staking with 5% hard bankroll cap
      5. Log every skip with an explicit reason
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.22,
        max_stake_pct: float = 0.05,
        min_odds: float = 1.50,
        max_odds: float = 10.0,
        parlay_min_net_ev: float = 0.0,    # parlay vig gate floor
    ):
        self.bankroll           = bankroll
        self.ev_threshold       = ev_threshold
        self.kelly_fraction     = kelly_fraction
        self.max_stake_pct      = max_stake_pct
        self.min_odds           = min_odds
        self.max_odds           = max_odds
        self.parlay_min_net_ev  = parlay_min_net_ev

        self._value_bets:  list[dict] = []
        self._skipped_log: list[dict] = []

    # ──────────────────────── bet identification ──────────────────────────────

    def evaluate_match(
        self,
        prediction: dict,
        odds: dict,
        league: str = "",
        match_date: str = "",
        match_id: str = "",
    ) -> list[dict]:
        """
        Identify value bets for a single match.

        prediction : output of PredictorModel.predict()
        odds       : dict with keys best_home, best_draw, best_away
                     (and optionally odds_home, odds_draw, odds_away for margin calc)
        """
        if not prediction:
            self._log_skip(odds, "Empty prediction — team not in model")
            return []

        home_team = prediction["home_team"]
        away_team = prediction["away_team"]

        # Compute bookmaker margin on this market
        bm_margin = 0.0
        if all(k in odds for k in ("best_home", "best_draw", "best_away")):
            try:
                from predictor_model import PredictorModel
                bm_margin = PredictorModel.overround(
                    odds["best_home"], odds["best_draw"], odds["best_away"]
                )
            except Exception:
                pass

        found = []
        selections = [
            ("home", prediction["home_win"], odds.get("best_home", 0)),
            ("draw", prediction["draw"],     odds.get("best_draw", 0)),
            ("away", prediction["away_win"], odds.get("best_away", 0)),
        ]

        for selection, model_prob, decimal_odds in selections:
            reason = self._pre_filter(model_prob, decimal_odds, home_team, away_team, selection)
            if reason:
                self._log_skip(
                    {"home_team": home_team, "away_team": away_team, "selection": selection},
                    reason,
                )
                continue

            from predictor_model import PredictorModel
            ev = PredictorModel.calculate_ev(model_prob, decimal_odds)

            if ev < self.ev_threshold:
                self._log_skip(
                    {"home_team": home_team, "away_team": away_team, "selection": selection},
                    f"Edge below threshold: EV={ev:.4f} < {self.ev_threshold}",
                )
                continue

            kelly  = PredictorModel.kelly_fraction(model_prob, decimal_odds, self.kelly_fraction)
            stake  = min(kelly * self.bankroll, self.max_stake_pct * self.bankroll)

            bet = make_value_bet(
                home_team=home_team,
                away_team=away_team,
                selection=selection,
                model_prob=model_prob,
                decimal_odds=decimal_odds,
                ev=ev,
                kelly_stake=stake,
                bookmaker_margin=bm_margin,
                league=league,
                match_date=match_date,
                match_id=match_id,
            )
            found.append(bet)
            logger.info(
                f"✅ VALUE BET: {home_team} vs {away_team} | {selection.upper()} "
                f"@ {decimal_odds:.2f} | EV={ev:.4f} | margin={bm_margin:.3f} | "
                f"Stake=£{stake:.2f}"
            )

        return found

    def _pre_filter(
        self,
        model_prob: float,
        decimal_odds: float,
        home_team: str,
        away_team: str,
        selection: str,
    ) -> Optional[str]:
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

    # ──────────────────────── betslip generation ─────────────────────────────

    def generate_betslips(
        self,
        value_bets: Optional[list[dict]] = None,
        games_per_slip: int = 3,
        slip_type: str = "auto",
        top_n: int = 10,
    ) -> list[dict]:
        """
        Package value bets into betslips with strict vig gating on parlays.

        slip_type: "single" | "trixie" | "accumulator" | "auto"
            auto → trixie if games_per_slip==3, else accumulator

        Trixie and Accumulator combinations are REJECTED if the compounded
        vig cost exceeds the summed individual EVs.
        """
        bets = value_bets if value_bets is not None else self._value_bets
        if not bets:
            logger.warning("No value bets to package into betslips")
            return []

        # Deduplicate: one selection per match
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

        logger.info(
            f"Generating {slip_type} betslips | "
            f"{len(unique)} candidates | {games_per_slip} legs"
        )

        if slip_type == "single":
            return [self._make_single(b) for b in unique]

        elif slip_type == "trixie":
            if len(unique) < 3:
                logger.warning("Need ≥3 bets for Trixie — reverting to singles")
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
                logger.info(
                    f"Trixie vig gate: {rejected} combos rejected | "
                    f"{len(slips)} accepted"
                )
            return slips

        elif slip_type == "accumulator":
            selected = unique[:games_per_slip]
            if len(selected) < games_per_slip:
                logger.warning(
                    f"Only {len(selected)} bets available "
                    f"(needed {games_per_slip}) — using available"
                )
            if not selected:
                return []

            ok, detail = _parlay_passes_vig_gate(selected, self.parlay_min_net_ev)
            if not ok:
                logger.warning(f"Accumulator rejected by vig gate: {detail}")
                self._log_skip(
                    {"home_team": "Acca", "away_team": "", "selection": "combo"},
                    detail,
                )
                # Degrade gracefully to singles
                logger.info("Falling back to singles for this round")
                return [self._make_single(b) for b in selected]

            return [self._make_accumulator(selected)]

        else:
            raise ValueError(f"Unknown slip_type: '{slip_type}'")

    # ──────────────────────── slip constructors ───────────────────────────────

    def _make_single(self, bet: dict) -> dict:
        stake = self._cap_stake(bet["kelly_stake"])
        net_ev = bet["ev"] - bet.get("bookmaker_margin", 0.0)
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
        Trixie = 3 doubles (C(3,2)=3) + 1 treble = 4 unit bets.
        Stake per unit = kelly/4, so total outlay = kelly.
        """
        assert len(legs) == 3
        base_stake = self._cap_stake(min(b["kelly_stake"] for b in legs) / 4)
        vig  = _compounded_vig(legs)
        nev  = _parlay_ev_net(legs)

        components = []
        total_potential = 0.0

        for a, b in itertools.combinations(legs, 2):
            odds  = a["decimal_odds"] * b["decimal_odds"]
            pot   = base_stake * odds
            components.append({
                "type":          "Double",
                "legs":          [f"{a['home_team']} ({a['selection']})",
                                  f"{b['home_team']} ({b['selection']})"],
                "combined_odds": round(odds, 3),
                "potential":     round(pot, 2),
            })
            total_potential += pot

        treble_odds = math.prod(b["decimal_odds"] for b in legs)
        pot = base_stake * treble_odds
        components.append({
            "type":          "Treble",
            "legs":          [f"{b['home_team']} ({b['selection']})" for b in legs],
            "combined_odds": round(treble_odds, 3),
            "potential":     round(pot, 2),
        })
        total_potential += pot

        return {
            "slip_type":         "Trixie",
            "legs":              legs,
            "components":        components,
            "stake_per_unit":    round(base_stake, 2),
            "total_stake":       round(base_stake * 4, 2),
            "combined_odds":     round(treble_odds, 3),
            "potential_return":  round(total_potential, 2),
            "combined_ev":       round(sum(b["ev"] for b in legs) / 3, 4),
            "net_ev":            round(nev, 4),
            "compounded_vig":    round(vig, 4),
            "vig_passed":        True,
        }

    def _make_accumulator(self, legs: list[dict]) -> dict:
        combined_odds = math.prod(b["decimal_odds"] for b in legs)
        vig  = _compounded_vig(legs)
        nev  = _parlay_ev_net(legs)

        # Stake discount: more legs = more compounded variance
        # Use: stake = kelly_min × 0.5^(N-1) as variance discount
        variance_discount = 0.5 ** (len(legs) - 1)
        base_stake = self._cap_stake(
            min(b["kelly_stake"] for b in legs) * variance_discount
        )

        return {
            "slip_type":         f"{len(legs)}-Fold Accumulator",
            "legs":              legs,
            "stake":             round(base_stake, 2),
            "combined_odds":     round(combined_odds, 3),
            "potential_return":  round(base_stake * combined_odds, 2),
            "combined_ev":       round(sum(b["ev"] for b in legs) / len(legs), 4),
            "net_ev":            round(nev, 4),
            "compounded_vig":    round(vig, 4),
            "vig_passed":        True,
        }

    def _cap_stake(self, kelly_stake: float) -> float:
        return min(float(kelly_stake), self.max_stake_pct * self.bankroll)

    # ──────────────────────── logging & reporting ─────────────────────────────

    def _log_skip(self, bet_info: dict, reason: str):
        entry = {
            "match":     f"{bet_info.get('home_team','')} vs {bet_info.get('away_team','')}",
            "selection": bet_info.get("selection", ""),
            "reason":    reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._skipped_log.append(entry)
        logger.debug(f"⏭  SKIP — {entry['match']} [{entry['selection']}]: {reason}")

    def get_skip_log(self) -> list[dict]:
        return list(self._skipped_log)

    def get_value_bets(self) -> list[dict]:
        return list(self._value_bets)

    def summary(self) -> dict:
        total_exp = sum(b["kelly_stake"] for b in self._value_bets)
        avg_ev    = (sum(b["ev"] for b in self._value_bets) / len(self._value_bets)
                     if self._value_bets else 0.0)
        avg_margin = (sum(b.get("bookmaker_margin", 0) for b in self._value_bets)
                      / len(self._value_bets) if self._value_bets else 0.0)
        return {
            "value_bets_found":   len(self._value_bets),
            "bets_skipped":       len(self._skipped_log),
            "bankroll":           self.bankroll,
            "total_exposure":     round(total_exp, 2),
            "avg_ev":             round(avg_ev, 4),
            "avg_bookie_margin":  round(avg_margin, 4),
        }
