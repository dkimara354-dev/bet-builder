"""
Backtester — Simulate the Dixon-Coles framework over historical seasons.
Calculates ROI, Win Rate, Maximum Drawdown, Sharpe-like ratio, and per-league breakdown.
"""

import logging
import math
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("betting_framework.backtester")


class Backtester:
    """
    Walk-forward backtesting engine.

    Strategy:
      - Split historical data into rolling windows
      - Fit Dixon-Coles on training window
      - Evaluate bets on test window using bookmaker odds from CSV
      - Track bankroll evolution, calculate performance metrics
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.22,
        max_stake_pct: float = 0.05,
        min_odds: float = 1.50,
        max_odds: float = 10.0,
        train_window: int = 200,   # matches to train on
        test_window: int = 50,     # matches to test before refit
    ):
        self.initial_bankroll = initial_bankroll
        self.ev_threshold     = ev_threshold
        self.kelly_fraction   = kelly_fraction
        self.max_stake_pct    = max_stake_pct
        self.min_odds         = min_odds
        self.max_odds         = max_odds
        self.train_window     = train_window
        self.test_window      = test_window

        self.results: list[dict] = []
        self.bankroll_curve: list[float] = []

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run_backtest(self, historical_data: list[dict]) -> dict:
        """
        Simulate the full betting algorithm over historical_data.

        historical_data: list of match dicts from DataIngestor.fetch_historical_csv()
          Required keys: home_team, away_team, home_goals, away_goals,
                         result (H/D/A), odds_home, odds_draw, odds_away

        Returns performance dict with ROI, win_rate, max_drawdown, etc.
        """
        from predictor_model import PredictorModel

        if len(historical_data) < self.train_window + self.test_window:
            raise ValueError(
                f"Need ≥{self.train_window + self.test_window} matches for backtesting. "
                f"Got {len(historical_data)}."
            )

        # Sort by date if available
        data = self._sort_by_date(historical_data)

        bankroll = self.initial_bankroll
        self.results = []
        self.bankroll_curve = [bankroll]
        bets_placed = 0
        bets_won = 0
        total_staked = 0.0
        total_profit = 0.0

        # Walk-forward loop
        cursor = self.train_window
        refit_counter = 0

        model = PredictorModel(half_life_days=90.0)

        logger.info(
            f"Starting walk-forward backtest | "
            f"Matches={len(data)} | Train={self.train_window} | Test={self.test_window}"
        )

        while cursor < len(data):
            # Refit model every test_window matches
            if refit_counter == 0:
                train_data = data[max(0, cursor - self.train_window): cursor]
                try:
                    # Anchor time-decay to the last date in the training window,
                    # not to wall-clock now() — prevents look-ahead in backtest.
                    ref_date = Backtester._latest_date(train_data)
                    model.fit(train_data, reference_date=ref_date)
                    logger.debug(f"Model refitted at cursor={cursor} | ref_date={ref_date}")
                except Exception as e:
                    logger.warning(f"Model fit failed at cursor={cursor}: {e}")
                    cursor += self.test_window
                    continue

            test_batch = data[cursor: cursor + self.test_window]

            for match in test_batch:
                if bankroll <= 0:
                    logger.warning("Bankroll exhausted — stopping backtest")
                    break

                try:
                    bet_results = self._evaluate_historical_match(match, model, bankroll)
                    for br in bet_results:
                        bankroll += br["profit"]
                        total_staked += br["stake"]
                        total_profit += br["profit"]
                        bets_placed += 1
                        if br["won"]:
                            bets_won += 1
                        self.results.append(br)
                        self.bankroll_curve.append(bankroll)

                except Exception as e:
                    logger.debug(f"Error evaluating match {match.get('home_team')} vs "
                                 f"{match.get('away_team')}: {e}")
                    continue

            cursor += self.test_window
            refit_counter = (refit_counter + 1) % 3  # refit every 3 windows

        metrics = self._calculate_metrics(
            bets_placed, bets_won, total_staked, total_profit, bankroll
        )
        logger.info(
            f"Backtest complete | Bets={bets_placed} | ROI={metrics['roi_pct']:.2f}% | "
            f"Win Rate={metrics['win_rate_pct']:.1f}% | "
            f"Max Drawdown={metrics['max_drawdown_pct']:.1f}%"
        )
        return metrics

    # ------------------------------------------------------------------ #
    #  Internal evaluation                                                 #
    # ------------------------------------------------------------------ #

    def _evaluate_historical_match(
        self, match: dict, model, bankroll: float
    ) -> list[dict]:
        """Apply the betting strategy to a single historical match."""
        from predictor_model import PredictorModel

        home = match["home_team"]
        away = match["away_team"]
        actual_result = match.get("result", "")  # H / D / A

        prediction = model.predict(home, away)
        if not prediction:
            return []

        placed = []
        selections = [
            ("home", prediction["home_win"], match.get("odds_home", 0), "H"),
            ("draw", prediction["draw"],     match.get("odds_draw", 0), "D"),
            ("away", prediction["away_win"], match.get("odds_away", 0), "A"),
        ]

        for selection, model_prob, decimal_odds, result_code in selections:
            if decimal_odds < self.min_odds or decimal_odds > self.max_odds:
                continue
            if model_prob <= 0:
                continue

            ev = PredictorModel.calculate_ev(model_prob, decimal_odds)
            if ev < self.ev_threshold:
                continue

            kelly = PredictorModel.kelly_fraction(model_prob, decimal_odds, self.kelly_fraction)
            stake = min(kelly * bankroll, self.max_stake_pct * bankroll)
            stake = max(stake, 0.01)

            won = (actual_result == result_code)
            profit = (stake * (decimal_odds - 1)) if won else -stake

            placed.append({
                "home_team":    home,
                "away_team":    away,
                "selection":    selection,
                "model_prob":   round(model_prob, 4),
                "decimal_odds": round(decimal_odds, 3),
                "ev":           round(ev, 4),
                "stake":        round(stake, 2),
                "profit":       round(profit, 2),
                "won":          won,
                "actual_result": actual_result,
                "date":         match.get("date", ""),
                "league":       match.get("league", ""),
            })

        return placed

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def _calculate_metrics(
        self,
        bets_placed: int,
        bets_won: int,
        total_staked: float,
        total_profit: float,
        final_bankroll: float,
    ) -> dict:
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0.0
        max_dd = self._max_drawdown()
        sharpe = self._sharpe_ratio()
        avg_ev = (
            sum(r["ev"] for r in self.results) / len(self.results)
            if self.results else 0.0
        )

        # Per-league breakdown
        by_league: dict = defaultdict(lambda: {"bets": 0, "profit": 0.0, "staked": 0.0, "wins": 0})
        for r in self.results:
            lg = r.get("league", "unknown")
            by_league[lg]["bets"] += 1
            by_league[lg]["profit"] += r["profit"]
            by_league[lg]["staked"] += r["stake"]
            by_league[lg]["wins"] += int(r["won"])

        league_breakdown = {}
        for lg, stats in by_league.items():
            league_breakdown[lg] = {
                "bets":      stats["bets"],
                "wins":      stats["wins"],
                "roi_pct":   round(stats["profit"] / stats["staked"] * 100
                                   if stats["staked"] > 0 else 0, 2),
                "win_rate":  round(stats["wins"] / stats["bets"] * 100
                                   if stats["bets"] > 0 else 0, 1),
            }

        return {
            "bets_placed":       bets_placed,
            "bets_won":          bets_won,
            "total_staked":      round(total_staked, 2),
            "total_profit":      round(total_profit, 2),
            "roi_pct":           round(roi, 2),
            "win_rate_pct":      round(win_rate, 1),
            "max_drawdown_pct":  round(max_dd, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "avg_ev":            round(avg_ev, 4),
            "final_bankroll":    round(final_bankroll, 2),
            "initial_bankroll":  self.initial_bankroll,
            "bankroll_growth_pct": round(
                (final_bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 2
            ),
            "league_breakdown":  league_breakdown,
            "bankroll_curve":    self.bankroll_curve,
        }

    def _max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown as a percentage of peak bankroll."""
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
        """
        Simplified Sharpe ratio based on per-bet profit/stake returns.
        Uses profit-per-unit-staked as the return series.
        """
        if not self.results:
            return 0.0
        returns = [r["profit"] / r["stake"] for r in self.results if r["stake"] > 0]
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 1e-10
        return (mean_r - risk_free) / std_r

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _latest_date(data: list[dict]):
        """
        Return the datetime of the most recent match in a list.
        Used to anchor time-decay reference_date in walk-forward backtest.
        Falls back to datetime.utcnow() if no dates are parseable.
        """
        from datetime import datetime
        DATE_FMTS = ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y", "%Y/%m/%d",
                     "%m/%d/%y", "%Y-%m-%dT%H:%M:%S"]
        best = None
        for m in data:
            raw = str(m.get("date", "") or "").split(" ")[0].split("T")[0].strip()
            for fmt in DATE_FMTS:
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
        """Sort match list by date string (best-effort — leaves unsorted if parse fails)."""
        def date_key(m):
            d = m.get("date", "")
            for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y", "%Y/%m/%d"]:
                try:
                    from datetime import datetime
                    return datetime.strptime(d, fmt)
                except ValueError:
                    pass
            return d  # fall back to string sort

        try:
            return sorted(data, key=date_key)
        except Exception:
            return data

    def print_report(self, metrics: dict):
        """Pretty-print backtest report to logger."""
        sep = "=" * 60
        logger.info(sep)
        logger.info("BACKTEST REPORT")
        logger.info(sep)
        logger.info(f"  Bets placed:        {metrics['bets_placed']}")
        logger.info(f"  Win rate:           {metrics['win_rate_pct']:.1f}%")
        logger.info(f"  ROI:                {metrics['roi_pct']:.2f}%")
        logger.info(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.1f}%")
        logger.info(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Avg EV per bet:     {metrics['avg_ev']:.4f}")
        logger.info(f"  Total profit:       £{metrics['total_profit']:.2f}")
        logger.info(f"  Final bankroll:     £{metrics['final_bankroll']:.2f} "
                    f"({metrics['bankroll_growth_pct']:+.2f}%)")
        logger.info("")
        logger.info("  League Breakdown:")
        for lg, stats in metrics.get("league_breakdown", {}).items():
            logger.info(
                f"    {lg:15s}  Bets={stats['bets']:4d}  "
                f"WR={stats['win_rate']:5.1f}%  ROI={stats['roi_pct']:+6.2f}%"
            )
        logger.info(sep)
