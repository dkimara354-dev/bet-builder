"""
betting_framework — Main Orchestrator
Wires DataIngestor → PredictorModel → BetManager → Backtester into a single pipeline.

Usage:
    python main.py --mode live   --league EPL --api-key YOUR_KEY --bankroll 1000
    python main.py --mode backtest --league EPL --season 2223
    python main.py --mode demo    # uses synthetic data, no API key required
"""

import logging
import argparse
import json
import sys
import math
import random
from datetime import datetime

# ------------------------------------------------------------------ #
#  Logging setup                                                       #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("betting_framework.log", mode="a"),
    ],
)
logger = logging.getLogger("betting_framework.main")


# ------------------------------------------------------------------ #
#  Import modules                                                      #
# ------------------------------------------------------------------ #

from data_ingestor   import DataIngestor
from predictor_model import PredictorModel
from bet_manager     import BetManager
from backtester      import Backtester


# ------------------------------------------------------------------ #
#  Pipeline                                                            #
# ------------------------------------------------------------------ #

class BettingFramework:
    """
    Top-level orchestrator that connects all four modules.
    """

    def __init__(
        self,
        odds_api_key: str = "",
        bankroll: float = 1000.0,
        ev_threshold: float = 0.05,
        kelly_fraction: float = 0.22,
        max_stake_pct: float = 0.05,
    ):
        self.ingestor   = DataIngestor(odds_api_key=odds_api_key)
        self.model      = PredictorModel(rho=-0.13)
        self.bet_mgr    = BetManager(
            bankroll=bankroll,
            ev_threshold=ev_threshold,
            kelly_fraction=kelly_fraction,
            max_stake_pct=max_stake_pct,
        )
        self.backtester = Backtester(
            initial_bankroll=bankroll,
            ev_threshold=ev_threshold,
            kelly_fraction=kelly_fraction,
            max_stake_pct=max_stake_pct,
        )

    # -------- Live pipeline -------- #

    def run_live(
        self,
        league: str = "EPL",
        season: int = 2024,
        games_per_slip: int = 3,
        slip_type: str = "auto",
    ) -> dict:
        """Full live pipeline: fetch xG → fetch odds → predict → find value → generate slips."""
        logger.info(f"{'='*60}")
        logger.info(f"LIVE PIPELINE: {league} | Season {season}")
        logger.info(f"{'='*60}")

        # 1. Fetch xG training data from understat
        logger.info("Step 1/4 — Fetching xG data from understat.com...")
        xg_matches = self.ingestor.fetch_understat_xg(league, season)
        if len(xg_matches) < 30:
            logger.warning(f"Only {len(xg_matches)} matches with xG data — model may be weak")

        # 2. Fit model
        logger.info(f"Step 2/4 — Fitting Dixon-Coles model on {len(xg_matches)} matches...")
        try:
            self.model.fit(xg_matches)
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            return {"error": str(e)}

        # 3. Fetch live odds
        logger.info("Step 3/4 — Fetching live odds from The Odds API...")
        upcoming = self.ingestor.fetch_odds(league)
        if not upcoming:
            logger.warning("No odds data — check API key or league name")
            return {"error": "No odds data available"}

        # 4. Evaluate each match
        logger.info(f"Step 4/4 — Evaluating {len(upcoming)} upcoming matches...")
        self.bet_mgr.clear_bets()
        all_value_bets = []

        for event in upcoming:
            prediction = self.model.predict(
                event["home_team"], event["away_team"]
            )
            value_bets = self.bet_mgr.evaluate_match(
                prediction=prediction,
                odds=event,
                league=league,
                match_date=event.get("commence", ""),
                match_id=event.get("event_id", ""),
            )
            all_value_bets.extend(value_bets)

        self.bet_mgr.collect_value_bets(all_value_bets)

        # 5. Generate betslips
        betslips = self.bet_mgr.generate_betslips(
            games_per_slip=games_per_slip,
            slip_type=slip_type,
        )

        summary = self.bet_mgr.summary()
        logger.info(f"\nSUMMARY: {summary}")

        return {
            "league":           league,
            "season":           season,
            "matches_analysed": len(upcoming),
            "value_bets":       all_value_bets,
            "betslips":         betslips,
            "skipped":          self.bet_mgr.get_skip_log(),
            "summary":          summary,
        }

    # -------- Backtest pipeline -------- #

    def run_backtest_pipeline(
        self,
        league: str = "EPL",
        season_codes: list[str] = None,
        local_csv: str = None,
    ) -> dict:
        """Backtest pipeline: ingest historical CSVs → walk-forward simulation."""
        logger.info(f"{'='*60}")
        logger.info(f"BACKTEST PIPELINE: {league}")
        logger.info(f"{'='*60}")

        all_matches = []

        if local_csv:
            logger.info(f"Loading local CSV: {local_csv}")
            all_matches = self.ingestor.ingest_local_csv(local_csv, league)
        else:
            season_codes = season_codes or ["2223", "2324"]
            for sc in season_codes:
                logger.info(f"Fetching historical CSV: {league} {sc}")
                matches = self.ingestor.fetch_historical_csv(league, sc)
                all_matches.extend(matches)

        logger.info(f"Total historical matches: {len(all_matches)}")
        if not all_matches:
            return {"error": "No historical data loaded"}

        metrics = self.backtester.run_backtest(all_matches)
        self.backtester.print_report(metrics)
        return metrics

    # -------- Demo pipeline (no API key needed) -------- #

    def run_demo(self) -> dict:
        """Generate synthetic data to demonstrate full pipeline without external APIs."""
        logger.info("DEMO MODE — generating synthetic Premier League data")

        teams = [
            "Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd",
            "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
            "Brentford", "Fulham", "Wolves", "Everton", "Crystal Palace",
            "Nottm Forest", "Bournemouth", "Leicester", "Luton", "Sheffield Utd",
        ]

        # Assign synthetic attack/defence strengths
        random.seed(42)
        strengths = {t: (random.uniform(0.8, 2.0), random.uniform(0.8, 2.0)) for t in teams}

        # Generate synthetic match results
        matches = []
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i == j:
                    continue
                ha, hd = strengths[home]
                aa, ad = strengths[away]
                mu_h = ha / ad * 1.25  # home advantage
                mu_a = aa / hd

                hg = min(int(random.expovariate(1/mu_h)), 8)
                ag = min(int(random.expovariate(1/mu_a)), 8)
                result = "H" if hg > ag else ("D" if hg == ag else "A")

                # Synthetic bookmaker odds
                total_goals = mu_h + mu_a
                odds_h = round(1 / (ha / (ha + aa) + 0.05) + random.uniform(-0.1, 0.1), 2)
                odds_d = round(3.2 + random.uniform(-0.3, 0.3), 2)
                odds_a = round(1 / (aa / (ha + aa) + 0.05) + random.uniform(-0.1, 0.1), 2)

                matches.append({
                    "home_team":  home,
                    "away_team":  away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_xg":    round(mu_h + random.gauss(0, 0.2), 2),
                    "away_xg":    round(mu_a + random.gauss(0, 0.2), 2),
                    "home_shots": int(mu_h * 5),
                    "away_shots": int(mu_a * 5),
                    "result":     result,
                    "odds_home":  max(1.10, odds_h),
                    "odds_draw":  max(1.10, odds_d),
                    "odds_away":  max(1.10, odds_a),
                    "date":       f"01/{random.randint(8,12):02d}/2023",
                    "league":     "EPL",
                    "season":     "demo",
                })

        logger.info(f"Generated {len(matches)} synthetic matches")

        # Fit model on first 300 matches
        train = matches[:300]
        self.model.fit(train)

        # Generate synthetic upcoming matches with odds
        self.bet_mgr.clear_bets()
        upcoming_pairs = [(teams[i], teams[i+1]) for i in range(0, min(10, len(teams)-1), 2)]
        all_value_bets = []

        for home, away in upcoming_pairs:
            prediction = self.model.predict(home, away)
            synthetic_odds = {
                "best_home": round(random.uniform(1.8, 4.5), 2),
                "best_draw": round(random.uniform(2.8, 4.2), 2),
                "best_away": round(random.uniform(1.8, 4.5), 2),
            }
            vbets = self.bet_mgr.evaluate_match(
                prediction=prediction,
                odds=synthetic_odds,
                league="EPL",
                match_date="demo",
            )
            all_value_bets.extend(vbets)

        self.bet_mgr.collect_value_bets(all_value_bets)
        betslips = self.bet_mgr.generate_betslips(games_per_slip=3, slip_type="trixie")

        # Run backtest on all data (walk-forward handles train/test split internally)
        self.backtester.train_window = 100
        self.backtester.test_window  = 30
        metrics = self.backtester.run_backtest(matches)
        self.backtester.print_report(metrics)

        return {
            "mode":       "demo",
            "value_bets": all_value_bets,
            "betslips":   betslips,
            "backtest":   metrics,
            "summary":    self.bet_mgr.summary(),
        }


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Dixon-Coles Sports Betting Framework")
    parser.add_argument("--mode",     choices=["live", "backtest", "demo"], default="demo")
    parser.add_argument("--league",   default="EPL",
                        choices=["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"])
    parser.add_argument("--season",   type=int, default=2024)
    parser.add_argument("--seasons",  nargs="+", default=["2223", "2324"],
                        help="Season codes for backtesting, e.g. 2223 2324")
    parser.add_argument("--csv",      default=None, help="Path to local football-data CSV")
    parser.add_argument("--api-key",  default="", help="The Odds API key")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--ev",       type=float, default=0.05, help="Min EV threshold")
    parser.add_argument("--kelly",    type=float, default=0.22, help="Kelly fraction")
    parser.add_argument("--slip-size",type=int,   default=3,    help="Games per betslip")
    parser.add_argument("--slip-type",default="auto",
                        choices=["auto", "single", "trixie", "accumulator"])
    parser.add_argument("--output",   default=None, help="Save results to JSON file")
    args = parser.parse_args()

    framework = BettingFramework(
        odds_api_key=args.api_key,
        bankroll=args.bankroll,
        ev_threshold=args.ev,
        kelly_fraction=args.kelly,
    )

    if args.mode == "demo":
        results = framework.run_demo()
    elif args.mode == "live":
        if not args.api_key:
            logger.error("--api-key required for live mode")
            sys.exit(1)
        results = framework.run_live(
            league=args.league,
            season=args.season,
            games_per_slip=args.slip_size,
            slip_type=args.slip_type,
        )
    elif args.mode == "backtest":
        results = framework.run_backtest_pipeline(
            league=args.league,
            season_codes=args.seasons,
            local_csv=args.csv,
        )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()