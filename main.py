"""
main.py  —  Betting Framework Orchestrator  (v2)

New in v2
─────────
1.  Cold-start fix:  --seasons flag in live mode now ingests the last 2 full
    seasons from understat PLUS the current season, giving the model 1,000+
    matches to work from at the start of a campaign.

2.  Time-decay:  --half-life (default 90 days) is passed to PredictorModel
    so recent matches dominate the MLE fit.

3.  --mode update_clv:  Fetches latest odds from The Odds API for all pending
    bet legs in the database and writes CLV figures to SQLite.

4.  --mode settle:  Mark a specific slip as won/lost (CLI helper).

Usage examples
──────────────
# Demo (no external APIs)
python main.py --mode demo

# Live — cold-start aware: pulls 2023 + 2024 seasons before fitting
python main.py --mode live \
    --league EPL --season 2024 --cold-start-seasons 2023 2022 \
    --api-key YOUR_KEY --bankroll 1000 --half-life 90 \
    --slip-type trixie --db bets.db

# CLV update (run before kick-off)
python main.py --mode update_clv --league EPL --api-key YOUR_KEY --db bets.db

# Backtest over two historical seasons
python main.py --mode backtest --league EPL --seasons 2223 2324

# Print CLV report
python main.py --mode clv_report --db bets.db
"""

import logging
import argparse
import json
import sys
import math
import random
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
#  Module imports
# ─────────────────────────────────────────────────────────────────────────────

from data_ingestor   import DataIngestor
from predictor_model import PredictorModel
from bet_manager     import BetManager
from backtester      import Backtester
from database        import BettingDatabase

# understat season int → FDCO season code mapping for cold-start CSV fallback
_SEASON_CODE_MAP = {
    2024: "2425",
    2023: "2324",
    2022: "2223",
    2021: "2122",
    2020: "2021",
    2019: "1920",
}


# ─────────────────────────────────────────────────────────────────────────────
#  BettingFramework
# ─────────────────────────────────────────────────────────────────────────────

class BettingFramework:

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

        self.ingestor = DataIngestor(odds_api_key=odds_api_key)

        # v2: PredictorModel now receives time-decay and xG-blend params
        self.model = PredictorModel(
            half_life_days=half_life_days,
            xg_blend=xg_blend,
            use_xg=use_xg,
        )

        self.bet_mgr = BetManager(
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

        self.db = BettingDatabase(db_path)

    # ──────────────────────────────── live ───────────────────────────────────

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
        Full live pipeline with cold-start multi-season ingestion.

        Cold-start fix
        ──────────────
        If cold_start_seasons is provided (e.g. [2023, 2022]), those seasons
        are fetched from understat FIRST, then the current season is appended.
        The combined corpus is passed to PredictorModel.fit().

        This solves the sparse-data problem at season start when only 5–10
        rounds have been played.
        """
        sep = "=" * 65
        logger.info(sep)
        logger.info(f"LIVE PIPELINE  {league} | Current season: {season}")
        if cold_start_seasons:
            logger.info(f"Cold-start seasons: {cold_start_seasons}")
        logger.info(sep)

        # ── Step 1: Ingest historical seasons (cold-start) ───────────────────
        all_xg_matches = []

        historical_seasons = list(cold_start_seasons or [])
        for hist_season in historical_seasons:
            logger.info(f"  [Cold-start] Fetching understat xG: {league} {hist_season}…")
            hist_matches = self.ingestor.fetch_understat_xg(league, hist_season)
            if not hist_matches:
                # Fallback to FDCO CSV for cold-start seasons
                sc = _SEASON_CODE_MAP.get(hist_season)
                if sc:
                    logger.info(f"  [Fallback] Fetching FDCO CSV: {league} {sc}…")
                    hist_matches = self.ingestor.fetch_historical_csv(league, sc)
            logger.info(f"    → {len(hist_matches)} matches from season {hist_season}")
            all_xg_matches.extend(hist_matches)

        # ── Step 2: Ingest current season ────────────────────────────────────
        logger.info(f"  [Current] Fetching understat xG: {league} {season}…")
        current_matches = self.ingestor.fetch_understat_xg(league, season)
        if not current_matches:
            logger.warning(f"  No current-season xG — falling back to FDCO CSV")
            sc = _SEASON_CODE_MAP.get(season)
            if sc:
                current_matches = self.ingestor.fetch_historical_csv(league, sc)

        logger.info(f"  Current season: {len(current_matches)} matches")
        all_xg_matches.extend(current_matches)

        total = len(all_xg_matches)
        logger.info(f"  Total training corpus: {total} matches")

        if total < 30:
            logger.warning(
                f"Only {total} training matches — model will be unstable. "
                "Consider adding more cold-start seasons."
            )

        # ── Step 3: Fit model ─────────────────────────────────────────────────
        logger.info("Fitting Dixon-Coles (joint MLE)…")
        try:
            self.model.fit(all_xg_matches)
        except Exception as exc:
            logger.error(f"Model fit failed: {exc}")
            return {"error": str(exc)}

        logger.info(f"Fit stats: {self.model.fit_summary()}")

        # ── Step 4: Fetch live odds ───────────────────────────────────────────
        logger.info("Fetching live H2H odds from The Odds API…")
        upcoming = self.ingestor.fetch_odds(league)
        if not upcoming:
            logger.warning("No odds data returned — check API key or quota")
            return {"error": "No odds data available"}

        # ── Step 5: Evaluate each upcoming match ──────────────────────────────
        logger.info(f"Evaluating {len(upcoming)} upcoming matches…")
        self.bet_mgr.clear_bets()
        all_value_bets = []

        for event in upcoming:
            pred = self.model.predict(event["home_team"], event["away_team"])
            vbets = self.bet_mgr.evaluate_match(
                prediction=pred,
                odds=event,
                league=league,
                match_date=event.get("commence", ""),
                match_id=event.get("event_id", ""),
            )
            all_value_bets.extend(vbets)

        self.bet_mgr.collect_value_bets(all_value_bets)

        # ── Step 6: Generate betslips with strict parlay vig gate ─────────────
        betslips = self.bet_mgr.generate_betslips(
            games_per_slip=games_per_slip,
            slip_type=slip_type,
        )

        # ── Step 7: Persist to SQLite ─────────────────────────────────────────
        saved_uuids = []
        if save_to_db and betslips:
            saved_uuids = self.db.save_betslips(betslips)
            logger.info(f"Saved {len(saved_uuids)} betslips to database")

        summary = self.bet_mgr.summary()
        logger.info(f"Summary: {summary}")

        return {
            "league":            league,
            "season":            season,
            "training_matches":  total,
            "matches_analysed":  len(upcoming),
            "value_bets":        all_value_bets,
            "betslips":          betslips,
            "saved_uuids":       saved_uuids,
            "skipped":           self.bet_mgr.get_skip_log(),
            "summary":           summary,
            "model_stats":       self.model.fit_summary(),
        }

    # ──────────────────────────── update_clv ─────────────────────────────────

    def run_update_clv(self, league: str = "EPL") -> dict:
        """
        Fetch current market odds and update CLV for all pending bet legs.

        Typically run 15–30 minutes before kick-off when the market is
        at its most efficient (closing line).
        """
        logger.info(f"{'='*65}")
        logger.info(f"CLV UPDATE  {league}")
        logger.info(f"{'='*65}")

        if not self.odds_api_key:
            return {"error": "Odds API key required for CLV update"}

        logger.info("Fetching closing odds from The Odds API…")
        live_events = self.ingestor.fetch_odds(league)

        if not live_events:
            return {"warning": "No live events returned — check API quota"}

        odds_updates = self.db.build_clv_updates_from_odds(live_events)
        n_updated = self.db.update_clv(odds_updates)

        report = self.db.clv_report()
        logger.info(f"CLV report: {json.dumps(report, indent=2)}")

        return {
            "events_fetched": len(live_events),
            "legs_updated":   n_updated,
            "clv_report":     report,
        }

    # ──────────────────────────── backtest ───────────────────────────────────

    def run_backtest_pipeline(
        self,
        league: str = "EPL",
        season_codes: list[str] = None,
        local_csv: str = None,
    ) -> dict:
        logger.info(f"{'='*65}")
        logger.info(f"BACKTEST PIPELINE  {league}")
        logger.info(f"{'='*65}")

        all_matches = []
        if local_csv:
            logger.info(f"Loading local CSV: {local_csv}")
            all_matches = self.ingestor.ingest_local_csv(local_csv, league)
        else:
            season_codes = season_codes or ["2223", "2324"]
            for sc in season_codes:
                logger.info(f"Fetching FDCO CSV: {league} {sc}")
                all_matches.extend(self.ingestor.fetch_historical_csv(league, sc))

        logger.info(f"Total historical matches: {len(all_matches)}")
        if not all_matches:
            return {"error": "No historical data loaded"}

        metrics = self.backtester.run_backtest(all_matches)
        self.backtester.print_report(metrics)
        return metrics

    # ──────────────────────────── demo ───────────────────────────────────────

    def run_demo(self) -> dict:
        """
        Fully self-contained demo — no external API calls.
        Generates synthetic EPL data, fits the upgraded model, and runs
        the full pipeline including SQLite persistence and CLV scaffolding.
        """
        logger.info("DEMO MODE — synthetic EPL data")

        teams = [
            "Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd",
            "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
            "Brentford", "Fulham", "Wolves", "Everton", "Crystal Palace",
            "Nottm Forest", "Bournemouth", "Leicester", "Luton", "Sheffield Utd",
        ]

        random.seed(42)
        strengths = {t: (random.uniform(0.8, 2.0), random.uniform(0.8, 2.0)) for t in teams}

        # Generate THREE synthetic seasons (cold-start simulation)
        all_matches = []
        for season_offset, year in enumerate([2022, 2023, 2024]):
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i == j:
                        continue
                    ha, hd = strengths[home]
                    aa, ad = strengths[away]
                    mu_h = max(ha / ad * 1.25, 0.1)
                    mu_a = max(aa / hd, 0.1)

                    hg = min(int(random.expovariate(1 / mu_h)), 8)
                    ag = min(int(random.expovariate(1 / mu_a)), 8)
                    result = "H" if hg > ag else ("D" if hg == ag else "A")

                    # Random matchday within the season window
                    month = random.randint(8, 5 + 12)  # Aug–May spread
                    actual_month = ((month - 1) % 12) + 1
                    actual_year  = year if actual_month >= 8 else year + 1
                    date_str = f"{random.randint(1,28):02d}/{actual_month:02d}/{actual_year}"

                    odds_h = max(1.10, round(1 / (ha / (ha + aa) + 0.05) + random.uniform(-0.1, 0.1), 2))
                    odds_d = max(1.10, round(3.2 + random.uniform(-0.3, 0.3), 2))
                    odds_a = max(1.10, round(1 / (aa / (ha + aa) + 0.05) + random.uniform(-0.1, 0.1), 2))

                    all_matches.append({
                        "home_team":  home,
                        "away_team":  away,
                        "home_goals": hg,
                        "away_goals": ag,
                        "home_xg":    round(mu_h + random.gauss(0, 0.15), 2),
                        "away_xg":    round(mu_a + random.gauss(0, 0.15), 2),
                        "home_shots": int(mu_h * 5),
                        "away_shots": int(mu_a * 5),
                        "result":     result,
                        "odds_home":  odds_h,
                        "odds_draw":  odds_d,
                        "odds_away":  odds_a,
                        "date":       date_str,
                        "league":     "EPL",
                        "season":     str(year),
                    })

        logger.info(f"Generated {len(all_matches)} synthetic matches across 3 seasons")

        # Fit on all data (time-decay naturally down-weights 2022 season)
        self.model.fit(all_matches)
        logger.info(f"Fit stats: {self.model.fit_summary()}")

        # Synthetic upcoming fixtures
        self.bet_mgr.clear_bets()
        upcoming_pairs = [(teams[i], teams[i + 1]) for i in range(0, min(10, len(teams) - 1), 2)]
        all_value_bets = []

        for home, away in upcoming_pairs:
            pred = self.model.predict(home, away)
            # Include all three odds so overround can be computed
            ho = round(random.uniform(1.8, 4.5), 2)
            do = round(random.uniform(2.8, 4.2), 2)
            ao = round(random.uniform(1.8, 4.5), 2)
            synthetic_odds = {
                "best_home": ho, "best_draw": do, "best_away": ao,
            }
            vbets = self.bet_mgr.evaluate_match(
                prediction=pred, odds=synthetic_odds, league="EPL", match_date="demo",
            )
            all_value_bets.extend(vbets)

        self.bet_mgr.collect_value_bets(all_value_bets)
        betslips = self.bet_mgr.generate_betslips(games_per_slip=3, slip_type="trixie")

        # Save to DB
        saved_uuids = self.db.save_betslips(betslips) if betslips else []
        logger.info(f"Saved {len(saved_uuids)} betslips to demo DB")

        # Print vig gate stats
        for slip in betslips:
            logger.info(
                f"Slip {slip['slip_type']:20s} | "
                f"netEV={slip.get('net_ev', 'N/A'):.4f} | "
                f"vig={slip.get('compounded_vig', 'N/A'):.4f} | "
                f"stake=£{slip.get('total_stake', slip.get('stake', 0)):.2f}"
            )

        # Backtest
        self.backtester.train_window = 100
        self.backtester.test_window  = 30
        metrics = self.backtester.run_backtest(all_matches)
        self.backtester.print_report(metrics)

        return {
            "mode":          "demo",
            "training_matches": len(all_matches),
            "value_bets":    all_value_bets,
            "betslips":      betslips,
            "saved_uuids":   saved_uuids,
            "backtest":      metrics,
            "summary":       self.bet_mgr.summary(),
            "model_stats":   self.model.fit_summary(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dixon-Coles Betting Framework v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "demo", "update_clv", "clv_report", "settle"],
        default="demo",
    )
    parser.add_argument("--league", default="EPL",
                        choices=["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"])
    parser.add_argument("--season",  type=int, default=2024,
                        help="Current season (understat int, e.g. 2024)")
    parser.add_argument("--cold-start-seasons", nargs="*", type=int, default=[2023, 2022],
                        dest="cold_start_seasons",
                        help="Historical understat seasons for cold-start (e.g. 2023 2022)")
    parser.add_argument("--seasons", nargs="+", default=["2223", "2324"],
                        help="FDCO season codes for backtest mode (e.g. 2223 2324)")
    parser.add_argument("--csv",     default=None, help="Path to local football-data CSV")
    parser.add_argument("--api-key", default="",   help="The Odds API key")
    parser.add_argument("--bankroll",type=float,   default=1000.0)
    parser.add_argument("--ev",      type=float,   default=0.05, help="Min EV threshold")
    parser.add_argument("--kelly",   type=float,   default=0.22, help="Kelly fraction")
    parser.add_argument("--half-life", type=float, default=90.0, dest="half_life",
                        help="Time-decay half-life in days (inf = no decay)")
    parser.add_argument("--xg-blend", type=float, default=0.0, dest="xg_blend",
                        help="xG blend fraction 0–1 (0=goals only, 1=xG only)")
    parser.add_argument("--use-xg",  action="store_true", dest="use_xg",
                        help="Enable xG blending in model")
    parser.add_argument("--slip-size",type=int,  default=3, help="Legs per betslip")
    parser.add_argument("--slip-type", default="auto",
                        choices=["auto", "single", "trixie", "accumulator"])
    parser.add_argument("--db",     default="betting_framework.db",
                        help="SQLite database file path")
    parser.add_argument("--slip-uuid", default=None,
                        help="Betslip UUID for settle mode")
    parser.add_argument("--results", default=None,
                        help='JSON string of results for settle, e.g. \'{"event_id": "H"}\'')
    parser.add_argument("--no-save", action="store_true", dest="no_save",
                        help="Do not persist betslips to database")
    parser.add_argument("--output",  default=None, help="Save JSON output to this file")
    parser.add_argument("--verbose", action="store_true",
                        help="Set logging level to DEBUG")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    results = {}

    if args.mode == "demo":
        results = framework.run_demo()

    elif args.mode == "live":
        if not args.api_key:
            logger.error("--api-key is required for live mode")
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
            logger.error("--api-key is required for update_clv mode")
            sys.exit(1)
        results = framework.run_update_clv(league=args.league)

    elif args.mode == "clv_report":
        results = framework.db.clv_report()
        results["profit_summary"] = framework.db.profit_summary()
        logger.info(json.dumps(results, indent=2))

    elif args.mode == "settle":
        if not args.slip_uuid:
            logger.error("--slip-uuid required for settle mode")
            sys.exit(1)
        if not args.results:
            logger.error("--results required for settle mode (JSON: {match_id: H/D/A})")
            sys.exit(1)
        res_map = json.loads(args.results)
        results = framework.db.settle_bet(args.slip_uuid, res_map)
        logger.info(f"Settlement: {results}")

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results written to {args.output}")

    return results


if __name__ == "__main__":
    main()
