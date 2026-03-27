"""
database.py  —  SQLite Persistence Layer  (betslips + CLV tracking)

Schema
──────
betslips      : one row per betslip (single/trixie/acca)
bet_legs      : one row per leg within a betslip
clv_snapshots : timestamped odds snapshots for CLV calculation

CLV (Closing Line Value)
────────────────────────
CLV measures whether you beat the closing (final pre-match) odds.
For each pending bet leg:

    CLV = (opening_odds / closing_odds) − 1

Positive CLV means you got better odds than the market closed at —
widely regarded as the best long-run proxy for edge.

    clv_pct > 0  →  you beat the close  (good)
    clv_pct < 0  →  market moved against you (reassess model)
"""

import sqlite3
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional
from pathlib import Path

logger = logging.getLogger("betting_framework.database")

# ─────────────────────────────────────────────────────────────────────────────
#  Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS betslips (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slip_uuid       TEXT    NOT NULL UNIQUE,    -- generated UUID
    slip_type       TEXT    NOT NULL,           -- Single / Trixie / Acca
    created_at      TEXT    NOT NULL,
    league          TEXT,
    total_stake     REAL    NOT NULL,
    combined_odds   REAL,
    potential_return REAL,
    combined_ev     REAL,
    net_ev          REAL,
    compounded_vig  REAL,
    status          TEXT    DEFAULT 'pending',  -- pending / won / lost / void
    profit          REAL,
    settled_at      TEXT
);

CREATE TABLE IF NOT EXISTS bet_legs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slip_uuid       TEXT    NOT NULL REFERENCES betslips(slip_uuid),
    home_team       TEXT    NOT NULL,
    away_team       TEXT    NOT NULL,
    selection       TEXT    NOT NULL,           -- home / draw / away
    model_prob      REAL,
    opening_odds    REAL    NOT NULL,
    closing_odds    REAL,                       -- filled by update_clv
    clv_pct         REAL,                       -- (opening/closing − 1) × 100
    bookmaker_margin REAL,
    ev              REAL,
    kelly_stake     REAL,
    match_date      TEXT,
    match_id        TEXT,
    league          TEXT,
    result          TEXT,                       -- H / D / A (filled at settlement)
    leg_status      TEXT    DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS clv_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    leg_id      INTEGER NOT NULL REFERENCES bet_legs(id),
    snapshot_at TEXT    NOT NULL,
    odds        REAL    NOT NULL,
    source      TEXT                            -- e.g. "the_odds_api"
);

CREATE INDEX IF NOT EXISTS idx_legs_slip   ON bet_legs(slip_uuid);
CREATE INDEX IF NOT EXISTS idx_legs_match  ON bet_legs(match_id);
CREATE INDEX IF NOT EXISTS idx_legs_status ON bet_legs(leg_status);
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Database class
# ─────────────────────────────────────────────────────────────────────────────

class BettingDatabase:
    """
    SQLite-backed persistence for betslips, legs, and CLV tracking.

    Usage
    ─────
        db = BettingDatabase("bets.db")
        db.save_betslips(betslips)
        db.update_clv(odds_updates)
        report = db.clv_report()
    """

    def __init__(self, db_path: str = "betting_framework.db"):
        self.db_path = Path(db_path)
        self._init_db()
        logger.info(f"Database initialised: {self.db_path.resolve()}")

    # ──────────────────────── connection ─────────────────────────────────────

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

    # ──────────────────────── save betslips ──────────────────────────────────

    def save_betslips(self, betslips: list[dict]) -> list[str]:
        """
        Persist a list of betslip dicts (output of BetManager.generate_betslips).
        Returns the list of slip_uuids saved.
        """
        import uuid
        saved_uuids = []

        with self._conn() as conn:
            for slip in betslips:
                slip_uuid = str(uuid.uuid4())
                now = datetime.utcnow().isoformat()

                # Resolve stake — Singles use 'stake', Trixies use 'total_stake'
                total_stake = slip.get("total_stake") or slip.get("stake") or 0.0

                conn.execute("""
                    INSERT INTO betslips
                        (slip_uuid, slip_type, created_at, league,
                         total_stake, combined_odds, potential_return,
                         combined_ev, net_ev, compounded_vig)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    slip_uuid,
                    slip.get("slip_type", "Unknown"),
                    now,
                    slip.get("legs", [{}])[0].get("league", ""),
                    round(float(total_stake), 4),
                    slip.get("combined_odds"),
                    slip.get("potential_return"),
                    slip.get("combined_ev"),
                    slip.get("net_ev"),
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
                        leg.get("home_team", ""),
                        leg.get("away_team", ""),
                        leg.get("selection", ""),
                        leg.get("model_prob"),
                        leg.get("decimal_odds"),
                        leg.get("bookmaker_margin"),
                        leg.get("ev"),
                        leg.get("kelly_stake"),
                        leg.get("match_date", ""),
                        leg.get("match_id", ""),
                        leg.get("league", ""),
                    ))

                saved_uuids.append(slip_uuid)
                logger.info(
                    f"Saved betslip {slip_uuid[:8]}… | "
                    f"{slip.get('slip_type')} | "
                    f"Stake=£{total_stake:.2f}"
                )

        return saved_uuids

    # ──────────────────────── CLV update ─────────────────────────────────────

    def update_clv(self, odds_updates: list[dict], source: str = "the_odds_api") -> int:
        """
        Update closing odds and calculate CLV for pending bet legs.

        odds_updates: list of dicts with keys:
            match_id  (str)  — used to join bet_legs
            selection (str)  — "home" | "draw" | "away"
            closing_odds (float)

        Returns number of legs updated.
        """
        updated = 0
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            for upd in odds_updates:
                mid     = upd.get("match_id", "")
                sel     = upd.get("selection", "")
                cl_odds = upd.get("closing_odds", 0.0)

                if not mid or not sel or cl_odds <= 1.0:
                    continue

                # Fetch pending legs matching this match+selection
                rows = conn.execute("""
                    SELECT id, opening_odds
                    FROM   bet_legs
                    WHERE  match_id = ? AND selection = ? AND leg_status = 'pending'
                """, (mid, sel)).fetchall()

                for row in rows:
                    leg_id = row["id"]
                    op     = row["opening_odds"]
                    clv    = ((op / cl_odds) - 1.0) * 100 if cl_odds > 0 and op else None

                    conn.execute("""
                        UPDATE bet_legs
                        SET    closing_odds = ?, clv_pct = ?
                        WHERE  id = ?
                    """, (cl_odds, clv, leg_id))

                    conn.execute("""
                        INSERT INTO clv_snapshots (leg_id, snapshot_at, odds, source)
                        VALUES (?, ?, ?, ?)
                    """, (leg_id, now, cl_odds, source))

                    updated += 1
                    sign = "✅" if (clv or 0) >= 0 else "⚠️"
                    logger.info(
                        f"{sign} CLV update | leg_id={leg_id} | match_id={mid} | "
                        f"{sel} | open={op:.2f} → close={cl_odds:.2f} | "
                        f"CLV={clv:.2f}%" if clv is not None else
                        f"CLV update | leg_id={leg_id} | CLV=N/A"
                    )

        logger.info(f"CLV update complete: {updated} legs updated")
        return updated

    # ──────────────────────── convenience updater ─────────────────────────────

    def build_clv_updates_from_odds(
        self, live_events: list[dict]
    ) -> list[dict]:
        """
        Convert Odds API event list into the format expected by update_clv().

        live_events: output of DataIngestor.fetch_odds()
        """
        updates = []
        for ev in live_events:
            event_id = ev.get("event_id", "")
            for sel, odds_key in [("home", "best_home"), ("draw", "best_draw"), ("away", "best_away")]:
                closing = ev.get(odds_key, 0.0)
                if closing > 1.0:
                    updates.append({
                        "match_id":     event_id,
                        "selection":    sel,
                        "closing_odds": closing,
                    })
        return updates

    # ──────────────────────── settling ───────────────────────────────────────

    def settle_bet(
        self,
        slip_uuid: str,
        results: dict[str, str],   # {match_id: "H"/"D"/"A"}
    ) -> dict:
        """
        Settle a betslip.  Marks legs won/lost and calculates profit.

        For Singles: profit = stake × (odds − 1) if won, else −stake
        For Trixies/Accas: the slip as a whole is won/lost based on leg results.
        """
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            slip = conn.execute(
                "SELECT * FROM betslips WHERE slip_uuid = ?", (slip_uuid,)
            ).fetchone()
            if not slip:
                return {"error": f"Slip {slip_uuid} not found"}

            legs = conn.execute(
                "SELECT * FROM bet_legs WHERE slip_uuid = ?", (slip_uuid,)
            ).fetchall()

            RESULT_MAP = {"home": "H", "draw": "D", "away": "A"}
            legs_won = 0

            for leg in legs:
                actual = results.get(leg["match_id"], "")
                expected = RESULT_MAP.get(leg["selection"], "?")
                won = (actual == expected)
                conn.execute("""
                    UPDATE bet_legs
                    SET    result = ?, leg_status = ?
                    WHERE  id = ?
                """, (actual, "won" if won else "lost", leg["id"]))
                if won:
                    legs_won += 1

            total_legs = len(legs)
            slip_won = (legs_won == total_legs)
            stake    = slip["total_stake"]
            profit   = (
                round(stake * (slip["combined_odds"] or 1.0) - stake, 2)
                if slip_won else -stake
            )

            conn.execute("""
                UPDATE betslips
                SET    status = ?, profit = ?, settled_at = ?
                WHERE  slip_uuid = ?
            """, (
                "won" if slip_won else "lost",
                profit,
                now,
                slip_uuid,
            ))

        return {
            "slip_uuid":  slip_uuid,
            "status":     "won" if slip_won else "lost",
            "legs_won":   legs_won,
            "total_legs": total_legs,
            "profit":     profit,
        }

    # ──────────────────────── reports ────────────────────────────────────────

    def clv_report(self) -> dict:
        """
        Aggregate CLV statistics across all settled and pending legs.

        Returns summary dict with avg_clv, pct_positive, by_league, etc.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT bl.selection, bl.league, bl.clv_pct,
                       bl.opening_odds, bl.closing_odds, bl.leg_status
                FROM   bet_legs bl
                WHERE  bl.clv_pct IS NOT NULL
            """).fetchall()

        if not rows:
            return {"message": "No CLV data yet — run --mode update_clv first"}

        clvs = [r["clv_pct"] for r in rows]
        pos  = [c for c in clvs if c > 0]

        by_league: dict[str, list[float]] = {}
        for r in rows:
            lg = r["league"] or "unknown"
            by_league.setdefault(lg, []).append(r["clv_pct"])

        return {
            "total_legs":      len(clvs),
            "avg_clv_pct":     round(sum(clvs) / len(clvs), 3),
            "pct_positive_clv": round(len(pos) / len(clvs) * 100, 1),
            "max_clv_pct":     round(max(clvs), 3),
            "min_clv_pct":     round(min(clvs), 3),
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
        """Return all betslips with status='pending' and their legs."""
        with self._conn() as conn:
            slips = conn.execute(
                "SELECT * FROM betslips WHERE status = 'pending' ORDER BY created_at DESC"
            ).fetchall()
            result = []
            for slip in slips:
                legs = conn.execute(
                    "SELECT * FROM bet_legs WHERE slip_uuid = ?",
                    (slip["slip_uuid"],)
                ).fetchall()
                result.append({
                    **dict(slip),
                    "legs": [dict(l) for l in legs],
                })
        return result

    def profit_summary(self) -> dict:
        """Overall P&L summary across all settled slips."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                              AS total_slips,
                    SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) AS wins,
                    SUM(total_stake)                      AS total_staked,
                    SUM(COALESCE(profit, 0))              AS total_profit
                FROM betslips
                WHERE status IN ('won', 'lost')
            """).fetchone()

        if not row or not row["total_slips"]:
            return {"message": "No settled bets yet"}

        return {
            "total_slips":   row["total_slips"],
            "wins":          row["wins"],
            "win_rate_pct":  round(row["wins"] / row["total_slips"] * 100, 1),
            "total_staked":  round(row["total_staked"] or 0, 2),
            "total_profit":  round(row["total_profit"] or 0, 2),
            "roi_pct":       round(
                (row["total_profit"] or 0) / (row["total_staked"] or 1) * 100, 2
            ),
        }
