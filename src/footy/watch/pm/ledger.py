"""Imaginary positions for the prediction-market strategies, settled from Polymarket's resolution."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from datetime import datetime, timedelta

from ..core import parse_time
from . import Signal, gamma

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pm_positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, placed TEXT, strategy TEXT, key TEXT, event_id TEXT, title TEXT,
  slug TEXT, end_date TEXT, kind TEXT, legs TEXT, cost REAL, payout_min REAL, expected REAL, edge REAL,
  apy REAL, prob REAL, detail TEXT, status TEXT DEFAULT 'open', payout REAL, profit REAL, settled TEXT,
  UNIQUE(strategy, key)
);
CREATE INDEX IF NOT EXISTS pm_status ON pm_positions(status);
"""
STRATEGIES = {"consistency": "Price consistency", "weather": "Weather"}


class PMBook:
    def __init__(self, path: str, cfg: dict):
        self.cfg = cfg
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def stake(self, strategy: str) -> float:
        return float(self.cfg.get(strategy, {}).get("stake", 10.0))

    # -------------------------------------------------------- placement
    def place(self, s: Signal, now: datetime) -> dict | None:
        """Record the position at the signal's prices, sized to the stake and the money on offer."""
        per_set = sum(lg["price"] + lg["fee"] for lg in s.legs)
        if per_set <= 0:
            return None
        shares = self.stake(s.strategy) / per_set
        avail = min((lg.get("size") or math.inf) for lg in s.legs)
        shares = min(shares, avail)
        if shares * per_set < float(self.cfg.get("pm", {}).get("min_position", 2.0)):
            return None
        legs = [{**lg, "shares": round(shares, 4)} for lg in s.legs]
        cost = round(shares * per_set, 4)
        payout_min = round(shares * s.payout_per_set, 4) if s.payout_per_set is not None else None
        expected = round(cost * (1 + s.edge), 4)
        row = {"placed": now.isoformat(timespec="seconds"), "strategy": s.strategy, "key": s.key,
               "event_id": s.event_id, "title": s.title, "slug": s.slug,
               "end_date": s.end_date.isoformat() if s.end_date else None, "kind": s.kind, "legs": json.dumps(legs),
               "cost": cost, "payout_min": payout_min, "expected": expected, "edge": s.edge, "apy": s.apy,
               "prob": s.prob, "detail": s.detail}
        with self.lock:
            cur = self.db.execute(f"INSERT OR IGNORE INTO pm_positions ({','.join(row)}) VALUES "
                                  f"({','.join('?' * len(row))})", list(row.values()))
            self.db.commit()
        if not cur.rowcount:
            return None
        log.info("pm position [%s] %s %s: cost $%.2f, edge %+.1f%%", s.strategy, s.kind, s.title, cost, 100 * s.edge)
        return row

    # ------------------------------------------------------- settlement
    def settle(self, now: datetime, fetch=gamma.markets_by_id) -> int:
        """Settle open positions whose markets have all resolved."""
        rows = self.positions(status="open")
        due = [r for r in rows if not r["end_date"] or parse_time(r["end_date"]) <= now + timedelta(days=60)]
        if not due:
            return 0
        ids = {lg["market_id"] for r in due for lg in r["legs"]}
        try:
            markets = fetch(ids)
        except Exception as e:  # noqa: BLE001
            log.warning("pm settlement lookup failed: %s", e)
            return 0
        n = 0
        for r in due:
            res = [gamma.resolution(markets.get(str(lg["market_id"]))) for lg in r["legs"]]
            if any(x is None for x in res):
                continue
            payout = sum(lg["shares"] * (x if lg["side"] == "YES" else 1 - x) for lg, x in zip(r["legs"], res))
            with self.lock:
                self.db.execute("UPDATE pm_positions SET status='settled', payout=?, profit=?, settled=? WHERE id=?",
                                (round(payout, 4), round(payout - r["cost"], 4), now.isoformat(timespec="seconds"),
                                 r["id"]))
                self.db.commit()
            n += 1
        if n:
            log.info("settled %d prediction-market positions", n)
        return n

    # ---------------------------------------------------------- reading
    def positions(self, strategy: str | None = None, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM pm_positions WHERE 1=1", []
        if strategy:
            q += " AND strategy=?"
            args.append(strategy)
        if status:
            q += " AND status=?"
            args.append(status)
        with self.lock:
            rows = [dict(r) for r in self.db.execute(q + " ORDER BY COALESCE(settled, placed), id", args).fetchall()]
        for r in rows:
            r["legs"] = json.loads(r["legs"] or "[]")
        return rows

    def reset(self) -> int:
        with self.lock:
            n = self.db.execute("DELETE FROM pm_positions").rowcount
            self.db.commit()
        return n


def summarize(rows: list[dict], start: float) -> dict:
    """Headline numbers and chart series (settled positions in settlement order)."""
    settled = [r for r in rows if r["status"] == "settled"]
    open_ = [r for r in rows if r["status"] == "open"]
    cost = sum(r["cost"] for r in settled)
    pl = sum(r["profit"] for r in settled)
    series, cum, exp, var = [], 0.0, 0.0, 0.0
    for i, r in enumerate(settled, 1):
        cum += r["profit"]
        exp += r["expected"] - r["cost"]
        if r["payout_min"] is None and r["prob"]:  # single bet: payout is all or nothing
            p, win = r["prob"], r["expected"] / max(r["prob"], 1e-9)
            var += win * win * p * (1 - p)
        series.append({"n": i, "actual": round(cum, 2), "ev": round(exp, 2), "clv": round(exp, 2),
                       "band": round(2 * math.sqrt(var), 2), "kickoff": r["settled"], "match": r["title"],
                       "status": "won" if r["profit"] > 0 else "lost" if r["profit"] < 0 else "even",
                       "profit": round(r["profit"], 2)})
    return {"positions": len(rows), "settled": len(settled), "open": len(open_),
            "open_cost": sum(r["cost"] for r in open_), "staked": cost, "profit": pl,
            "roi": pl / cost if cost else None,
            "expected_profit": sum(r["expected"] - r["cost"] for r in settled),
            "avg_edge": (sum(r["edge"] * r["cost"] for r in rows) / sum(r["cost"] for r in rows)) if rows else None,
            "won": sum(1 for r in settled if r["profit"] > 0), "bankroll": start + pl, "series": series}
