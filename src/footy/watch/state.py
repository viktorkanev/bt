"""SQLite store for sent alerts (de-duplication and history)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta

from .rules import Alert

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  key TEXT PRIMARY KEY, kind TEXT, match TEXT, kickoff TEXT, league TEXT, market TEXT, line REAL,
  outcome TEXT, book TEXT, price REAL, fair_odds REAL, edge REAL, detail TEXT, extra TEXT,
  first_sent TEXT, last_sent TEXT, sent_edge REAL, times INTEGER
);
"""


class State:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def should_send(self, a: Alert, now: datetime, cooldown_min: float, step: float) -> bool:
        with self.lock:
            row = self.db.execute("SELECT last_sent, sent_edge FROM alerts WHERE key=?", (a.key,)).fetchone()
        if row is None:
            return True
        last, sent_edge = datetime.fromisoformat(row[0]), row[1]
        return a.edge >= sent_edge + step or (now - last) >= timedelta(minutes=cooldown_min)

    def record(self, a: Alert, now: datetime) -> None:
        with self.lock:
            prev = self.db.execute("SELECT first_sent, times FROM alerts WHERE key=?", (a.key,)).fetchone()
            first, times = (prev[0], prev[1] + 1) if prev else (now.isoformat(), 1)
            self.db.execute(
                "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (a.key, a.kind, a.match, a.kickoff.isoformat(), a.league, a.market, a.line, a.outcome,
                 a.book, a.price, a.fair_odds, a.edge, a.detail, json.dumps(a.extra), first,
                 now.isoformat(), a.edge, times))
            self.db.commit()

    def recent(self, since: datetime, limit: int = 200) -> list[dict]:
        with self.lock:
            cur = self.db.execute(
                "SELECT * FROM alerts WHERE last_sent >= ? ORDER BY last_sent DESC LIMIT ?",
                (since.isoformat(), limit))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def prune(self, before: datetime) -> None:
        with self.lock:
            self.db.execute("DELETE FROM alerts WHERE kickoff < ?", (before.isoformat(),))
            self.db.commit()
