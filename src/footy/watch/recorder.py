"""Feed recorder: what each odds feed delivers, so you can decide which ones are worth paying for.

Kept in its own SQLite file (``recorder.db``), pruned after ``keep_days``:

* ``polls``  one row per poll: provider, time taken, prices, matches, requests/credits used, error.
* ``prices`` a row whenever a feed reports a new price for a selection (unchanged prices
  are not written again, which keeps the file small on a Pi's SD card).
* ``events`` every match seen, with its sport, league and which feeds listed it.

``report()`` turns that into a per-feed comparison: coverage (matches only that feed
had), books it supplies, how often its prices change, which feed shows a move first
when two feeds carry the same bookmaker, how many paper bets depended on books only it
supplies, and the requests a month it needs at a given polling rate.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .core import parse_time

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS polls (ts TEXT, provider TEXT, seconds REAL, quotes INTEGER, events INTEGER,
  requests REAL, error TEXT);
CREATE TABLE IF NOT EXISTS prices (ts TEXT, provider TEXT, book TEXT, event TEXT, market TEXT, line REAL,
  outcome TEXT, price REAL, lay REAL, size REAL);
CREATE INDEX IF NOT EXISTS prices_ts ON prices(ts);
CREATE INDEX IF NOT EXISTS prices_sel ON prices(event, market, line, outcome, book);
CREATE TABLE IF NOT EXISTS events (event TEXT PRIMARY KEY, sport TEXT, league TEXT, home TEXT, away TEXT,
  kickoff TEXT, providers TEXT, first_seen TEXT);
"""


class Recorder:
    def __init__(self, cfg: dict):
        r = cfg.get("recorder", {})
        self.path = r.get("db") or str(Path(cfg["server"]["db"]).with_name("footy-feeds.db"))
        self.keep_days = float(r.get("keep_days", 30))
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self._last: dict[tuple, float] = {}
        self._events: dict[str, set] = {}
        self._pruned_at = 0.0

    # ---------------------------------------------------------- writing
    def record(self, provider: str, now: datetime, quotes, index, seconds: float = 0.0,
               requests: float | None = None, error: str = "") -> int:
        """Store one poll: its stats, the prices that changed, and the matches it listed."""
        ts = now.isoformat(timespec="seconds")
        rows, seen = [], set()
        for q in quotes:
            key = (provider, q.book, q.event, q.market, q.line, q.outcome)
            seen.add(q.event)
            if self._last.get(key) == q.price:
                continue
            self._last[key] = q.price
            rows.append((ts, provider, q.book, q.event, q.market, q.line, q.outcome, q.price, q.lay, q.size))
        ev_rows = []
        for eid in seen:
            provs = self._events.setdefault(eid, set())
            if provider in provs:
                continue
            provs.add(provider)
            ev = index.events.get(eid) if index is not None else None
            if ev is not None:
                ev_rows.append((eid, ev.sport, ev.league, ev.home, ev.away, ev.kickoff.isoformat(), provider, ts))
        with self.lock:
            self.db.execute("INSERT INTO polls VALUES (?,?,?,?,?,?,?)",
                            (ts, provider, round(seconds, 2), len(quotes), len(seen), requests, error[:300]))
            if rows:
                self.db.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            for r in ev_rows:
                cur = self.db.execute("SELECT providers FROM events WHERE event=?", (r[0],)).fetchone()
                if cur is None:
                    self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", r)
                elif provider not in cur[0].split(","):
                    self.db.execute("UPDATE events SET providers=? WHERE event=?", (f"{cur[0]},{provider}", r[0]))
            self.db.commit()
        if time.time() - self._pruned_at > 6 * 3600:
            self.prune(now, index)
        return len(rows)

    def record_scan(self, name: str, now: datetime, seconds: float, events: int, signals: int, error: str) -> None:
        """A prediction-market scan, as a poll of feed "pm:<strategy>" (quotes = opportunities found)."""
        with self.lock:
            self.db.execute("INSERT INTO polls VALUES (?,?,?,?,?,?,?)",
                            (now.isoformat(timespec="seconds"), f"pm:{name}", seconds, signals, events, None,
                             (error or "")[:300]))
            self.db.commit()

    def prune(self, now: datetime, index=None) -> None:
        cut = (now - timedelta(days=self.keep_days)).isoformat(timespec="seconds")
        with self.lock:
            for table in ("polls", "prices"):
                self.db.execute(f"DELETE FROM {table} WHERE ts < ?", (cut,))
            self.db.execute("DELETE FROM events WHERE first_seen < ?", (cut,))
            self.db.commit()
        if index is not None:  # forget finished matches so the change-tracking stays small in memory
            live = set(index.events)
            self._last = {k: v for k, v in self._last.items() if k[2] in live}
            self._events = {k: v for k, v in self._events.items() if k in live}
        self._pruned_at = time.time()

    def size_mb(self) -> float:
        try:
            return Path(self.path).stat().st_size / 1e6
        except OSError:
            return 0.0

    # ---------------------------------------------------------- reading
    def report(self, now: datetime, days: float = 7.0, paper=None) -> dict:
        """Per-feed comparison over the last ``days``."""
        since = (now - timedelta(days=days)).isoformat(timespec="seconds")
        with self.lock:
            polls = self.db.execute("SELECT provider, ts, seconds, quotes, events, requests, error FROM polls "
                                    "WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
            events = self.db.execute("SELECT event, sport, providers FROM events WHERE first_seen >= ?",
                                     (since,)).fetchall()
            books = self.db.execute("SELECT provider, book, COUNT(*), COUNT(DISTINCT event) FROM prices "
                                    "WHERE ts >= ? GROUP BY provider, book", (since,)).fetchall()
            changes = self.db.execute("SELECT provider, COUNT(*), COUNT(DISTINCT event || market || "
                                      "COALESCE(line, '') || outcome || book) FROM prices WHERE ts >= ? "
                                      "GROUP BY provider", (since,)).fetchall()
            shared = self.db.execute(
                "SELECT book, event, market, line, outcome, provider, ts, price FROM prices WHERE ts >= ? AND "
                "book IN (SELECT book FROM prices WHERE ts >= ? GROUP BY book HAVING COUNT(DISTINCT provider) > 1) "
                "ORDER BY ts", (since, since)).fetchall()
        span_h = max(days * 24, 1e-9)
        out: dict[str, dict] = {}
        for prov, ts, secs, nq, nev, req, err in polls:
            d = out.setdefault(prov, {"polls": 0, "errors": 0, "seconds": [], "requests": [], "quotes": 0,
                                      "first": ts, "last": ts})
            d["polls"] += 1
            d["errors"] += 1 if err else 0
            d["seconds"].append(secs or 0)
            if req is not None:
                d["requests"].append(req)
            d["quotes"] = max(d["quotes"], nq or 0)
            d["last"] = ts
        # coverage
        by_prov_sport: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        only: dict[str, int] = defaultdict(int)
        for eid, sport, provs in events:
            ps = provs.split(",")
            for p in ps:
                by_prov_sport[p][sport or "soccer"] += 1
            if len(ps) == 1:
                only[ps[0]] += 1
        book_rows: dict[str, list] = defaultdict(list)
        supplied_by: dict[str, set] = defaultdict(set)
        for prov, book, n, nev in books:
            book_rows[prov].append({"book": book, "changes": n, "matches": nev})
            supplied_by[book].add(prov)
        change_rate = {p: (n / max(sel, 1)) / span_h for p, n, sel in changes}
        lead = self._lead(shared)
        for prov in set(out) | set(by_prov_sport):
            d = out.setdefault(prov, {"polls": 0, "errors": 0, "seconds": [], "requests": [], "quotes": 0,
                                      "first": None, "last": None})
            d["avg_seconds"] = round(statistics.mean(d["seconds"]), 1) if d["seconds"] else None
            reqs = d.pop("requests")
            per_poll = statistics.mean(reqs) if reqs else None
            d["requests_per_poll"] = round(per_poll, 2) if per_poll is not None else None
            d["requests_per_day"] = round(sum(reqs) / days, 1) if reqs else None
            d.pop("seconds")
            d["matches"] = dict(by_prov_sport.get(prov, {}))
            d["matches_total"] = sum(d["matches"].values())
            d["only_here"] = only.get(prov, 0)
            d["books"] = sorted(book_rows.get(prov, []), key=lambda b: -b["matches"])
            d["exclusive_books"] = sorted(b["book"] for b in d["books"] if supplied_by[b["book"]] == {prov})
            d["changes_per_selection_hour"] = round(change_rate.get(prov, 0.0), 3)
            d["lead"] = lead.get(prov, {})
        if paper is not None:
            self._contribution(out, paper, since, supplied_by)
        return {"days": days, "providers": out, "db_mb": round(self.size_mb(), 1)}

    @staticmethod
    def _lead(shared) -> dict[str, dict]:
        """For bookmakers carried by more than one feed: who shows a new price first, and by how long.

        For each selection and each new price value, the first time every feed reported it;
        a feed "leads" when it had the value earlier than the other feed (median minutes).
        """
        first: dict[tuple, dict[str, str]] = defaultdict(dict)
        sighted: set = set()
        for book, ev, mk, ln, oc, prov, ts, price in shared:
            sel = (prov, book, ev, mk, ln, oc)
            if sel not in sighted:  # a feed's first sight of a selection isn't a move
                sighted.add(sel)
                continue
            first[(book, ev, mk, ln, oc, price)].setdefault(prov, ts)
        gaps: dict[tuple, list[float]] = defaultdict(list)
        for provs in first.values():
            if len(provs) < 2:
                continue
            items = sorted(provs.items(), key=lambda kv: kv[1])
            for a, ta in items:
                for b, tb in items:
                    if a < b:
                        gaps[(a, b)].append((parse_time(tb) - parse_time(ta)).total_seconds() / 60)
        out: dict[str, dict] = defaultdict(dict)
        for (a, b), g in gaps.items():
            med = statistics.median(g)  # > 0: a had the price first
            out[a][b] = {"median_minutes_ahead": round(med, 1), "samples": len(g)}
            out[b][a] = {"median_minutes_ahead": round(-med, 1), "samples": len(g)}
        return out

    @staticmethod
    def _contribution(out: dict, paper, since: str, supplied_by: dict) -> None:
        """Paper bets placed at books only this feed supplies: what you'd lose without it."""
        try:
            bets = [b for b in paper.bets() if b["placed"] >= since]
        except Exception:  # noqa: BLE001
            return
        for prov, d in out.items():
            mine = [b for b in bets if supplied_by.get(b["book"]) == {prov}]
            settled = [b for b in mine if b["status"] not in ("open", "void")]
            clv = [b["clv"] for b in mine if b["clv"] is not None]
            d["paper_bets_only_here"] = len(mine)
            d["paper_ev_only_here"] = round(sum(b["stake"] * b["ev"] for b in mine), 2)
            d["paper_profit_only_here"] = round(sum(b["profit"] for b in settled), 2)
            d["paper_avg_clv_only_here"] = round(statistics.mean(clv), 4) if clv else None


def monthly_requests(requests_per_poll: float, minutes_between_polls: float) -> int:
    """Requests a 30-day month needs at a polling rate."""
    return int(math.ceil(requests_per_poll * 30 * 24 * 60 / max(minutes_between_polls, 1e-9)))


def minutes_between(requests_per_poll: float, monthly: int, reserve: int = 0) -> float:
    """How often a monthly allowance lets a feed poll."""
    return 30 * 24 * 60 * requests_per_poll / max(monthly - reserve, 1)
