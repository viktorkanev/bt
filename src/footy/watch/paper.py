"""Paper trading: a forward test with an imaginary bankroll.

Every alert the watcher raises becomes an imaginary bet, recorded separately at
three places, each with its own bankroll:

* ``best``      the best price among the books you allow (``paper.best_books``)
* ``pinnacle``  Pinnacle's own price
* ``betfair``   the Betfair Exchange back price, after commission
* ``polymarket`` the Polymarket best ask, after the taker fee
* ``arb``       arbitrages: every outcome backed at the best book, stake split so
                each result returns the same amount

For each bet we keep the expected value when it was placed (EV), the closing
line just before kickoff (CLV = our price against the margin-free closing price
of Pinnacle, else Betfair) and, once results arrive, the profit.

CLV is the early verdict: hundreds of bets are needed before profit alone tells
skill from luck, while beating the closing line shows up much sooner.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import sqlite3
import threading
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .core import (OUTCOMES, SPORTS, EventIndex, Quote, book_net_odds, fit_goals, grid_fair_odds, match_event,
                   name_score,
                   parse_time, profit, score_grid, settle)
from .rules import Alert, ref_probs

log = logging.getLogger(__name__)

VENUES = {"best": "Best price", "pinnacle": "Pinnacle", "betfair": "Betfair Exchange",
          "polymarket": "Polymarket", "arb": "Arbitrage"}
STATUS = {1.0: "won", 0.5: "half won", 0.0: "push", -0.5: "half lost", -1.0: "lost"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_bets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  placed TEXT, strategy TEXT, strategy_name TEXT, venue TEXT,
  event_id TEXT, home TEXT, away TEXT, league TEXT, kickoff TEXT,
  market TEXT, line REAL, outcome TEXT, book TEXT, price REAL, net_price REAL,
  fair_odds REAL, ev REAL, stake REAL, detail TEXT, meta TEXT,
  close_fair REAL, close_ref TEXT, close_how TEXT, close_price REAL, close_book TEXT, close_at TEXT, clv REAL,
  status TEXT DEFAULT 'open', home_goals INTEGER, away_goals INTEGER, result REAL, profit REAL,
  settled TEXT, result_source TEXT,
  UNIQUE(strategy, venue, event_id, market, outcome)
);
CREATE INDEX IF NOT EXISTS paper_status ON paper_bets(status, kickoff);
"""


def _same_line(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) < 1e-6)


def _balanced(pairs: dict[float, list[float]]) -> tuple[float, list[float]] | None:
    """The line whose two-way no-margin probabilities are closest to 50/50."""
    if not pairs:
        return None
    line = min(pairs, key=lambda k: abs(pairs[k][0] - 0.5))
    return line, pairs[line]


def closing_fair(quotes: list[Quote], market: str, line, outcome: str, refs: list[str],
                 max_spread: float = 0.05, sport: str = "soccer") -> tuple[float, str, str] | None:
    """No-margin price of a selection from a reference book: (odds, book, how).

    Uses the same line when the reference offers it, otherwise moves the
    reference's main handicap / total to our line with a Poisson model.
    """
    outs_needed = OUTCOMES[market]
    k = outs_needed.index(outcome)
    for rb in refs:
        outs = {q.outcome: q for q in quotes if q.book == rb and q.market == market and _same_line(q.line, line)}
        probs = ref_probs(outs, outs_needed, max_spread)
        if probs:
            return 1 / probs[k], rb, "same line"
    if market in ("1x2", "ml") or sport != "soccer":  # the goals model only fits football
        return None
    for rb in refs:
        rq = [q for q in quotes if q.book == rb]
        by: dict[tuple, dict[str, Quote]] = defaultdict(dict)
        for q in rq:
            by[(q.market, q.line)][q.outcome] = q
        x12 = ref_probs(by.get(("1x2", None), {}), OUTCOMES["1x2"], max_spread)
        ah_lines, ou_lines = {}, {}
        for (m, ln), outs in by.items():
            if m in ("ah", "ou") and ln is not None:
                p = ref_probs(outs, OUTCOMES[m], max_spread)
                if p:
                    (ah_lines if m == "ah" else ou_lines)[ln] = p
        ah, ou = _balanced(ah_lines), _balanced(ou_lines)
        if market == "ah" and not (ah or x12):
            continue
        if market == "ou" and not ou:
            continue
        lh, la = fit_goals(x12=tuple(x12) if x12 else None,
                           ah=(ah[0], 1 / ah[1][0]) if ah else None,
                           ou=(ou[0], 1 / ou[1][0]) if ou else None)
        return grid_fair_odds(score_grid(lh, la), market, line, outcome), rb, "converted"
    return None


# ------------------------------------------------------------- results feeds
FOOTBALL_DATA_CODES = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2",
                       "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"]


class FootballDataResults:
    """Final scores from football-data.co.uk season files (updated a few times a week)."""

    name = "football-data"
    URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

    def __init__(self, refresh_minutes: float = 360, fetch=None):
        self.refresh_s = refresh_minutes * 60
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._fetch = fetch or self._http
        self.last_error = ""

    @staticmethod
    def _http(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "footy-watch/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8-sig", errors="replace")

    @staticmethod
    def season(dt: datetime) -> str:
        y = dt.year if dt.month >= 7 else dt.year - 1
        return f"{y % 100:02d}{(y + 1) % 100:02d}"

    @staticmethod
    def parse(text: str) -> list[dict]:
        from zoneinfo import ZoneInfo
        uk = ZoneInfo("Europe/London")
        out = []
        for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
            try:
                hg, ag = int(float(r["FTHG"])), int(float(r["FTAG"]))
                d = r["Date"].strip()
                day = datetime.strptime(d, "%d/%m/%Y" if len(d.split("/")[-1]) == 4 else "%d/%m/%y")
                hh, mm = (r.get("Time") or "15:00").strip().split(":")[:2]
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            ko = day.replace(hour=int(hh), minute=int(mm), tzinfo=uk).astimezone(timezone.utc)
            out.append({"source": "football-data", "home": r["HomeTeam"], "away": r["AwayTeam"],
                        "kickoff": ko, "home_goals": hg, "away_goals": ag})
        return out

    def results(self, bets: list[dict], now: datetime) -> list[dict]:
        bets = [b for b in bets if (b.get("sport") or "soccer") == "soccer"]  # football only
        if not bets:
            return []
        codes = {b.get("league") for b in bets} & set(FOOTBALL_DATA_CODES)
        if any(b.get("league") not in FOOTBALL_DATA_CODES for b in bets):
            codes = set(FOOTBALL_DATA_CODES)  # unknown league (e.g. a Betfair-only match): look everywhere
        seasons = {self.season(parse_time(b["kickoff"])) for b in bets}
        out = []
        for season in sorted(seasons):
            for code in sorted(codes):
                key = f"{season}/{code}"
                cached = self._cache.get(key)
                if cached is None or time.time() - cached[0] > self.refresh_s:
                    try:
                        cached = (time.time(), self.parse(self._fetch(self.URL.format(season=season, code=code))))
                    except Exception as e:  # noqa: BLE001
                        self.last_error = f"{key}: {e}"
                        log.debug("football-data %s: %s", key, e)
                        cached = (time.time(), [])
                    self._cache[key] = cached
                out += cached[1]
        return out


def find_result(bet: dict, results: list[dict], tol_hours: float = 30.0) -> dict | None:
    """The result that settles this bet.

    Winner results (Betfair) settle only the same match and market. Final scores settle
    only markets a score decides in that sport (not hockey's regular-time 1X2, not tennis).
    A score listed the other way round (away team first) is turned around.
    """
    ko = parse_time(bet["kickoff"])
    sport = bet.get("sport") or "soccer"
    oid = (bet.get("meta") or {}).get("sources", {}).get("oddsapi")
    market = bet.get("market") or "1x2"
    score_ok = market in SPORTS.get(sport, SPORTS["soccer"])["score"]
    best, best_s = None, 0.0
    for r in results:
        if "winner" in r:
            if r.get("event_id") == bet.get("event_id") and r.get("market") == market:
                return r
            continue
        if not score_ok:
            continue
        if oid and r.get("id") == oid:
            return r
        if abs((r["kickoff"] - ko).total_seconds()) > tol_hours * 3600:
            continue
        for flip in (False, True):
            rh, ra = (r["away"], r["home"]) if flip else (r["home"], r["away"])
            sh, sa = name_score(bet["home"], rh), name_score(bet["away"], ra)
            if min(sh, sa) >= 0.72 and sh + sa > best_s:
                best = {**r, "home_goals": r["away_goals"], "away_goals": r["home_goals"]} if flip else r
                best_s = sh + sa
    return best


# ------------------------------------------------------------------ ledger
class PaperBook:
    def __init__(self, path: str, cfg: dict):
        self.cfg = cfg
        self.p = cfg["paper"]
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(paper_bets)")}
        if "sport" not in cols:  # databases from before multi-sport
            self.db.execute("ALTER TABLE paper_bets ADD COLUMN sport TEXT DEFAULT 'soccer'")
            self.db.commit()
        self.lock = threading.Lock()
        self._close_written: dict[int, tuple] = {}

    # ---------------------------------------------------------- helpers
    def venue_books(self, venue: str) -> list[str]:
        return list(self.p.get(f"{venue}_books", ["*"]))

    def net(self, price: float, book: str) -> float:
        """Odds after the book's commission or fees."""
        return book_net_odds(price, book, self.cfg)

    @staticmethod
    def strategy_of(a: Alert) -> tuple[str, str]:
        if a.kind == "arb":
            return "arb", "Arbitrage"
        if a.kind == "value":
            return "value", "Wrong odds"
        code = a.extra.get("rule_code") or "pick"
        return code, a.extra.get("rule") or code

    def strategy_enabled(self, code: str) -> bool:
        allowed = self.p.get("strategies", ["*"])
        return "*" in allowed or code in allowed or (code not in ("value", "arb") and "picks" in allowed)

    def settled_profit(self, venue: str) -> float:
        with self.lock:
            r = self.db.execute("SELECT COALESCE(SUM(profit),0) FROM paper_bets WHERE venue=? AND status!='open'",
                                (venue,)).fetchone()
        return float(r[0])

    def stake_for(self, venue: str, a: Alert) -> float:
        if self.p.get("staking", "flat") == "kelly":
            bank = float(self.p["bankroll"]) + self.settled_profit(venue)
            net = self.net(a.price, a.book)
            frac = float(self.p.get("kelly_fraction", 0.25)) * max(a.edge, 0.0) / max(net - 1, 1e-9)
            return round(max(0.0, bank * min(frac, float(self.p.get("max_stake_pct", 0.03)))), 2)
        return float(self.p.get("flat_stake", 10.0))

    # -------------------------------------------------------- placement
    def place(self, alerts: list[Alert], index: EventIndex, now: datetime) -> list[dict]:
        """Turn alerts into imaginary bets: one per signal per place, at that place's best price."""
        min_before = timedelta(minutes=float(self.p.get("min_minutes_before_kickoff", 5)))
        groups: dict[tuple, list[Alert]] = defaultdict(list)
        for a in alerts:
            if a.kickoff - now < min_before:
                continue
            code, _ = self.strategy_of(a)
            if self.strategy_enabled(code):
                groups[(code, a.event_id, a.market, a.outcome)].append(a)
        placed = self._place_arbs([a for a in alerts if a.kind == "arb"], index, now)
        groups = {k: v for k, v in groups.items() if k[0] != "arb"}
        for (code, eid, market, outcome), als in groups.items():
            ev = index.events.get(eid)
            for venue in self.p.get("venues", list(VENUES)):
                if venue == "arb":
                    continue
                books = self.venue_books(venue)
                cands = [a for a in als if "*" in books or a.book in books]
                if not cands:
                    continue
                a = max(cands, key=lambda x: x.edge)
                home, away = (ev.home, ev.away) if ev else (a.match.split(" v ", 1) + [""])[:2]
                meta = {"sources": dict(ev.sources) if ev else {}, **(ev.meta if ev else {})}
                row = {
                    "sport": ev.sport if ev else a.sport,
                    "placed": now.isoformat(timespec="seconds"), "strategy": code,
                    "strategy_name": self.strategy_of(a)[1], "venue": venue, "event_id": eid,
                    "home": home, "away": away, "league": a.extra.get("league") or a.league,
                    "kickoff": a.kickoff.isoformat(), "market": market, "line": a.line, "outcome": outcome,
                    "book": a.book, "price": a.price, "net_price": self.net(a.price, a.book),
                    "fair_odds": a.fair_odds, "ev": a.edge, "stake": self.stake_for(venue, a),
                    "detail": a.detail, "meta": json.dumps(meta)}
                cols = ",".join(row)
                with self.lock:
                    cur = self.db.execute(f"INSERT OR IGNORE INTO paper_bets ({cols}) VALUES "
                                          f"({','.join('?' * len(row))})", list(row.values()))
                    self.db.commit()
                if cur.rowcount:
                    placed.append(row)
                    log.info("paper bet [%s/%s] %s v %s %s %s %s @ %.2f %s (EV %+.1f%%)", venue, code, home,
                             away, market, a.line, outcome, a.price, a.book, 100 * a.edge)
        return placed

    def _place_arbs(self, arbs: list[Alert], index: EventIndex, now: datetime) -> list[dict]:
        """Every leg of an arbitrage, stakes split so each result returns the same; all legs or none."""
        if "arb" not in self.p.get("venues", list(VENUES)) or not self.strategy_enabled("arb"):
            return []
        min_before = timedelta(minutes=float(self.p.get("min_minutes_before_kickoff", 5)))
        placed = []
        for a in sorted(arbs, key=lambda x: -x.edge):
            if a.kickoff - now < min_before:
                continue
            with self.lock:
                seen = self.db.execute("SELECT 1 FROM paper_bets WHERE strategy='arb' AND event_id=? AND market=?",
                                       (a.event_id, a.market)).fetchone()
            if seen:
                continue
            ev = index.events.get(a.event_id)
            home, away = (ev.home, ev.away) if ev else (a.match.split(" v ", 1) + [""])[:2]
            meta = json.dumps({"sources": dict(ev.sources) if ev else {}, **(ev.meta if ev else {})})
            total = float(self.p.get("arb_stake", 100.0))
            rows = [{"sport": ev.sport if ev else a.sport,
                     "placed": now.isoformat(timespec="seconds"), "strategy": "arb", "strategy_name": "Arbitrage",
                     "venue": "arb", "event_id": a.event_id, "home": home, "away": away, "league": a.league,
                     "kickoff": a.kickoff.isoformat(), "market": a.market, "line": a.line, "outcome": lg["outcome"],
                     "book": lg["book"], "price": lg["price"], "net_price": lg["net"], "fair_odds": a.price,
                     "ev": a.edge, "stake": round(total * lg["share"], 2), "detail": a.detail, "meta": meta}
                    for lg in a.extra.get("legs", [])]
            if not rows:
                continue
            cols = ",".join(rows[0])
            with self.lock:
                self.db.executemany(f"INSERT OR IGNORE INTO paper_bets ({cols}) VALUES ({','.join('?' * len(rows[0]))})",
                                    [list(r.values()) for r in rows])
                self.db.commit()
            placed += rows
            log.info("paper arbitrage %s %s %s: %+.2f%% (%s)", a.match, a.market, a.line, 100 * a.edge, a.detail)
        return placed

    # ---------------------------------------------------------- closing
    def open_bets(self, before_kickoff_of: datetime | None = None) -> list[dict]:
        q = "SELECT * FROM paper_bets WHERE status='open'"
        args: tuple = ()
        if before_kickoff_of is not None:
            q += " AND kickoff > ?"
            args = (before_kickoff_of.isoformat(),)
        with self.lock:
            rows = [dict(r) for r in self.db.execute(q + " ORDER BY kickoff", args).fetchall()]
        for r in rows:
            r["meta"] = json.loads(r["meta"] or "{}")
        return rows

    def update_closing(self, index: EventIndex, quotes: list[Quote], now: datetime) -> int:
        """Record the latest pre-kickoff reference price for every open bet (becomes the closing line)."""
        rows = self.open_bets(before_kickoff_of=now)
        if not rows:
            return 0
        a = self.cfg["alerts"]
        refs = list(a.get("reference_books", ["pinnacle"]))
        max_age = timedelta(minutes=float(a.get("max_quote_age_minutes", 30)))
        by_event: dict[str, list[Quote]] = defaultdict(list)
        for q in quotes:
            by_event[q.event].append(q)
        n = 0
        for b in rows:
            ko = parse_time(b["kickoff"])
            eid = b["event_id"]
            if eid not in index.events:
                ev = match_event(b["home"], b["away"], ko, index.events.values(), sport=b.get("sport") or "soccer")
                if ev is None:
                    continue
                eid = ev.id
            qs = [q for q in by_event.get(eid, [])
                  if (q.updated is None or (q.updated < ko and now - q.updated <= max_age))]
            if not qs:
                continue
            fair = closing_fair(qs, b["market"], b["line"], b["outcome"], refs, a.get("max_exchange_spread", 0.05),
                                b.get("sport") or "soccer")
            books = self.venue_books(b["venue"])
            vq = [q for q in qs if q.market == b["market"] and q.outcome == b["outcome"]
                  and _same_line(q.line, b["line"]) and ("*" in books or q.book in books)]
            vbest = max(vq, key=lambda q: self.net(q.price, q.book), default=None)
            if fair is None and vbest is None:
                continue
            key = (round(fair[0], 4) if fair else None, vbest.price if vbest else None)
            last = self._close_written.get(b["id"])
            if last and last[0] == key and (now - last[1]) < timedelta(minutes=10):
                continue
            clv = b["net_price"] / fair[0] - 1 if fair and math.isfinite(fair[0]) else None
            with self.lock:
                self.db.execute(
                    "UPDATE paper_bets SET close_fair=COALESCE(?,close_fair), close_ref=COALESCE(?,close_ref), "
                    "close_how=COALESCE(?,close_how), clv=COALESCE(?,clv), close_price=COALESCE(?,close_price), "
                    "close_book=COALESCE(?,close_book), close_at=? WHERE id=?",
                    (fair[0] if fair else None, fair[1] if fair else None, fair[2] if fair else None, clv,
                     vbest.price if vbest else None, vbest.book if vbest else None,
                     now.isoformat(timespec="seconds"), b["id"]))
                self.db.commit()
            self._close_written[b["id"]] = (key, now)
            n += 1
        return n

    # ------------------------------------------------------- settlement
    def due(self, now: datetime) -> list[dict]:
        cutoff = now - timedelta(minutes=float(self.p.get("settle_after_minutes", 135)))
        return [b for b in self.open_bets() if parse_time(b["kickoff"]) < cutoff]

    def settle(self, now: datetime, sources: list) -> int:
        due = self.due(now)
        if not due:
            return 0
        found: dict[int, dict] = {}
        for src in sources:  # cheapest / fastest first; later sources only see what is still missing
            todo = [b for b in due if b["id"] not in found]
            if not todo:
                break
            try:
                results = src.results(todo, now)
            except Exception as e:  # noqa: BLE001
                log.warning("results from %s failed: %s", getattr(src, "name", src), e)
                continue
            for b in todo:
                r = find_result(b, results)
                if r is not None:
                    found[b["id"]] = r
        void_after = timedelta(days=float(self.p.get("void_after_days", 7)))
        n = 0
        for b in due:
            r = found.get(b["id"])
            if r is not None and "winner" in r:
                res = 1.0 if r["winner"] == b["outcome"] else -1.0
                vals = (STATUS[res], None, None, res, round(b["stake"] * profit(res, b["net_price"]), 4),
                        r.get("source", ""))
            elif r is not None:
                res = settle(b["market"], b["line"], b["outcome"], r["home_goals"], r["away_goals"])
                vals = (STATUS.get(res, "settled"), r["home_goals"], r["away_goals"], res,
                        round(b["stake"] * profit(res, b["net_price"]), 4), r.get("source", ""))
            elif now - parse_time(b["kickoff"]) > void_after:
                vals = ("void", None, None, 0.0, 0.0, "no result found")
            else:
                continue
            with self.lock:
                self.db.execute("UPDATE paper_bets SET status=?, home_goals=?, away_goals=?, result=?, profit=?, "
                                "result_source=?, settled=? WHERE id=?",
                                (*vals, now.isoformat(timespec="seconds"), b["id"]))
                self.db.commit()
            n += 1
        if n:
            log.info("settled %d paper bets", n)
        return n

    # ---------------------------------------------------------- reading
    def bets(self, venue: str | None = None, strategy: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM paper_bets WHERE 1=1", []
        if venue:
            q += " AND venue=?"
            args.append(venue)
        if strategy and strategy != "all":
            q += " AND strategy=?"
            args.append(strategy)
        with self.lock:
            rows = [dict(r) for r in self.db.execute(q + " ORDER BY kickoff, id", args).fetchall()]
        return rows

    def strategies(self) -> list[tuple[str, str]]:
        with self.lock:
            rows = self.db.execute("SELECT strategy, MAX(strategy_name) FROM paper_bets GROUP BY strategy "
                                   "ORDER BY strategy='value' DESC, strategy").fetchall()
        return [(r[0], r[1]) for r in rows]

    def reset(self) -> int:
        with self.lock:
            n = self.db.execute("DELETE FROM paper_bets").rowcount
            self.db.commit()
        self._close_written.clear()
        return n


def summarize(bets: list[dict], start_bankroll: float) -> dict:
    """Headline numbers and the chart series for a set of bets."""
    settled = [b for b in bets if b["status"] not in ("open", "void")]
    open_ = [b for b in bets if b["status"] == "open"]
    staked = sum(b["stake"] for b in settled)
    pl = sum(b["profit"] for b in settled)
    exp_ev = sum(b["stake"] * b["ev"] for b in settled)
    with_clv = [b for b in settled if b["clv"] is not None]
    all_clv = [b for b in bets if b["clv"] is not None and b["status"] != "void"]
    w_ev = sum(b["stake"] for b in bets if b["status"] != "void")
    var = 0.0
    series = []
    cum = {"actual": 0.0, "ev": 0.0, "clv": 0.0}
    for i, b in enumerate(settled, 1):
        o = b["net_price"]
        p = min(max((1 + b["ev"]) / o, 0.01), 0.99)
        var += (b["stake"] * o) ** 2 * p * (1 - p)
        cum["actual"] += b["profit"]
        cum["ev"] += b["stake"] * b["ev"]
        cum["clv"] += b["stake"] * (b["clv"] or 0.0)
        series.append({"n": i, "actual": round(cum["actual"], 2), "ev": round(cum["ev"], 2),
                       "clv": round(cum["clv"], 2), "band": round(2 * math.sqrt(var), 2),
                       "kickoff": b["kickoff"], "match": f"{b['home']} v {b['away']}", "status": b["status"],
                       "profit": round(b["profit"], 2)})
    clvs = [b["clv"] for b in all_clv]
    clv_mean = sum(clvs) / len(clvs) if clvs else None
    clv_t = None
    if len(clvs) > 2:
        sd = math.sqrt(sum((c - clv_mean) ** 2 for c in clvs) / (len(clvs) - 1))
        clv_t = clv_mean / (sd / math.sqrt(len(clvs))) if sd > 0 else None
    return {
        "bets": len(bets), "settled": len(settled), "open": len(open_),
        "void": sum(1 for b in bets if b["status"] == "void"),
        "open_stake": sum(b["stake"] for b in open_), "staked": staked, "profit": pl,
        "roi": pl / staked if staked else None,
        "avg_ev": sum(b["stake"] * b["ev"] for b in bets if b["status"] != "void") / w_ev if w_ev else None,
        "expected_profit": exp_ev,
        "clv_expected_profit": sum(b["stake"] * b["clv"] for b in with_clv),
        "avg_clv": clv_mean, "clv_t": clv_t, "clv_n": len(clvs),
        "beat_close": sum(1 for c in clvs if c > 0) / len(clvs) if clvs else None,
        "luck_sd": math.sqrt(var), "bankroll": start_bankroll + pl, "start_bankroll": start_bankroll,
        "series": series,
    }


def verdict(s: dict) -> str:
    """One plain sentence on what the numbers say so far."""
    n, nc = s["settled"], s["clv_n"]
    if s["bets"] == 0:
        return "No bets yet. They are placed automatically when an alert fires."
    parts = []
    if nc >= 30 and s["clv_t"] is not None:
        if s["clv_t"] >= 2:
            parts.append(f"Beating the closing line by {s['avg_clv']:+.1%} on average over {nc} bets, "
                         "which is the best early sign of a real edge.")
        elif s["clv_t"] <= -2:
            parts.append(f"Losing to the closing line by {s['avg_clv']:+.1%} on average over {nc} bets: "
                         "the prices taken are worse than the market's final view.")
        else:
            parts.append(f"Closing-line value {s['avg_clv']:+.1%} over {nc} bets, not yet clearly above zero.")
    else:
        parts.append(f"Only {nc} bets with a closing line so far; about 30 are needed for a first read on CLV.")
    if n >= 50 and s["luck_sd"] > 0:
        z = s["profit"] / s["luck_sd"]
        if z >= 2:
            parts.append("Profit is above what luck alone would give.")
        elif z <= -2:
            parts.append("The loss is bigger than bad luck alone would explain.")
        else:
            parts.append("Profit so far is within the normal luck range, so it can't prove anything yet.")
    elif n:
        parts.append(f"{n} settled bets is too few to judge profit.")
    return " ".join(parts)
