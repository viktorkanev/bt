"""Odds-watcher core: data model, margin removal, settlement and team matching.

Standard library only, so it runs on a Raspberry Pi Zero 2 W without pandas.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

OUTCOMES = {"1x2": ("home", "draw", "away"), "ou": ("over", "under"), "ah": ("home", "away")}


@dataclass
class Event:
    id: str                 # canonical id (shared across providers)
    home: str
    away: str
    kickoff: datetime       # timezone-aware UTC
    league: str | None = None
    sources: dict = field(default_factory=dict)  # provider -> provider event id
    meta: dict = field(default_factory=dict)     # provider extras, e.g. the-odds-api sport key


@dataclass
class Quote:
    event: str              # canonical event id
    market: str             # "1x2" | "ou" | "ah"
    line: float | None      # ou: goals line; ah: HOME handicap
    outcome: str            # home | draw | away | over | under
    book: str
    price: float            # decimal back price
    updated: datetime | None = None
    lay: float | None = None
    size: float | None = None

    @property
    def key(self) -> tuple:
        return (self.event, self.market, self.line, self.book)


# --------------------------------------------------------------- margins
def devig(prices: list[float], method: str = "power") -> list[float]:
    """Remove the bookmaker margin. ``power`` handles the favourite-longshot bias."""
    q = [1.0 / p for p in prices]
    if method == "proportional" or len(q) < 2:
        s = sum(q)
        return [x / s for x in q]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        s = sum(x ** k for x in q)
        if s < 1:
            hi = k
        else:
            lo = k
    k = (lo + hi) / 2
    p = [x ** k for x in q]
    s = sum(p)
    return [x / s for x in p]


def net_odds(price: float, commission: float = 0.0) -> float:
    """Decimal odds after commission on net winnings (exchanges)."""
    return 1 + (price - 1) * (1 - commission)


def gross_needed(net: float, commission: float = 0.0) -> float:
    """Quoted price needed to get ``net`` odds after commission."""
    return 1 + (net - 1) / (1 - commission)


def book_net_odds(price: float, book: str, cfg: dict) -> float:
    """Decimal odds you really get at ``book``.

    Exchanges charge commission on winnings (``cfg["commission"]``). Prediction
    markets such as Polymarket charge takers ``rate * p * (1 - p)`` per share
    bought at price ``p`` (``cfg["fees"]``), so a share costs ``p + fee``.
    """
    rate = float(cfg.get("fees", {}).get(book, 0.0) or 0.0)
    if rate:
        p = 1.0 / price
        return 1.0 / (p + rate * p * (1 - p))
    return net_odds(price, float(cfg.get("commission", {}).get(book, 0.0) or 0.0))


def book_gross_needed(net: float, book: str, cfg: dict) -> float:
    """Quoted price needed at ``book`` to get ``net`` odds after its fees."""
    if not math.isfinite(net):
        return net
    rate = float(cfg.get("fees", {}).get(book, 0.0) or 0.0)
    if rate:
        # solve p + rate*p*(1-p) = 1/net for the share price p
        a, b, c = rate, -(1 + rate), 1.0 / net
        p = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
        return 1.0 / p
    return gross_needed(net, float(cfg.get("commission", {}).get(book, 0.0) or 0.0))


# ------------------------------------------------------------ settlement
def _sign(x: float) -> float:
    return 1.0 if x > 1e-9 else -1.0 if x < -1e-9 else 0.0


def ah_result(goal_diff: int, home_line: float, side: str) -> float:
    """Fraction of stake won (+) or lost (-) for an Asian handicap bet.

    ``home_line`` is the home team's handicap (-0.5 = home gives half a goal);
    quarter lines are split into two half-stakes.
    """
    if round(home_line * 4) % 2 != 0:
        r = 0.5 * _sign(goal_diff + home_line - 0.25) + 0.5 * _sign(goal_diff + home_line + 0.25)
    else:
        r = _sign(goal_diff + home_line)
    return r if side == "home" else -r


def ah_ev(gd_dist: dict[int, float], home_line: float, side: str, odds: float) -> float:
    """Expected profit per unit stake given a goal-difference distribution."""
    ev = 0.0
    for gd, p in gd_dist.items():
        r = ah_result(int(gd), home_line, side)
        ev += p * (r * (odds - 1) if r > 0 else r)
    return ev


def ah_break_even(gd_dist: dict[int, float], home_line: float, side: str, edge: float = 0.0) -> float:
    """Lowest (net) odds at which the bet's expected profit reaches ``edge``; 0 = break-even."""
    win = lose = 0.0
    for gd, p in gd_dist.items():
        r = ah_result(int(gd), home_line, side)
        if r > 0:
            win += p * r
        elif r < 0:
            lose += p * -r
    return float("inf") if win <= 0 else 1 + (lose + edge) / win


def ou_result(total_goals: int, line: float, side: str) -> float:
    """Over/under settlement, quarter lines included (2.25, 2.75, ...)."""
    r = ah_result(total_goals, -line, "home")
    return r if side == "over" else -r


def settle(market: str, line: float | None, outcome: str, home_goals: int, away_goals: int) -> float:
    """Fraction of the stake won (+1, +0.5), pushed (0) or lost (-0.5, -1)."""
    if market == "1x2":
        res = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
        return 1.0 if res == outcome else -1.0
    if market == "ou":
        return ou_result(home_goals + away_goals, float(line), outcome)
    return ah_result(home_goals - away_goals, float(line), outcome)


def profit(result: float, odds: float) -> float:
    """Profit per unit stake for a settled result at (net) decimal odds."""
    return result * (odds - 1) if result > 0 else result


# ----------------------------------------------- converting between lines
def score_grid(lh: float, la: float, max_goals: int = 10) -> list[tuple[int, int, float]]:
    """Independent Poisson scorelines, used to move a price from one line to another."""
    def pois(lam):
        p, out = math.exp(-lam), []
        for k in range(max_goals + 1):
            out.append(p)
            p *= lam / (k + 1)
        return out
    ph, pa = pois(lh), pois(la)
    return [(i, j, ph[i] * pa[j]) for i in range(max_goals + 1) for j in range(max_goals + 1)]


def grid_fair_odds(grid, market: str, line: float | None, outcome: str) -> float:
    """No-margin price of a selection under a scoreline distribution (pushes handled)."""
    win = lose = 0.0
    for i, j, p in grid:
        r = settle(market, line, outcome, i, j)
        if r > 0:
            win += p * r
        elif r < 0:
            lose += p * -r
    return float("inf") if win <= 0 else 1 + lose / win


def _bisect(f, lo: float, hi: float, iters: int = 40) -> float:
    """Root of an increasing function on [lo, hi] (clamped to the ends)."""
    if f(lo) >= 0:
        return lo
    if f(hi) <= 0:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_goals(x12: tuple | None = None, ah: tuple | None = None, ou: tuple | None = None,
              total: float = 2.6) -> tuple[float, float]:
    """Home/away goal means matching a market's no-margin prices.

    x12 = (p_home, p_draw, p_away); ah = (home_line, fair_home_odds); ou = (line, fair_over_odds).
    Supremacy comes from the handicap (or 1X2), the total from over/under (else ``total``).
    """
    sup = 0.0

    def lams(t, s):
        return max((t + s) / 2, 0.02), max((t - s) / 2, 0.02)

    for _ in range(3):
        if ou:
            line, fair_over = ou
            # fair over price falls as the total rises -> increasing function of t
            total = _bisect(lambda t: fair_over - grid_fair_odds(score_grid(*lams(t, sup)), "ou", line, "over"),
                            0.5, 7.0)
        if ah:
            line, fair_home = ah
            sup = _bisect(lambda s: fair_home - grid_fair_odds(score_grid(*lams(total, s)), "ah", line, "home"),
                          -total + 0.05, total - 0.05)
        elif x12:
            target = x12[0] - x12[2]

            def diff(s):
                g = score_grid(*lams(total, s))
                return sum(p for i, j, p in g if i > j) - sum(p for i, j, p in g if i < j) - target
            sup = _bisect(diff, -total + 0.05, total - 0.05)
    return lams(total, sup)


# --------------------------------------------------------------- names
_STOP = {"fc", "afc", "cf", "sc", "ssc", "cd", "ud", "sd", "rcd", "club", "calcio", "fk", "sk",
         "bk", "if", "ff", "the", "de", "1", "ac", "as", "us", "sv", "vfl", "vfb", "tsg", "fsv"}
_ALIASES = {
    "man united": "manchester united", "man utd": "manchester united", "manchester utd": "manchester united",
    "man city": "manchester city", "nott m forest": "nottingham forest", "nottm forest": "nottingham forest",
    "sheffield weds": "sheffield wednesday", "sheffield utd": "sheffield united", "qpr": "queens park rangers",
    "wolves": "wolverhampton wanderers", "wolverhampton": "wolverhampton wanderers",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur", "brighton": "brighton and hove albion",
    "west brom": "west bromwich albion", "newcastle": "newcastle united", "leeds": "leeds united",
    "ath bilbao": "athletic bilbao", "athletic club": "athletic bilbao", "ath madrid": "atletico madrid",
    "sociedad": "real sociedad", "betis": "real betis", "celta": "celta vigo", "espanol": "espanyol",
    "m gladbach": "borussia monchengladbach", "monchengladbach": "borussia monchengladbach",
    "ein frankfurt": "eintracht frankfurt", "fc koln": "koln", "fc cologne": "koln", "cologne": "koln",
    "paris sg": "paris saint germain", "psg": "paris saint germain", "st etienne": "saint etienne",
    "inter": "inter milan", "internazionale": "inter milan", "milan": "ac milan",
    "dortmund": "borussia dortmund", "leverkusen": "bayer leverkusen", "bayern munich": "bayern munchen",
    "bayern": "bayern munchen", "rb leipzig": "leipzig", "rasenballsport leipzig": "leipzig",
    "sp lisbon": "sporting lisbon", "sporting cp": "sporting lisbon", "porto": "fc porto",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ALIASES.get(s, s)
    toks = [t for t in s.split() if t not in _STOP]
    s = " ".join(toks) if toks else s
    s = s.replace(" utd", " united")
    return _ALIASES.get(s, s)


def name_score(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    contain = 0.9 if short and short <= long_ else 0.0
    return max(SequenceMatcher(None, a, b).ratio(), contain)


def match_event(home: str, away: str, kickoff: datetime, events, tol_hours: float = 3.0,
                min_score: float = 0.72):
    """Best event with the same teams kicking off within ``tol_hours``."""
    best, best_s = None, 0.0
    for ev in events:
        if abs((ev.kickoff - kickoff).total_seconds()) > tol_hours * 3600:
            continue
        sh, sa = name_score(home, ev.home), name_score(away, ev.away)
        if min(sh, sa) < min_score:
            continue
        if sh + sa > best_s:
            best, best_s = ev, sh + sa
    return best


class EventIndex:
    """Merges the same match seen by different providers into one canonical event."""

    def __init__(self):
        self.events: dict[str, Event] = {}
        self._by_source: dict[tuple[str, str], str] = {}

    def add(self, provider: str, provider_id: str, home: str, away: str, kickoff: datetime,
            league: str | None = None) -> Event:
        cid = self._by_source.get((provider, provider_id))
        if cid:
            ev = self.events[cid]
            ev.kickoff = kickoff
            return ev
        ev = match_event(home, away, kickoff, self.events.values(), tol_hours=2.0, min_score=0.8)
        if ev is None:
            cid = f"{kickoff:%Y%m%d}-{normalize(home)}-{normalize(away)}".replace(" ", "_")
            ev = Event(cid, home, away, kickoff, league)
            self.events[cid] = ev
        ev.sources[provider] = provider_id
        if league and not ev.league:
            ev.league = league
        self._by_source[(provider, provider_id)] = ev.id
        return ev

    def prune(self, before: datetime) -> None:
        for cid in [k for k, e in self.events.items() if e.kickoff < before]:
            ev = self.events.pop(cid)
            for prov, pid in ev.sources.items():
                self._by_source.pop((prov, pid), None)


def parse_time(s: str) -> datetime:
    """ISO-8601 (with Z or offset) to an aware UTC datetime."""
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def within(kickoff: datetime, now: datetime, min_minutes: float, hours: float) -> bool:
    return now + timedelta(minutes=min_minutes) <= kickoff <= now + timedelta(hours=hours)
