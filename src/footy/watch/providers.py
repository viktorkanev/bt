"""Live odds providers (standard library HTTP only).

* :class:`OddsAPIProvider`  the-odds-api.com v4: one feed with Pinnacle, Betfair
  Exchange and many other bookmakers. Credits: markets x regions per odds call;
  the events list is used for scheduling.
* :class:`BetfairProvider`  Betfair Exchange API-NG (needs an app key; the free
  delayed key returns prices 1-180 s late, a live key is a one-off fee).
* :class:`FileProvider`     replays a JSON snapshot (tests, demos, or a feed
  written by another tool, e.g. a BetInAsia/Mollybet script).

Each provider's ``poll(index)`` returns a list of :class:`Quote` and registers
events in the shared :class:`EventIndex`.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from .core import EventIndex, Quote, parse_time, utcnow

log = logging.getLogger(__name__)
UA = {"User-Agent": "footy-watch/0.1"}


def http_json(url: str, data: bytes | None = None, headers: dict | None = None,
              timeout: float = 30.0) -> tuple[object, dict]:
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8")
        return json.loads(body) if body else None, {k.lower(): v for k, v in r.headers.items()}


# ------------------------------------------------------------- the-odds-api
class OddsAPIProvider:
    name = "oddsapi"
    MARKETS = {"h2h": "1x2", "totals": "ou", "spreads": "ah"}

    def __init__(self, cfg: dict):
        self.key = cfg.get("api_key", "")
        self.base = cfg.get("base_url", "https://api.the-odds-api.com/v4")
        self.sports: dict[str, str] = dict(cfg.get("sports", {}))
        self.regions = cfg.get("regions", "eu")
        self.markets = list(cfg.get("markets", ["h2h", "totals", "spreads"]))
        self.bookmakers = list(cfg.get("bookmakers", []))
        self.monthly_credits = int(cfg.get("monthly_credits", 500))
        self.min_interval = float(cfg.get("min_interval_seconds", 300))
        self.lookahead_h = float(cfg.get("lookahead_hours", 48))
        self.events_refresh_s = float(cfg.get("events_refresh_minutes", 60)) * 60
        self.remaining: int | None = None
        self.used: int | None = None
        self._events: dict[str, list[dict]] = {}
        self._events_at = 0.0
        self.bad_sports: set[str] = set()
        self.scores_refresh_s = float(cfg.get("scores_refresh_minutes", 180)) * 60
        self._scores: dict[str, tuple[float, list[dict]]] = {}
        self.last_error = ""

    def _url(self, path: str, **params) -> str:
        params = {k: v for k, v in params.items() if v not in (None, "", [])}
        return f"{self.base}{path}?" + urllib.parse.urlencode({"apiKey": self.key, **params})

    def _track(self, headers: dict) -> None:
        if "x-requests-remaining" in headers:
            try:
                self.remaining = int(float(headers["x-requests-remaining"]))
                self.used = int(float(headers.get("x-requests-used", 0)))
            except ValueError:
                pass

    def list_sports(self) -> list[dict]:
        data, h = http_json(self._url("/sports", all="true"))
        self._track(h)
        return [s for s in data if str(s.get("key", "")).startswith("soccer")]

    def refresh_events(self, force: bool = False) -> None:
        if not force and time.time() - self._events_at < self.events_refresh_s:
            return
        for sport in self.sports:
            if sport in self.bad_sports:
                continue
            try:
                data, h = http_json(self._url(f"/sports/{sport}/events", dateFormat="iso"))
                self._track(h)
                self._events[sport] = data or []
            except urllib.error.HTTPError as e:
                if e.code in (404, 422):
                    log.warning("the-odds-api: unknown sport key %s (check `footy-watch sports`)", sport)
                    self.bad_sports.add(sport)
                else:
                    self.last_error = f"events {sport}: HTTP {e.code}"
                    log.warning(self.last_error)
            except Exception as e:  # noqa: BLE001
                self.last_error = f"events {sport}: {e}"
                log.warning(self.last_error)
        self._events_at = time.time()

    def active_sports(self, now: datetime) -> list[str]:
        horizon = now + timedelta(hours=self.lookahead_h)
        out = []
        for sport, evs in self._events.items():
            if any(now - timedelta(hours=2) <= parse_time(e["commence_time"]) <= horizon for e in evs):
                out.append(sport)
        return out

    def cost_per_sweep(self, n_sports: int) -> int:
        regions = max(1, -(-len(self.bookmakers) // 10)) if self.bookmakers else len(self.regions.split(","))
        return n_sports * len(self.markets) * regions

    def interval(self, n_sports: int, now: datetime) -> float:
        """Seconds between sweeps so the monthly credits last until the reset."""
        if n_sports == 0:
            return self.min_interval
        nxt = (now.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0)
        secs_left = max((nxt - now).total_seconds(), 3600)
        credits = self.remaining if self.remaining is not None else self.monthly_credits
        sweeps = max(credits - 10, 1) / self.cost_per_sweep(n_sports)
        return max(self.min_interval, secs_left / sweeps)

    def poll(self, index: EventIndex, now: datetime | None = None) -> list[Quote]:
        now = now or utcnow()
        self.refresh_events()
        quotes: list[Quote] = []
        for sport in self.active_sports(now):
            try:
                data, h = http_json(self._url(
                    f"/sports/{sport}/odds", regions=None if self.bookmakers else self.regions,
                    bookmakers=",".join(self.bookmakers), markets=",".join(self.markets),
                    oddsFormat="decimal", dateFormat="iso",
                    commenceTimeTo=(now + timedelta(hours=self.lookahead_h)).strftime("%Y-%m-%dT%H:%M:%SZ")))
                self._track(h)
            except urllib.error.HTTPError as e:
                self.last_error = f"odds {sport}: HTTP {e.code}"
                log.warning(self.last_error)
                if e.code in (401, 429):
                    break
                continue
            except Exception as e:  # noqa: BLE001
                self.last_error = f"odds {sport}: {e}"
                log.warning(self.last_error)
                continue
            quotes += self.parse(data or [], index, self.sports.get(sport), sport)
        return quotes

    def event_odds(self, index: EventIndex, sport: str, event_id: str) -> list[Quote]:
        """One match's prices (closing-line snapshot for paper bets). Same credit cost per market."""
        data, h = http_json(self._url(
            f"/sports/{sport}/events/{event_id}/odds", regions=None if self.bookmakers else self.regions,
            bookmakers=",".join(self.bookmakers), markets=",".join(self.markets),
            oddsFormat="decimal", dateFormat="iso"))
        self._track(h)
        return self.parse([data] if isinstance(data, dict) else (data or []), index, self.sports.get(sport), sport)

    def results(self, bets: list[dict], now: datetime) -> list[dict]:
        """Final scores for paper bets (scores endpoint, 2 credits per league per call)."""
        recent = [b for b in bets if now - parse_time(b["kickoff"]) < timedelta(days=3)]  # the API's limit
        sports = {b["meta"].get("oddsapi_sport") for b in recent if b.get("meta", {}).get("oddsapi_sport")}
        out = []
        for sport in sorted(sports):
            cached = self._scores.get(sport)
            if cached is None or time.time() - cached[0] > self.scores_refresh_s:
                try:
                    data, h = http_json(self._url(f"/sports/{sport}/scores", daysFrom=3, dateFormat="iso"))
                    self._track(h)
                    cached = (time.time(), self.parse_scores(data or []))
                    self._scores[sport] = cached
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"scores {sport}: {e}"
                    log.warning(self.last_error)
                    continue
            out += cached[1]
        return out

    @staticmethod
    def parse_scores(data: list[dict]) -> list[dict]:
        out = []
        for e in data:
            if not e.get("completed") or not e.get("scores"):
                continue
            sc = {s["name"]: s["score"] for s in e["scores"]}
            try:
                hg, ag = int(sc[e["home_team"]]), int(sc[e["away_team"]])
            except (KeyError, TypeError, ValueError):
                continue
            out.append({"source": "the-odds-api", "id": e.get("id"), "home": e["home_team"], "away": e["away_team"],
                        "kickoff": parse_time(e["commence_time"]), "home_goals": hg, "away_goals": ag})
        return out

    @classmethod
    def parse(cls, data: list[dict], index: EventIndex, league: str | None = None,
              sport: str | None = None) -> list[Quote]:
        out: list[Quote] = []
        for e in data:
            home, away = e["home_team"], e["away_team"]
            ev = index.add(cls.name, e["id"], home, away, parse_time(e["commence_time"]), league)
            if sport:
                ev.meta["oddsapi_sport"] = sport
            for bk in e.get("bookmakers", []):
                book = bk["key"]
                for mk in bk.get("markets", []):
                    market = cls.MARKETS.get(mk["key"])
                    if market is None:
                        continue
                    upd = parse_time(mk.get("last_update") or bk.get("last_update")) \
                        if (mk.get("last_update") or bk.get("last_update")) else None
                    for o in mk.get("outcomes", []):
                        name, price, point = o.get("name"), float(o["price"]), o.get("point")
                        if market == "1x2":
                            outcome = "home" if name == home else "away" if name == away else "draw"
                            out.append(Quote(ev.id, "1x2", None, outcome, book, price, upd))
                        elif market == "ou":
                            out.append(Quote(ev.id, "ou", float(point), str(name).lower(), book, price, upd))
                        else:  # spreads: point is that team's handicap -> express as home line
                            if name == home:
                                out.append(Quote(ev.id, "ah", float(point), "home", book, price, upd))
                            elif name == away:
                                out.append(Quote(ev.id, "ah", -float(point), "away", book, price, upd))
        return out


# ------------------------------------------------------------------ Betfair
class BetfairProvider:
    name = "betfair"
    LOGIN = "https://identitysso.betfair.com/api/login"
    KEEPALIVE = "https://identitysso.betfair.com/api/keepAlive"
    API = "https://api.betfair.com/exchange/betting/rest/v1.0"
    MARKET_TYPES = ("MATCH_ODDS", "OVER_UNDER_25", "ASIAN_HANDICAP")

    def __init__(self, cfg: dict):
        self.app_key = cfg.get("app_key", "")
        self.username = cfg.get("username", "")
        self.password = cfg.get("password", "")
        self.book = cfg.get("book_name", "betfair")
        self.competitions = [str(c) for c in cfg.get("competitions", [])]
        self.lookahead_h = float(cfg.get("lookahead_hours", 48))
        self.min_interval = float(cfg.get("interval_seconds", 30))
        self.catalogue_refresh_s = float(cfg.get("catalogue_refresh_minutes", 15)) * 60
        self.token = ""
        self._login_at = 0.0
        self._catalogue: dict[str, dict] = {}
        self._catalogue_at = 0.0
        self.last_error = ""

    def interval(self, n: int = 0, now: datetime | None = None) -> float:
        return self.min_interval

    def _login(self) -> None:
        if self.token and time.time() - self._login_at < 3 * 3600:
            return
        body = urllib.parse.urlencode({"username": self.username, "password": self.password}).encode()
        data, _ = http_json(self.LOGIN, body, {"X-Application": self.app_key, "Accept": "application/json",
                                               "Content-Type": "application/x-www-form-urlencoded"})
        if not isinstance(data, dict) or data.get("status") != "SUCCESS":
            raise RuntimeError(f"Betfair login failed: {data}")
        self.token, self._login_at = data["token"], time.time()

    def _call(self, op: str, params: dict):
        self._login()
        data, _ = http_json(f"{self.API}/{op}/", json.dumps(params).encode(), {
            "X-Application": self.app_key, "X-Authentication": self.token,
            "Content-Type": "application/json", "Accept": "application/json"})
        return data

    def _refresh_catalogue(self, now: datetime) -> None:
        if time.time() - self._catalogue_at < self.catalogue_refresh_s and self._catalogue:
            return
        flt = {"eventTypeIds": ["1"], "marketTypeCodes": list(self.MARKET_TYPES),
               "marketStartTime": {"from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   "to": (now + timedelta(hours=self.lookahead_h)).strftime("%Y-%m-%dT%H:%M:%SZ")}}
        if self.competitions:
            flt["competitionIds"] = self.competitions
        cat = self._call("listMarketCatalogue", {
            "filter": flt, "maxResults": 1000, "sort": "FIRST_TO_START",
            "marketProjection": ["EVENT", "COMPETITION", "MARKET_START_TIME", "RUNNER_DESCRIPTION",
                                 "MARKET_DESCRIPTION"]})
        self._catalogue = {m["marketId"]: m for m in cat or []}
        self._catalogue_at = time.time()

    def poll(self, index: EventIndex, now: datetime | None = None) -> list[Quote]:
        now = now or utcnow()
        self._refresh_catalogue(now)
        ids = list(self._catalogue)
        books = []
        for i in range(0, len(ids), 40):  # EX_BEST_OFFERS weight allows ~40 markets per call
            books += self._call("listMarketBook", {
                "marketIds": ids[i:i + 40],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"],
                                    "exBestOffersOverrides": {"bestPricesDepth": 1}}}) or []
        return self.parse(list(self._catalogue.values()), books, index, self.book)

    @classmethod
    def parse(cls, catalogue: list[dict], books: list[dict], index: EventIndex,
              book: str = "betfair") -> list[Quote]:
        cat = {m["marketId"]: m for m in catalogue}
        out: list[Quote] = []
        for mb in books:
            m = cat.get(mb.get("marketId"))
            if not m or mb.get("status") not in (None, "OPEN") or mb.get("inplay"):
                continue
            ev_name = m["event"]["name"]
            if " v " not in ev_name:
                continue
            home, away = [x.strip() for x in ev_name.split(" v ", 1)]
            ev = index.add(cls.name, m["event"]["id"], home, away,
                           parse_time(m.get("marketStartTime") or m["event"]["openDate"]))
            names = {(r["selectionId"], float(r.get("handicap", 0.0))): r["runnerName"]
                     for r in m.get("runners", [])}
            sel_names = {r["selectionId"]: r["runnerName"] for r in m.get("runners", [])}
            mtype = m.get("description", {}).get("marketType") or m.get("marketType") or \
                ("MATCH_ODDS" if m.get("marketName") == "Match Odds" else "")
            if not mtype:
                mtype = {"Match Odds": "MATCH_ODDS", "Over/Under 2.5 Goals": "OVER_UNDER_25",
                         "Asian Handicap": "ASIAN_HANDICAP"}.get(m.get("marketName"), "")
            for r in mb.get("runners", []):
                if r.get("status") not in (None, "ACTIVE"):
                    continue
                ex = r.get("ex", {})
                back = ex.get("availableToBack") or []
                lay = ex.get("availableToLay") or []
                if not back:
                    continue
                price, size = float(back[0]["price"]), float(back[0].get("size", 0))
                lay_p = float(lay[0]["price"]) if lay else None
                hc = float(r.get("handicap", 0.0))
                rn = names.get((r["selectionId"], hc)) or sel_names.get(r["selectionId"], "")
                if mtype == "MATCH_ODDS":
                    outcome = "draw" if rn.lower() in ("the draw", "draw") else \
                        "home" if rn == home else "away" if rn == away else None
                    if outcome:
                        out.append(Quote(ev.id, "1x2", None, outcome, book, price, utcnow(), lay_p, size))
                elif mtype == "OVER_UNDER_25":
                    outcome = "over" if rn.lower().startswith("over") else "under"
                    out.append(Quote(ev.id, "ou", 2.5, outcome, book, price, utcnow(), lay_p, size))
                elif mtype == "ASIAN_HANDICAP":
                    if rn == home:
                        out.append(Quote(ev.id, "ah", hc, "home", book, price, utcnow(), lay_p, size))
                    elif rn == away:
                        out.append(Quote(ev.id, "ah", -hc, "away", book, price, utcnow(), lay_p, size))
        return out


# --------------------------------------------------------------------- file
class FileProvider:
    """Reads ``{"events": [...], "quotes": [...], "results": [...]}`` from a JSON file each poll.

    events: {id, home, away, kickoff (ISO), league?}
    quotes: {event, market, line, outcome, book, price, updated?, lay?, size?}
    """

    name = "file"

    def __init__(self, cfg: dict):
        self.path = cfg.get("path", "odds.json")
        self.min_interval = float(cfg.get("interval_seconds", 30))
        self.last_error = ""

    def interval(self, n: int = 0, now: datetime | None = None) -> float:
        return self.min_interval

    def poll(self, index: EventIndex, now: datetime | None = None) -> list[Quote]:
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        ids = {}
        for e in data.get("events", []):
            ev = index.add(self.name, str(e["id"]), e["home"], e["away"], parse_time(e["kickoff"]),
                           e.get("league"))
            ids[str(e["id"])] = ev.id
        out = []
        for q in data.get("quotes", []):
            out.append(Quote(ids[str(q["event"])], q["market"], q.get("line"), q["outcome"], q["book"],
                             float(q["price"]), parse_time(q["updated"]) if q.get("updated") else utcnow(),
                             q.get("lay"), q.get("size")))
        return out

    def results(self, bets: list[dict], now: datetime) -> list[dict]:
        """Optional ``"results": [{home, away, kickoff, home_goals, away_goals}]`` in the same file."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return []
        return [{"source": "file", "home": r["home"], "away": r["away"], "kickoff": parse_time(r["kickoff"]),
                 "home_goals": int(r["home_goals"]), "away_goals": int(r["away_goals"])}
                for r in data.get("results", [])]


PROVIDERS = {"oddsapi": OddsAPIProvider, "betfair": BetfairProvider, "file": FileProvider}


def build(cfg: dict) -> list:
    out = []
    for name, cls in PROVIDERS.items():
        section = cfg.get(name, {})
        if section.get("enabled"):
            out.append(cls(section))
    return out

