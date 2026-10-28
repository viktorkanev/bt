"""OddsPapi (oddspapi.io) feed: 350+ bookmakers in any sport, including Pinnacle and the
Asian books BetInAsia offers (SBOBet, Singbet).

The free plan allows 250 requests a month, so this provider:

* caches the sport list, market catalogue and fixture list on disk (they change slowly);
* fetches odds with ``odds-by-tournaments``: up to 10 leagues per request, one bookmaker each
  (the API refuses more than one);
* counts its requests and spaces sweeps so the monthly allowance lasts until the 1st.

Market ids differ per sport, so they are resolved from the catalogue (``/markets``)
by outcome names and handicap: 1/X/2 -> "1x2", 1/2 -> "ml" (or "ah" with a
handicap), Over/Under -> "ou". Halves, periods, sets, corners and the like are skipped.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

from . import providers as P
from .core import EventIndex, Quote, orient, parse_time, utcnow

log = logging.getLogger(__name__)

SPORT_NAMES = {"soccer": ("soccer", "football"), "basketball": ("basketball",), "baseball": ("baseball",),
               "icehockey": ("ice hockey", "hockey", "icehockey"), "tennis": ("tennis",),
               "volleyball": ("volleyball",), "rugbyunion": ("rugby union", "rugby"),
               "rugbyleague": ("rugby league",), "americanfootball": ("american football", "nfl")}
HOME = {"1", "home", "w1", "player 1", "{$competitor1}", "competitor1", "team 1"}
AWAY = {"2", "away", "w2", "player 2", "{$competitor2}", "competitor2", "team 2"}
DRAW = {"x", "draw", "tie"}
PART_WORDS = ("half", "1st", "2nd", "3rd", "4th", "5th", "quarter", "period", " set", "set ", "map ", "frame",
              "first 5", "first five", "f5 ",
              "corner", "card", "booking", "team total", "home total", "away total", "exact", "race to", "first",
              "player", "odd/even", "odd / even", "margin", "both teams", "double chance", "next", "minute",
              "goalscorer", "range", "anytime", "highest", "method", "round")
FULL_PERIODS = {"", "ft", "fulltime", "full time", "full-time", "result", "match", "game", "regular time", "rt",
                "ot", "incl. overtime", "including overtime", "fullgame", "full game", "0"}


# OddsPapi marketType codes of the full-match markets we compare (soccer catalogue, 2026-10-28).
# Other types reuse the same outcome names but settle differently: team totals ("Over Under
# Team 1"), "2Up" early payout, "To Win From Behind", "To Qualify" (extra time and penalties).
FULL_MATCH_TYPES = {"1x2": {"1x2"}, "ou": {"totals"}, "ah": {"spreads", "drawnobet"}}
NOT_MATCH_TYPES = ("teamtotals", "2up", "winfrombehind", "toqualify")
MAP_VERSION = 2  # bump when the mapping rules change, so market maps cached on disk are rebuilt


def map_market(m: dict) -> dict | None:
    """Catalogue entry -> {"market", "line", "outcomes": {outcomeId: our outcome}} or None."""
    mm = _map_by_names(m)
    mtype = str(m.get("marketType") or "").lower()
    if mm is None or not mtype:  # a catalogue without type codes: the names decide
        return mm
    if any(t in mtype for t in NOT_MATCH_TYPES):
        return None
    return mm if mtype in FULL_MATCH_TYPES.get(mm["market"], {mtype}) else None


def _map_by_names(m: dict) -> dict | None:
    name = str(m.get("marketName") or m.get("name") or "").lower()
    mtype = str(m.get("marketType") or "").lower()
    period = str(m.get("period") or "").lower().strip()
    if any(w in f" {name} " for w in PART_WORDS) or period not in FULL_PERIODS:
        return None
    outs = {}
    for o in m.get("outcomes") or []:
        outs[str(o.get("outcomeId"))] = str(o.get("outcomeName") or o.get("name") or "").strip().lower()
    names = set(outs.values())
    hc = m.get("handicap")
    try:
        hc = None if hc in (None, "") else float(hc)
    except (TypeError, ValueError):
        return None
    is_hcp = any(w in name for w in ("handicap", "spread", "run line", "puck line", "point line")) or \
        any(w in mtype for w in ("handicap", "spread", "line"))

    def side(v):
        return "home" if v in HOME else "away" if v in AWAY else "draw" if v in DRAW else None
    if len(outs) == 3 and {side(v) for v in names} == {"home", "draw", "away"} and not hc:
        if "handicap" in name:
            return None  # European (3-way) handicap
        return {"market": "1x2", "line": None, "outcomes": {k: side(v) for k, v in outs.items()}}
    if len(outs) == 2 and {side(v) for v in names} == {"home", "away"}:
        if is_hcp or "draw no bet" in name:
            line = 0.0 if hc is None else hc   # draw no bet = handicap 0
            return {"market": "ah", "line": line, "outcomes": {k: side(v) for k, v in outs.items()}}
        if hc in (None, 0.0):
            return {"market": "ml", "line": None, "outcomes": {k: side(v) for k, v in outs.items()}}
        return None
    if len(outs) == 2 and names == {"over", "under"} and hc is not None:
        return {"market": "ou", "line": hc, "outcomes": dict(outs)}
    return None


def _latest(o) -> dict | None:
    """The current price object of an outcome (a dict, or a list of snapshots oldest first)."""
    if not isinstance(o, dict):
        return None
    if "price" in o and not isinstance(o.get("price"), (dict, list)):
        return o
    players = o.get("players") or {}
    x = players.get("0") if isinstance(players, dict) else None
    if x is None and isinstance(players, dict) and players:
        x = next(iter(players.values()))
    if isinstance(x, list):
        x = x[-1] if x else None
    return x if isinstance(x, dict) else None


class OddsPapiProvider:
    name = "oddspapi"
    BASE = "https://api.oddspapi.io/v4"

    def __init__(self, cfg: dict):
        self.key = cfg.get("api_key", "")
        self.base = cfg.get("base_url", self.BASE)
        self.sports: dict[str, list] = {k: [str(t) for t in (v or [])] for k, v in cfg.get("sports", {}).items()}
        self.bookmakers = list(cfg.get("bookmakers", ["pinnacle", "singbet", "sbobet"]))[:3]
        self.book_names = dict(cfg.get("book_names", {"betfair-ex": "betfair"}))
        self.monthly = int(cfg.get("monthly_requests", 250))
        self.reserve = int(cfg.get("reserve_requests", 10))
        self.min_interval = float(cfg.get("min_interval_seconds", 1800))
        self.lookahead_h = float(cfg.get("lookahead_hours", 36))
        self.fixtures_refresh_s = float(cfg.get("fixtures_refresh_hours", 24)) * 3600
        self.max_tournaments = int(cfg.get("max_tournaments", 20))
        self.per_request = int(cfg.get("tournaments_per_request", 10))
        self.verbosity = cfg.get("verbosity", 3)
        self.cache_file = Path(cfg.get("cache_file", "oddspapi-cache.json"))
        self.dump_dir: Path | None = None
        self.last_error = ""
        self.cache = self._load()
        self._sweep_cost = 1

    # ------------------------------------------------------------ cache
    def _load(self) -> dict:
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self.cache), encoding="utf-8")
        except OSError as e:
            log.warning("oddspapi cache not saved: %s", e)

    @property
    def used(self) -> int:
        u = self.cache.get("usage", {})
        return int(u.get("used", 0)) if u.get("month") == utcnow().strftime("%Y-%m") else 0

    @property
    def remaining(self) -> int:
        return max(self.monthly - self.used, 0)

    def _count(self) -> None:
        month = utcnow().strftime("%Y-%m")
        u = self.cache.get("usage", {})
        self.cache["usage"] = {"month": month, "used": (u.get("used", 0) if u.get("month") == month else 0) + 1}

    # ------------------------------------------------------------- http
    def get(self, path: str, **params):
        if not self.key:
            raise RuntimeError("oddspapi: no api_key (set it in watch.toml or FOOTY_ODDSPAPI_KEY)")
        if self.remaining <= 0:
            raise RuntimeError("oddspapi: monthly request allowance used up")
        params = {k: v for k, v in params.items() if v not in (None, "", [])}
        url = f"{self.base}{path}?" + urllib.parse.urlencode({**params, "apiKey": self.key})
        data = []
        try:
            for attempt in (1, 2):
                self._count()
                try:
                    data, _ = P.http_json(url, timeout=60)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt == 1:
                        try:
                            wait = float(json.loads(e.read().decode() or "{}").get("retryMs", 1500)) / 1000
                        except (ValueError, AttributeError):
                            wait = 1.5
                        time.sleep(min(max(wait, 1.0), 10.0))
                        continue
                    if e.code == 404:  # nothing in that window
                        data = []
                        break
                    raise
        finally:
            self._save()
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            fname = path.strip("/").replace("/", "_") + "_" + "_".join(
                f"{k}-{v}" for k, v in sorted(params.items()))[:80].replace(",", "+") + ".json"
            (self.dump_dir / fname).write_text(json.dumps(data, indent=1), encoding="utf-8")
        return data

    # ------------------------------------------------------ catalogues
    def sport_ids(self) -> dict[str, int]:
        """our sport name -> OddsPapi sportId (cached for 30 days)."""
        c = self.cache.get("sports")
        if not c or time.time() - c.get("at", 0) > 30 * 86400:
            data = self.get("/sports")
            c = {"at": time.time(), "items": data if isinstance(data, list) else []}
            self.cache["sports"] = c
            self._save()
        out = {}
        for s in c["items"]:
            nm = str(s.get("sportName") or s.get("name") or s.get("slug") or "").strip().lower()
            for kind, aliases in SPORT_NAMES.items():
                if nm in aliases and kind not in out:
                    out[kind] = s.get("sportId")
        return out

    def market_map(self, sport_id) -> dict[str, dict]:
        """marketId -> mapping, built from the catalogue (cached for 30 days)."""
        key = str(sport_id)
        c = self.cache.setdefault("markets", {}).get(key)
        if not c or c.get("version") != MAP_VERSION or time.time() - c.get("at", 0) > 30 * 86400:
            data = self.get("/markets", sportId=sport_id)
            items = data if isinstance(data, list) else (data or {}).get("markets", []) if isinstance(data, dict) else []
            mapped = {}
            for m in items:
                mm = map_market(m)
                if mm:
                    mapped[str(m.get("marketId"))] = mm
            c = {"at": time.time(), "map": mapped, "total": len(items), "version": MAP_VERSION}
            self.cache["markets"][key] = c
            self._save()
        return c["map"]

    def fixtures(self, kind: str, sport_id, now: datetime, force: bool = False) -> dict[str, dict]:
        """fixtureId -> {home, away, start, tournamentId, tournament} in the lookahead window."""
        c = self.cache.setdefault("fixtures", {}).get(kind)
        if force or not c or time.time() - c.get("at", 0) > self.fixtures_refresh_s:
            to = now + timedelta(hours=self.lookahead_h) + timedelta(seconds=self.fixtures_refresh_s)  # covers until the next refresh
            data = self.get("/fixtures", sportId=sport_id, **{"from": now.strftime("%Y-%m-%d"),
                                                               "to": to.strftime("%Y-%m-%d")})
            items = data if isinstance(data, list) else (data or {}).get("fixtures", []) if isinstance(data, dict) else []
            fx = {}
            for f in items:
                try:
                    start = parse_time(str(f.get("startTime") or f.get("start_time")))
                except (ValueError, TypeError):
                    continue
                home, away = f.get("participant1Name"), f.get("participant2Name")
                if not home or not away or f.get("hasOdds") is False:
                    continue
                fx[str(f.get("fixtureId"))] = {"home": home, "away": away, "start": start.isoformat(),
                                               "tournamentId": str(f.get("tournamentId", "")),
                                               "tournament": f.get("tournamentName") or f.get("tournamentSlug") or ""}
            c = {"at": time.time(), "items": fx}
            self.cache["fixtures"][kind] = c
            self._save()
        return c["items"]

    def tournaments(self, kind: str, fx: dict[str, dict], now: datetime) -> list[str]:
        """The configured tournament ids, or the busiest ones in the lookahead window."""
        if self.sports.get(kind):
            return self.sports[kind]
        hi = now + timedelta(hours=self.lookahead_h)
        counts: dict[str, int] = {}
        for f in fx.values():
            if now <= parse_time(f["start"]) <= hi and f["tournamentId"]:
                counts[f["tournamentId"]] = counts.get(f["tournamentId"], 0) + 1
        return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:self.max_tournaments]

    # ------------------------------------------------------------ polling
    def interval(self, n: int = 0, now: datetime | None = None) -> float:
        now = now or utcnow()
        nxt = (now.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        secs_left = max((nxt - now).total_seconds(), 3600)
        sweeps = max(self.remaining - self.reserve, 1) / max(self._sweep_cost, 1)
        return max(self.min_interval, secs_left / sweeps)

    def poll(self, index: EventIndex, now: datetime | None = None) -> list[Quote]:
        now = now or utcnow()
        ids = self.sport_ids()
        out: list[Quote] = []
        before = self.used
        for kind in self.sports:
            sid = ids.get(kind)
            if sid is None:
                log.warning("oddspapi: sport %s not found in /sports", kind)
                continue
            mmap = self.market_map(sid)
            fx = self.fixtures(kind, sid, now)
            tids = self.tournaments(kind, fx, now)
            for i in range(0, len(tids), self.per_request):
                for bk in self.bookmakers:  # the API takes exactly one bookmaker per request
                    data = self.get("/odds-by-tournaments", tournamentIds=",".join(tids[i:i + self.per_request]),
                                    bookmaker=bk, verbosity=self.verbosity)
                    out += self.parse(data, fx, mmap, index, kind, now, self.book_names, self.lookahead_h)
        self._sweep_cost = max(self.used - before, 1)
        return out

    @classmethod
    def parse(cls, data, fixtures: dict[str, dict], mmap: dict[str, dict], index: EventIndex, kind: str,
              now: datetime, book_names: dict | None = None, lookahead_h: float = 36) -> list[Quote]:
        book_names = book_names or {}
        items = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
        out: list[Quote] = []
        for f in items:
            if not isinstance(f, dict):
                continue
            fid = str(f.get("fixtureId"))
            fx = fixtures.get(fid) or {}
            home = fx.get("home") or f.get("participant1Name")
            away = fx.get("away") or f.get("participant2Name")
            start = fx.get("start") or f.get("startTime")
            if not home or not away or not start:
                continue
            ko = parse_time(str(start))
            if not now - timedelta(hours=3) <= ko <= now + timedelta(hours=lookahead_h):
                continue
            ev, swapped = index.add_oriented(cls.name, fid, home, away, ko, fx.get("tournament") or None, kind)
            for slug, bdata in (f.get("bookmakerOdds") or {}).items():
                if not isinstance(bdata, dict) or bdata.get("bookmakerIsActive") is False or bdata.get("suspended"):
                    continue
                book = book_names.get(slug, slug)
                for mid, mdata in (bdata.get("markets") or {}).items():
                    spec = mmap.get(str(mid))
                    if not spec or not isinstance(mdata, dict):
                        continue
                    for oid, odata in (mdata.get("outcomes") or {}).items():
                        outcome = spec["outcomes"].get(str(oid))
                        p = _latest(odata)
                        if not outcome or not p or p.get("active") is False:
                            continue
                        try:
                            price = float(p.get("price") or 0)
                        except (TypeError, ValueError):
                            continue
                        if price <= 1.0 or not math.isfinite(price):
                            continue
                        line, oc = orient(spec["market"], spec["line"], outcome, swapped)
                        out.append(Quote(ev.id, spec["market"], line, oc, book, price, now))
        return out
