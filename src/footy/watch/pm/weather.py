"""Daily temperature markets on Polymarket priced from weather-model ensembles.

Each event ("Highest temperature in NYC on November 8?") has one Yes/No market per
range ("62-63°F", "61°F or below", ...) and settles on one airport station's readings.

* Station: read from the rules (the NOAA/Weather Underground link names it, e.g. KLGA).
* Forecast: Open-Meteo's ensemble API (GFS, ECMWF and ICON members, free), hourly
  temperatures at the station's coordinates; each member's high (or low) over the
  local day, each model weighted equally.
* Today: the highest temperature already observed at the station (METAR from
  aviationweather.gov) is a floor; only the remaining hours can raise it.
* Each member is smoothed with a normal error (``sigma_f``) for station-vs-grid and
  rounding differences, shifted by the station's learned bias.

A range is a bet when the model's probability beats the price (after the taker fee) by
``min_edge``. Markets where the model and the market disagree by many degrees are
skipped: that usually means a wrong station or unit, not value.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import statistics
import threading
from datetime import datetime, timedelta, timezone

from .. import providers as P
from ..core import utcnow
from ..providers import _pm_time
from . import Signal, gamma
from .consistency import MONTHS

log = logging.getLogger(__name__)

TITLE_RE = re.compile(r"(highest|lowest) temperature in (.+?) on ([A-Za-z]+)\.? (\d{1,2})(?:,? (\d{4}))?", re.I)
STATION_RES = (re.compile(r"[?&]site=([A-Z0-9]{4})\b"),
               re.compile(r"wunderground\.com/history/daily/[^\s\"'<>]*?/([A-Z]{4})(?:/|\b)"),
               re.compile(r"\(([A-Z]{4})\)"),
               re.compile(r"\b(?:station|airport)\b[^.\n]{0,80}?\b([KCEPLRVWYZ][A-Z]{3})\b"))
OPEN_METEO = "https://ensemble-api.open-meteo.com/v1/ensemble"
METAR = "https://aviationweather.gov/api/data/metar"


def parse_title(title: str) -> dict | None:
    m = TITLE_RE.search(str(title))
    if not m or m.group(3).lower().rstrip(".") not in MONTHS:
        return None
    return {"kind": "max" if m.group(1).lower() == "highest" else "min", "city": m.group(2).strip(),
            "month": MONTHS[m.group(3).lower().rstrip(".")], "day": int(m.group(4)),
            "year": int(m.group(5)) if m.group(5) else None}


def parse_bucket(label: str) -> tuple[float, float, str] | None:
    """(low, high, unit) with inclusive whole-degree bounds; open ends are +-inf."""
    t = str(label).replace("º", "°").replace("−", "-").strip()
    unit = "F" if re.search(r"°\s*F|\bF\b", t) else "C" if re.search(r"°\s*C|\bC\b", t) else None
    low = t.lower()
    rng = re.search(r"(-?\d+)\s*(?:-|–|to)\s*(-?\d+)", t)
    one = re.findall(r"-?\d+", t)
    if rng:
        a, b = float(rng.group(1)), float(rng.group(2))
        return (min(a, b), max(a, b), unit) if unit else None
    if len(one) != 1 or not unit:
        return None
    v = float(one[0])
    if re.search(r"or (below|lower|less|under)|≤|<=|\bbelow\b|\bunder\b", low):
        return (-math.inf, v, unit)
    if re.search(r"or (above|higher|more|over)|≥|>=|\+|\babove\b", low):
        return (v, math.inf, unit)
    return (v, v, unit)


def station_of(e: dict, overrides: dict | None = None) -> str | None:
    texts = [e.get("resolutionSource") or "", e.get("description") or ""] + \
            [m.get("description") or "" for m in e.get("markets", [])[:2]]
    for rx in STATION_RES:
        for t in texts:
            hit = rx.search(t)
            if hit:
                return hit.group(1)
    info = parse_title(e.get("title", ""))
    return (overrides or {}).get(info["city"]) if info else None


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bucket_probs(values: list[tuple[float, float]], buckets: list[tuple[float, float]], sigma: float,
                 floor: float | None = None) -> list[float]:
    """P(rounded reading falls in each bucket) for weighted member values (weight, degrees)."""
    out = []
    for lo, hi in buckets:
        p = 0.0
        for w, v in values:
            upper = 1.0 if hi == math.inf else _phi((hi + 0.5 - v) / sigma)
            lower = 0.0 if lo == -math.inf else _phi((lo - 0.5 - v) / sigma)
            p += w * max(upper - lower, 0.0)
        out.append(p)
    if floor is not None:  # today's reading can't end below what has already been observed
        out = [0.0 if hi + 0.5 <= floor else p for (lo, hi), p in zip(buckets, out)]
    s = sum(out)
    return [p / s for p in out] if s > 0 else out


def members(ens: dict, day: str, kind: str, now_local: datetime | None = None,
            observed: float | None = None) -> list[tuple[float, float]]:
    """(weight, high or low of the local day) per ensemble member; models weighted equally.

    ``now_local``/``observed`` (today only): only hours from now on are forecast, and the
    observed extreme so far bounds the result.
    """
    times = ens.get("time", [])
    idx = [i for i, t in enumerate(times) if str(t).startswith(day)]
    if now_local is not None:
        later = [i for i in idx if str(times[i]) >= now_local.strftime("%Y-%m-%dT%H:00")]
        idx = later
    by_model: dict[str, list[float]] = {}
    for model, series in ens.get("series", []):
        vals = [series[i] for i in idx if i < len(series) and series[i] is not None]
        if now_local is None and len(vals) < 20:
            continue
        if not vals and observed is None:
            continue
        ext = (max(vals) if kind == "max" else min(vals)) if vals else observed
        if observed is not None:
            ext = max(ext, observed) if kind == "max" else min(ext, observed)
        by_model.setdefault(model, []).append(ext)
    out = []
    for vals in by_model.values():
        out += [(1 / len(by_model) / len(vals), v) for v in vals]
    return out


def parse_ensemble(data: dict) -> dict:
    """Open-Meteo hourly response -> {"time", "timezone", "series": [(model, values)]}."""
    hourly = (data or {}).get("hourly") or {}
    series = []
    for k, v in hourly.items():
        if not k.startswith("temperature_2m") or not isinstance(v, list):
            continue
        rest = k[len("temperature_2m"):]
        rest = re.sub(r"^_member\d+", "", rest)
        series.append((rest.lstrip("_") or "default", v))
    return {"time": hourly.get("time", []), "timezone": (data or {}).get("timezone") or "UTC", "series": series}


class WeatherScanner:
    name = "weather"

    def __init__(self, cfg: dict):
        w = cfg.get("weather", {})
        self.fee_rate = float(cfg.get("pm", {}).get("fee_rate", 0.05))
        self.interval_s = float(w.get("interval_minutes", 30)) * 60
        self.lookahead_days = float(w.get("lookahead_days", 2))
        self.min_edge = float(w.get("min_edge", 0.08))
        self.min_prob = float(w.get("min_prob", 0.10))
        self.sigma_f = float(w.get("sigma_f", 1.6))
        self.max_disagreement_f = float(w.get("max_disagreement_f", 6.0))
        self.models = list(w.get("models", ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]))
        self.use_metar = bool(w.get("use_metar", True))
        self.stations = dict(w.get("stations", {}))
        self.tag_slug = str(w.get("tag_slug", "weather"))
        self.tag = str(w["tag_id"]) if w.get("tag_id") else None
        self.max_signals = int(w.get("max_signals", 10))
        self.db = sqlite3.connect(cfg["server"]["db"], check_same_thread=False, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS pm_weather (station TEXT, day TEXT, kind TEXT, unit TEXT, "
                        "forecast REAL, observed REAL, PRIMARY KEY (station, day, kind, unit))")
        self.db.commit()
        self.lock = threading.Lock()
        self.last_error = ""
        self.last_count = 0

    # ---------------------------------------------------------- sources
    def fetch(self, now: datetime) -> list[dict]:
        if self.tag is None:
            self.tag = gamma.tag_id(self.tag_slug)
        return [e for e in gamma.events({"tag_id": self.tag, "closed": "false",
                                         "end_date_min": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                         "end_date_max": (now + timedelta(days=self.lookahead_days + 1)).strftime(
                                             "%Y-%m-%dT%H:%M:%SZ")}, keep_text=True)
                if parse_title(e.get("title", ""))]

    def metar(self, station: str) -> dict:
        """{"lat", "lon", "obs": [(utc time, °C)]} from the last 36 hours of reports."""
        data, _ = P.http_json(f"{METAR}?ids={station}&format=json&hours=36", timeout=30)
        rows = data if isinstance(data, list) else []
        obs = []
        for r in rows:
            t, c = r.get("obsTime"), r.get("temp")
            if t is None or c is None:
                continue
            try:
                when = datetime.fromtimestamp(int(t), tz=timezone.utc) if str(t).isdigit() else _pm_time(t)
                obs.append((when, float(c)))
            except (TypeError, ValueError):
                continue
        first = rows[0] if rows else {}
        return {"lat": first.get("lat"), "lon": first.get("lon"), "obs": obs}

    def ensemble(self, lat: float, lon: float, unit: str) -> dict:
        url = (f"{OPEN_METEO}?latitude={lat}&longitude={lon}&hourly=temperature_2m&models={','.join(self.models)}"
               f"&timezone=auto&forecast_days={int(self.lookahead_days) + 2}"
               f"&temperature_unit={'fahrenheit' if unit == 'F' else 'celsius'}")
        data, _ = P.http_json(url, timeout=60)
        return parse_ensemble(data)

    # ------------------------------------------------------------- bias
    def bias(self, station: str, kind: str, unit: str) -> float:
        with self.lock:
            rows = self.db.execute("SELECT observed - forecast FROM pm_weather WHERE station=? AND kind=? AND unit=? "
                                   "AND observed IS NOT NULL AND forecast IS NOT NULL ORDER BY day DESC LIMIT 30",
                                   (station, kind, unit)).fetchall()
        errs = [r[0] for r in rows]
        return max(min(statistics.median(errs), 5.0), -5.0) if len(errs) >= 5 else 0.0

    def _learn(self, station: str, kind: str, unit: str, day: str, forecast: float | None, observed: float | None):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO pm_weather (station, day, kind, unit) VALUES (?,?,?,?)",
                            (station, day, kind, unit))
            if forecast is not None:
                self.db.execute("UPDATE pm_weather SET forecast=? WHERE station=? AND day=? AND kind=? AND unit=?",
                                (forecast, station, day, kind, unit))
            if observed is not None:
                self.db.execute("UPDATE pm_weather SET observed=? WHERE station=? AND day=? AND kind=? AND unit=? "
                                "AND observed IS NULL", (observed, station, day, kind, unit))
            self.db.commit()

    # -------------------------------------------------------------- run
    def run(self, now: datetime | None = None, events: list[dict] | None = None, books=None,
            ensembles: dict | None = None, metars: dict | None = None) -> list[Signal]:
        now = now or utcnow()
        events = self.fetch(now) if events is None else events
        self.last_count = len(events)
        ensembles = {} if ensembles is None else ensembles
        metars = {} if metars is None else metars
        cands = []
        for e in events:
            try:
                c = self.evaluate(e, now, ensembles, metars)
            except Exception as ex:  # noqa: BLE001 - one city's data problem shouldn't stop the others
                log.warning("weather %s: %s", e.get("title"), ex)
                c = []
            cands += c
        if not cands:
            return []
        bk = books if books is not None else gamma.books({c["leg"]["token"] for c in cands})
        out = []
        for c in cands:
            q = bk.get(c["leg"]["token"]) or {}
            ask = q.get("ask")
            if ask is None or not 0.02 <= ask <= 0.98:
                continue
            fee = gamma.fee(ask, self.fee_rate)
            edge = c["prob"] / (ask + fee) - 1
            if edge < self.min_edge:
                continue
            leg = {**c["leg"], "price": ask, "fee": round(fee, 6), "size": q.get("ask_size") or 0.0}
            e = c["event"]
            end = _pm_time(e.get("endDate"))
            days = max(((end - now).total_seconds() / 86400) if end else 1.0, 1 / 24)
            detail = f"{c['why']} · model {100 * c['prob']:.0f}% vs {100 * ask:.0f}¢"
            out.append(Signal("weather", f"weather|{e['id']}|{leg['market_id']}|{leg['side']}", e["id"],
                              e.get("title", ""), e.get("slug") or "", end, f"weather-{leg['side'].lower()}", [leg],
                              None, c["prob"], edge, edge * 365 / days, detail,
                              {"station": c["station"], "mean": c["mean"]}))
        best: dict[str, Signal] = {}  # one position per event: the best edge
        for s in out:
            if s.event_id not in best or s.edge > best[s.event_id].edge:
                best[s.event_id] = s
        return sorted(best.values(), key=lambda s: -s.edge)[:self.max_signals]

    def evaluate(self, e: dict, now: datetime, ensembles: dict, metars: dict) -> list[dict]:
        info = parse_title(e.get("title", ""))
        station = station_of(e, self.stations)
        if not info or not station:
            return []
        rows = []
        for m in e.get("markets", []):
            toks = gamma.yes_no(m)
            b = parse_bucket(m.get("groupItemTitle") or m.get("question") or "")
            if not toks or not b or m.get("closed"):
                return []  # every range must be readable, or the probabilities can't be trusted
            rows.append((b, m, toks))
        units = {b[2] for b, _, _ in rows}
        if len(units) != 1 or len(rows) < 2:
            return []
        unit = units.pop()
        if station not in metars:
            metars[station] = self.metar(station) if self.use_metar else {"lat": None, "lon": None, "obs": []}
        mt = metars[station]
        if mt.get("lat") is None:
            return []
        key = (station, unit)
        if key not in ensembles:
            ensembles[key] = self.ensemble(mt["lat"], mt["lon"], unit)
        ens = ensembles[key]
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(ens["timezone"])
        except Exception:  # noqa: BLE001
            tz = timezone.utc
        local_now = now.astimezone(tz)
        end = _pm_time(e.get("endDate"))
        year = info["year"] or (end.astimezone(tz).year if end else local_now.year)
        day = f"{year:04d}-{info['month']:02d}-{info['day']:02d}"
        today = local_now.strftime("%Y-%m-%d")
        if day < today:
            return []
        to_unit = (lambda c: c * 9 / 5 + 32) if unit == "F" else (lambda c: c)
        observed = None
        obs_today = [to_unit(c) for t, c in mt.get("obs", []) if t.astimezone(tz).strftime("%Y-%m-%d") == day]
        if day == today and obs_today:
            observed = (max(obs_today) - 0.5) if info["kind"] == "max" else (min(obs_today) + 0.5)
        # learning: yesterday's reading for the bias, today's forecast (made the day before)
        yday = (local_now - timedelta(days=1)).strftime("%Y-%m-%d")
        obs_y = [to_unit(c) for t, c in mt.get("obs", []) if t.astimezone(tz).strftime("%Y-%m-%d") == yday]
        if len(obs_y) >= 18:
            self._learn(station, info["kind"], unit, yday, None, max(obs_y) if info["kind"] == "max" else min(obs_y))
        vals = members(ens, day, info["kind"], local_now if day == today else None, observed)
        if len(vals) < 10:
            return []
        mean = sum(w * v for w, v in vals)
        if day > today:
            self._learn(station, info["kind"], unit, day, mean, None)
        bias = self.bias(station, info["kind"], unit)
        vals = [(w, v + bias) for w, v in vals]
        sigma = self.sigma_f if unit == "F" else self.sigma_f * 5 / 9
        buckets = [(b[0], b[1]) for b, _, _ in rows]
        floor = observed if (observed is not None and info["kind"] == "max") else None
        probs = bucket_probs(vals, buckets, sigma, floor)
        # sanity: the market's own expected reading must be close to the model's
        mids, mkt = [], []
        for (lo, hi), (b, m, _), p in zip(buckets, rows, probs):
            price = float(m.get("bestAsk") or 0) or float(m.get("bestBid") or 0)
            mid = (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else (hi - 1 if math.isfinite(hi) else lo + 1)
            mids.append(mid)
            mkt.append(price)
        if sum(mkt) > 0:
            mkt_mean = sum(x * p for x, p in zip(mids, mkt)) / sum(mkt)
            limit = self.max_disagreement_f if unit == "F" else self.max_disagreement_f * 5 / 9
            if abs(mkt_mean - (mean + bias)) > limit:
                log.info("weather %s: model %.1f vs market %.1f°%s, skipped (station or unit?)",
                         e.get("title"), mean + bias, mkt_mean, unit)
                return []
        why = (f"{info['city']} {day}: ensemble {mean + bias:.1f}°{unit} ({len(vals)} members"
               + (f", bias {bias:+.1f}" if bias else "") + ")"
               + (f", {'max' if info['kind'] == 'max' else 'min'} so far {max(obs_today) if info['kind'] == 'max' else min(obs_today):.0f}°{unit}"
                  if observed is not None else ""))
        out = []
        for (b, m, toks), p in zip(rows, probs):
            label = m.get("groupItemTitle") or ""
            for side, prob, tok in (("YES", p, toks[0]), ("NO", 1 - p, toks[1])):
                if prob < self.min_prob:
                    continue
                screen = float(m.get("bestAsk") or 0) if side == "YES" else 1 - float(m.get("bestBid") or 1)
                if not 0 < screen < 1 or prob / (screen + gamma.fee(screen, self.fee_rate)) - 1 < self.min_edge * 0.5:
                    continue
                out.append({"event": e, "station": station, "mean": round(mean + bias, 2), "prob": prob, "why": why,
                            "leg": {"market_id": m["id"], "label": label, "token": tok, "side": side}})
        return out
