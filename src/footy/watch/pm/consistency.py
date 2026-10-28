"""Price-consistency arbitrage on Polymarket.

1. **Outcomes that don't add up** (``negRisk`` events: exactly one outcome resolves YES)
   * all YES for less than $1 (after fees): one share of each pays exactly $1;
   * all NO for less than $(n-1): with n open outcomes, n-1 of the NO shares pay.
     This side stays safe even if an unlisted outcome wins (then all n pay).
2. **Ladders** - markets in one event that differ only by a threshold or a date
   ("above $100k" / "above $110k", "by March 31" / "by June 30"). The narrower one
   can't be more likely than the wider one, so YES on the wider plus NO on the
   narrower always pays at least $1; it's an arbitrage when both cost less than $1.

Gamma's best bid/ask screen the whole catalogue; order books then confirm the price
and the money available on every leg. Money is tied up until resolution, so an
opportunity must also clear a yearly return (``min_apy``).
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone

from ..core import utcnow
from ..providers import _pm_time
from . import Signal, gamma

log = logging.getLogger(__name__)

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
DATE_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
                     r"(?:,?\s+(\d{4}))?\b", re.I)
NUM_RE = re.compile(r"[-−]?\$?\d[\d,]*(?:\.\d+)?\s*(?:[kmb]\b|%|°[fc]?)?", re.I)
UP = ("above", "over", "more than", "at least", "greater", "higher", "reach", "hit", "exceed", "or more", "top")
DOWN = ("below", "under", "less than", "lower", "dip", "fall", "drop", "or less", "bottom")


def template(text: str) -> str:
    t = DATE_RE.sub(" <d> ", str(text).lower())
    t = NUM_RE.sub(" <n> ", t)
    return re.sub(r"\s+", " ", t).strip()


def number(text: str) -> float | None:
    """The single number in a label like "$110,000", "110k", "2.5%" (None if none or several)."""
    t = DATE_RE.sub(" ", str(text))
    hits = NUM_RE.findall(t)
    if len(hits) != 1:
        return None
    h = hits[0].lower().replace("−", "-").replace("$", "").replace(",", "").strip()
    mult = 1.0
    for suf, m in (("k", 1e3), ("m", 1e6), ("b", 1e9)):
        if h.endswith(suf):
            mult, h = m, h[:-1]
    h = h.rstrip("%°fc ").strip()
    try:
        return float(h) * mult
    except ValueError:
        return None


def date_of(text: str, year_hint: int) -> datetime | None:
    hits = DATE_RE.findall(str(text))
    if len(hits) != 1:
        return None
    mon, day, year = hits[0]
    try:
        return datetime(int(year) if year else year_hint, MONTHS[mon.lower().rstrip(".")], int(day), tzinfo=timezone.utc)
    except ValueError:
        return None


class ConsistencyScanner:
    name = "consistency"

    def __init__(self, cfg: dict):
        c = cfg.get("consistency", {})
        self.fee_rate = float(cfg.get("pm", {}).get("fee_rate", 0.05))
        self.interval_s = float(c.get("interval_minutes", 20)) * 60
        self.max_days = float(c.get("max_days", 45))
        self.min_edge = float(c.get("min_edge", 0.01))
        self.min_apy = float(c.get("min_apy", 0.15))
        self.min_leg_size = float(c.get("min_leg_size", 20))
        self.exclude_tags = [str(t) for t in c.get("exclude_tags", ["1"])]
        self.max_signals = int(c.get("max_signals", 20))
        self.last_error = ""
        self.last_count = 0

    # ----------------------------------------------------------- fetch
    def fetch(self, now: datetime) -> list[dict]:
        return gamma.events({"closed": "false", "active": "true",
                             "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "end_date_max": (now + timedelta(days=self.max_days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "exclude_tag_id": self.exclude_tags})

    def run(self, now: datetime | None = None, events: list[dict] | None = None, books=None) -> list[Signal]:
        now = now or utcnow()
        events = self.fetch(now) if events is None else events
        self.last_count = len(events)
        cands = []
        for e in events:
            cands += self.sum_to_one(e) + self.ladders(e)
        if not cands:
            return []
        tokens = {lg["token"] for c in cands for lg in c["legs"]}
        bk = books if books is not None else gamma.books(tokens)
        out = [s for c in cands if (s := self.confirm(c, bk, now))]
        out.sort(key=lambda s: -(s.apy or 0))
        return out[:self.max_signals]

    # ------------------------------------------------------- screening
    def _price(self, m: dict, side: str) -> float | None:
        """Gamma's summary price for buying YES (best ask) or NO (1 - best bid)."""
        try:
            if side == "YES":
                a = float(m.get("bestAsk") or 0)
                return a if 0 < a < 1 else None
            b = float(m.get("bestBid") or 0)
            return 1 - b if 0 < b < 1 else None
        except (TypeError, ValueError):
            return None

    def _leg(self, m: dict, side: str, label: str | None = None) -> dict | None:
        toks = gamma.yes_no(m)
        p = self._price(m, side)
        if not toks or p is None:
            return None
        return {"market_id": m["id"], "label": label or m.get("groupItemTitle") or m.get("question") or "",
                "token": toks[0] if side == "YES" else toks[1], "side": side, "price": p}

    def sum_to_one(self, e: dict) -> list[dict]:
        if not e.get("negRisk"):
            return []
        ms = [m for m in e["markets"] if gamma.yes_no(m)]
        if any(m.get("closed") and gamma.resolution(m) == 1.0 for m in ms):
            return []  # already decided
        live = [m for m in ms if not m.get("closed")]
        if len(live) < 2:
            return []
        out = []
        all_open = all(m.get("active") is not False and m.get("acceptingOrders") is not False for m in live)
        yes = [self._leg(m, "YES") for m in live]
        if all_open and all(yes) and sum(lg["price"] + gamma.fee(lg["price"], self.fee_rate) for lg in yes) < 1:
            out.append({"event": e, "kind": "all-yes", "legs": yes, "payout": 1.0})
        no = [self._leg(m, "NO") for m in live if m.get("acceptingOrders") is not False]
        if len(no) == len(live) and all(no) and \
                sum(lg["price"] + gamma.fee(lg["price"], self.fee_rate) for lg in no) < len(live) - 1:
            out.append({"event": e, "kind": "all-no", "legs": no, "payout": float(len(live) - 1)})
        return out

    def ladders(self, e: dict) -> list[dict]:
        ms = [m for m in e["markets"] if gamma.yes_no(m) and not m.get("closed")
              and m.get("acceptingOrders") is not False]
        groups: dict[str, list[dict]] = {}
        for m in ms:
            groups.setdefault(template(m.get("question") or ""), []).append(m)
        end = _pm_time(e.get("endDate"))
        year = (end or utcnow()).year
        out = []
        for tpl, grp in groups.items():
            if len(grp) < 2:
                continue
            up, down = any(w in tpl for w in UP), any(w in tpl for w in DOWN)
            if "<d>" in tpl and "<n>" not in tpl and re.search(r"\b(by|before)\b", tpl):
                texts = [m.get("groupItemTitle") or m.get("question") or "" for m in grp]
                with_year = {bool(h[2]) for t in texts for h in DATE_RE.findall(t)}
                if len(with_year) > 1:
                    continue  # some dates have a year and some don't: the order can't be trusted
                vals = [date_of(t, year) for t in texts]
                increasing = True                       # later deadline -> more likely
            elif "<n>" in tpl and up != down:
                vals = [number(m.get("groupItemTitle") or "") or number(m.get("question") or "") for m in grp]
                increasing = down                       # "below X": higher X -> more likely
            else:
                continue
            if any(v is None for v in vals) or len(set(vals)) != len(vals):
                continue
            ordered = sorted(zip(vals, grp), key=lambda x: x[0])
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    lo, hi = ordered[i][1], ordered[j][1]
                    narrow, wide = (lo, hi) if increasing else (hi, lo)  # narrow implies wide
                    a, b = self._leg(wide, "YES"), self._leg(narrow, "NO")
                    if a and b and a["price"] + b["price"] + gamma.fee(a["price"], self.fee_rate) + \
                            gamma.fee(b["price"], self.fee_rate) < 1:
                        out.append({"event": e, "kind": "ladder", "legs": [a, b], "payout": 1.0})
        return out

    # ----------------------------------------------------- confirmation
    def confirm(self, c: dict, bk: dict, now: datetime) -> Signal | None:
        legs = []
        for lg in c["legs"]:
            q = bk.get(lg["token"]) or {}
            ask = q.get("ask")
            if ask is None or not 0 < ask < 1:
                return None
            size = q.get("ask_size") or 0.0
            if size * ask < self.min_leg_size:
                return None
            legs.append({**lg, "price": ask, "fee": round(gamma.fee(ask, self.fee_rate), 6), "size": size})
        cost = sum(lg["price"] + lg["fee"] for lg in legs)
        edge = c["payout"] / cost - 1
        e = c["event"]
        end = _pm_time(e.get("endDate"))
        days = max(((end - now).total_seconds() / 86400) if end else self.max_days, 1 / 24)
        apy = edge * 365 / days
        if edge < self.min_edge or apy < self.min_apy:
            return None
        key = f"{c['kind']}|{e['id']}|" + ",".join(sorted(f"{lg['token']}:{lg['side']}" for lg in legs))
        what = {"all-yes": "every outcome YES", "all-no": "every outcome NO", "ladder": "ladder"}[c["kind"]]
        detail = (f"{what}: " + " + ".join(f"{lg['side']} {lg['label']} {100 * lg['price']:.1f}¢" for lg in legs[:6]) +
                  (f" + {len(legs) - 6} more" if len(legs) > 6 else "") +
                  f" = {100 * cost:.1f}¢ for a sure ${c['payout']:g}; {100 * edge:+.1f}% "
                  f"({100 * apy:.0f}%/yr, resolves in {days:.0f} d)")
        return Signal("consistency", key, e["id"], e.get("title") or "", e.get("slug") or "", end, c["kind"], legs,
                      c["payout"], None, edge, apy, detail)
