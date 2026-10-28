"""Polymarket's public Gamma (catalogue) and CLOB (order book) APIs, trimmed for a Pi Zero.

Gamma pages can be tens of MB (football side markets), so events are fetched in small
pages with filters and only the fields the strategies read are kept.
"""

from __future__ import annotations

import json
import logging
import urllib.parse

from .. import providers as P
from ..providers import _jlist

log = logging.getLogger(__name__)
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MARKET_FIELDS = ("id", "question", "groupItemTitle", "outcomes", "clobTokenIds", "outcomePrices", "bestBid",
                 "bestAsk", "active", "closed", "acceptingOrders", "negRiskOther", "endDate", "slug",
                 "umaResolutionStatus")


def get(path: str, **params):
    params = {k: v for k, v in params.items() if v not in (None, "", [])}
    url = f"{GAMMA}{path}" + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    data, _ = P.http_json(url, timeout=60)
    return data


def trim_event(e: dict, keep_text: bool = False) -> dict:
    ev = {k: e.get(k) for k in ("id", "title", "slug", "negRisk", "endDate")}
    ev["id"] = str(ev["id"])
    if keep_text:  # rules text names the weather station
        ev["description"] = str(e.get("description") or "")[:6000]
        ev["resolutionSource"] = str(e.get("resolutionSource") or "")
    ev["markets"] = []
    for m in e.get("markets") or []:
        mm = {k: m.get(k) for k in MARKET_FIELDS}
        mm["id"] = str(mm["id"])
        if keep_text:
            mm["description"] = str(m.get("description") or "")[:3000]
        ev["markets"].append(mm)
    return ev


def events(params: dict, page: int = 50, max_pages: int = 80, keep_text: bool = False) -> list[dict]:
    out: list[dict] = []
    for i in range(max_pages):
        rows = get("/events", **params, limit=page, offset=i * page)
        rows = rows if isinstance(rows, list) else (rows or {}).get("data", []) if isinstance(rows, dict) else []
        out += [trim_event(e, keep_text) for e in rows if isinstance(e, dict)]
        if len(rows) < page:
            break
    return out


def tag_id(slug: str) -> str | None:
    data = get(f"/tags/slug/{slug}")
    return str(data["id"]) if isinstance(data, dict) and data.get("id") is not None else None


def yes_no(m: dict) -> tuple[str, str] | None:
    """(YES token, NO token) of a Yes/No market."""
    outs = [str(o).strip().lower() for o in _jlist(m.get("outcomes"))]
    toks = [str(t) for t in _jlist(m.get("clobTokenIds"))]
    if outs != ["yes", "no"] or len(toks) != 2:
        return None
    return toks[0], toks[1]


def books(tokens) -> dict[str, dict]:
    """token -> {ask, ask_size, bid, bid_size}; sizes in shares at the best price."""
    tokens = sorted({str(t) for t in tokens})
    out: dict[str, dict] = {}
    for i in range(0, len(tokens), 400):  # the API takes up to 500 per call
        data, _ = P.http_json(f"{CLOB}/books", json.dumps([{"token_id": t} for t in tokens[i:i + 400]]).encode(),
                              {"Content-Type": "application/json"}, timeout=60)
        for b in data or []:
            asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks", []) if float(x.get("size", 0)) > 0]
            bids = [(float(x["price"]), float(x["size"])) for x in b.get("bids", []) if float(x.get("size", 0)) > 0]
            q = {"ask": None, "ask_size": 0.0, "bid": None, "bid_size": 0.0}
            if asks:
                q["ask"] = min(p for p, _ in asks)
                q["ask_size"] = sum(s for p, s in asks if p == q["ask"])
            if bids:
                q["bid"] = max(p for p, _ in bids)
                q["bid_size"] = sum(s for p, s in bids if p == q["bid"])
            out[str(b.get("asset_id"))] = q
    return out


def markets_by_id(ids) -> dict[str, dict]:
    ids = sorted({str(i) for i in ids})
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        rows = get("/markets", id=ids[i:i + 50], limit=50)
        for m in rows if isinstance(rows, list) else []:
            out[str(m.get("id"))] = {k: m.get(k) for k in MARKET_FIELDS}
    return out


def resolution(m: dict) -> float | None:
    """What a YES share paid (1, 0, or 0.5 for a 50/50 resolution); None while unresolved."""
    if not m or not m.get("closed"):
        return None
    try:
        prices = [float(x) for x in _jlist(m.get("outcomePrices"))]
    except (TypeError, ValueError):
        return None
    if len(prices) != 2:
        return None
    if prices[0] >= 0.99 or prices[0] <= 0.01 or (abs(prices[0] - 0.5) < 1e-9 and
                                                  str(m.get("umaResolutionStatus", "")).lower() == "resolved"):
        return round(prices[0], 2)
    return None


def fee(price: float, rate: float) -> float:
    """Taker fee per share bought at ``price``."""
    return rate * price * (1 - price)
