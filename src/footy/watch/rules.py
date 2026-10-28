"""Alert rules.

* ``value_alerts``  "wrong odds": a bookmaker's price (after commission) beats the
  margin-free price of a sharp reference (Pinnacle, or the Betfair Exchange) by
  at least ``min_edge``. Same idea as Trademate-style value betting.
* ``pick_alerts``   research picks from ``footy upcoming``: the live price gives a
  positive expected value under our model. Handicap picks carry the full
  goal-difference distribution, so any line on offer can be evaluated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .core import OUTCOMES, EventIndex, Quote, ah_break_even, ah_ev, devig, gross_needed, match_event, \
    net_odds, parse_time, within


@dataclass
class Alert:
    kind: str                # "value" | "pick"
    event_id: str
    match: str               # "Home v Away"
    kickoff: datetime
    league: str | None
    market: str
    line: float | None
    outcome: str
    book: str
    price: float
    fair_odds: float         # value: reference fair price; pick: our break-even price
    edge: float              # expected profit per unit stake (after commission)
    detail: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}|{self.event_id}|{self.market}|{self.line}|{self.outcome}|{self.book}"

    def selection(self, home: str = "", away: str = "") -> str:
        if self.market == "1x2":
            return {"home": f"{home} to win", "away": f"{away} to win", "draw": "Draw"}[self.outcome]
        if self.market == "ou":
            return f"{self.outcome.capitalize()} {self.line:g} goals"
        team = home if self.outcome == "home" else away
        line = self.line if self.outcome == "home" else -self.line
        return f"{team} {line:+.2f} (Asian handicap)"


def group_quotes(quotes: list[Quote]) -> dict[tuple, dict[str, dict[str, Quote]]]:
    """(event, market, line) -> book -> outcome -> latest quote."""
    g: dict[tuple, dict[str, dict[str, Quote]]] = defaultdict(lambda: defaultdict(dict))
    for q in quotes:
        g[(q.event, q.market, q.line)][q.book][q.outcome] = q
    return g


def _fresh(q: Quote, now: datetime, max_age_min: float) -> bool:
    return q.updated is None or (now - q.updated) <= timedelta(minutes=max_age_min)


def _ref_probs(outs: dict[str, Quote], outcomes: tuple, max_spread: float) -> list[float] | None:
    if not all(o in outs for o in outcomes):
        return None
    implied = []
    for o in outcomes:
        q = outs[o]
        pb = 1 / q.price
        if q.lay and q.lay > 1 and (q.lay / q.price - 1) <= max_spread:
            pb = (1 / q.price + 1 / q.lay) / 2  # exchange: back/lay midpoint
            implied.append(1 / pb)
        else:
            implied.append(q.price)
    return devig(implied, "power")


def value_alerts(index: EventIndex, quotes: list[Quote], cfg: dict, now: datetime) -> list[Alert]:
    a = cfg["alerts"]
    commission = cfg.get("commission", {})
    refs = list(a.get("reference_books", ["pinnacle"]))
    targets = list(a.get("target_books", ["*"]))
    out: list[Alert] = []
    for (eid, market, line), books in group_quotes(quotes).items():
        if market not in a.get("markets", ["1x2", "ou", "ah"]):
            continue
        ev = index.events.get(eid)
        if ev is None or not within(ev.kickoff, now, a.get("min_minutes_before_kickoff", 2),
                                    a.get("lookahead_hours", 48)):
            continue
        outcomes = OUTCOMES[market]
        ref_book, fair = None, None
        for rb in refs:
            outs = books.get(rb)
            if not outs or not all(_fresh(q, now, a.get("max_quote_age_minutes", 30)) for q in outs.values()):
                continue
            fair = _ref_probs(outs, outcomes, a.get("max_exchange_spread", 0.05))
            if fair:
                ref_book = rb
                break
        if not fair:
            continue
        ref_txt = " / ".join(f"{books[ref_book][o].price:.2f}" for o in outcomes)
        for book, outs in books.items():
            if book == ref_book or (targets != ["*"] and book not in targets):
                continue
            for o, p in zip(outcomes, fair):
                q = outs.get(o)
                if q is None or not _fresh(q, now, a.get("max_quote_age_minutes", 30)):
                    continue
                if q.size is not None and q.size < a.get("min_exchange_size", 0):
                    continue
                edge = net_odds(q.price, commission.get(book, 0.0)) * p - 1
                if a.get("min_edge", 0.03) <= edge <= a.get("max_edge", 0.25):
                    out.append(Alert("value", eid, f"{ev.home} v {ev.away}", ev.kickoff, ev.league, market, line,
                                     o, book, q.price, 1 / p, edge,
                                     f"{ref_book} {ref_txt} (no-margin {1 / p:.2f})",
                                     {"reference": ref_book}))
    return out


def pick_alerts(picks: list[dict], index: EventIndex, quotes: list[Quote], cfg: dict,
                now: datetime) -> tuple[list[Alert], list[dict]]:
    """Alerts for research picks, plus a live status row per pick for the web page."""
    commission = cfg.get("commission", {})
    pick_books = cfg["alerts"].get("pick_books", ["*"])
    by_event: dict[str, list[Quote]] = defaultdict(list)
    for q in quotes:
        by_event[q.event].append(q)
    alerts, status = [], []
    for pk in picks:
        kickoff = parse_time(pk["kickoff"])
        row = {"pick": pk, "matched": False, "best": None}
        if kickoff < now - timedelta(hours=2):
            continue
        ev = match_event(pk["home"], pk["away"], kickoff, index.events.values())
        if ev is None:
            status.append(row)
            continue
        row["matched"] = True
        best = None
        for q in by_event.get(ev.id, []):
            if q.market != pk["market"] or q.outcome != pk["side"]:
                continue
            if pick_books != ["*"] and q.book not in pick_books:
                continue
            c = commission.get(q.book, 0.0)
            net = net_odds(q.price, c)
            if pk["market"] == "ou":
                if q.line is None or abs(q.line - float(pk["line"])) > 1e-9:
                    continue
                p = float(pk["prob"])
                edge, need = p * net - 1, gross_needed(1 / p, c)
            elif pk["market"] == "ah":
                gd = {int(k): float(v) for k, v in pk["gd_dist"].items()}
                edge = ah_ev(gd, q.line, q.outcome, net)
                need = gross_needed(ah_break_even(gd, q.line, q.outcome), c)
            else:
                continue
            cand = {"book": q.book, "price": q.price, "line": q.line, "edge": edge, "need": need}
            if best is None or edge > best["edge"]:
                best = cand
            if edge >= float(pk.get("min_edge", 0.0)) + float(cfg["alerts"].get("pick_extra_edge", 0.0)):
                alerts.append(Alert("pick", ev.id, f"{ev.home} v {ev.away}", ev.kickoff, ev.league,
                                    q.market, q.line, q.outcome, q.book, q.price, need, edge,
                                    pk.get("reason", ""), {"rule": pk.get("rule_name", pk.get("rule", "")),
                                                           "pick_line": pk.get("line")}))
        row["best"] = best
        status.append(row)
    return alerts, status
