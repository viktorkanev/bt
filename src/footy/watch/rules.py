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

from .core import OUTCOMES, EventIndex, Quote, ah_break_even, ah_ev, book_gross_needed, book_net_odds, devig, \
    match_event, parse_time, sport_allows, within


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
    fair_odds: float         # value: reference fair price; pick: lowest price that still has the rule's edge
    edge: float              # expected profit per unit stake (after commission)
    detail: str = ""
    extra: dict = field(default_factory=dict)
    sport: str = "soccer"

    @property
    def key(self) -> str:
        if self.kind == "pick":  # one alert per research pick, at whichever book and line is best
            return f"pick|{self.event_id}|{self.market}|{self.outcome}|{self.extra.get('rule_code', '')}"
        return f"{self.kind}|{self.event_id}|{self.market}|{self.line}|{self.outcome}|{self.book}"

    def selection(self, home: str = "", away: str = "") -> str:
        return selection_text(self.market, self.line, self.outcome, home, away)


def selection_text(market: str, line: float | None, outcome: str, home: str = "", away: str = "") -> str:
    if outcome == "all":  # arbitrage: every outcome of the market
        if market == "1x2":
            return "Match result, all three outcomes"
        if market == "ml":
            return "Winner, both sides"
        if market == "ou":
            return f"Over/under {line:g}, both sides"
        return f"Handicap {home} {line:+g}, both sides"
    if market in ("1x2", "ml"):
        return {"home": f"{home} to win", "away": f"{away} to win", "draw": "Draw"}[outcome]
    if market == "ou":
        return f"{outcome.capitalize()} {line:g}"
    team = home if outcome == "home" else away
    line = line if outcome == "home" else -line
    return f"{team} {line:+g} (handicap)"


def group_quotes(quotes: list[Quote]) -> dict[tuple, dict[str, dict[str, Quote]]]:
    """(event, market, line) -> book -> outcome -> latest quote."""
    g: dict[tuple, dict[str, dict[str, Quote]]] = defaultdict(lambda: defaultdict(dict))
    for q in quotes:
        g[(q.event, q.market, q.line)][q.book][q.outcome] = q
    return g


def _fresh(q: Quote, now: datetime, max_age_min: float) -> bool:
    return q.updated is None or (now - q.updated) <= timedelta(minutes=max_age_min)


def ref_probs(outs: dict[str, Quote], outcomes: tuple, max_spread: float) -> list[float] | None:
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


def value_alerts(index: EventIndex, quotes: list[Quote], cfg: dict, now: datetime,
                 targets: list[str] | None = None) -> list[Alert]:
    """``targets`` overrides the configured target books (paper trading looks at every book)."""
    a = cfg["alerts"]
    refs = list(a.get("reference_books", ["pinnacle"]))
    targets = list(targets if targets is not None else a.get("target_books", ["*"]))
    out: list[Alert] = []
    for (eid, market, line), books in group_quotes(quotes).items():
        if market not in a.get("markets", ["1x2", "ml", "ou", "ah"]):
            continue
        ev = index.events.get(eid)
        if ev is None or not sport_allows(ev.sport, market) or not within(
                ev.kickoff, now, a.get("min_minutes_before_kickoff", 2), a.get("lookahead_hours", 48)):
            continue
        outcomes = OUTCOMES[market]
        ref_book, fair = None, None
        for rb in refs:
            outs = books.get(rb)
            if not outs or not all(_fresh(q, now, a.get("max_quote_age_minutes", 30)) for q in outs.values()):
                continue
            fair = ref_probs(outs, outcomes, a.get("max_exchange_spread", 0.05))
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
                edge = book_net_odds(q.price, book, cfg) * p - 1
                if a.get("min_edge", 0.03) <= edge <= a.get("max_edge", 0.25):
                    out.append(Alert("value", eid, f"{ev.home} v {ev.away}", ev.kickoff, ev.league, market, line,
                                     o, book, q.price, 1 / p, edge,
                                     f"{ref_book} {ref_txt} (no-margin {1 / p:.2f})",
                                     {"reference": ref_book, **({"link": q.link} if q.link else {})}, ev.sport))
    return out


def arb_alerts(index: EventIndex, quotes: list[Quote], cfg: dict, now: datetime,
               books: list[str] | None = None) -> list[Alert]:
    """Arbitrage: the best price for every outcome of a market, taken at different books,
    returns more than the total stake whatever happens (after commission and fees).

    ``books`` overrides the configured arb books (paper trading looks at every book).
    """
    a = cfg["alerts"]
    if not a.get("arb_alerts", True):
        return []
    allowed = list(books if books is not None else a.get("arb_books", ["*"]))
    max_age = a.get("max_quote_age_minutes", 30)
    min_size = a.get("min_exchange_size", 0)
    out: list[Alert] = []
    for (eid, market, line), books in group_quotes(quotes).items():
        if market not in a.get("markets", ["1x2", "ml", "ou", "ah"]):
            continue
        ev = index.events.get(eid)
        if ev is None or not sport_allows(ev.sport, market) or not within(
                ev.kickoff, now, a.get("min_minutes_before_kickoff", 2), a.get("lookahead_hours", 48)):
            continue
        best: dict[str, tuple[float, Quote]] = {}
        for book, outs in books.items():
            if "*" not in allowed and book not in allowed:
                continue
            for o, q in outs.items():
                if not _fresh(q, now, max_age) or (q.size is not None and q.size < min_size):
                    continue
                net = book_net_odds(q.price, book, cfg)
                if o not in best or net > best[o][0]:
                    best[o] = (net, q)
        outcomes = OUTCOMES[market]
        if not all(o in best for o in outcomes) or len({best[o][1].book for o in outcomes}) < 2:
            continue
        inv = sum(1 / best[o][0] for o in outcomes)
        edge = 1 / inv - 1
        if not a.get("min_arb", 0.005) <= edge <= a.get("max_arb", 0.10):
            continue
        legs = [{"outcome": o, "book": best[o][1].book, "price": best[o][1].price, "net": best[o][0],
                 "share": (1 / best[o][0]) / inv, **({"link": best[o][1].link} if best[o][1].link else {})}
                for o in outcomes]
        detail = " · ".join(f"{selection_text(market, line, lg['outcome'], ev.home, ev.away)} @ {lg['price']:.2f} "
                            f"{lg['book']} ({lg['share']:.0%})" for lg in legs)
        out.append(Alert("arb", eid, f"{ev.home} v {ev.away}", ev.kickoff, ev.league, market, line, "all",
                         "multi", round(1 / inv, 4), 0.0, edge, detail, {"legs": legs}, ev.sport))
    return out


def pick_alerts(picks: list[dict], index: EventIndex, quotes: list[Quote], cfg: dict,
                now: datetime, books: list[str] | None = None) -> tuple[list[Alert], list[dict]]:
    """Alerts for research picks, plus a live status row per pick for the web page.

    ``books`` overrides the configured pick books (paper trading looks at every book).
    """
    a = cfg["alerts"]
    pick_books = list(books if books is not None else a.get("pick_books", ["*"]))
    by_event: dict[str, list[Quote]] = defaultdict(list)
    for q in quotes:
        by_event[q.event].append(q)
    alerts, status = [], []
    for pk in picks:
        kickoff = parse_time(pk["kickoff"])
        row = {"pick": pk, "matched": False, "best": None}
        if kickoff < now - timedelta(hours=2):
            continue
        ev = match_event(pk["home"], pk["away"], kickoff, index.events.values(), sport="soccer")
        if ev is None:
            status.append(row)
            continue
        row["matched"] = True
        best = None
        open_for_bets = within(ev.kickoff, now, a.get("min_minutes_before_kickoff", 2), a.get("lookahead_hours", 48))
        need_edge = float(pk.get("min_edge", 0.0)) + float(a.get("pick_extra_edge", 0.0))
        for q in by_event.get(ev.id, []):
            if q.market != pk["market"] or q.outcome != pk["side"]:
                continue
            if not _fresh(q, now, a.get("max_quote_age_minutes", 30)):
                continue
            if pick_books != ["*"] and q.book not in pick_books:
                continue
            net = book_net_odds(q.price, q.book, cfg)
            if pk["market"] == "ou":
                if q.line is None or abs(q.line - float(pk["line"])) > 1e-9:
                    continue
                p = float(pk["prob"])
                edge, need = p * net - 1, book_gross_needed((1 + need_edge) / p, q.book, cfg)
            elif pk["market"] == "ah":
                gd = {int(k): float(v) for k, v in pk["gd_dist"].items()}
                edge = ah_ev(gd, q.line, q.outcome, net)
                need = book_gross_needed(ah_break_even(gd, q.line, q.outcome, need_edge), q.book, cfg)
            else:
                continue
            cand = {"book": q.book, "price": q.price, "line": q.line, "edge": edge, "need": need, "link": q.link}
            if best is None or edge > best["edge"]:
                best = cand
            if open_for_bets and edge >= need_edge:
                alerts.append(Alert("pick", ev.id, f"{ev.home} v {ev.away}", ev.kickoff, ev.league,
                                    q.market, q.line, q.outcome, q.book, q.price, need, edge,
                                    pk.get("reason", ""), {"rule": pk.get("rule_name", pk.get("rule", "")),
                                                           "rule_code": pk.get("rule", ""),
                                                           "league": pk.get("league"),
                                                           "pick_line": pk.get("line"),
                                                           **({"link": q.link} if q.link else {})}))
        row["best"] = best
        status.append(row)
    return alerts, status


def best_per_pick(alerts: list[Alert]) -> list[Alert]:
    """Keep each research pick's best-priced alert; "wrong odds" alerts stay one per book."""
    best: dict[str, Alert] = {}
    for a in alerts:
        if a.kind == "pick" and (a.key not in best or a.edge > best[a.key].edge):
            best[a.key] = a
    return [a for a in alerts if a.kind != "pick"] + list(best.values())
