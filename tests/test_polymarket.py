"""Polymarket feed, prediction-market fees and arbitrage alerts."""

import json
from datetime import datetime, timedelta, timezone

from footy.watch import config as C
from footy.watch import providers as P
from footy.watch.core import EventIndex, Quote, book_gross_needed, book_net_odds, devig
from footy.watch.providers import PolymarketProvider, _pm_time
from footy.watch.rules import arb_alerts, value_alerts

NOW = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)
KO = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)


def _m(question, outcomes, tokens, smt, **kw):
    return {"question": question, "outcomes": json.dumps(outcomes), "clobTokenIds": json.dumps(tokens),
            "sportsMarketType": smt, "gameStartTime": "2026-11-07 15:00:00+00", "active": True, "closed": False,
            "acceptingOrders": True, **kw}


EVENTS = [
    {"id": "9001", "title": "Arsenal FC vs. Chelsea FC", "startTime": "2026-11-07T15:00:00Z", "markets": [
        _m("Will Arsenal FC win on 2026-11-07?", ["Yes", "No"], ["111", "112"], "moneyline", groupItemTitle="Arsenal FC"),
        _m("Will Arsenal FC vs. Chelsea FC end in a draw?", ["Yes", "No"], ["121", "122"], "moneyline",
           groupItemTitle="Draw (Arsenal FC vs. Chelsea FC)"),
        _m("Will Chelsea FC win on 2026-11-07?", ["Yes", "No"], ["131", "132"], "moneyline", groupItemTitle="Chelsea FC"),
    ]},
    {"id": "9002", "title": "Arsenal FC vs. Chelsea FC - More Markets", "markets": [
        _m("Arsenal FC vs. Chelsea FC: O/U 2.5", ["Over", "Under"], ["211", "212"], "totals", line=2.5),
        _m("Spread: Chelsea FC (-1.5)", ["Chelsea FC", "Arsenal FC"], ["311", "312"], "spreads", line=-1.5),
        _m("Both Teams to Score", ["Yes", "No"], ["411", "412"], "both_teams_to_score"),
        _m("Arsenal FC vs. Chelsea FC: O/U 3.5", ["Over", "Under"], ["511", "512"], "totals", line=3.5, closed=True),
    ]},
]


def _book(tok, ask, ask_size, bid):
    return {"asset_id": tok, "asks": [{"price": str(ask + 0.02), "size": "500"}, {"price": str(ask), "size": str(ask_size)}],
            "bids": [{"price": str(bid - 0.01), "size": "800"}, {"price": str(bid), "size": "200"}]}


BOOKS = {t: _book(t, a, s, b) for t, a, s, b in [
    ("111", 0.48, 120, 0.46), ("121", 0.27, 300, 0.25), ("131", 0.27, 80, 0.25),
    ("211", 0.55, 400, 0.53), ("212", 0.47, 400, 0.45), ("311", 0.20, 100, 0.18), ("312", 0.82, 100, 0.80)]}


def test_time_formats():
    assert _pm_time("2026-11-07 15:00:00+00") == KO
    assert _pm_time("2026-11-07T15:00:00.000Z") == KO and _pm_time("") is None


def test_parse_maps_every_market_to_one_match():
    idx = EventIndex()
    qs = PolymarketProvider.parse(EVENTS, BOOKS, idx, now=NOW)
    assert len(idx.events) == 1                       # main event and "More Markets" are the same match
    got = {(q.market, q.line, q.outcome): q for q in qs}
    assert set(got) == {("1x2", None, "home"), ("1x2", None, "draw"), ("1x2", None, "away"),
                        ("ou", 2.5, "over"), ("ou", 2.5, "under"), ("ah", 1.5, "home"), ("ah", 1.5, "away")}
    home = got[("1x2", None, "home")]
    assert abs(home.price - 1 / 0.48) < 1e-3 and abs(home.lay - 1 / 0.46) < 1e-9 and home.size == round(120 * 0.48, 2)
    # "Chelsea -1.5" is the away team giving 1.5 -> home line +1.5; the home side is the Arsenal share
    assert abs(got[("ah", 1.5, "home")].price - 1 / 0.82) < 1e-3 and abs(got[("ah", 1.5, "away")].price - 5.0) < 1e-9


def test_follows_other_feeds_home_away_order():
    idx = EventIndex()
    idx.add("oddsapi", "x", "Chelsea", "Arsenal", KO)  # say the other feed has Chelsea at home
    qs = PolymarketProvider.parse(EVENTS[:1], BOOKS, idx, now=NOW)
    away = next(q for q in qs if q.outcome == "away")
    assert len(idx.events) == 1 and abs(away.price - 1 / 0.48) < 1e-3   # "Arsenal win" is now the away outcome


def test_fees_turn_price_into_real_odds_and_back():
    cfg = C.load(None)
    p = 0.48
    net = book_net_odds(1 / p, "polymarket", cfg)
    assert abs(net - 1 / (p + 0.05 * p * (1 - p))) < 1e-12 and net < 1 / p
    assert abs(book_gross_needed(net, "polymarket", cfg) - 1 / p) < 1e-9
    assert abs(book_net_odds(2.0, "betfair_ex_eu", cfg) - 1.95) < 1e-12      # exchange commission unchanged
    assert book_net_odds(2.0, "pinnacle", cfg) == 2.0


def test_polymarket_as_a_target_for_wrong_odds():
    cfg = C.load(None)
    idx = EventIndex()
    qs = PolymarketProvider.parse(EVENTS[:1], BOOKS, idx, now=NOW)
    eid = next(iter(idx.events))
    qs += [Quote(eid, "1x2", None, o, "pinnacle", pr, NOW) for o, pr in (("home", 2.25), ("draw", 3.30), ("away", 3.40))]
    al = [a for a in value_alerts(idx, qs, cfg, NOW) if a.book == "polymarket"]
    # Pinnacle no-margin draw ~0.289 vs Polymarket draw at 27c plus the taker fee -> about +3.3%
    fair_draw = devig([2.25, 3.30, 3.40])[1]
    draw = next(a for a in al if a.outcome == "draw")
    assert abs(draw.edge - (book_net_odds(1 / 0.27, "polymarket", cfg) * fair_draw - 1)) < 1e-9
    assert 0.03 < draw.edge < 0.04
    cfg["fees"]["polymarket"] = 0.10                       # a higher fee removes the edge
    assert not [a for a in value_alerts(idx, qs, cfg, NOW) if a.book == "polymarket" and a.outcome == "draw"]


def test_arbitrage_across_books_after_fees():
    cfg = C.load(None)
    idx = EventIndex()
    ev = idx.add("t", "1", "Arsenal", "Chelsea", KO)
    q = lambda o, b, p, line=None, m="1x2": Quote(ev.id, m, line, o, b, p, NOW)  # noqa: E731
    quotes = [q("home", "pinnacle", 2.20), q("draw", "pinnacle", 3.40), q("away", "pinnacle", 3.50),
              q("home", "polymarket", 1 / 0.40), q("draw", "polymarket", 1 / 0.24), q("away", "sport888", 3.80)]
    arbs = arb_alerts(idx, quotes, cfg, NOW)
    assert len(arbs) == 1
    legs = {lg["outcome"]: lg for lg in arbs[0].extra["legs"]}
    assert legs["home"]["book"] == "polymarket" and legs["away"]["book"] == "sport888"
    inv = sum(1 / lg["net"] for lg in legs.values())
    assert abs(arbs[0].edge - (1 / inv - 1)) < 1e-12 and abs(sum(lg["share"] for lg in legs.values()) - 1) < 1e-12
    # every result returns the same amount
    pay = [lg["share"] * lg["net"] for lg in legs.values()]
    assert max(pay) - min(pay) < 1e-12
    # without the Polymarket legs there is no arbitrage
    assert not arb_alerts(idx, [x for x in quotes if x.book != "polymarket"], cfg, NOW)
    cfg["alerts"]["arb_books"] = ["pinnacle", "polymarket"]              # only books you can use
    only = arb_alerts(idx, quotes, cfg, NOW)
    assert len(only) == 1 and {lg["book"] for lg in only[0].extra["legs"]} == {"pinnacle", "polymarket"}
    assert only[0].edge < arbs[0].edge
    cfg["alerts"]["arb_books"] = ["*"]
    one_book = [q("home", "polymarket", 1 / 0.30), q("draw", "polymarket", 1 / 0.30), q("away", "polymarket", 1 / 0.30)]
    assert not arb_alerts(idx, one_book, cfg, NOW)                       # one book alone is a data error


def test_poll_discovers_football_series_and_batches_books(monkeypatch):
    calls = []

    def fake(url, data=None, headers=None, timeout=30.0):
        calls.append((url, data))
        if url.endswith("/sports"):
            return [{"sport": "epl", "tags": "1,82,306,100639,100350", "series": "10188"},
                    {"sport": "nba", "tags": "1,745,100639", "series": "10345"}], {}
        if "/events?" in url:
            assert "series_id=10188" in url
            return EVENTS, {}
        if url.endswith("/books"):
            toks = [x["token_id"] for x in json.loads(data)]
            return [BOOKS[t] for t in toks if t in BOOKS], {}
        raise AssertionError(url)
    monkeypatch.setattr(P, "http_json", fake)
    prov = PolymarketProvider({"lookahead_hours": 36})
    qs = prov.poll(EventIndex(), NOW)
    assert len(qs) == 7 and prov.football_series() == ["10188"]
    assert sum(1 for u, _ in calls if u.endswith("/books")) == 1
    # outside the lookahead window -> nothing fetched from the order books
    prov2 = PolymarketProvider({"lookahead_hours": 1})
    assert prov2.poll(EventIndex(), NOW) == []
