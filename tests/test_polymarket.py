"""Polymarket feed, prediction-market fees and arbitrage alerts."""

import json
import urllib.parse
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


def test_joins_the_other_feeds_match_when_polymarket_names_differ():
    idx = EventIndex()
    idx.add("oddsapi", "x", "Olympiacos", "PAOK Thessaloniki", KO)
    idx.add("oddsapi", "y", "Aris", "AEK Athens", KO)                  # same kickoff, different match
    ev = {"id": "9201", "title": "Olympiakós SFP vs. PAOK FC", "markets": [
        _m("Will Olympiakós SFP win on 2026-11-07?", ["Yes", "No"], ["611", "612"], "moneyline",
           groupItemTitle="Olympiakós SFP"),
        _m("Will Olympiakós SFP vs. PAOK FC end in a draw?", ["Yes", "No"], ["621", "622"], "moneyline",
           groupItemTitle="Draw (Olympiakós SFP vs. PAOK FC)"),
        _m("Will PAOK FC win on 2026-11-07?", ["Yes", "No"], ["631", "632"], "moneyline", groupItemTitle="PAOK FC")]}
    books = {t: _book(t, a, 100, a - 0.02) for t, a in (("611", 0.50), ("621", 0.27), ("631", 0.26))}
    qs = PolymarketProvider.parse([ev], books, idx, now=NOW)
    assert len(idx.events) == 2                                         # no duplicate "Olympiakos SFP v PAOK"
    olympiacos = next(e.id for e in idx.events.values() if e.home == "Olympiacos")
    assert {q.event for q in qs} == {olympiacos}
    assert {q.outcome: round(1 / q.price, 2) for q in qs} == {"home": 0.50, "draw": 0.27, "away": 0.26}


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


# A match only Polymarket lists (no other feed has it, so it can't be compared with anything).
OTHER = [{"id": "9101", "title": "AS Trenčín vs. ŠK Slovan Bratislava", "markets": [
    _m("Will AS Trenčín win on 2026-11-07?", ["Yes", "No"], ["811", "812"], "moneyline", groupItemTitle="AS Trenčín"),
    _m("Will AS Trenčín vs. ŠK Slovan Bratislava end in a draw?", ["Yes", "No"], ["821", "822"], "moneyline",
       groupItemTitle="Draw (AS Trenčín vs. ŠK Slovan Bratislava)"),
    _m("Will ŠK Slovan Bratislava win on 2026-11-07?", ["Yes", "No"], ["831", "832"], "moneyline",
       groupItemTitle="ŠK Slovan Bratislava")]}]


SLUGS = {"9001": "epl-ars-che-2026-11-07", "9002": "epl-ars-che-2026-11-07-more-markets",
         "9101": "svk1-tre-sba-2026-11-07"}


def _rows(events):
    """Gamma /markets rows: flat markets, each naming its parent event, as the real API returns them."""
    return [{**m, "slug": f"{SLUGS[e['id']]}-m{i}", "endDate": "2026-11-07T15:00:00Z", "feesEnabled": True,
             "feeSchedule": {"exponent": 1, "rate": 0.05, "takerOnly": True, "rebateRate": 0.15},
             "events": [{"id": e["id"], "title": e["title"], "slug": SLUGS[e["id"]], "seriesSlug": "soccer-epl",
                         "startTime": "2026-11-07T15:00:00Z", "endDate": "2026-11-07T15:00:00Z"}]}
            for e in events for i, m in enumerate(e["markets"])]


def _fake_gamma(rows):
    def fake(url, data=None, headers=None, timeout=30.0):
        parts = urllib.parse.urlsplit(url)
        if parts.path == "/markets":
            q = urllib.parse.parse_qs(parts.query)
            want = [r for r in rows if r["sportsMarketType"] == q["sports_market_types"][0]]
            return want[int(q["offset"][0]):int(q["offset"][0]) + int(q["limit"][0])], {}
        if parts.path == "/books":
            return [BOOKS[t] for t in (x["token_id"] for x in json.loads(data)) if t in BOOKS], {}
        raise AssertionError(f"unexpected request {url}")
    return fake


def test_prices_link_to_their_polymarket_page(monkeypatch):
    monkeypatch.setattr(P, "http_json", _fake_gamma(_rows(EVENTS)))
    idx = EventIndex()
    idx.add("oddsapi", "x", "Arsenal", "Chelsea", KO)
    got = {(q.market, q.line, q.outcome): q.link for q in PolymarketProvider({}).poll(idx, NOW)}
    # the draw is the main event's 2nd market; O/U 2.5 the first of "More Markets"
    assert got[("1x2", None, "draw")] == "https://polymarket.com/event/epl-ars-che-2026-11-07/epl-ars-che-2026-11-07-m1"
    assert got[("ou", 2.5, "over")] == got[("ou", 2.5, "under")] == \
        "https://polymarket.com/event/epl-ars-che-2026-11-07-more-markets/epl-ars-che-2026-11-07-more-markets-m0"


def test_poll_asks_only_for_needed_markets_and_prices_only_matches_other_feeds_have(monkeypatch):
    rows, calls = _rows(EVENTS + OTHER), []

    def fake(url, data=None, headers=None, timeout=30.0):
        calls.append(url)
        parts = urllib.parse.urlsplit(url)
        if parts.path == "/markets":
            q = urllib.parse.parse_qs(parts.query)
            assert q["tag_id"] == ["100350"] and q["closed"] == ["false"]
            assert _pm_time(q["end_date_min"][0]) <= NOW and _pm_time(q["end_date_max"][0]) == NOW + timedelta(hours=36)
            want = [r for r in rows if r["sportsMarketType"] == q["sports_market_types"][0]]
            off, lim = int(q["offset"][0]), int(q["limit"][0])
            return want[off:off + lim], {}
        if parts.path == "/books":
            toks = [x["token_id"] for x in json.loads(data)]
            assert not [t for t in toks if t.startswith("8")]          # the Polymarket-only match isn't priced
            return [BOOKS[t] for t in toks if t in BOOKS], {}
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(P, "http_json", fake)
    idx = EventIndex()
    idx.add("oddsapi", "x", "Arsenal", "Chelsea", KO)                  # the match the Odds API also has
    qs = PolymarketProvider({"lookahead_hours": 36}).poll(idx, NOW)
    assert len(qs) == 7 and len(idx.events) == 1
    assert {u.split("?")[0].rsplit("/", 1)[-1] for u in calls} == {"markets", "books"}   # no crawl of every series
    assert sum(1 for u in calls if u.endswith("/books")) == 1


def test_match_is_priced_once_another_feed_lists_it(monkeypatch):
    rows = _rows(EVENTS)

    def fake(url, data=None, headers=None, timeout=30.0):
        parts = urllib.parse.urlsplit(url)
        if parts.path == "/markets":
            q = urllib.parse.parse_qs(parts.query)
            want = [r for r in rows if r["sportsMarketType"] == q["sports_market_types"][0]]
            return want[int(q["offset"][0]):int(q["offset"][0]) + int(q["limit"][0])], {}
        if parts.path == "/books":
            return [BOOKS[t] for t in (x["token_id"] for x in json.loads(data)) if t in BOOKS], {}
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(P, "http_json", fake)
    idx, prov = EventIndex(), PolymarketProvider({"lookahead_hours": 36})
    assert prov.poll(idx, NOW) == []                                   # no other feed has the match yet
    idx.add("oddsapi", "x", "Arsenal", "Chelsea", KO)                  # the next Odds API sweep adds it
    assert len(prov.poll(idx, NOW)) == 7
    assert len(prov.poll(idx, NOW)) == 7 and len(idx.events) == 1      # and keeps pricing it, no duplicate


def test_arbitrage_paper_trades_every_book_while_alerts_stay_at_yours(tmp_path):
    from footy.watch.server import Watcher
    cfg = C.load(None)
    cfg["server"].update({"db": str(tmp_path / "w.db"), "picks_file": str(tmp_path / "picks.json")})
    cfg["alerts"].update({"arb_books": ["pinnacle", "polymarket"], "value_alerts": False})
    cfg["notify"] = {"console": False}
    w = Watcher(cfg, provider_list=[])
    ev = w.index.add("t", "1", "Arsenal", "Chelsea", KO)
    quotes = [Quote(ev.id, "1x2", None, o, b, p, NOW) for o, b, p in (
        ("home", "pinnacle", 2.20), ("draw", "pinnacle", 3.40), ("away", "pinnacle", 3.50),
        ("home", "polymarket", 1 / 0.40), ("draw", "polymarket", 1 / 0.24), ("away", "sport888", 3.80))]
    w.quotes = {(q.event, q.market, q.line, q.book, q.outcome): q for q in quotes}
    sent = [a for a in w.evaluate(NOW) if a.kind == "arb"]
    assert len(sent) == 1 and {lg["book"] for lg in sent[0].extra["legs"]} == {"pinnacle", "polymarket"}
    legs = {b["outcome"]: b["book"] for b in w.paper.bets(venue="arb")}
    assert legs == {"home": "polymarket", "draw": "polymarket", "away": "sport888"}   # best prices anywhere
