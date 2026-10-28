"""Multi-sport support and the OddsPapi feed."""

import io
import json
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest

from footy.watch import config as C
from footy.watch import providers as P
from footy.watch.core import EventIndex, Quote, name_score, orient, settle, sport_allows
from footy.watch.oddspapi import OddsPapiProvider, _latest, map_market
from footy.watch.paper import PaperBook, find_result
from footy.watch.providers import BetfairProvider, OddsAPIProvider, PolymarketProvider
from footy.watch.rules import arb_alerts, value_alerts

NOW = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=6)
ISO = KO.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ core
def test_orientation_and_sport_separation():
    assert orient("ah", -1.5, "home", True) == (1.5, "away") and orient("ou", 8.5, "over", True) == (8.5, "over")
    assert orient("ml", None, "away", True) == (None, "home") and orient("1x2", None, "draw", True) == (None, "draw")
    idx = EventIndex()
    a = idx.add("x", "1", "Real Madrid", "Barcelona", KO)                           # football
    b = idx.add("y", "2", "Real Madrid", "Barcelona", KO, sport="basketball")       # basketball, same names
    assert a.id != b.id and b.id.startswith("basketball-") and b.sport == "basketball"
    ev, swapped = idx.add_oriented("z", "3", "Alcaraz C.", "Djokovic N.", KO, sport="tennis")
    ev2, swapped2 = idx.add_oriented("w", "4", "Novak Djokovic", "Carlos Alcaraz", KO, sport="tennis")
    assert ev2 is ev and swapped2 and not swapped


def test_names_and_ties():
    assert name_score("Djokovic, Novak", "Novak Djokovic") == 1.0
    assert name_score("A. Zverev", "Mischa Zverev") < 0.5 <= name_score("A. Zverev", "Alexander Zverev")
    assert settle("ml", None, "home", 20, 20) == 0.0 and settle("ml", None, "away", 3, 5) == 1.0
    assert sport_allows("tennis", "ml") and not sport_allows("tennis", "ah") and not sport_allows("soccer", "ml")


# ------------------------------------------------------------- the-odds-api
def test_oddsapi_two_way_sports_and_orientation():
    idx = EventIndex()
    idx.add("betfair", "b1", "Boston Red Sox", "New York Yankees", KO, sport="baseball")  # Betfair: Red Sox home
    data = [{"id": "m1", "commence_time": ISO, "home_team": "New York Yankees", "away_team": "Boston Red Sox",
             "bookmakers": [{"key": "pinnacle", "markets": [
                 {"key": "h2h", "outcomes": [{"name": "New York Yankees", "price": 1.80},
                                             {"name": "Boston Red Sox", "price": 2.10}]},
                 {"key": "spreads", "outcomes": [{"name": "New York Yankees", "price": 2.2, "point": -1.5},
                                                 {"name": "Boston Red Sox", "price": 1.7, "point": 1.5}]}]}]}]
    qs = OddsAPIProvider.parse(data, idx, "baseball_mlb", "baseball_mlb")
    assert len(idx.events) == 1
    got = {(q.market, q.line, q.outcome): q.price for q in qs}
    # Yankees are the canonical AWAY side: their -1.5 becomes home line +1.5, outcome away
    assert got[("ml", None, "away")] == 1.80 and got[("ml", None, "home")] == 2.10
    assert got[("ah", 1.5, "away")] == 2.2 and got[("ah", 1.5, "home")] == 1.7


# ------------------------------------------------------------------ Betfair
def _bf_cat(mid, mtype, evtype, name, runners):
    return {"marketId": mid, "marketStartTime": ISO, "eventType": {"id": evtype},
            "event": {"id": "e" + mid, "name": name, "openDate": ISO}, "description": {"marketType": mtype},
            "runners": [{"selectionId": i, "runnerName": r, "handicap": 0} for i, r in enumerate(runners, 1)]}


def _bf_book(mid, prices):
    return {"marketId": mid, "status": "OPEN", "inplay": False, "runners": [
        {"selectionId": i, "status": "ACTIVE", "ex": {"availableToBack": [{"price": p, "size": 100}],
                                                      "availableToLay": [{"price": p + 0.02, "size": 50}]}}
        for i, p in enumerate(prices, 1)]}


def test_betfair_sports_markets_and_winner_results(tmp_path):
    cat = [_bf_cat("1.1", "MATCH_ODDS", "7511", "NY Yankees @ Boston Red Sox", ["NY Yankees", "Boston Red Sox"]),
           _bf_cat("1.2", "MATCH_ODDS", "7524", "Rangers v Bruins", ["Rangers", "Bruins", "The Draw"]),
           _bf_cat("1.3", "MONEY_LINE", "7524", "Rangers v Bruins", ["Rangers", "Bruins"])]
    books = [_bf_book("1.1", [1.9, 2.0]), _bf_book("1.2", [2.5, 2.6, 4.2]), _bf_book("1.3", [1.95, 1.95])]
    idx = EventIndex()
    qs = BetfairProvider.parse(cat, books, idx)
    evs = {e.sport: e for e in idx.events.values()}
    assert set(evs) == {"baseball", "icehockey"} and evs["baseball"].home == "Boston Red Sox"  # "Away @ Home"
    mk = {(idx.events[q.event].sport, q.market, q.outcome) for q in qs}
    assert ("baseball", "ml", "away") in mk and ("icehockey", "1x2", "draw") in mk and ("icehockey", "ml", "home") in mk
    assert evs["icehockey"].meta["betfair_markets"]["1x2"]["id"] == "1.2"

    cfg = C.load(None)
    cfg["server"]["db"] = str(tmp_path / "w.db")
    book = PaperBook(cfg["server"]["db"], cfg)
    from footy.watch.rules import Alert
    ev = evs["baseball"]
    al = [Alert("value", ev.id, f"{ev.home} v {ev.away}", ev.kickoff, None, "ml", None, "away", "sport888", 2.2,
                2.0, 0.05, "", {}, "baseball"),
          Alert("value", ev.id, f"{ev.home} v {ev.away}", ev.kickoff, None, "ou", 8.5, "over", "sport888", 2.2,
                2.0, 0.05, "", {}, "baseball")]
    book.place(al, idx, NOW)
    assert {b["sport"] for b in book.bets()} == {"baseball"}

    prov = BetfairProvider({"app_key": "k"})
    prov._call = lambda op, params: [{"marketId": "1.1", "status": "CLOSED", "runners": [
        {"selectionId": 1, "status": "WINNER"}, {"selectionId": 2, "status": "LOSER"}]}]
    later = KO + timedelta(hours=4)
    book.settle(later, [prov])
    st = {(b["market"], b["venue"]): b["status"] for b in book.bets()}
    assert all(v == "won" for (m, _), v in st.items() if m == "ml")       # NY Yankees (away) won
    assert all(v == "open" for (m, _), v in st.items() if m == "ou")      # a winner can't settle a total


# ------------------------------------------------------------------- rules
def test_rules_only_compare_safe_markets_per_sport():
    cfg = C.load(None)
    idx = EventIndex()
    ev = idx.add("x", "1", "Novak Djokovic", "Carlos Alcaraz", KO, sport="tennis")
    q = lambda m, ln, o, b, p: Quote(ev.id, m, ln, o, b, p, NOW)  # noqa: E731
    games = [q("ah", -2.5, "home", "pinnacle", 1.9), q("ah", -2.5, "away", "pinnacle", 1.9),
             q("ah", -2.5, "home", "sport888", 2.4)]                       # games vs sets mix-ups: ignored
    assert not value_alerts(idx, games, cfg, NOW) and not arb_alerts(idx, games, cfg, NOW)
    ml = [q("ml", None, "home", "pinnacle", 1.80), q("ml", None, "away", "pinnacle", 2.10),
          q("ml", None, "away", "polymarket", 1 / 0.40)]
    arbs = arb_alerts(idx, ml, cfg, NOW)
    assert len(arbs) == 1 and arbs[0].sport == "tennis" and arbs[0].market == "ml"


def test_polymarket_us_moneyline():
    events = [{"id": "77", "title": "Yankees vs. Red Sox", "_sport": "baseball", "markets": [
        {"question": "Yankees vs. Red Sox", "outcomes": json.dumps(["Yankees", "Red Sox"]),
         "clobTokenIds": json.dumps(["a1", "a2"]), "sportsMarketType": "moneyline",
         "gameStartTime": KO.strftime("%Y-%m-%d %H:%M:%S+00"), "active": True, "closed": False}]}]
    books = {"a1": {"asks": [{"price": "0.55", "size": "100"}], "bids": [{"price": "0.53", "size": "100"}]},
             "a2": {"asks": [{"price": "0.47", "size": "100"}], "bids": [{"price": "0.45", "size": "100"}]}}
    idx = EventIndex()
    qs = PolymarketProvider.parse(events, books, idx, now=NOW)
    ev = next(iter(idx.events.values()))
    assert ev.sport == "baseball" and {(q.market, q.outcome) for q in qs} == {("ml", "home"), ("ml", "away")}
    prov = PolymarketProvider({"sports": ["mlb", "epl"]})
    assert prov.kind_of({"sport": "mlb", "tags": "1,3"}) == "baseball"
    assert prov.kind_of({"sport": "epl", "tags": "1,100350"}) == "soccer"
    assert prov.kind_of({"sport": "nba", "tags": "1"}) is None                  # not asked for


# ---------------------------------------------------------------- OddsPapi
MARKETS = [
    {"marketId": 101, "marketName": "Full Time Result", "period": "fulltime", "handicap": 0,
     "outcomes": [{"outcomeId": 101, "outcomeName": "1"}, {"outcomeId": 102, "outcomeName": "X"},
                  {"outcomeId": 103, "outcomeName": "2"}]},
    {"marketId": 1010, "marketName": "Over/Under", "period": "fulltime", "handicap": 2.5,
     "outcomes": [{"outcomeId": 1010, "outcomeName": "Over"}, {"outcomeId": 1011, "outcomeName": "Under"}]},
    {"marketId": 1068, "marketName": "Asian Handicap", "period": "fulltime", "handicap": -0.5,
     "outcomes": [{"outcomeId": 1068, "outcomeName": "1"}, {"outcomeId": 1069, "outcomeName": "2"}]},
    {"marketId": 104, "marketName": "Draw No Bet", "period": "fulltime",
     "outcomes": [{"outcomeId": 104, "outcomeName": "1"}, {"outcomeId": 105, "outcomeName": "2"}]},
    {"marketId": 1100, "marketName": "1st Half Result", "period": "fulltime",
     "outcomes": [{"outcomeId": 1, "outcomeName": "1"}, {"outcomeId": 2, "outcomeName": "X"},
                  {"outcomeId": 3, "outcomeName": "2"}]},
    {"marketId": 1200, "marketName": "Total Corners", "period": "fulltime", "handicap": 9.5,
     "outcomes": [{"outcomeId": 1, "outcomeName": "Over"}, {"outcomeId": 2, "outcomeName": "Under"}]},
    {"marketId": 1300, "marketName": "European Handicap", "period": "fulltime", "handicap": -1,
     "outcomes": [{"outcomeId": 1, "outcomeName": "1"}, {"outcomeId": 2, "outcomeName": "X"},
                  {"outcomeId": 3, "outcomeName": "2"}]},
    {"marketId": 1400, "marketName": "Over/Under", "period": "p1", "handicap": 1.5,
     "outcomes": [{"outcomeId": 1, "outcomeName": "Over"}, {"outcomeId": 2, "outcomeName": "Under"}]},
]


def test_market_catalogue_mapping():
    got = {m["marketId"]: map_market(m) for m in MARKETS}
    assert got[101]["market"] == "1x2" and got[101]["outcomes"] == {"101": "home", "102": "draw", "103": "away"}
    assert got[1010] == {"market": "ou", "line": 2.5, "outcomes": {"1010": "over", "1011": "under"}}
    assert got[1068]["market"] == "ah" and got[1068]["line"] == -0.5
    assert got[104]["market"] == "ah" and got[104]["line"] == 0.0           # draw no bet = handicap 0
    assert got[1100] is None and got[1200] is None and got[1300] is None and got[1400] is None
    winner = {"marketId": 9, "marketName": "Winner (incl. overtime)", "outcomes": [
        {"outcomeId": 1, "outcomeName": "1"}, {"outcomeId": 2, "outcomeName": "2"}]}
    run_line = {"marketId": 10, "marketName": "Run Line", "handicap": -1.5, "outcomes": [
        {"outcomeId": 1, "outcomeName": "1"}, {"outcomeId": 2, "outcomeName": "2"}]}
    assert map_market(winner)["market"] == "ml" and map_market(run_line)["line"] == -1.5
    extras = {"marketId": 11, "marketName": "Winner (incl. extra innings)", "outcomes": [
        {"outcomeId": 1, "outcomeName": "1"}, {"outcomeId": 2, "outcomeName": "2"}]}
    f5 = {"marketId": 12, "marketName": "1st 5 Innings Winner", "outcomes": [
        {"outcomeId": 1, "outcomeName": "1"}, {"outcomeId": 2, "outcomeName": "2"}]}
    assert map_market(extras)["market"] == "ml" and map_market(f5) is None


def test_latest_price_shapes():
    assert _latest({"players": {"0": {"price": 2.1, "active": True}}})["price"] == 2.1
    assert _latest({"players": {"0": [{"price": 2.0}, {"price": 2.2, "active": True}]}})["price"] == 2.2
    assert _latest({"price": 1.9})["price"] == 1.9 and _latest(None) is None


def _odds_payload(fid="f1"):
    def o(p):
        return {"players": {"0": {"price": p, "active": True}}}
    return [{"fixtureId": fid, "participant1Id": 1, "participant2Id": 2, "tournamentId": 17, "startTime": ISO,
             "bookmakerOdds": {
                 "pinnacle": {"markets": {"101": {"outcomes": {"101": o(2.10), "102": o(3.50), "103": o(3.60)}},
                                          "1010": {"outcomes": {"1010": o(1.95), "1011": o(1.92)}},
                                          "1100": {"outcomes": {"1": o(3.0)}}}},
                 "singbet": {"markets": {"101": {"outcomes": {"101": o(2.20), "102": {"players": {"0": {
                     "price": 3.4, "active": False}}}}}}}}}]


def test_parse_odds_by_tournaments():
    mmap = {str(m["marketId"]): map_market(m) for m in MARKETS if map_market(m)}
    fx = {"f1": {"home": "Arsenal", "away": "Chelsea", "start": KO.isoformat(), "tournamentId": "17",
                 "tournament": "Premier League"}}
    idx = EventIndex()
    qs = OddsPapiProvider.parse(_odds_payload(), fx, mmap, idx, "soccer", NOW)
    got = {(q.book, q.market, q.line, q.outcome): q.price for q in qs}
    assert got[("pinnacle", "1x2", None, "home")] == 2.10 and got[("pinnacle", "ou", 2.5, "under")] == 1.92
    assert got[("singbet", "1x2", None, "home")] == 2.20
    assert ("singbet", "1x2", None, "draw") not in got                      # suspended price skipped
    assert len(qs) == 6 and next(iter(idx.events.values())).league == "Premier League"


def test_poll_caches_counts_requests_and_spaces_sweeps(tmp_path, monkeypatch):
    calls = []

    def fake(url, data=None, headers=None, timeout=30.0):
        calls.append(url)
        assert "apiKey=secret" in url
        if "/sports?" in url:
            return [{"sportId": 10, "sportName": "Soccer"}, {"sportId": 13, "sportName": "Baseball"}], {}
        if "/markets?" in url:
            return MARKETS, {}
        if "/fixtures?" in url:
            return [{"fixtureId": "f1", "participant1Name": "Arsenal", "participant2Name": "Chelsea",
                     "startTime": ISO, "tournamentId": 17, "tournamentName": "Premier League", "hasOdds": True},
                    {"fixtureId": "f2", "participant1Name": "A", "participant2Name": "B", "startTime": ISO,
                     "tournamentId": 99, "hasOdds": False}], {}
        if "/odds-by-tournaments?" in url:
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            if len(q.get("bookmaker", [])) != 1 or "," in q["bookmaker"][0] or "bookmakers" in q:
                # what the real API answers (2026-10-28) when asked for several bookmakers at once
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(json.dumps({"error": {
                    "message": "Invalid number of bookmakers specified.", "code": "INVALID_PARAMETER",
                    "details": "Please provide exactly one bookmaker using the 'bookmaker' query parameter."}}).encode()))
            assert q["tournamentIds"] == ["17"]
            payload = _odds_payload()
            for f in payload:  # the API returns only the bookmaker asked for
                f["bookmakerOdds"] = {k: v for k, v in f["bookmakerOdds"].items() if k == q["bookmaker"][0]}
            return payload, {}
        raise AssertionError(url)
    monkeypatch.setattr(P, "http_json", fake)
    cfg = {"api_key": "secret", "cache_file": str(tmp_path / "c.json"), "sports": {"soccer": []}}
    prov = OddsPapiProvider(cfg)
    qs = prov.poll(EventIndex(), NOW)
    odds_calls = [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["bookmaker"][0] for u in calls
                  if "/odds-by-tournaments" in u]
    assert sorted(odds_calls) == ["pinnacle", "sbobet", "singbet"]      # one request per bookmaker
    assert len(qs) == 6 and prov.used == 6 and len(calls) == 6          # sports, markets, fixtures + 3 odds
    # a second sweep only fetches odds; the catalogue and fixtures come from the disk cache
    prov2 = OddsPapiProvider(cfg)
    prov2.poll(EventIndex(), NOW)
    assert prov2.used == 9 and sum("/odds-by-tournaments" in u for u in calls) == 6
    # 241 requests left, 3 per sweep, spread over the rest of the month
    assert prov2.interval(0, NOW) == pytest.approx(
        (datetime(2026, 12, 1, tzinfo=timezone.utc) - NOW).total_seconds() / ((241 - 10) / 3), rel=1e-6)
    prov2.cache["usage"]["used"] = 250
    with pytest.raises(RuntimeError):
        prov2.get("/sports")


def test_rate_limit_retry_and_missing_key(tmp_path, monkeypatch):
    seen = []

    def fake(url, data=None, headers=None, timeout=30.0):
        seen.append(url)
        if len(seen) == 1:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, None)
        return [{"sportId": 10, "sportName": "Soccer"}], {}
    monkeypatch.setattr(P, "http_json", fake)
    monkeypatch.setattr("footy.watch.oddspapi.time.sleep", lambda s: None)
    prov = OddsPapiProvider({"api_key": "k", "cache_file": str(tmp_path / "c.json")})
    assert prov.sport_ids() == {"soccer": 10} and prov.used == 2
    with pytest.raises(RuntimeError):
        OddsPapiProvider({"api_key": "", "cache_file": str(tmp_path / "d.json")}).get("/sports")


def test_scores_settle_only_what_they_decide():
    r = {"source": "x", "home": "Bruins", "away": "Rangers", "kickoff": KO, "home_goals": 3, "away_goals": 2}
    hockey_1x2 = {"home": "Rangers", "away": "Bruins", "kickoff": KO.isoformat(), "market": "1x2",
                  "sport": "icehockey", "meta": {}}
    assert find_result(hockey_1x2, [r]) is None                 # the final score may include overtime
    ml = {**hockey_1x2, "market": "ml"}
    got = find_result(ml, [r])                                   # listed the other way round: turned around
    assert (got["home_goals"], got["away_goals"]) == (2, 3)
    tennis = {**ml, "sport": "tennis", "home": "Bruins", "away": "Rangers"}
    assert find_result(tennis, [r]) is None
