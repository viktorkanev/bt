"""Tests for paper trading (forward test with an imaginary bankroll)."""

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from footy.watch import config as C
from footy.watch.core import EventIndex, Quote, devig, fit_goals, grid_fair_odds, net_odds, ou_result, score_grid, \
    settle
from footy.watch.paper import FootballDataResults, PaperBook, closing_fair, find_result, summarize, verdict
from footy.watch.rules import pick_alerts, value_alerts

NOW = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=5)


def _cfg(tmp_path, **paper):
    cfg = C.load(None)
    cfg["server"].update({"db": str(tmp_path / "w.db"), "picks_file": str(tmp_path / "picks.json"), "port": 0})
    cfg["paper"].update(paper)
    return cfg


def _market(idx, quotes_spec, when=NOW):
    ev = idx.add("t", "1", "Bolton Wanderers", "Wigan Athletic", KO, "E2")
    return ev, [Quote(ev.id, m, ln, o, b, p, when, lay) for m, ln, o, b, p, lay in quotes_spec]


BASE = [  # market, line, outcome, book, price, lay
    ("1x2", None, "home", "pinnacle", 2.00, None), ("1x2", None, "draw", "pinnacle", 3.60, None),
    ("1x2", None, "away", "pinnacle", 4.00, None),
    ("1x2", None, "home", "sport888", 2.30, None),                       # soft book out of line
    ("1x2", None, "home", "betfair_ex_eu", 2.26, 2.30),                  # exchange, also value after 5%
    ("ou", 2.5, "over", "pinnacle", 1.80, None), ("ou", 2.5, "under", "pinnacle", 2.10, None),
    ("ou", 2.5, "over", "betfair_ex_eu", 1.85, 1.87), ("ou", 2.5, "over", "sport888", 1.90, None),
]


# ------------------------------------------------------------------ maths
def test_over_under_quarter_lines_and_settlement():
    assert ou_result(3, 2.75, "over") == 0.5 and ou_result(2, 2.25, "under") == 0.5
    assert ou_result(3, 3.0, "over") == 0.0 and ou_result(4, 3.0, "over") == 1.0
    assert settle("1x2", None, "draw", 1, 1) == 1.0 and settle("1x2", None, "home", 1, 1) == -1.0
    assert settle("ah", -0.25, "home", 0, 0) == -0.5 and settle("ah", -0.25, "away", 0, 0) == 0.5


def test_goal_model_recovers_market_and_moves_lines():
    g = score_grid(1.7, 1.05)
    ah, ou = (-0.5, grid_fair_odds(g, "ah", -0.5, "home")), (2.5, grid_fair_odds(g, "ou", 2.5, "over"))
    lh, la = fit_goals(ah=ah, ou=ou)
    assert abs(lh - 1.7) < 1e-4 and abs(la - 1.05) < 1e-4
    # a bigger handicap must cost more, pushes included
    assert grid_fair_odds(g, "ah", -1.0, "home") > grid_fair_odds(g, "ah", -0.75, "home") > ah[1]


def test_closing_fair_same_line_and_converted():
    idx = EventIndex()
    ev, qs = _market(idx, [("ah", -0.75, "home", "pinnacle", 2.05, None), ("ah", -0.75, "away", "pinnacle", 1.85, None),
                           ("ou", 2.5, "over", "pinnacle", 1.90, None), ("ou", 2.5, "under", "pinnacle", 2.00, None)])
    same = closing_fair(qs, "ah", -0.75, "home", ["pinnacle"])
    assert same[2] == "same line" and abs(same[0] - 1 / devig([2.05, 1.85])[0]) < 1e-12
    conv = closing_fair(qs, "ah", -0.5, "home", ["pinnacle"])
    assert conv[2] == "converted" and conv[0] < same[0]          # -0.5 is easier than -0.75 -> shorter
    assert closing_fair(qs, "1x2", None, "home", ["pinnacle"]) is None


# ---------------------------------------------------------------- ledger
def test_places_one_bet_per_place_at_that_places_price(tmp_path):
    cfg = _cfg(tmp_path)
    idx = EventIndex()
    ev, qs = _market(idx, BASE)
    book = PaperBook(cfg["server"]["db"], cfg)
    al = value_alerts(idx, qs, cfg, NOW, targets=["*"])
    book.place(al, idx, NOW)
    bets = {(b["venue"], b["market"]): b for b in book.bets()}
    assert bets[("best", "1x2")]["book"] == "sport888" and bets[("best", "1x2")]["price"] == 2.30
    bf = bets[("betfair", "1x2")]
    assert bf["book"] == "betfair_ex_eu" and abs(bf["net_price"] - net_odds(2.26, 0.05)) < 1e-12
    assert ("pinnacle", "1x2") not in bets            # Pinnacle is the reference: no edge against itself
    fair_home = devig([2.00, 3.60, 4.00])[0]
    assert abs(bets[("best", "1x2")]["ev"] - (2.30 * fair_home - 1)) < 1e-12
    n = len(book.bets())
    book.place(value_alerts(idx, qs, cfg, NOW + timedelta(minutes=5), targets=["*"]), idx, NOW + timedelta(minutes=5))
    assert len(book.bets()) == n                     # one bet per signal and place, never repeated
    late = KO - timedelta(minutes=3)
    book.reset()
    book.place(value_alerts(idx, [Quote(q.event, q.market, q.line, q.outcome, q.book, q.price, late) for q in qs],
                            cfg, late, targets=["*"]), idx, late)
    assert book.bets() == []                         # too close to kickoff


def test_research_pick_is_bet_at_all_three_places(tmp_path):
    cfg = _cfg(tmp_path)
    idx = EventIndex()
    ev, qs = _market(idx, BASE)
    picks = [{"rule": "F1", "rule_name": "F1 lower-league totals", "league": "E2", "kickoff": KO.isoformat(),
              "home": "Bolton", "away": "Wigan", "market": "ou", "side": "over", "line": 2.5, "prob": 0.60,
              "min_edge": 0.03}]
    book = PaperBook(cfg["server"]["db"], cfg)
    book.place(pick_alerts(picks, idx, qs, cfg, NOW, books=["*"])[0], idx, NOW)
    got = {b["venue"]: (b["book"], b["price"], round(b["ev"], 4)) for b in book.bets(strategy="F1")}
    assert got == {"best": ("sport888", 1.90, round(0.6 * 1.90 - 1, 4)),
                   "pinnacle": ("pinnacle", 1.80, round(0.6 * 1.80 - 1, 4)),
                   "betfair": ("betfair_ex_eu", 1.85, round(0.6 * net_odds(1.85, 0.05) - 1, 4))}
    assert all(b["league"] == "E2" and b["strategy_name"] == "F1 lower-league totals" for b in book.bets())


def test_kelly_stake_uses_the_places_bankroll(tmp_path):
    cfg = _cfg(tmp_path, staking="kelly", kelly_fraction=0.5, max_stake_pct=0.05, bankroll=2000)
    idx = EventIndex()
    ev, qs = _market(idx, BASE)
    book = PaperBook(cfg["server"]["db"], cfg)
    book.place(value_alerts(idx, qs, cfg, NOW, targets=["*"]), idx, NOW)
    b = next(x for x in book.bets(venue="best") if x["market"] == "1x2")
    assert abs(b["stake"] - round(2000 * min(0.5 * b["ev"] / (2.30 - 1), 0.05), 2)) < 1e-9


def test_closing_line_and_settlement(tmp_path):
    cfg = _cfg(tmp_path)
    idx = EventIndex()
    ev, qs = _market(idx, BASE)
    book = PaperBook(cfg["server"]["db"], cfg)
    book.place(value_alerts(idx, qs, cfg, NOW, targets=["*"]), idx, NOW)
    # the market moves our way: Pinnacle home shortens to 1.90 before kickoff
    t = KO - timedelta(minutes=8)
    close = [Quote(ev.id, "1x2", None, o, "pinnacle", p, t) for o, p in (("home", 1.90), ("draw", 3.70), ("away", 4.40))]
    assert book.update_closing(idx, close, t) >= 1
    b = next(x for x in book.bets(venue="best") if x["market"] == "1x2")
    fair = 1 / devig([1.90, 3.70, 4.40])[0]
    assert abs(b["close_fair"] - fair) < 1e-9 and abs(b["clv"] - (2.30 / fair - 1)) < 1e-9
    # quotes after kickoff are in-play and must not move the closing line
    book.update_closing(idx, [Quote(ev.id, "1x2", None, o, "pinnacle", 9.0, KO + timedelta(minutes=5))
                              for o in ("home", "draw", "away")], KO + timedelta(minutes=5))
    assert abs(book.bets(venue="best")[0]["close_fair"] - fair) < 1e-9

    class Src:
        name = "test"

        def results(self, bets, now):
            return [{"source": "test", "home": "Bolton", "away": "Wigan", "kickoff": KO,
                     "home_goals": 2, "away_goals": 1}]
    later = KO + timedelta(hours=3)
    assert book.settle(later, [Src()]) == len(book.bets())
    for b in book.bets():
        exp = b["stake"] * (b["net_price"] - 1) if (b["market"] == "1x2" or b["outcome"] == "over") else -b["stake"]
        assert b["status"] in ("won", "lost") and abs(b["profit"] - exp) < 1e-6


def test_void_when_no_result_arrives(tmp_path):
    cfg = _cfg(tmp_path, void_after_days=3)
    idx = EventIndex()
    ev, qs = _market(idx, BASE)
    book = PaperBook(cfg["server"]["db"], cfg)
    book.place(value_alerts(idx, qs, cfg, NOW, targets=["*"]), idx, NOW)
    assert book.settle(KO + timedelta(days=1), []) == 0
    assert book.settle(KO + timedelta(days=4), []) == len(book.bets())
    assert {b["status"] for b in book.bets()} == {"void"} and sum(b["profit"] for b in book.bets()) == 0


def test_football_data_results_parse_and_match():
    csv_text = ("﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
                "E2,07/11/2026,15:00,Bolton,Wigan,2,1,H\n"
                "E2,07/11/2026,15:00,Man United,Leeds,,,\n")
    rows = FootballDataResults.parse(csv_text)
    assert len(rows) == 1 and rows[0]["kickoff"] == datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
    src = FootballDataResults(fetch=lambda url: csv_text)
    assert FootballDataResults.season(KO) == "2627"
    bet = {"home": "Bolton Wanderers", "away": "Wigan Athletic", "kickoff": KO.isoformat(), "league": "E2", "meta": {}}
    r = find_result(bet, src.results([bet], NOW))
    assert r["home_goals"] == 2 and r["away_goals"] == 1


def test_summary_series_and_verdict(tmp_path):
    bets = [{"status": "won", "stake": 10, "profit": 9.0, "ev": 0.05, "clv": 0.02, "net_price": 1.9,
             "kickoff": KO.isoformat(), "home": "A", "away": "B"},
            {"status": "lost", "stake": 10, "profit": -10.0, "ev": 0.05, "clv": -0.01, "net_price": 2.0,
             "kickoff": KO.isoformat(), "home": "C", "away": "D"},
            {"status": "open", "stake": 10, "profit": None, "ev": 0.04, "clv": None, "net_price": 2.0,
             "kickoff": KO.isoformat(), "home": "E", "away": "F"}]
    s = summarize(bets, 1000)
    assert s["settled"] == 2 and s["open"] == 1 and s["profit"] == -1.0 and s["bankroll"] == 999.0
    assert [p["actual"] for p in s["series"]] == [9.0, -1.0] and s["series"][-1]["ev"] == 1.0
    assert s["beat_close"] == 0.5 and s["series"][-1]["band"] > 0
    assert "too few" in verdict(s)


def test_paper_page_and_exports(tmp_path):
    from footy.watch.server import Watcher, serve
    odds = {"events": [{"id": "1", "home": "Bolton Wanderers", "away": "Wigan Athletic", "kickoff": KO.isoformat(),
                        "league": "E2"}],
            "quotes": [{"event": "1", "market": m, "line": ln, "outcome": o, "book": b, "price": p, "lay": lay,
                        "updated": NOW.isoformat()} for m, ln, o, b, p, lay in BASE],
            "results": [{"home": "Bolton", "away": "Wigan", "kickoff": KO.isoformat(), "home_goals": 0,
                         "away_goals": 0}]}
    (tmp_path / "odds.json").write_text(json.dumps(odds))
    cfg = _cfg(tmp_path, football_data_results=False)
    cfg["file"].update({"enabled": True, "path": str(tmp_path / "odds.json")})
    cfg["notify"]["console"] = False
    w = Watcher(cfg)
    w.poll_once(NOW)                                   # places the bets
    # sport888 2.30 / Pinnacle 3.60 / 4.00 is also a 3.9% arbitrage
    assert {b["venue"] for b in w.paper.bets()} == {"best", "betfair", "arb"}
    w.next_poll = {k: 0 for k in w.next_poll}
    w.next_settle = 0
    w.poll_once(KO + timedelta(hours=3))               # result in the file -> settled (draw: home bets lose)
    assert all(b["status"] == "lost" for b in w.paper.bets() if b["market"] == "1x2" and b["venue"] != "arb")
    arb = w.paper.bets(venue="arb")
    assert len(arb) == 3 and abs(sum(b["stake"] for b in arb) - 100) < 0.05
    assert 3.0 < sum(b["profit"] for b in arb) < 4.5      # ~3.9% locked in, whatever the score
    srv = serve(w)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/paper?venue=betfair").read().decode()
        assert "Paper trading" in page and "pchart-data" in page and "Betfair Exchange" in page
        assert "Paper trading" in urllib.request.urlopen(base + "/").read().decode()
        data = json.loads(urllib.request.urlopen(base + "/api/paper").read())
        assert data["venues"]["best"]["settled"] >= 1 and len(data["bets"]) == len(w.paper.bets())
        text = urllib.request.urlopen(base + "/api/paper.csv").read().decode()
        assert text.splitlines()[0].startswith("id,placed,venue,strategy")
        assert urllib.request.urlopen(base + "/paper?venue=nope&strategy=zzz").status == 200
    finally:
        srv.shutdown()


@pytest.mark.parametrize("staking", ["flat", "kelly"])
def test_cli_paper_summary(tmp_path, staking, capsys):
    from footy.watch.__main__ import main
    cfgf = tmp_path / "watch.json"
    cfgf.write_text(json.dumps({"server": {"db": str(tmp_path / "w.db")}, "paper": {"staking": staking}}))
    cfg = C.load(str(cfgf))
    idx = EventIndex()
    ev, qs = _market(idx, BASE)
    PaperBook(cfg["server"]["db"], cfg).place(value_alerts(idx, qs, cfg, NOW, targets=["*"]), idx, NOW)
    assert main(["paper", "--config", str(cfgf)]) == 0
    out = capsys.readouterr().out
    assert "Best price" in out and "Wrong odds" in out
    assert main(["paper-reset", "--config", str(cfgf)]) == 1
    assert main(["paper-reset", "--config", str(cfgf), "--yes"]) == 0


def test_oddsapi_scores_parse_and_match_by_id():
    from footy.watch.providers import OddsAPIProvider
    data = [{"id": "abc", "completed": True, "commence_time": KO.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "home_team": "Bolton Wanderers", "away_team": "Wigan Athletic",
             "scores": [{"name": "Bolton Wanderers", "score": "1"}, {"name": "Wigan Athletic", "score": "3"}]},
            {"id": "live", "completed": False, "commence_time": KO.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "home_team": "X", "away_team": "Y", "scores": None}]
    rows = OddsAPIProvider.parse_scores(data)
    assert len(rows) == 1 and (rows[0]["home_goals"], rows[0]["away_goals"]) == (1, 3)
    bet = {"home": "Bolton", "away": "Wigan", "kickoff": KO.isoformat(), "meta": {"sources": {"oddsapi": "abc"}}}
    assert find_result(bet, rows)["id"] == "abc"
    assert settle("ah", 0.75, "home", 1, 3) == -1.0 and settle("ou", 3.75, "over", 1, 3) == 0.5
