"""Tests for the standard-library odds watcher."""

import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from footy.ratings import poisson_math as pm
from footy.watch import config as C
from footy.watch.core import EventIndex, Quote, ah_ev, ah_result, devig, match_event, name_score, net_odds
from footy.watch.providers import BetfairProvider, OddsAPIProvider
from footy.watch.rules import pick_alerts, value_alerts
from footy.watch.state import State

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=3)


def test_devig_and_commission():
    p = devig([1.5, 4.2, 7.0])
    assert abs(sum(p) - 1) < 1e-9 and p[2] < (1 / 7.0) / sum(1 / x for x in (1.5, 4.2, 7.0))
    assert abs(net_odds(2.0, 0.05) - 1.95) < 1e-12


def test_ah_settlement_matches_numpy_version():
    for line in (-1.75, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.25):
        for gd in range(-3, 4):
            ref = float(pm.ah_settle(np.array([gd]), np.array([line]))[0])
            assert ah_result(gd, line, "home") == ref
            assert ah_result(gd, line, "away") == -ref


def test_ah_ev_matches_numpy_version():
    m = pm.score_matrix([1.6], [1.1], -0.07)
    diffs, pgd = pm.goal_diff_dist(m)
    dist = {int(d): float(p) for d, p in zip(diffs, pgd[0])}
    for line in (-0.75, -0.25, 0.5):
        eh, ea = pm.ah_ev(diffs, pgd, np.array([line]), np.array([1.95]), np.array([1.90]))
        assert abs(ah_ev(dist, line, "home", 1.95) - eh[0]) < 1e-9
        assert abs(ah_ev(dist, line, "away", 1.90) - ea[0]) < 1e-9


@pytest.mark.parametrize("a,b", [("Man United", "Manchester United"), ("Ath Bilbao", "Athletic Bilbao"),
                                 ("Nott'm Forest", "Nottingham Forest"), ("M'gladbach", "Borussia Monchengladbach"),
                                 ("Paris SG", "Paris Saint Germain"), ("Sheffield Weds", "Sheffield Wednesday"),
                                 ("Wigan", "Wigan Athletic"), ("Inter", "Inter Milan")])
def test_team_names_match(a, b):
    assert name_score(a, b) >= 0.85


def test_match_event_picks_the_right_fixture():
    idx = EventIndex()
    idx.add("x", "1", "Manchester United", "Aston Villa", KO)
    idx.add("x", "2", "Manchester City", "Arsenal", KO)
    idx.add("x", "3", "Manchester City", "Arsenal", KO + timedelta(days=7))
    ev = match_event("Man City", "Arsenal", KO + timedelta(hours=1), idx.events.values())
    assert ev.home == "Manchester City" and ev.kickoff == KO


ODDSAPI = [{
    "id": "e1", "sport_key": "soccer_england_league1", "commence_time": KO.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "home_team": "Bolton Wanderers", "away_team": "Wigan Athletic",
    "bookmakers": [
        {"key": "pinnacle", "last_update": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": [
            {"key": "h2h", "outcomes": [{"name": "Bolton Wanderers", "price": 2.10},
                                        {"name": "Wigan Athletic", "price": 3.60},
                                        {"name": "Draw", "price": 3.40}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.95, "point": 2.5},
                                           {"name": "Under", "price": 1.93, "point": 2.5}]},
            {"key": "spreads", "outcomes": [{"name": "Bolton Wanderers", "price": 1.98, "point": -0.25},
                                            {"name": "Wigan Athletic", "price": 1.90, "point": 0.25}]}]},
        {"key": "betfair_ex_eu", "last_update": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "price": 2.12, "point": 2.5},
                                           {"name": "Under", "price": 1.88, "point": 2.5}]}]}]}]


def test_oddsapi_parse():
    idx = EventIndex()
    qs = OddsAPIProvider.parse(ODDSAPI, idx, "E2")
    ah = {(q.outcome, q.line) for q in qs if q.market == "ah"}
    assert ah == {("home", -0.25), ("away", -0.25)}  # both expressed as the HOME line
    assert {q.outcome for q in qs if q.market == "1x2"} == {"home", "draw", "away"}
    ev = next(iter(idx.events.values()))
    assert ev.league == "E2" and ev.kickoff == KO


def test_betfair_parse():
    cat = [{"marketId": "1.1", "marketName": "Match Odds", "marketStartTime": KO.isoformat(),
            "event": {"id": "9", "name": "Bolton v Wigan", "openDate": KO.isoformat()},
            "description": {"marketType": "MATCH_ODDS"},
            "runners": [{"selectionId": 1, "runnerName": "Bolton", "handicap": 0},
                        {"selectionId": 2, "runnerName": "Wigan", "handicap": 0},
                        {"selectionId": 3, "runnerName": "The Draw", "handicap": 0}]},
           {"marketId": "1.2", "marketName": "Asian Handicap", "marketStartTime": KO.isoformat(),
            "event": {"id": "9", "name": "Bolton v Wigan", "openDate": KO.isoformat()},
            "description": {"marketType": "ASIAN_HANDICAP"},
            "runners": [{"selectionId": 1, "runnerName": "Bolton", "handicap": -0.25},
                        {"selectionId": 2, "runnerName": "Wigan", "handicap": 0.25}]}]
    book = [{"marketId": "1.1", "status": "OPEN", "inplay": False, "runners": [
        {"selectionId": 1, "handicap": 0, "status": "ACTIVE",
         "ex": {"availableToBack": [{"price": 2.12, "size": 150}], "availableToLay": [{"price": 2.16, "size": 90}]}},
        {"selectionId": 3, "handicap": 0, "status": "ACTIVE",
         "ex": {"availableToBack": [{"price": 3.45, "size": 50}], "availableToLay": []}}]},
        {"marketId": "1.2", "status": "OPEN", "inplay": False, "runners": [
            {"selectionId": 2, "handicap": 0.25, "status": "ACTIVE",
             "ex": {"availableToBack": [{"price": 1.95, "size": 80}], "availableToLay": [{"price": 1.98}]}}]}]
    qs = BetfairProvider.parse(cat, book, EventIndex())
    assert {(q.market, q.outcome) for q in qs} == {("1x2", "home"), ("1x2", "draw"), ("ah", "away")}
    ah = next(q for q in qs if q.market == "ah")
    assert ah.line == -0.25 and ah.lay == 1.98 and ah.size == 80


def _cfg(tmp_path, **alerts):
    cfg = C.load(None)
    cfg["server"].update({"db": str(tmp_path / "w.db"), "picks_file": str(tmp_path / "picks.json"), "port": 0})
    cfg["alerts"].update(alerts)
    return cfg


def test_value_alert_uses_reference_and_commission(tmp_path):
    idx = EventIndex()
    qs = OddsAPIProvider.parse(ODDSAPI, idx)
    cfg = _cfg(tmp_path, min_edge=0.02)
    al = value_alerts(idx, qs, cfg, NOW)
    over = [a for a in al if a.market == "ou" and a.outcome == "over"]
    assert len(over) == 1 and over[0].book == "betfair_ex_eu"
    fair_over = devig([1.95, 1.93])[0]
    assert abs(over[0].edge - (net_odds(2.12, 0.05) * fair_over - 1)) < 1e-12
    cfg["commission"]["betfair_ex_eu"] = 0.10       # more commission -> no longer value
    assert not [a for a in value_alerts(idx, qs, cfg, NOW) if a.market == "ou"]
    stale = [Quote(q.event, q.market, q.line, q.outcome, q.book, q.price, NOW - timedelta(hours=2))
             if q.book == "pinnacle" else q for q in qs]
    assert not value_alerts(idx, stale, cfg, NOW)  # stale reference -> no alerts


def test_pick_alert_recalculates_moved_handicap_line(tmp_path):
    idx = EventIndex()
    qs = OddsAPIProvider.parse(ODDSAPI, idx)
    m = pm.score_matrix([1.2], [1.5], -0.07)        # our model likes Wigan
    diffs, pgd = pm.goal_diff_dist(m)
    gd = {str(int(d)): float(p) for d, p in zip(diffs, pgd[0]) if p > 1e-6}
    picks = [{"rule": "F2", "rule_name": "Luck fade", "kickoff": KO.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "home": "Bolton", "away": "Wigan", "market": "ah", "side": "away", "line": 0.0,
              "min_edge": 0.0, "gd_dist": gd, "selection": "Wigan +0.00"},
             {"rule": "F1", "kickoff": KO.strftime("%Y-%m-%dT%H:%M:%SZ"), "home": "Bolton", "away": "Wigan",
              "market": "ou", "side": "over", "line": 2.5, "prob": 0.40, "min_edge": 0.03}]
    alerts, status = pick_alerts(picks, idx, qs, _cfg(tmp_path), NOW)
    ah = [a for a in alerts if a.market == "ah"]
    assert ah and ah[0].line == -0.25 and ah[0].outcome == "away"   # offered line differs from the pick's
    assert status[0]["matched"] and status[0]["best"]["edge"] > 0
    assert not [a for a in alerts if a.market == "ou"]                # 40% at ~2 is no value
    assert status[1]["best"]["need"] > 2.4


def test_state_deduplicates(tmp_path):
    idx = EventIndex()
    qs = OddsAPIProvider.parse(ODDSAPI, idx)
    a = value_alerts(idx, qs, _cfg(tmp_path, min_edge=0.02), NOW)[0]
    st = State(str(tmp_path / "s.db"))
    assert st.should_send(a, NOW, 240, 0.01)
    st.record(a, NOW)
    assert not st.should_send(a, NOW + timedelta(minutes=5), 240, 0.01)
    a.edge += 0.02
    assert st.should_send(a, NOW + timedelta(minutes=5), 240, 0.01)
    assert st.should_send(a, NOW + timedelta(hours=5), 240, 0.5)


def test_server_endpoints(tmp_path):
    from footy.watch.server import Watcher, serve
    odds = {"events": [{"id": "e1", "home": "Bolton", "away": "Wigan", "kickoff": KO.isoformat()}],
            "quotes": []}
    (tmp_path / "odds.json").write_text(json.dumps(odds))
    cfg = _cfg(tmp_path)
    cfg["server"]["token"] = "s3cret"
    cfg["file"].update({"enabled": True, "path": str(tmp_path / "odds.json")})
    w = Watcher(cfg)
    srv = serve(w)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Odds watch" in urllib.request.urlopen(base + "/").read()
        body = json.dumps({"picks": [{"home": "Bolton", "away": "Wigan", "kickoff": KO.isoformat(),
                                      "market": "ou", "side": "over", "line": 2.5, "prob": 0.6}]}).encode()
        req = urllib.request.Request(base + "/api/picks", data=body, headers={"X-Token": "wrong"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req)
        req = urllib.request.Request(base + "/api/picks", data=body, headers={"X-Token": "s3cret"})
        assert json.loads(urllib.request.urlopen(req).read())["picks"] == 1
        st = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert st["picks"] == 1 and st["providers"][0]["name"] == "file"
    finally:
        srv.shutdown()


def test_watcher_imports_without_heavy_dependencies():
    """The Pi install has no pandas/numpy: footy.watch must not import them."""
    import subprocess
    import sys
    code = ("import sys, importlib; sys.modules['pandas'] = None; sys.modules['numpy'] = None; "
            "[importlib.import_module(m) for m in ('footy.watch.core', 'footy.watch.providers', "
            "'footy.watch.rules', 'footy.watch.server', 'footy.watch.__main__')]")
    subprocess.run([sys.executable, "-c", code], check=True)
