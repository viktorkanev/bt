import numpy as np
import pandas as pd

from footy import upcoming
from footy.research import forward

COEF = {c: 0.0 for c in forward.F2_COLS}
COEF["d_fin_luck"] = -0.5  # home side "luckier" -> market over-rates it
SPEC = {
    "F1": {"model": "oublend", "tiers": [2, 3, 4, 5], "min_edge": 0.03, "max_edge": 0.10,
           "max_odds": 4.0, "prices": ["max", "b365"]},
    "F2": {"coef": COEF, "min_signal": 0.05, "rho": -0.07, "prices": ["ps", "b365"]},
}


def fixture_frame():
    n = 3
    d = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)], "date": pd.Timestamp("2026-11-01"), "time": "15:00",
        "league": ["E1", "E0", "E2"], "tier": [2, 1, 3], "home": ["A", "C", "E"], "away": ["B", "D", "F"],
        "x_lh": [1.5, 1.4, 1.3], "x_la": [1.1, 1.2, 1.2],
        "mkt_ph": [0.45, 0.42, 0.40], "mkt_pd": [0.27, 0.28, 0.29], "mkt_pa": [0.28, 0.30, 0.31],
        "ah_line": [-0.25, -0.25, 0.0], "ps_ahh": np.nan, "ps_aha": np.nan,
        "b365_ahh": [1.95, 1.95, 1.90], "b365_aha": [1.95, 1.95, 2.00],
        "oublend_po25": [0.60, 0.60, 0.50], "max_o25": [1.80, 1.80, 1.95], "max_u25": [2.10, 2.10, 1.95],
        "b365_o25": [1.75, 1.75, 1.90], "b365_u25": [2.05, 2.05, 1.90],
    })
    for c in forward.F2_COLS:
        d[c] = 0.0
    d.loc[0, "d_fin_luck"] = 0.6   # strongly lucky home team in match 0
    d.loc[2, "d_fin_luck"] = np.nan  # no shot data -> no research layer
    return d


def test_research_layer_shifts_probabilities_and_picks():
    fx = upcoming.research_layer(fixture_frame(), SPEC)
    res = fx[["res_ph", "res_pd", "res_pa"]].to_numpy()
    assert np.allclose(res[:2].sum(axis=1), 1)
    assert fx.loc[0, "res_ph"] < fx.loc[0, "mkt_ph"]          # luck fade lowers the lucky home side
    assert np.isclose(fx.loc[1, "res_ph"], fx.loc[1, "mkt_ph"], atol=1e-6)  # no signal, no change
    assert np.isnan(fx.loc[2, "res_ph"])
    assert fx.loc[0, "f2_pick"] == "B +0.25" and fx.loc[0, "f2_side"] == "away"
    assert fx.loc[1, "f2_pick"] == ""
    # F1: tier-2 match, P(over)=0.60 at 1.80 -> EV +8% -> pick; tier-1 match never picked
    assert fx.loc[0, "f1_pick"] == "Over 2.5" and fx.loc[0, "f1_book"] == "max"
    assert fx.loc[1, "f1_pick"] == ""


def test_ledger_settlement_and_closing_value(tmp_path):
    fx = upcoming.research_layer(fixture_frame(), SPEC)
    led, summ = upcoming.update_ledger(fx, pd.DataFrame({"match_id": [], "fthg": [], "ftag": []}),
                                       path=tmp_path / "ledger.csv")
    assert summ["pending"] == 2 and summ["settled"] == 0
    results = pd.DataFrame({"match_id": ["m0"], "fthg": [1.0], "ftag": [1.0],
                            "avgc_o25": [1.70], "avgc_u25": [2.20],
                            "avgc_ahh": [2.05], "avgc_aha": [1.85], "ahc_line": [-0.25]})
    # Same fixtures logged again later at worse odds: first price must be kept.
    fx2 = fx.copy()
    fx2["f1_odds"] = 1.5
    led, summ = upcoming.update_ledger(fx2, results, path=tmp_path / "ledger.csv")
    assert len(led) == 2 and led.loc[led["rule"] == "F1", "odds"].iloc[0] == 1.80
    # 1-1: under wins? no -> Over 2.5 loses (-1); away +0.25 on a draw = half win
    assert np.isclose(summ["F1"]["roi"], -1.0)
    assert np.isclose(summ["F2"]["roi"], 0.5 * (1.95 - 1))
    assert "avg_clv" in summ["F1"] and "avg_clv" in summ["F2"]


def test_exchange_commission_and_venue_preference():
    assert np.isclose(forward.effective_odds("bfe", 2.0), 1.95)
    assert np.isclose(forward.effective_odds("b365", 2.0), 2.0)
    d = fixture_frame()
    d["bfe_ahh"], d["bfe_aha"] = [2.02, 2.00, 1.95], [2.02, 2.00, 2.05]
    fx = upcoming.research_layer(d, SPEC)
    assert fx.loc[0, "f2_book"] == "bfe" and fx.loc[0, "f2_odds"] == 2.02


def test_settle_applies_exchange_commission(tmp_path):
    led = pd.DataFrame([{"logged_at": "x", "match_id": "m0", "date": pd.Timestamp("2026-11-01"),
                         "league": "E1", "home": "A", "away": "B", "rule": "F2", "pick": "A -0.50",
                         "side": "home", "line": -0.5, "book": "bfe", "odds": 2.0, "prob": np.nan,
                         "ev": 0.05, "signal": 0.06}])
    res = pd.DataFrame({"match_id": ["m0"], "fthg": [2.0], "ftag": [0.0]})
    summ = upcoming.settle(led, res)
    assert np.isclose(summ["F2"]["roi"], 0.95)


def _scored():
    d = fixture_frame()
    d["time"], d["oumkt_po25"] = "15:00", [0.52, 0.55, 0.47]
    for side in ("h", "a"):
        d[f"{side}_fin_luck"], d[f"{side}_gk_luck"] = 0.0, 0.0
    d.loc[0, "h_fin_luck"] = 0.4
    return upcoming.research_layer(d, SPEC)


def test_pick_details_handicap_numbers_are_consistent():
    pk = upcoming.pick_details(_scored(), SPEC)
    f2 = pk[pk["rule"] == "F2"].iloc[0]
    assert f2["selection"].startswith("B +0.25")
    assert f2["our_prob"] > f2["market_prob"]                  # the research favours the pick
    assert f2["breakeven_odds"] < f2["market_fair_odds"]        # so it needs a lower price
    assert np.isclose(f2["our_prob"] + f2["push_prob"] + f2["lose_prob"], 1, atol=1e-6)
    assert (f2["edge_at_best"] > 0) == (f2["best_odds"] > f2["breakeven_odds"])
    assert f2["min_odds_exchange"] > f2["min_odds"]             # commission needs a higher price
    assert "A" in f2["reason"] and "luck" in f2["reason"]


def test_pick_details_goals_minimum_odds():
    pk = upcoming.pick_details(_scored(), SPEC)
    f1 = pk[(pk["rule"] == "F1") & (pk["status"] == "Bet")].iloc[0]
    assert f1["selection"] == "Over 2.5 goals"
    assert np.isclose(f1["breakeven_odds"], 1 / f1["our_prob"])
    assert np.isclose(f1["min_odds"], 1.03 / f1["our_prob"])
    assert f1["best_odds"] >= f1["min_odds"]


def test_pick_details_empty():
    assert upcoming.pick_details(pd.DataFrame(), SPEC).empty
