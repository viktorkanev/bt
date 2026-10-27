import numpy as np
import pandas as pd

from footy.ratings.poisson_math import outcome_probs, score_matrix
from footy.research import confirm, signals


def test_implied_goals_round_trip():
    ig = signals.ImpliedGoals()
    lh, la = np.array([1.7, 0.9, 1.3]), np.array([0.8, 1.6, 1.3])
    ph, _, pa = outcome_probs(score_matrix(lh, la, -0.07))
    out = ig(ph, pa)
    assert np.allclose(out["lh"], lh, atol=0.05) and np.allclose(out["la"], la, atol=0.05)


def _frame(toy):
    d = toy.copy()
    d["mkt_ph"], d["mkt_pd"], d["mkt_pa"] = 0.45, 0.27, 0.28
    d["sot_lh"], d["sot_la"] = 4.5, 3.8
    return d


def test_surprise_features_use_only_past_matches(toy_league):
    d = _frame(toy_league)
    f1 = signals.surprise_features(d)
    k = len(d) // 2
    d2 = d.copy()
    d2.loc[k, ["fthg", "ftag", "hst", "ast", "ftr"]] = [9, 0, 20, 0, "H"]
    f2 = signals.surprise_features(d2)
    upto = d["date"] <= d.loc[k, "date"]
    assert np.allclose(f1[upto].fillna(-9), f2[upto].fillna(-9))
    assert not np.allclose(f1[~upto].fillna(-9), f2[~upto].fillna(-9))


def test_team_history_lags_red_cards(toy_league):
    d = toy_league.copy()
    d["hr"], d["ar"], d["hc"], d["ac"] = 0.0, 0.0, 5.0, 4.0
    d.loc[0, "hr"] = 1.0
    h = signals.team_history(d)
    assert h.loc[0, "h_red_prev"] != 1.0  # a match's own red card is never its own feature
    team = d.loc[0, "home"]
    nxt = d.index[(d.index > 0) & ((d["home"] == team) | (d["away"] == team))][0]
    side = "h" if d.loc[nxt, "home"] == team else "a"
    assert h.loc[nxt, f"{side}_red_prev"] == 1.0


def test_holm_adjustment():
    adj = confirm.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)
