import numpy as np
import pandas as pd

from footy import backtest as bt


def frame():
    return pd.DataFrame({
        "match_id": ["a", "b", "c"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "league": "E0", "season": "2019-20", "home": ["x", "y", "z"], "away": ["p", "q", "r"],
        "ftr": ["H", "A", "D"], "fthg": [2, 0, 1], "ftag": [0, 1, 1],
        "m_ph": [0.60, 0.30, 0.40], "m_pd": [0.25, 0.30, 0.30], "m_pa": [0.15, 0.40, 0.30],
        "max_h": [1.90, 3.00, 2.40], "max_d": [3.5, 3.3, 3.2], "max_a": [6.0, 2.30, 3.0],
    })


def test_selects_only_value_and_settles():
    s = bt.Strategy(model="m", price="max", min_edge=0.05, min_odds=1.01, max_odds=10)
    b = bt.run(frame(), s)
    # a: EV home = 0.6*1.9-1 = +0.14 -> bet H, wins 0.9; b and c: no outcome reaches +5%
    assert list(b["match_id"]) == ["a"]
    assert b.loc[0, "pick"] == "h" and np.isclose(b.loc[0, "ret"], 0.9)


def test_kelly_is_capped():
    s = bt.Strategy(model="m", price="max", min_edge=0.0, min_odds=1.01, max_odds=10,
                    staking="kelly", kelly_fraction=1.0, kelly_cap=0.05)
    b = bt.run(frame(), s)
    assert (b["stake"] <= 0.05 * 100 * 1.2).all()


def test_summary_statistics():
    b = pd.DataFrame({"ret": [1.0, -1.0, 1.0, -1.0], "win": [1, 0, 1, 0], "odds": [2.0] * 4,
                      "ev": [0.05] * 4, "stake": [1.0] * 4, "profit": [1.0, -1.0, 1.0, -1.0],
                      "bank": [101, 100, 101, 100]})
    sm = bt.summarise(b)
    assert sm["bets"] == 4 and sm["roi"] == 0 and sm["max_drawdown_units"] == 1


def test_max_edge_filters_outliers():
    s = bt.Strategy(model="m", price="max", min_edge=0.05, max_edge=0.10, min_odds=1.01, max_odds=10)
    assert bt.run(frame(), s).empty
