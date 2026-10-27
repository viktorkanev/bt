import numpy as np
import pandas as pd

from footy.ratings.engine import RatingEngine


def test_strong_team_rated_highest(toy_league):
    eng = RatingEngine()
    eng.run(toy_league)
    r = eng.ratings_table().set_index("team")
    assert r["elo"].idxmax() == "Alpha"
    assert r["pxg_att"].idxmax() == "Alpha"
    assert r["elo"].idxmin() == "Foxtrot"


def test_no_lookahead(toy_league):
    """Changing results must not change any feature on or before that match day."""
    f1 = RatingEngine().run(toy_league)
    altered = toy_league.copy()
    k = len(altered) // 2
    altered.loc[k:, ["fthg", "ftag"]] = altered.loc[k:, ["ftag", "fthg"]].to_numpy() + 3
    f2 = RatingEngine().run(altered)
    day = altered.loc[k, "date"]
    upto = altered["date"] <= day
    num = f1.select_dtypes("number").columns
    assert np.allclose(f1.loc[upto, num].fillna(-9).to_numpy(),
                       f2.loc[upto, num].fillna(-9).to_numpy())
    later = ~upto
    assert not np.allclose(f1.loc[later, "elo_diff"], f2.loc[later, "elo_diff"])


def test_unplayed_rows_get_features(toy_league):
    fx = toy_league.tail(3).copy()
    fx["date"] = toy_league["date"].max() + pd.Timedelta(days=7)
    fx[["fthg", "ftag", "ftr"]] = np.nan
    d = pd.concat([toy_league, fx], ignore_index=True)
    f = RatingEngine().run(d)
    assert f.tail(3)["pxg_ph"].notna().all()


def test_relegated_team_is_rebased(toy_league):
    """A team dropping a division gets a rating relative to its new pool, above that pool's average."""
    last = toy_league[toy_league["season_start"] == 2020]
    lower = last.copy()
    lower["league"] = "E1"
    lower["date"] = lower["date"] + pd.Timedelta(days=365)
    lower["home"] = lower["home"].str.replace("Alpha", "Zulu")
    lower["away"] = lower["away"].str.replace("Alpha", "Zulu")
    # Foxtrot (weakest) is relegated into E1, taking Zulu's place in one fixture
    move = lower.head(1).copy()
    move["home"], move["away"] = "Foxtrot", "Bravo"
    move["date"] = lower["date"].max() + pd.Timedelta(days=1)
    d = pd.concat([toy_league, lower, move], ignore_index=True).sort_values("date").reset_index(drop=True)
    eng = RatingEngine()
    eng.run(d)
    assert eng.teams["ENG|Foxtrot"].league == "E1"
    assert eng.teams["ENG|Foxtrot"].entry_dir == "down"
