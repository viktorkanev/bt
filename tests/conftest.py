import numpy as np
import pandas as pd
import pytest

from footy.data.schema import finalise


@pytest.fixture
def toy_league():
    """Three seasons of a 6-team double round robin with one dominant team."""
    rng = np.random.default_rng(0)
    teams = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    strength = dict(zip(teams, [1.0, 0.3, 0.1, 0.0, -0.2, -0.5]))
    rows = []
    for season in range(2018, 2021):
        day = pd.Timestamp(f"{season}-08-01")
        for rnd in range(2):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    lh = np.exp(0.25 + 0.15 + strength[h] - strength[a] * 0.8)
                    la = np.exp(0.25 + strength[a] - strength[h] * 0.8)
                    rows.append({"league": "E0", "date": day, "time": "", "home": h, "away": a,
                                 "fthg": rng.poisson(lh), "ftag": rng.poisson(la),
                                 "hst": rng.poisson(lh * 3), "ast": rng.poisson(la * 3),
                                 "hs": rng.poisson(lh * 8), "as_": rng.poisson(la * 8)})
                    day += pd.Timedelta(days=1)
    return finalise(pd.DataFrame(rows))
