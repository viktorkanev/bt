import numpy as np

from footy.models import market


ODDS = np.array([[1.50, 4.20, 7.00], [2.10, 3.40, 3.60], [1.10, 9.0, 21.0]])


def test_devig_sums_to_one():
    for m in ("proportional", "power", "shin"):
        p = market.devig(ODDS, m)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert (p > 0).all()


def test_shin_and_power_shade_longshots():
    prop = market.devig(ODDS, "proportional")
    for m in ("power", "shin"):
        p = market.devig(ODDS, m)
        # the favourite-longshot bias: longshots get a lower probability than proportional
        assert p[2, 2] < prop[2, 2]
        assert p[2, 0] > prop[2, 0]


def test_devig_handles_missing():
    odds = np.array([[2.0, np.nan, 3.0], [2.0, 3.0, 4.0]])
    p = market.devig(odds)
    assert np.isnan(p[0]).all() and np.isfinite(p[1]).all()


def test_overround():
    assert np.isclose(market.overround(np.array([[2.0, 2.0]]))[0], 0.0)
    assert market.overround(ODDS)[0] > 0
