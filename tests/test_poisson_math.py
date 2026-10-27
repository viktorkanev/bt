import numpy as np

from footy.ratings import poisson_math as pm


def test_score_matrix_normalised_and_probs():
    m = pm.score_matrix([1.6, 0.8], [1.1, 1.9], rho=-0.07)
    assert np.allclose(m.sum(axis=(1, 2)), 1)
    ph, pd_, pa = pm.outcome_probs(m)
    assert np.allclose(ph + pd_ + pa, 1)
    assert ph[0] > pa[0] and pa[1] > ph[1]


def test_rho_increases_draws():
    m0 = pm.score_matrix([1.3], [1.1], rho=0.0)
    m1 = pm.score_matrix([1.3], [1.1], rho=-0.1)
    assert pm.outcome_probs(m1)[1][0] > pm.outcome_probs(m0)[1][0]


def test_over_prob_matches_poisson_total():
    from scipy.stats import poisson
    m = pm.score_matrix([1.4], [1.2], rho=0.0)
    expected = 1 - poisson.cdf(2, 2.6)
    assert abs(pm.over_prob(m)[0] - expected) < 1e-6


def test_ah_settle_rules():
    def s(gd, line):
        return float(pm.ah_settle(np.array([gd]), np.array([line]))[0])
    assert s(1, -0.5) == 1 and s(0, -0.5) == -1
    assert s(0, 0.0) == 0 and s(1, -1.0) == 0 and s(2, -1.0) == 1
    # quarter lines split the stake in two halves
    assert s(0, -0.25) == -0.5
    assert s(1, -0.75) == 0.5
    assert s(0, 0.25) == 0.5
    assert s(-1, 0.75) == -0.5


def test_ah_ev_zero_for_fair_odds():
    m = pm.score_matrix([1.5], [1.0], rho=0.0)
    diffs, pgd = pm.goal_diff_dist(m)
    p_home = pgd[0][diffs > 0.5].sum()  # line -0.5
    ev_h, ev_a = pm.ah_ev(diffs, pgd, np.array([-0.5]), np.array([1 / p_home]),
                          np.array([1 / (1 - p_home)]))
    assert abs(ev_h[0]) < 1e-9 and abs(ev_a[0]) < 1e-9
