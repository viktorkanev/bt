"""Vectorised Dixon-Coles scoreline maths.

Given expected goals (lambda_home, lambda_away) and the Dixon-Coles low-score
dependence parameter rho, build the full scoreline probability matrix and derive
1X2, over/under and Asian-handicap probabilities from it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10


def score_matrix(lh, la, rho: float = -0.06, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Return P[i, x, y] = P(home scores x, away scores y) for each match i."""
    lh = np.atleast_1d(np.asarray(lh, dtype=float))
    la = np.atleast_1d(np.asarray(la, dtype=float))
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g[None, :], lh[:, None])
    pa = poisson.pmf(g[None, :], la[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    # Dixon-Coles tau adjustment for 0-0, 1-0, 0-1, 1-1.
    m[:, 0, 0] *= 1 - lh * la * rho
    m[:, 0, 1] *= 1 + lh * rho
    m[:, 1, 0] *= 1 + la * rho
    m[:, 1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    m /= m.sum(axis=(1, 2), keepdims=True)
    return m


def outcome_probs(m: np.ndarray):
    home = np.tril(np.ones(m.shape[1:]), -1)  # x > y
    away = np.triu(np.ones(m.shape[1:]), 1)
    ph = (m * home).sum(axis=(1, 2))
    pa = (m * away).sum(axis=(1, 2))
    return ph, 1 - ph - pa, pa


def over_prob(m: np.ndarray, line: float = 2.5) -> np.ndarray:
    g = np.arange(m.shape[1])
    tot = g[:, None] + g[None, :]
    return (m * (tot > line)).sum(axis=(1, 2))


def goal_diff_dist(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (diffs, P[i, k]) where diffs[k] = home goals - away goals."""
    n = m.shape[1]
    diffs = np.arange(-(n - 1), n)
    g = np.arange(n)
    d = g[:, None] - g[None, :]
    out = np.zeros((m.shape[0], diffs.size))
    for k, dv in enumerate(diffs):
        out[:, k] = (m * (d == dv)).sum(axis=(1, 2))
    return diffs, out


def dc_probs(lh, la, rho: float = -0.06, chunk: int = 40000) -> dict[str, np.ndarray]:
    """1X2 + over 2.5 probabilities for many matches (chunked to bound memory)."""
    lh = np.asarray(lh, dtype=float)
    la = np.asarray(la, dtype=float)
    out = {k: np.full(lh.size, np.nan) for k in ("ph", "pd", "pa", "po25")}
    ok = np.isfinite(lh) & np.isfinite(la)
    idx = np.flatnonzero(ok)
    for s in range(0, idx.size, chunk):
        j = idx[s:s + chunk]
        m = score_matrix(lh[j], la[j], rho)
        ph, pd, pa = outcome_probs(m)
        out["ph"][j], out["pd"][j], out["pa"][j] = ph, pd, pa
        out["po25"][j] = over_prob(m)
    return out


def ah_settle(gd: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Per-unit-stake return multiplier class for an Asian handicap bet on HOME.

    Returns a value in {-1, -0.5, 0, 0.5, 1}: fraction of stake lost (-), refunded (0)
    or won (+) before applying odds. Quarter lines are split in two half-stakes.
    """
    gd = np.asarray(gd, dtype=float)
    line = np.asarray(line, dtype=float)
    out = np.zeros(np.broadcast(gd, line).shape)
    q = (np.round(line * 4) % 2) != 0  # quarter line (x.25 / x.75)
    for sub, w in (((line - 0.25), 0.5), ((line + 0.25), 0.5)):
        adj = gd + sub
        out = out + np.where(q, w * np.sign(adj), 0.0)
    adj = gd + line
    out = np.where(q, out, np.sign(adj))
    return out


def ah_ev(diffs: np.ndarray, pgd: np.ndarray, line: np.ndarray, odds_home: np.ndarray,
          odds_away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected profit per unit stake for AH home and away bets."""
    line = np.asarray(line, dtype=float)[:, None]
    res_home = ah_settle(diffs[None, :], line)
    res_away = -ah_settle(diffs[None, :], line)

    def ev(res, odds):
        win = np.clip(res, 0, None)
        lose = np.clip(-res, 0, None)
        return (pgd * (win * (odds[:, None] - 1) - lose)).sum(axis=1)

    return ev(res_home, np.asarray(odds_home, float)), ev(res_away, np.asarray(odds_away, float))
