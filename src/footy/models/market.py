"""Turn bookmaker odds into probabilities (remove the margin / "vig").

* ``proportional`` - divide implied probabilities by the overround
* ``power``        - find k with sum(p_i ** k) = 1 (handles favourite-longshot bias)
* ``shin``         - Shin (1993) insider-trading model; Strumbelj (2014) found it
                     the most accurate simple method for football 1X2
"""

from __future__ import annotations

import numpy as np


def _implied(odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / odds


def overround(odds: np.ndarray) -> np.ndarray:
    return np.nansum(_implied(odds), axis=1) - 1.0 + np.where(np.isnan(odds).any(axis=1), np.nan, 0)


def proportional(odds: np.ndarray) -> np.ndarray:
    q = _implied(odds)
    return q / q.sum(axis=1, keepdims=True)


def power(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    q = _implied(odds)
    lo = np.full(q.shape[0], 0.5)
    hi = np.full(q.shape[0], 3.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        s = (q ** k[:, None]).sum(axis=1)
        hi = np.where(s < 1, k, hi)
        lo = np.where(s >= 1, k, lo)
    k = (lo + hi) / 2
    p = q ** k[:, None]
    return p / p.sum(axis=1, keepdims=True)


def shin(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    q = _implied(odds)
    b = q.sum(axis=1, keepdims=True)

    def probs(z):
        z = z[:, None]
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / b) - z) / (2 * (1 - z))

    lo = np.zeros(q.shape[0])
    hi = np.full(q.shape[0], 0.4)
    for _ in range(iters):
        z = (lo + hi) / 2
        s = probs(z).sum(axis=1)
        # sum decreases as z grows
        lo = np.where(s > 1, z, lo)
        hi = np.where(s <= 1, z, hi)
    p = probs((lo + hi) / 2)
    return p / p.sum(axis=1, keepdims=True)


METHODS = {"proportional": proportional, "power": power, "shin": shin}


def devig(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    out = np.full(odds.shape, np.nan)
    ok = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    if ok.any():
        out[ok] = METHODS[method](odds[ok])
    return out
