"""Probability scoring rules and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


def onehot(ftr) -> np.ndarray:
    ftr = np.asarray(ftr)
    return np.stack([(ftr == o).astype(float) for o in OUTCOMES], axis=1)


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score per match (lower is better). p, y: (n, 3) in H/D/A order."""
    cp = np.cumsum(p, axis=1)[:, :2]
    cy = np.cumsum(y, axis=1)[:, :2]
    return ((cp - cy) ** 2).sum(axis=1) / 2


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip((p * y).sum(axis=1), 1e-12, 1))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - y) ** 2).sum(axis=1)


def score_table(df: pd.DataFrame, models: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """Mean RPS / log loss / Brier / accuracy per model (rows where all models exist)."""
    y = onehot(df["ftr"])
    cols = {m: [f"{m}_ph", f"{m}_pd", f"{m}_pa"] for m in models}
    mask = np.ones(len(df), bool)
    for m in models:
        mask &= df[cols[m]].notna().all(axis=1).to_numpy()
    rows = []
    sub = df[mask]
    ysub = y[mask]
    groups = [((), sub.index)] if not by else sub.groupby(by).groups.items()
    for key, idx in groups:
        pos = sub.index.get_indexer(idx)
        for m in models:
            p = sub[cols[m]].to_numpy()[pos]
            yy = ysub[pos]
            row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}
            row.update({
                "model": m, "n": len(pos),
                "rps": rps(p, yy).mean(), "logloss": logloss(p, yy).mean(),
                "brier": brier(p, yy).mean(),
                "accuracy": (p.argmax(1) == yy.argmax(1)).mean(),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def calibration(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table for a single outcome's probability."""
    p = np.asarray(p)
    hit = np.asarray(hit)
    ok = np.isfinite(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p[ok], edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p[ok], "hit": hit[ok]})
    g = df.groupby("bin").agg(pred=("p", "mean"), obs=("hit", "mean"), n=("p", "size"))
    return g.reset_index()
