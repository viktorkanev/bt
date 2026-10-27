"""Unconventional pre-match signals.

Every function returns features aligned with the input frame and uses only
information that existed before kick-off (previous matches, or odds quoted
before the match).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..ratings.poisson_math import outcome_probs, over_prob, score_matrix


# --------------------------------------------------------------------------
# 1. Cross-market implied totals: "the draw price knows the total"
# --------------------------------------------------------------------------
class ImpliedGoals:
    """Back out expected goals (lambda_h, lambda_a) from 1X2 probabilities.

    Under a Dixon-Coles scoreline model the three 1X2 probabilities pin down
    both expected goal rates: the home/away split sets the difference, and the
    draw probability sets the total (low-scoring games draw more often). From
    those we can price over/under 2.5 using *only the 1X2 market*, then compare
    it with what the over/under market is charging.
    """

    def __init__(self, rho: float = -0.07, step: float = 0.02, lo: float = 0.15, hi: float = 4.6):
        g = np.arange(lo, hi + 1e-9, step)
        lh, la = np.meshgrid(g, g, indexing="ij")
        lh, la = lh.ravel(), la.ravel()
        m = score_matrix(lh, la, rho)
        ph, pd_, pa = outcome_probs(m)
        self.lh, self.la = lh, la
        self.po = over_prob(m, 2.5)
        self.tree = cKDTree(np.c_[ph, pa])

    def __call__(self, ph: np.ndarray, pa: np.ndarray, k: int = 4):
        ph = np.asarray(ph, float)
        pa = np.asarray(pa, float)
        ok = np.isfinite(ph) & np.isfinite(pa)
        out = {k_: np.full(ph.size, np.nan) for k_ in ("lh", "la", "po25")}
        if not ok.any():
            return out
        dist, idx = self.tree.query(np.c_[ph[ok], pa[ok]], k=k)
        w = 1.0 / np.maximum(dist, 1e-9)
        w /= w.sum(axis=1, keepdims=True)
        out["lh"][ok] = (self.lh[idx] * w).sum(axis=1)
        out["la"][ok] = (self.la[idx] * w).sum(axis=1)
        out["po25"][ok] = (self.po[idx] * w).sum(axis=1)
        return out


# --------------------------------------------------------------------------
# Team-level rolling history (red cards, finishing luck, corners)
# --------------------------------------------------------------------------
def _team_long(d: pd.DataFrame) -> pd.DataFrame:
    base = d[["date", "country", "league", "fthg", "ftag", "hst", "ast", "hr", "ar", "hc", "ac"]].copy()
    base["row"] = np.arange(len(d))
    home = base.assign(team=d["country"] + "|" + d["home"], side="h", gf=base["fthg"], ga=base["ftag"],
                       sf=base["hst"], sa=base["ast"], red=base["hr"], cf=base["hc"], ca=base["ac"])
    away = base.assign(team=d["country"] + "|" + d["away"], side="a", gf=base["ftag"], ga=base["fthg"],
                       sf=base["ast"], sa=base["hst"], red=base["ar"], cf=base["ac"], ca=base["hc"])
    cols = ["row", "date", "league", "team", "side", "gf", "ga", "sf", "sa", "red", "cf", "ca"]
    t = pd.concat([home[cols], away[cols]], ignore_index=True)
    return t.sort_values(["team", "date", "row"]).reset_index(drop=True)


def team_history(d: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """Per-match features for home (h_) and away (a_) teams from *previous* matches.

    * ``red_prev``   red cards the team received in its previous league match
    * ``days_since`` days since that match (suspensions only bite if it was recent)
    * ``fin_luck``   EWMA of (goals scored - expected from shots on target)
    * ``gk_luck``    EWMA of (expected conceded from shots on target - goals conceded)
    """
    t = _team_long(d)
    played = t["gf"].notna()
    # league conversion rate (goals per shot on target), expanding & lagged
    lg = t[played & t["sf"].notna()].groupby("league")
    conv = (lg["gf"].cumsum() / lg["sf"].cumsum().replace(0, np.nan))
    t["conv"] = np.nan
    t.loc[conv.index, "conv"] = conv
    t["conv"] = t.groupby("league")["conv"].transform(lambda s: s.ffill().shift(1)).fillna(0.30)
    t["fin"] = t["gf"] - t["conv"] * t["sf"]
    t["gk"] = t["conv"] * t["sa"] - t["ga"]
    g = t.groupby("team", sort=False)
    t["red_prev"] = g["red"].shift(1)
    t["days_since"] = (t["date"] - g["date"].shift(1)).dt.days
    for col in ("fin", "gk"):
        t[f"{col}_luck"] = g[col].transform(
            lambda s: s.shift(1).ewm(span=span, min_periods=5, ignore_na=True).mean())
    t["n_shots_hist"] = g["sf"].transform(lambda s: s.notna().astype(float).shift(1).rolling(10, min_periods=1).sum())
    out = pd.DataFrame(index=np.arange(len(d)))
    for side in ("h", "a"):
        s = t[t["side"] == side].set_index("row")
        for col in ("red_prev", "days_since", "fin_luck", "gk_luck", "n_shots_hist"):
            out[f"{side}_{col}"] = s[col].reindex(out.index).to_numpy()
    out.index = d.index
    return out


# --------------------------------------------------------------------------
# Odds dispersion: is Bet365 the best price on an outcome?
# --------------------------------------------------------------------------
def dispersion(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for o in ("h", "d", "a"):
        out[f"disp_{o}"] = d[f"max_{o}"] / d[f"b365_{o}"]
    for o in ("o25", "u25"):
        out[f"disp_{o}"] = d[f"max_{o}"] / d[f"b365_{o}"]
    return out


# --------------------------------------------------------------------------
# 2. Market-surprise decomposition (skill vs luck relative to the market)
# --------------------------------------------------------------------------
SURPRISE_FEATURES = ["surp_pts_10", "surp_sot_10", "fin_luck", "gk_luck"]


def surprise_features(d: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """How each team has been doing *relative to what the market expected*.

    For every previous match of a team we know what the bookmaker expected
    before kick-off (from that match's odds) and what our shots model expected.
    We then split the team's recent over/under-performance into:

    * ``surp_pts``  points won minus market-expected points (3*P(win) + P(draw))
    * ``surp_sot``  shots-on-target difference minus the pre-match expectation
                    (genuine territorial dominance the market may under-weight)
    * ``fin_luck``  goals scored minus league conversion x shots on target
    * ``gk_luck``   expected goals conceded from shots on target minus goals conceded

    Each is an exponentially weighted mean over previous matches only
    (``span`` matches), so nothing from the match being predicted is used.
    Requires columns from :func:`add_all` (``x_lh``/``x_la``) and the rating
    engine (``sot_lh``/``sot_la``, ``mkt_*``).
    """
    n = len(d)
    rows = np.arange(n)
    played = d["ftr"].notna().to_numpy()
    hp = np.select([d["fthg"] > d["ftag"], d["fthg"] == d["ftag"]], [3.0, 1.0], 0.0)
    ap = np.select([d["ftag"] > d["fthg"], d["fthg"] == d["ftag"]], [3.0, 1.0], 0.0)
    common = {"row": rows, "date": d["date"].to_numpy(), "played": played}
    home = pd.DataFrame({**common, "team": (d["country"] + "|" + d["home"]).to_numpy(), "side": "h",
                         "pts": hp, "xpts": (3 * d["mkt_ph"] + d["mkt_pd"]).to_numpy(),
                         "sotd": (d["hst"] - d["ast"]).to_numpy(),
                         "xsotd": (d["sot_lh"] - d["sot_la"]).to_numpy()})
    away = pd.DataFrame({**common, "team": (d["country"] + "|" + d["away"]).to_numpy(), "side": "a",
                         "pts": ap, "xpts": (3 * d["mkt_pa"] + d["mkt_pd"]).to_numpy(),
                         "sotd": (d["ast"] - d["hst"]).to_numpy(),
                         "xsotd": (d["sot_la"] - d["sot_lh"]).to_numpy()})
    t = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "row"])
    t.loc[~t["played"], ["pts", "sotd"]] = np.nan
    t["surp_pts"] = t["pts"] - t["xpts"]
    t["surp_sot"] = t["sotd"] - t["xsotd"]
    g = t.groupby("team", sort=False)
    for c in ("surp_pts", "surp_sot"):
        t[f"{c}_{span}"] = g[c].transform(
            lambda s: s.shift(1).ewm(span=span, min_periods=3, ignore_na=True).mean())
    out = pd.DataFrame(index=np.arange(n))
    for side in ("h", "a"):
        s = t[t["side"] == side].set_index("row")
        for c in (f"surp_pts_{span}", f"surp_sot_{span}"):
            out[f"{side}_{c}"] = s[c].reindex(out.index).to_numpy()
    out.index = d.index
    return out


def add_all(d: pd.DataFrame) -> pd.DataFrame:
    """Add every research signal (implied goals, team history, dispersion, surprise)."""
    d = d.copy()
    ig = ImpliedGoals()
    imp = ig(d["mkt_ph"].to_numpy(), d["mkt_pa"].to_numpy())
    d["x_lh"], d["x_la"], d["x_po25"] = imp["lh"], imp["la"], imp["po25"]
    d = pd.concat([d, team_history(d), dispersion(d)], axis=1)
    d = pd.concat([d, surprise_features(d)], axis=1)
    for c in SURPRISE_FEATURES:
        d[f"d_{c}"] = d[f"h_{c}"] - d[f"a_{c}"]
    d["month"] = d["date"].dt.month
    return d
