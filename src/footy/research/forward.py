"""Forward test: score frozen rules on matches the research never saw.

Everything up to ``FREEZE_DATE`` (the last result in the data used for the
research) was available when the rules were designed. Only matches after it
count as evidence. Run ``footy forward`` after each ``footy fetch`` + ``footy
build``; settled bets accumulate and the summary shows ROI, p-values and
closing-line value (CLV) against Pinnacle when closing odds are present.

Rules
-----
F1  Lower-division over/under 2.5: model+market blend (``oublend``), tiers 2-5,
    edge 3%-10% at the best available price, odds <= 4.0.
F2  Luck-fade Asian handicap: frozen OLS supremacy shift from the four
    market-surprise features (+ tier >= 2 interactions), bet the side whose
    expected value improves by >= 0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..data import store
from ..models import market
from ..ratings.poisson_math import ah_ev, ah_settle, goal_diff_dist, score_matrix
from . import confirm

FREEZE_DATE = pd.Timestamp("2026-09-03")
FROZEN = store.ROOT / "config" / "forward_rules.json"
OUT_DIR = store.ROOT / "reports" / "research" / "forward"
F2_COLS = confirm.FEATS + [f"low_{c}" for c in confirm.FEATS]


def freeze(d: pd.DataFrame | None = None, path: Path = FROZEN) -> dict:
    """Fit F2 on everything up to FREEZE_DATE and write the coefficients."""
    d = confirm.prepare(d)
    d = d[d["date"] <= FREEZE_DATE]
    ok = d[F2_COLS + ["res_gd"]].notna().all(axis=1)
    r = sm.OLS(d.loc[ok, "res_gd"], sm.add_constant(d.loc[ok, F2_COLS])).fit()
    spec = {
        "freeze_date": str(FREEZE_DATE.date()),
        "fitted_on_matches": int(ok.sum()),
        "F1": {"model": "oublend", "tiers": [2, 3, 4, 5], "min_edge": 0.03, "max_edge": 0.10,
               "max_odds": 4.0, "prices": ["max", "ps", "avg", "b365"]},
        "F2": {"coef": {c: float(r.params[c]) for c in F2_COLS}, "min_signal": 0.05, "rho": -0.07,
               "prices": ["ps", "b365", "max", "avg"]},
    }
    Path(path).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def _summ(ret: np.ndarray, odds: np.ndarray, clv: np.ndarray | None = None) -> dict:
    n = len(ret)
    if n == 0:
        return {"bets": 0}
    out = {"bets": n, "roi": float(ret.mean()), "profit_units": float(ret.sum()),
           "avg_odds": float(np.mean(odds))}
    if n > 1 and ret.std(ddof=1) > 0:
        t = ret.mean() / (ret.std(ddof=1) / np.sqrt(n))
        out["p_value"] = float(stats.t.sf(t, n - 1))
    if clv is not None and np.isfinite(clv).sum() > 1:
        c = clv[np.isfinite(clv)]
        out["clv_bets"] = int(c.size)
        out["avg_clv"] = float(c.mean())
        out["clv_p_value"] = float(stats.t.sf(c.mean() / (c.std(ddof=1) / np.sqrt(c.size)), c.size - 1))
    return out


def _f1(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    s = d[d["tier"].isin(spec["tiers"]) & d[f"{spec['model']}_po25"].notna()].copy()
    if s.empty:
        return s
    po = s[f"{spec['model']}_po25"].to_numpy()
    rows = []
    for price in spec["prices"]:
        co, cu = f"{price}_o25", f"{price}_u25"
        if co not in s or s[co].isna().all():
            continue
        odds = s[[co, cu]].to_numpy()
        ev = np.stack([po, 1 - po], 1) * odds - 1
        ok = (ev >= spec["min_edge"]) & (ev <= spec["max_edge"]) & (odds <= spec["max_odds"])
        ev = np.where(ok, ev, -np.inf)
        j = ev.argmax(1)
        keep = np.isfinite(ev[np.arange(len(s)), j])
        sub = s[keep].copy()
        jj = j[keep]
        sub["price_book"] = price
        sub["pick"] = np.where(jj == 0, "over 2.5", "under 2.5")
        sub["odds"] = odds[keep][np.arange(keep.sum()), jj]
        sub["ev"] = ev[keep][np.arange(keep.sum()), jj]
        played = sub["fthg"].notna()
        over = (sub["fthg"] + sub["ftag"]) > 2.5
        win = np.where(jj == 0, over, ~over)
        sub["ret"] = np.where(played, np.where(win, sub["odds"] - 1, -1.0), np.nan)
        if {"psc_o25", "psc_u25"} <= set(sub.columns):
            fair = market.devig(sub[["psc_o25", "psc_u25"]].to_numpy(), "power")
            sub["clv"] = sub["odds"] * fair[np.arange(len(sub)), jj] - 1
        rows.append(sub)
    return pd.concat(rows) if rows else pd.DataFrame()


def _f2(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    s = d[d[F2_COLS + ["x_lh", "x_la", "ah_line"]].notna().all(axis=1)].copy()
    if s.empty:
        return s
    beta = np.array([spec["coef"][c] for c in F2_COLS])
    delta = s[F2_COLS].to_numpy() @ beta
    lh = np.clip(s["x_lh"] + delta / 2, 0.05, None).to_numpy()
    la = np.clip(s["x_la"] - delta / 2, 0.05, None).to_numpy()
    diffs, pgd = goal_diff_dist(score_matrix(lh, la, spec["rho"]))
    _, pgd0 = goal_diff_dist(score_matrix(s["x_lh"].to_numpy(), s["x_la"].to_numpy(), spec["rho"]))
    rows = []
    for price in spec["prices"]:
        ch, ca = f"{price}_ahh", f"{price}_aha"
        if ch not in s or s[ch].isna().all():
            continue
        has = s[[ch, ca]].notna().all(axis=1).to_numpy()
        if not has.any():
            continue
        a = (s["ah_line"].to_numpy()[has], s[ch].to_numpy()[has], s[ca].to_numpy()[has])
        eh, ea = ah_ev(diffs, pgd[has], *a)
        eh0, ea0 = ah_ev(diffs, pgd0[has], *a)
        sh, sa = eh - eh0, ea - ea0
        side = np.where(sh >= sa, 0, 1)
        sig = np.maximum(sh, sa)
        sub = s[has].copy()
        sub["price_book"], sub["signal"] = price, sig
        sub["pick"] = np.where(side == 0, "home ", "away ") + sub["ah_line"].map(lambda x: f"{x:+.2f}")
        sub["odds"] = np.where(side == 0, sub[ch], sub[ca])
        sub["ev"] = np.where(side == 0, eh, ea)
        sub["side"] = side
        sub = sub[sub["signal"] >= spec["min_signal"]]
        played = sub["fthg"].notna()
        stl = ah_settle((sub["fthg"] - sub["ftag"]).fillna(0).to_numpy(), sub["ah_line"].to_numpy())
        stl = np.where(sub["side"] == 0, stl, -stl)
        sub["ret"] = np.where(played, np.where(stl > 0, stl * (sub["odds"] - 1), stl), np.nan)
        if {"psc_ahh", "psc_aha"} <= set(sub.columns) and sub["psc_ahh"].notna().any():
            fair = market.devig(sub[["psc_ahh", "psc_aha"]].to_numpy(), "power")
            same_line = sub["ahc_line"].isna() | (sub["ahc_line"] == sub["ah_line"])
            sub["clv"] = np.where(same_line, sub["odds"] * fair[np.arange(len(sub)), sub["side"]] - 1, np.nan)
        rows.append(sub)
    return pd.concat(rows) if rows else pd.DataFrame()


def evaluate(d: pd.DataFrame | None = None, out_dir: Path = OUT_DIR,
             since: pd.Timestamp = FREEZE_DATE) -> dict:
    spec = json.loads(FROZEN.read_text(encoding="utf-8"))
    from ..pipeline import load_predictions
    from ..models.probability import eligible
    from . import signals

    d = load_predictions() if d is None else d
    d = signals.add_all(d.reset_index(drop=True))
    d["low"] = (d["tier"] >= 2).astype(float)
    for c in confirm.FEATS:
        d[f"low_{c}"] = d[c] * d["low"]
    d = d[(d["date"] > since) & (eligible(d) | d["ftr"].isna())]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"since": str(pd.Timestamp(since).date()), "matches_after_freeze": int(len(d)),
           "played_after_freeze": int(d["ftr"].notna().sum())}
    keep = ["match_id", "date", "league", "tier", "home", "away", "price_book", "pick", "odds", "ev",
            "ret", "clv"]
    for name, fn in (("F1", _f1), ("F2", _f2)):
        b = fn(d, spec[name])
        if b.empty:
            res[name] = {"note": "no qualifying matches yet"}
            continue
        b = b[[c for c in keep if c in b.columns]].sort_values(["date", "league"])
        b.to_csv(out_dir / f"{name}_bets.csv", index=False)
        settled = b[b["ret"].notna()]
        res[name] = {"pending": int(b["ret"].isna().sum())}
        for price, g in settled.groupby("price_book"):
            res[name][price] = _summ(g["ret"].to_numpy(), g["odds"].to_numpy(),
                                     g["clv"].to_numpy() if "clv" in g else None)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res
