"""Closing-line value (CLV) of the frozen research rules.

CLV compares the price a bet was struck at with the closing price (margin
removed). A bettor who keeps beating the close has an edge, and CLV shows it
after a few hundred bets because it does not depend on how matches end.

Secondary analysis: the rules and thresholds are the pre-registered ones and
nothing is re-fitted here except the walk-forward models defined in
``confirm`` (each season fitted on earlier seasons only).
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
from . import confirm, forward, signals

OUT_DIR = store.ROOT / "reports" / "research" / "clv"
CLOSE_1X2 = forward.CLOSING_BOOKS


def clv_1x2(bets: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{bk}_{o}" for bk in CLOSE_1X2 for o in "hda"]
    m = bets.merge(d[["match_id"] + [c for c in cols if c in d]], on="match_id", how="left")
    j = m["pick"].map({"h": 0, "d": 1, "a": 2}).to_numpy()
    clv = np.full(len(m), np.nan)
    used = np.array([""] * len(m), dtype=object)
    for bk in CLOSE_1X2:
        c3 = [f"{bk}_{o}" for o in "hda"]
        if not set(c3) <= set(m.columns):
            continue
        fair = market.devig(m[c3].to_numpy(dtype=float), "power")
        p = fair[np.arange(len(m)), j]
        ok = np.isnan(clv) & np.isfinite(p)
        clv = np.where(ok, m["odds"].to_numpy() * p - 1, clv)
        used = np.where(ok, bk, used)
    m["clv"], m["clv_book"] = clv, used
    return m


def clv_ah(bets: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    cols = ["ahc_line"] + [f"{bk}_{o}" for bk in CLOSE_1X2 for o in ("ahh", "aha")]
    m = bets.merge(d[["match_id", "ah_line"] + [c for c in cols if c in d]], on="match_id", how="left")
    side = np.where(m["pick"].str.startswith("home"), 0, 1)
    same = (m["ahc_line"].isna() | (m["ahc_line"] == m["ah_line"])).to_numpy()
    m["clv"], m["clv_book"] = forward.closing_value(m, ("ahh", "aha"), side, same)
    return m


def summary(b: pd.DataFrame, commission: float = 0.0) -> dict:
    out = {"bets": int(len(b))}
    if len(b) and "ret" in b and b["ret"].notna().any():
        r = b["ret"].dropna().to_numpy()
        if commission:
            r = np.where(r > 0, r * (1 - commission), r)
        out.update({"roi": float(r.mean()), "roi_p": float(stats.t.sf(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), len(r) - 1))
                    if len(r) > 2 else np.nan})
    c = b["clv"].dropna() if "clv" in b else pd.Series(dtype=float)
    if len(c) > 2:
        t = c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))
        out.update({"clv_bets": int(len(c)), "avg_clv": float(c.mean()), "clv_t": float(t),
                    "clv_p": float(stats.t.sf(t, len(c) - 1)), "beat_close": float((c > 0).mean()),
                    "benchmarks": b.loc[c.index, "clv_book"].value_counts().to_dict()})
    return out


def by_season(b: pd.DataFrame) -> list[dict]:
    if b.empty:
        return []
    y = confirm.season(pd.to_datetime(b["date"]))
    g = b.assign(y=y).groupby("y")
    return (pd.DataFrame({"bets": g.size(), "roi": g["ret"].mean(),
                          "clv_bets": g["clv"].count(), "avg_clv": g["clv"].mean()})
            .round(4).reset_index().to_dict("records"))


def effect_vs_close(d: pd.DataFrame) -> dict:
    """Is the luck over-reaction still there against Pinnacle's closing price?"""
    h = d[(d["date"] >= confirm.HOLDOUT_START) & d[["psc_h", "psc_d", "psc_a"]].notna().all(axis=1)]
    if h.empty:
        return {}
    pc = market.devig(h[["psc_h", "psc_d", "psc_a"]].to_numpy(dtype=float), "power")
    imp = signals.ImpliedGoals()(pc[:, 0], pc[:, 2])
    y = (h["fthg"] - h["ftag"]) - (imp["lh"] - imp["la"])
    ok = h[confirm.FEATS].notna().all(axis=1) & np.isfinite(y)
    r = sm.OLS(y[ok], sm.add_constant(h.loc[ok, confirm.FEATS])).fit(cov_type="HC1")
    return {"n": int(ok.sum()), "t": r.tvalues[confirm.FEATS].round(2).to_dict(),
            "coef": r.params[confirm.FEATS].round(4).to_dict()}


def baselines(d: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """CLV of betting every side (the cost of the margin) and of model-free line shopping."""
    out = {}
    x = d[(d["date"] >= confirm.HOLDOUT_START) & d[["ps_ahh", "psc_ahh"]].notna().all(axis=1)
          & (d["ahc_line"] == d["ah_line"])]
    if len(x):
        f = market.devig(x[["psc_ahh", "psc_aha"]].to_numpy(dtype=float), "power")
        out["ah_pinnacle_every_side_clv"] = float(np.nanmean(np.r_[x["ps_ahh"] * f[:, 0] - 1,
                                                                    x["ps_aha"] * f[:, 1] - 1]))
    y = raw[(raw["date"] >= confirm.HOLDOUT_START) & (raw["tier"] >= 2)
            & raw[["max_o25", "max_u25", "b365_o25", "b365_u25"]].notna().all(axis=1)]
    yc = y[y[["psc_o25", "psc_u25"]].notna().all(axis=1)]
    if len(yc):
        f = market.devig(yc[["psc_o25", "psc_u25"]].to_numpy(dtype=float), "power")
        out["ou_max_tiers2_5_every_side_clv"] = float(np.nanmean(np.r_[yc["max_o25"] * f[:, 0] - 1,
                                                                        yc["max_u25"] * f[:, 1] - 1]))
    if len(y):
        pb = market.devig(y[["b365_o25", "b365_u25"]].to_numpy(dtype=float), "power")
        mx = y[["max_o25", "max_u25"]].to_numpy(dtype=float)
        ev = pb * mx - 1
        ev = np.where((ev >= 0.03) & (ev <= 0.10) & (mx <= 4.0), ev, -np.inf)
        j = ev.argmax(axis=1)
        keep = np.isfinite(ev[np.arange(len(y)), j])
        sel = y[keep].assign(odds=mx[keep, :][np.arange(keep.sum()), j[keep]])
        over = ((sel["fthg"] + sel["ftag"]) > 2.5).to_numpy()
        win = np.where(j[keep] == 0, over, ~over)
        sel["ret"] = np.where(win, sel["odds"] - 1, -1.0)
        sel["clv"], sel["clv_book"] = forward.closing_value(sel, ("o25", "u25"), j[keep])
        out["ou_line_shopping_no_model"] = summary(sel)
    return out


def run(d: pd.DataFrame | None = None, out_dir: Path = OUT_DIR) -> dict:
    from ..pipeline import load_predictions

    raw = load_predictions() if d is None else d
    d = confirm.prepare(raw)
    years = [y for y in sorted(d["season_y"].unique()) if y >= 2016]
    res: dict = {}
    # H2: skill/luck model, 1X2 at Bet365
    b2 = clv_1x2(confirm.bets_1x2(confirm._walk_1x2(d, years, use_max=False), confirm.H2), d)
    res["H2_1x2_bet365"] = summary(b2)
    # H3 / F2: luck-fade Asian handicap at Bet365, Pinnacle and Betfair Exchange
    bets = {}
    for price in ("b365", "ps", "bfe"):
        if f"{price}_ahh" in d and d[f"{price}_ahh"].notna().any():
            bets[price] = clv_ah(confirm._walk_ah(d, years, price=price), d)
            res[f"F2_ah_{price}"] = summary(bets[price], commission=forward.COMMISSION.get(price, 0.0))
    # F2 bet just before kick-off, at closing prices on the closing line
    for price in ("psc", "bfec", "avgc"):
        if f"{price}_ahh" in d and d[f"{price}_ahh"].notna().any():
            b = confirm._walk_ah(d, years, price=price, line_col="ahc_line")
            res[f"F2_ah_at_close_{price}"] = summary(b, commission=0.05 if price == "bfec" else 0.0)
    res["luck_effect_vs_pinnacle_close"] = effect_vs_close(d)
    res["baselines"] = baselines(d, raw)
    # F1: lower-division over/under 2.5 (walk-forward oublend model from the build)
    spec = json.loads(forward.FROZEN.read_text(encoding="utf-8"))
    hold = raw[(raw["date"] >= confirm.HOLDOUT_START) & raw["ftr"].notna()]
    f1 = forward._f1(hold, spec["F1"])
    for price, g in f1.groupby("price_book"):
        res[f"F1_ou_{price}"] = summary(g)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["by_season"] = {"H2_1x2_bet365": by_season(b2),
                        **{f"F2_ah_{k}": by_season(v) for k, v in bets.items()},
                        **{f"F1_ou_{k}": by_season(g) for k, g in f1.groupby("price_book")}}
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res
