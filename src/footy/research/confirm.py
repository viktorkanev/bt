"""Pre-registered confirmation test of the market-surprise strategies.

Rules are frozen in ``docs/PREREGISTRATION.md``. This module implements them
exactly; ``run(window="dev")`` reproduces the development numbers that informed
the rules and ``run(window="holdout")`` is the one-shot confirmation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import store
from ..models import market
from ..models.probability import eligible
from ..ratings.poisson_math import ah_ev, ah_settle, goal_diff_dist, score_matrix
from . import signals

OUT_DIR = store.ROOT / "reports" / "research"
DEV_START = "2005-07-01"
HOLDOUT_START = "2016-07-01"
FEATS = [f"d_{c}" for c in signals.SURPRISE_FEATURES]
EXPECTED_SIGN = {"d_surp_pts_10": -1, "d_surp_sot_10": +1, "d_fin_luck": -1, "d_gk_luck": -1}


@dataclass(frozen=True)
class Rule1X2:
    name: str
    use_max: bool          # add best-price (max) log-probabilities as features
    min_ev: float
    min_odds: float = 1.7
    max_odds: float = 7.0
    outcomes: tuple = ("h", "a")


H2 = Rule1X2("H2: skill/luck model, 1X2 at Bet365", use_max=False, min_ev=0.03)
H4 = Rule1X2("H4: skill/luck + best-price model, 1X2 at Bet365", use_max=True, min_ev=0.04)
H3_MIN_SIGNAL = 0.05


def season(dates: pd.Series) -> np.ndarray:
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def prepare(d: pd.DataFrame | None = None) -> pd.DataFrame:
    if d is None:
        from ..pipeline import load_predictions
        d = load_predictions()
    d = signals.add_all(d.reset_index(drop=True))
    d = d[d["ftr"].notna() & eligible(d) & (d["date"] >= DEV_START)].copy()
    d["low"] = (d["tier"] >= 2).astype(float)
    pb = market.devig(d[["b365_h", "b365_d", "b365_a"]].to_numpy(), "power")
    pm = market.devig(d[["max_h", "max_d", "max_a"]].to_numpy(), "power")
    for j, o in enumerate("hda"):
        d[f"lb_{o}"] = np.log(np.clip(pb[:, j], 1e-4, 1))
        d[f"lm_{o}"] = np.log(np.clip(pm[:, j], 1e-4, 1))
    for c in FEATS:
        d[f"low_{c}"] = d[c] * d["low"]
    d["res_gd"] = (d["fthg"] - d["ftag"]) - (d["x_lh"] - d["x_la"])
    d["season_y"] = season(d["date"])
    return d


def _design(d: pd.DataFrame, use_max: bool) -> list[str]:
    cols = ["lb_h", "lb_d", "lb_a"] + FEATS + [f"low_{c}" for c in FEATS]
    if use_max:
        cols += ["lm_h", "lm_d", "lm_a"]
    return cols


def _complete(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    return d[cols].notna().all(axis=1) & np.isfinite(d[cols]).all(axis=1)


# ------------------------------------------------------------- H1: the effect
def h1_effect(test: pd.DataFrame) -> dict:
    ok = _complete(test, FEATS + ["res_gd"])
    X = sm.add_constant(test.loc[ok, FEATS])
    r = sm.OLS(test.loc[ok, "res_gd"], X).fit(cov_type="HC1")
    signs_ok = all(np.sign(r.params[c]) == s for c, s in EXPECTED_SIGN.items())
    ftest = r.wald_test(" = 0, ".join(FEATS) + " = 0", scalar=True)
    one_sided = {c: float(stats.norm.sf(r.tvalues[c] * EXPECTED_SIGN[c])) for c in FEATS}
    return {"n": int(ok.sum()), "coef": r.params[FEATS].round(5).to_dict(),
            "t": r.tvalues[FEATS].round(2).to_dict(), "one_sided_p": one_sided,
            "signs_as_predicted": bool(signs_ok), "joint_p": float(ftest.pvalue),
            "p_value": float(ftest.pvalue) if signs_ok else 1.0}


# ---------------------------------------------------- H2 / H4: 1X2 at Bet365
def _walk_1x2(d: pd.DataFrame, test_years: list[int], use_max: bool) -> pd.DataFrame:
    cols = _design(d, use_max)
    ok = _complete(d, cols) & d[["b365_h", "b365_d", "b365_a"]].notna().all(axis=1)
    y = d["ftr"].map({"H": 0, "D": 1, "A": 2})
    parts = []
    for Y in test_years:
        tr = ok & (d["season_y"] < Y)
        te = ok & (d["season_y"] == Y)
        if te.sum() == 0 or tr.sum() < 5000:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        m.fit(d.loc[tr, cols], y[tr])
        p = m.predict_proba(d.loc[te, cols])
        sub = d.loc[te, ["match_id", "date", "league", "tier", "home", "away", "ftr",
                         "b365_h", "b365_d", "b365_a"]].copy()
        sub[["p_h", "p_d", "p_a"]] = p
        parts.append(sub)
    return pd.concat(parts) if parts else pd.DataFrame()


def bets_1x2(pred: pd.DataFrame, rule: Rule1X2) -> pd.DataFrame:
    P = pred[["p_h", "p_d", "p_a"]].to_numpy()
    O = pred[["b365_h", "b365_d", "b365_a"]].to_numpy()
    E = P * O - 1
    allowed = np.array([o in rule.outcomes for o in "hda"])
    ok = allowed[None, :] & (O >= rule.min_odds) & (O <= rule.max_odds) & np.isfinite(E)
    E = np.where(ok, E, -np.inf)
    j = E.argmax(axis=1)
    e = E[np.arange(len(E)), j]
    keep = e >= rule.min_ev
    sel = pred[keep].copy()
    jj = j[keep]
    sel["pick"] = np.array(list("hda"))[jj]
    sel["p"] = P[keep][np.arange(keep.sum()), jj]
    sel["odds"] = O[keep][np.arange(keep.sum()), jj]
    sel["ev"] = e[keep]
    sel["win"] = (sel["ftr"].str.lower() == sel["pick"]).astype(float)
    sel["ret"] = np.where(sel["win"] == 1, sel["odds"] - 1, -1.0)
    return sel


# ------------------------------------------------- H3: Asian handicap at Bet365
def _walk_ah(d: pd.DataFrame, test_years: list[int], price: str = "b365",
             line_col: str = "ah_line") -> pd.DataFrame:
    cols = FEATS + [f"low_{c}" for c in FEATS]
    ok = _complete(d, cols + ["res_gd", "x_lh", "x_la"])
    parts = []
    for Y in test_years:
        tr = ok & (d["season_y"] < Y)
        te = ok & (d["season_y"] == Y) & d[line_col].notna() & d[f"{price}_ahh"].notna() & d[f"{price}_aha"].notna()
        if te.sum() == 0 or tr.sum() < 5000:
            continue
        r = sm.OLS(d.loc[tr, "res_gd"], sm.add_constant(d.loc[tr, cols])).fit()
        delta = (d.loc[te, cols].to_numpy() @ r.params[cols].to_numpy())
        s = d.loc[te].copy()
        lh = np.clip(s["x_lh"] + delta / 2, 0.05, None).to_numpy()
        la = np.clip(s["x_la"] - delta / 2, 0.05, None).to_numpy()
        diffs, pgd = goal_diff_dist(score_matrix(lh, la, -0.07))
        _, pgd0 = goal_diff_dist(score_matrix(s["x_lh"].to_numpy(), s["x_la"].to_numpy(), -0.07))
        args = (s[line_col].to_numpy(), s[f"{price}_ahh"].to_numpy(), s[f"{price}_aha"].to_numpy())
        eh, ea = ah_ev(diffs, pgd, *args)
        eh0, ea0 = ah_ev(diffs, pgd0, *args)
        sh, sa = eh - eh0, ea - ea0
        side = np.where(sh >= sa, 0, 1)
        sig = np.maximum(sh, sa)
        stl = ah_settle((s["fthg"] - s["ftag"]).to_numpy(), s[line_col].to_numpy())
        stl = np.where(side == 0, stl, -stl)
        odds = np.where(side == 0, s[f"{price}_ahh"], s[f"{price}_aha"])
        s["odds_other"] = np.where(side == 0, s[f"{price}_aha"], s[f"{price}_ahh"])
        s["pick"] = np.where(side == 0, "home ", "away ") + s[line_col].map(lambda x: f"{x:+.2f}")
        s["signal"], s["odds"] = sig, odds
        s["win"] = (stl > 0).astype(float)
        s["ret"] = np.where(stl > 0, stl * (odds - 1), stl)
        s["settle"] = stl
        parts.append(s[["match_id", "date", "league", "tier", "home", "away", "pick", "signal",
                        "odds", "odds_other", "settle", "win", "ret"]])
    out = pd.concat(parts) if parts else pd.DataFrame()
    return out[out["signal"] >= H3_MIN_SIGNAL] if not out.empty else out


# ------------------------------------------------------------------ summaries
def summarise(b: pd.DataFrame, seed: int = 0, n_boot: int = 5000) -> dict:
    if b.empty:
        return {"bets": 0, "roi": np.nan, "p_value": 1.0}
    r = b["ret"].to_numpy()
    n = len(r)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(n_boot, n), replace=True).mean(axis=1)
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max())
    return {"bets": n, "roi": float(r.mean()), "roi_lo": float(np.percentile(boots, 2.5)),
            "roi_hi": float(np.percentile(boots, 97.5)), "p_value": float(stats.t.sf(t, n - 1)),
            "profit_units": float(r.sum()), "avg_odds": float(b["odds"].mean()),
            "hit_rate": float(b["win"].mean()), "max_drawdown_units": dd}


def margin_projection(b: pd.DataFrame, margins=(0.036, 0.03, 0.02, 0.0)) -> list[dict]:
    """Post-hoc: ROI of the same bets if the price carried a different margin.

    Fair odds come from Bet365's two handicap prices with the margin removed
    proportionally. Descriptive only; not part of the pre-registered test.
    """
    if b.empty:
        return []
    over = 1 / b["odds"] + 1 / b["odds_other"]
    fair = b["odds"] * over
    out = []
    for m in margins:
        o = fair / (1 + m)
        r = np.where(b["settle"] > 0, b["settle"] * (o - 1), b["settle"])
        out.append({"margin": m, "roi": float(np.mean(r)), "bets": int(len(r))})
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def run(window: str = "holdout", out_dir: Path = OUT_DIR, d: pd.DataFrame | None = None) -> dict:
    d = prepare(d)
    years = sorted(d["season_y"].unique())
    if window == "dev":
        test_years = [y for y in years if 2009 <= y <= 2015]
        test = d[(d["season_y"] >= 2009) & (d["date"] < HOLDOUT_START)]
    else:
        test_years = [y for y in years if y >= 2016]
        test = d[d["date"] >= HOLDOUT_START]
    out_dir = Path(out_dir) / window
    out_dir.mkdir(parents=True, exist_ok=True)

    res: dict = {"window": window, "test_years": [int(y) for y in test_years]}
    res["H1"] = h1_effect(test)
    pred = _walk_1x2(d, test_years, use_max=False)
    b2 = bets_1x2(pred, H2)
    pred_m = _walk_1x2(d, test_years, use_max=True)
    b4 = bets_1x2(pred_m, H4)
    b3 = _walk_ah(d, test_years)
    res["H2"], res["H3"], res["H4"] = summarise(b2), summarise(b3), summarise(b4)
    res["H3_margin_projection"] = margin_projection(b3)
    res["holm_adjusted_p"] = holm({k: res[k]["p_value"] for k in ("H1", "H2", "H3", "H4")})
    # secondary (not part of the confirmatory test): H2 at other thresholds
    res["H2_thresholds"] = {f"{th:.2f}": summarise(bets_1x2(pred, Rule1X2("x", False, th)), n_boot=500)
                            for th in (0.0, 0.02, 0.04, 0.06, 0.08)}
    for name, b in (("H2", b2), ("H3", b3), ("H4", b4)):
        b.to_csv(out_dir / f"bets_{name}.csv", index=False)
        if not b.empty:
            y = season(pd.to_datetime(b["date"]))
            res[f"{name}_by_season"] = (pd.DataFrame({"y": y, "ret": b["ret"].to_numpy()})
                                        .groupby("y")["ret"].agg(["size", "mean"]).round(4)
                                        .reset_index().to_dict("records"))
            res[f"{name}_by_tier"] = (b.assign(tg=b["tier"].clip(upper=3)).groupby("tg")["ret"]
                                      .agg(["size", "mean"]).round(4).reset_index().to_dict("records"))
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res
