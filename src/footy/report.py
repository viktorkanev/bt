"""Build report tables (CSV/JSON) and the static HTML dashboard."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import evaluate as ev
from . import leagues as L
from .data import store
from .models import market
from .models.probability import eligible

log = logging.getLogger(__name__)

OUT_DIR = store.ROOT / "reports" / "latest"
WINDOW = bt.Window()
MODELS_1X2 = ["mkt", "elo", "pi", "pois", "pxg", "ens", "gbm", "blend"]
MODEL_LABELS = {
    "mkt": "Bet365 (margin removed)", "elo": "Elo", "pi": "Pi-ratings",
    "pois": "Dixon-Coles (goals)", "pxg": "Dixon-Coles (pseudo-xG)",
    "ens": "Ensemble (ratings only)", "gbm": "Gradient boosting (ratings only)",
    "blend": "Ensemble + market",
}

# Fixed, pre-declared strategies (not picked by looking at results).
NAMED = {
    "consensus_max": bt.Strategy(name="Consensus vs best price", model="mkt", price="max",
                                 min_edge=0.02, min_odds=1.3, max_odds=8.0),
    "blend_max": bt.Strategy(name="Model+market @ best price", model="blend", price="max",
                             min_edge=0.03, min_odds=1.3, max_odds=8.0),
    "blend_b365": bt.Strategy(name="Model+market @ Bet365", model="blend", price="b365",
                              min_edge=0.03, min_odds=1.3, max_odds=8.0),
    "ens_max": bt.Strategy(name="Ratings-only model @ best price", model="ens", price="max",
                           min_edge=0.05, min_odds=1.3, max_odds=8.0),
    "ou_blend_max": bt.Strategy(name="Over/Under 2.5 model+market @ best price", market="ou25",
                                model="oublend", price="max", min_edge=0.03, min_odds=1.3,
                                max_odds=4.0),
    "ah_pxg_b365": bt.Strategy(name="Asian handicap pseudo-xG @ Bet365", market="ah",
                               model="pxg", price="b365", min_edge=0.05, min_odds=1.5,
                               max_odds=3.0),
}


def _period(d: pd.DataFrame) -> pd.Series:
    p = pd.Series("", index=d.index)
    p[(d["date"] >= WINDOW.dev[0]) & (d["date"] < WINDOW.dev[1])] = "dev"
    p[(d["date"] >= WINDOW.holdout[0]) & (d["date"] < WINDOW.holdout[1])] = "holdout"
    return p


def eval_frame(d: pd.DataFrame) -> pd.DataFrame:
    m = d["ftr"].notna() & eligible(d) & (d["date"] >= WINDOW.dev[0])
    for mdl in MODELS_1X2:
        m &= d[f"{mdl}_ph"].notna()
    e = d[m].copy()
    e["period"] = _period(e)
    e["year"] = np.where(e["date"].dt.month >= 7, e["date"].dt.year, e["date"].dt.year - 1)
    e["b365_margin"] = market.overround(e[["b365_h", "b365_d", "b365_a"]].to_numpy())
    e["max_margin"] = market.overround(e[["max_h", "max_d", "max_a"]].to_numpy())
    return e


def accuracy(e: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    out["overall"] = ev.score_table(e, MODELS_1X2, by=["period"])
    hold = e[e["period"] == "holdout"]
    bl = ev.score_table(hold, MODELS_1X2, by=["league"])
    piv = bl.pivot(index="league", columns="model", values="rps")
    n = hold.groupby("league").size().rename("matches")
    marg = hold.groupby("league")[["b365_margin", "max_margin"]].mean()
    tbl = pd.concat([n, marg, piv.add_prefix("rps_")], axis=1).reset_index()
    for mdl in ("pxg", "ens", "gbm", "blend"):
        tbl[f"gap_{mdl}"] = tbl[f"rps_{mdl}"] - tbl["rps_mkt"]
    tbl["name"] = tbl["league"].map(L.league_name)
    tbl["tier"] = tbl["league"].map(lambda c: L.get(c).tier)
    out["by_league"] = tbl.sort_values(["tier", "league"])
    yr = ev.score_table(e, ["mkt", "pxg", "ens", "blend"], by=["year"])
    yr = yr.pivot(index="year", columns="model", values="rps").reset_index()
    yr = yr.merge(e.groupby("year")[["b365_margin"]].mean().reset_index(), on="year")
    yr = yr.merge(e.groupby("year").size().rename("matches").reset_index(), on="year")
    out["by_year"] = yr
    tier = ev.score_table(hold.assign(tier_grp=hold["tier"].clip(upper=3)), MODELS_1X2,
                          by=["tier_grp"])
    out["by_tier"] = tier
    cal = []
    y = ev.onehot(hold["ftr"])
    for mdl in ("mkt", "ens", "blend"):
        for j, o in enumerate(("h", "d", "a")):
            c = ev.calibration(hold[f"{mdl}_p{o}"].to_numpy(), y[:, j], bins=10)
            c["model"], c["outcome"] = mdl, o
            cal.append(c)
    out["calibration"] = pd.concat(cal, ignore_index=True)
    # Over/under 2.5
    ou = e[e["oumkt_po25"].notna() & e["oublend_po25"].notna() & e["ouens_po25"].notna()]
    rows = []
    over = ((ou["fthg"] + ou["ftag"]) > 2.5).astype(float).to_numpy()
    for per, g in ou.groupby("period"):
        yo = over[ou.index.get_indexer(g.index)]
        for mdl in ("oumkt", "pxg", "ouens", "oublend"):
            p = np.clip(g[f"{mdl}_po25"].to_numpy(), 1e-6, 1 - 1e-6)
            rows.append({"period": per, "model": mdl, "n": len(g),
                         "logloss": float(-(yo * np.log(p) + (1 - yo) * np.log(1 - p)).mean()),
                         "brier": float(((p - yo) ** 2).mean())})
    out["ou"] = pd.DataFrame(rows)
    return out


def strategy_grid(e: pd.DataFrame) -> pd.DataFrame:
    strats: list[bt.Strategy] = []
    for model in ("mkt", "pxg", "ens", "gbm", "blend"):
        for price in ("b365", "max"):
            if model == "mkt" and price == "b365":
                continue  # margin-free Bet365 vs Bet365 can never show value
            strats += bt.grid(bt.Strategy(model=model, price=price), min_edge=[0.02, 0.05, 0.10],
                              max_odds=[3.0, 8.0])
    for model in ("pxg", "ouens", "oublend"):
        for price in ("b365", "max"):
            strats += bt.grid(bt.Strategy(market="ou25", model=model, price=price, max_odds=4.0),
                              min_edge=[0.02, 0.05, 0.10])
    strats += bt.grid(bt.Strategy(market="ah", model="pxg", price="b365", min_odds=1.5,
                                  max_odds=3.0), min_edge=[0.03, 0.06, 0.10])
    res = bt.select_and_validate(e, strats, WINDOW, min_bets=300)
    res["market"] = [s.market for s in res["_s"]]
    res["model"] = [s.model for s in res["_s"]]
    res["price"] = [s.price for s in res["_s"]]
    res["min_edge"] = [s.min_edge for s in res["_s"]]
    res["max_odds"] = [s.max_odds for s in res["_s"]]
    return res.drop(columns="_s")


def named_strategies(e: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows, curves, per_league = [], {}, []
    dev, hold = bt.split(e, WINDOW)
    for key, s in NAMED.items():
        bd, bh = bt.run(dev, s), bt.run(hold, s)
        sd, sh = bt.summarise(bd), bt.summarise(bh)
        rows.append({"key": key, "name": s.name, "label": s.label(),
                     **{f"dev_{k}": v for k, v in sd.items()},
                     **{f"hold_{k}": v for k, v in sh.items()}})
        if not bh.empty:
            c = bh.groupby("date")["ret"].sum().cumsum()
            nb = bh.groupby("date").size().cumsum()
            idx = np.unique(np.linspace(0, len(c) - 1, min(len(c), 320)).astype(int))
            curves[key] = {"date": [str(x.date()) for x in c.index[idx]],
                           "units": [round(float(v), 2) for v in c.to_numpy()[idx]],
                           "bets": [int(v) for v in nb.to_numpy()[idx]]}
        for per, b in (("dev", bd), ("holdout", bh)):
            if b.empty:
                continue
            g = bt.by(b, "league")
            g["strategy"], g["period"] = key, per
            per_league.append(g)
    pl = pd.concat(per_league, ignore_index=True) if per_league else pd.DataFrame()
    return pd.DataFrame(rows), curves, pl


def league_edge_table(acc_league: pd.DataFrame, per_league: pd.DataFrame) -> pd.DataFrame:
    t = acc_league.copy()
    if per_league.empty:
        return t
    for key in ("consensus_max", "blend_max", "blend_b365", "ou_blend_max"):
        for per in ("dev", "holdout"):
            g = per_league[(per_league["strategy"] == key) & (per_league["period"] == per)]
            g = g.set_index("league")
            t[f"{key}_{per}_roi"] = t["league"].map(g["roi"])
            t[f"{key}_{per}_bets"] = t["league"].map(g["bets"]).fillna(0).astype(int)
    return t


def tier_strategy_table(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dev, hold = bt.split(e, WINDOW)
    for key in ("consensus_max", "blend_max", "blend_b365", "ou_blend_max"):
        s = NAMED[key]
        for per, df in (("dev", dev), ("holdout", hold)):
            b = bt.run(df, s)
            if b.empty:
                continue
            b["tier_grp"] = b["league"].map(lambda c: min(L.get(c).tier, 3))
            for t, g in b.groupby("tier_grp"):
                sm = bt.summarise(g, n_boot=300)
                rows.append({"strategy": key, "period": per, "tier_grp": int(t), **sm})
            # tier_grp 0 = all lower divisions (tiers 2-5) combined
            low = b[b["tier_grp"] >= 2]
            if not low.empty:
                rows.append({"strategy": key, "period": per, "tier_grp": 0,
                             **bt.summarise(low, n_boot=1000)})
    return pd.DataFrame(rows)


def edge_bands(e: pd.DataFrame) -> pd.DataFrame:
    """ROI by claimed edge: profits that live only in huge edges point to stale/outlier prices."""
    rows = []
    dev, hold = bt.split(e, WINDOW)
    bands = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 1.0]
    for key in ("consensus_max", "blend_max", "ou_blend_max"):
        for per, df in (("dev", dev), ("holdout", hold)):
            b = bt.run(df, NAMED[key])
            if b.empty:
                continue
            b["band"] = pd.cut(b["ev"], bands, include_lowest=True)
            for band, g in b.groupby("band", observed=True):
                rows.append({"strategy": key, "period": per, "band_lo": band.left, "band_hi": band.right,
                             "bets": len(g), "roi": g["ret"].mean(), "avg_odds": g["odds"].mean()})
    return pd.DataFrame(rows)


def capped_variants(e: pd.DataFrame, caps=(0.05, 0.08, 0.12)) -> pd.DataFrame:
    rows = []
    dev, hold = bt.split(e, WINDOW)
    for key in ("consensus_max", "blend_max", "ou_blend_max"):
        for cap in caps:
            s = bt.Strategy(**{**asdict(NAMED[key]), "max_edge": cap})
            sd, sh = bt.summarise(bt.run(dev, s), n_boot=300), bt.summarise(bt.run(hold, s), n_boot=300)
            rows.append({"strategy": key, "name": NAMED[key].name, "max_edge": cap,
                         **{f"dev_{k}": v for k, v in sd.items()},
                         **{f"hold_{k}": v for k, v in sh.items()}})
    return pd.DataFrame(rows)


def ratings_snapshot(as_of: pd.Timestamp) -> pd.DataFrame:
    from .pipeline import load_engine

    eng = load_engine()
    r = eng.ratings_table(as_of)
    r["league_name"] = r["league"].map(L.league_name)
    r["tier"] = r["league"].map(lambda c: L.get(c).tier)
    r = r.sort_values(["tier", "league", "elo"], ascending=[True, True, False])
    keep = ["team", "league", "league_name", "tier", "elo", "pi_home", "pi_away",
            "pxg_att", "pxg_def", "sot_att", "sot_def", "matches", "last_match"]
    return r[[c for c in keep if c in r.columns]]


def make_report(out_dir: Path = OUT_DIR) -> Path:
    from .pipeline import load_predictions, upcoming
    from .report_html import render

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_predictions()
    e = eval_frame(d)
    log.info("evaluation rows: %d", len(e))
    acc = accuracy(e)
    for k, v in acc.items():
        v.to_csv(out_dir / f"accuracy_{k}.csv", index=False)
    grid = strategy_grid(e)
    grid.to_csv(out_dir / "strategy_grid.csv", index=False)
    named, curves, per_league = named_strategies(e)
    named.to_csv(out_dir / "strategies.csv", index=False)
    per_league.to_csv(out_dir / "strategies_by_league.csv", index=False)
    edge = league_edge_table(acc["by_league"], per_league)
    edge.to_csv(out_dir / "league_edge.csv", index=False)
    tiers = tier_strategy_table(e)
    tiers.to_csv(out_dir / "strategies_by_tier.csv", index=False)
    edge_bands(e).to_csv(out_dir / "edge_bands.csv", index=False)
    capped_variants(e).to_csv(out_dir / "capped_strategies.csv", index=False)
    last = d.loc[d["ftr"].notna(), "date"].max()
    ratings = ratings_snapshot(last)
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    fx = upcoming(d)
    if not fx.empty:
        fx.to_csv(out_dir / "upcoming.csv", index=False)
    elif (out_dir / "upcoming.csv").exists():
        (out_dir / "upcoming.csv").unlink()
    meta = store.load_meta()
    summary = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_source": meta.get("source", "?"),
        "matches": int(d["ftr"].notna().sum()),
        "first_date": str(d["date"].min().date()), "last_result": str(last.date()),
        "leagues": int(d["league"].nunique()),
        "eval_matches": int(len(e)),
        "dev_window": list(WINDOW.dev), "holdout_window": list(WINDOW.holdout),
        "named_strategies": {k: asdict(s) for k, s in NAMED.items()},
        "model_labels": MODEL_LABELS,
        "upcoming": int(len(fx)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "curves.json").write_text(json.dumps(curves))
    html = render(out_dir, standalone=True)
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"
