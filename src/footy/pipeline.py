"""End-to-end pipeline: ratings -> walk-forward probabilities -> artefacts."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import store
from .models import probability as prob
from .ratings.engine import RatingEngine
from .ratings.poisson_math import dc_probs

log = logging.getLogger(__name__)

PREDICTIONS = store.DATA_DIR / "predictions.parquet"
ENGINE_PKL = store.DATA_DIR / "engine.pkl"
MODELS_PKL = store.DATA_DIR / "models.pkl"
ADJUSTMENTS = store.ROOT / "config" / "adjustments.yaml"


def build(matches: pd.DataFrame, fixtures: pd.DataFrame | None = None, use_gbm: bool = True,
          first_year: int = 2004) -> pd.DataFrame:
    """Run ratings + walk-forward models over history (and fixtures if given)."""
    data = matches
    if fixtures is not None and not fixtures.empty:
        fx = fixtures.copy()
        fx[["fthg", "ftag", "ftr"]] = np.nan
        known = set(matches["match_id"])
        fx = fx[~fx["match_id"].isin(known)]
        data = pd.concat([matches, fx], ignore_index=True)
    data = data.sort_values(["date", "league", "home"]).reset_index(drop=True)
    log.info("running rating engine over %d matches", len(data))
    engine = RatingEngine()
    feats = engine.run(data)
    d = pd.concat([data, feats], axis=1)
    log.info("walk-forward 1X2 models")
    d = prob.walk_forward(d, first_year=first_year, use_gbm=use_gbm)
    log.info("walk-forward over/under models")
    d = prob.walk_forward_ou(d, first_year=first_year)
    log.info("fitting final models on recent history")
    models = prob.fit_final(d[d["ftr"].notna()], use_gbm=use_gbm)
    store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    d.to_parquet(PREDICTIONS, index=False)
    with open(ENGINE_PKL, "wb") as fh:
        pickle.dump(engine, fh)
    with open(MODELS_PKL, "wb") as fh:
        pickle.dump(models, fh)
    return d


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        raise FileNotFoundError("run `footy build` first")
    return pd.read_parquet(PREDICTIONS)


def load_engine() -> RatingEngine:
    with open(ENGINE_PKL, "rb") as fh:
        return pickle.load(fh)


def load_models() -> dict:
    with open(MODELS_PKL, "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------- team news
def load_adjustments(path: Path = ADJUSTMENTS) -> list[dict]:
    if not Path(path).exists():
        return []
    items = yaml.safe_load(Path(path).read_text()) or []
    return [a for a in items if isinstance(a, dict) and a.get("team")]


def apply_adjustments(d: pd.DataFrame, adjustments: list[dict], rho: float = -0.06) -> pd.DataFrame:
    """Shift expected goals for teams with manual team-news adjustments.

    ``attack``  : relative change in the team's own expected goals (-0.1 = -10%)
    ``defence`` : relative change in goals *conceded* (+0.1 = concede 10% more)
    """
    if not adjustments or d.empty:
        return d
    d = d.copy()
    d["adjusted"] = ""
    for a in adjustments:
        until = pd.Timestamp(a["until"]) if a.get("until") else None
        att, dfn = float(a.get("attack", 0) or 0), float(a.get("defence", 0) or 0)
        for side, other in (("h", "a"), ("a", "h")):
            team_col = "home" if side == "h" else "away"
            m = d[team_col].eq(a["team"])
            if a.get("country"):
                m &= d["country"].eq(a["country"])
            if until is not None:
                m &= d["date"] <= until
            if not m.any():
                continue
            for mdl in ("pois", "pxg"):
                d.loc[m, f"{mdl}_l{side}"] *= np.exp(att)
                d.loc[m, f"{mdl}_l{other}"] *= np.exp(dfn)
            d.loc[m, "adjusted"] += f"{a['team']}: {a.get('note', '')} "
    for mdl in ("pois", "pxg"):
        p = dc_probs(d[f"{mdl}_lh"].to_numpy(), d[f"{mdl}_la"].to_numpy(), rho)
        for k, v in p.items():
            d[f"{mdl}_{k}"] = v
    return d


# ------------------------------------------------------------- predictions
def upcoming(d: pd.DataFrame | None = None, models: dict | None = None,
             adjustments: list[dict] | None = None) -> pd.DataFrame:
    """Predictions for unplayed matches present in the predictions table."""
    d = load_predictions() if d is None else d
    last_played = d.loc[d["ftr"].notna(), "date"].max()
    # Unplayed rows older than the latest result are postponed/abandoned games.
    fx = d[d["ftr"].isna() & (d["date"] >= last_played - pd.Timedelta(days=3))].copy()
    if fx.empty:
        return fx
    models = models or load_models()
    adjustments = load_adjustments() if adjustments is None else adjustments
    fx = apply_adjustments(fx, adjustments)
    fx = prob.predict_with(models, fx)
    return add_value_columns(fx)


def add_value_columns(fx: pd.DataFrame, model: str = "blend") -> pd.DataFrame:
    """Fair odds and expected value at the best available price for each outcome."""
    fx = fx.copy()
    src = model if fx[f"{model}_ph"].notna().any() else "ens"
    for o in ("h", "d", "a"):
        p = fx[f"{src}_p{o}"]
        fx[f"fair_{o}"] = 1 / p
        best = fx[[c for c in (f"max_{o}", f"avg_{o}", f"b365_{o}", f"ps_{o}") if c in fx]].max(axis=1)
        fx[f"best_{o}"] = best
        fx[f"ev_{o}"] = p * best - 1
    return fx


def what_if(engine: RatingEngine, models: dict, league: str, home: str, away: str,
            country: str | None = None, adjustments: list[dict] | None = None) -> pd.DataFrame:
    """Predict an arbitrary match-up from the latest ratings."""
    from . import leagues as L

    import copy

    lg = L.get(league)
    country = country or lg.country
    h, a = f"{country}|{home}", f"{country}|{away}"
    if h not in engine.teams or a not in engine.teams:
        raise KeyError(f"unknown team: {home if h not in engine.teams else away}")
    # Work on a copy so a team from another division is re-based exactly as the
    # engine would do it when that team actually plays in this league.
    engine = copy.deepcopy(engine)
    when = max(engine.teams[h].last_date, engine.teams[a].last_date) + pd.Timedelta(days=7)
    for team in (h, a):
        if engine.teams[team].league != league:
            log.info("%s last played in %s; re-basing to %s", team, engine.teams[team].league, league)
        engine._prepare(team, league, when)
    f: dict = {}
    for m in engine.models:
        f.update(m.pre(h, a, league))
    sh, sa = engine.teams[h], engine.teams[a]
    c = engine.ctx[league]
    f.update({
        "h_n": sh.n_total, "a_n": sa.n_total, "h_ndiv": sh.n_div, "a_ndiv": sa.n_div,
        "h_entry": "settled", "a_entry": "settled", "h_rest": 7, "a_rest": 7,
        "lg_home_rate": c.home_rate, "lg_draw_rate": c.draw_rate, "lg_goals": c.goals, "lg_n": c.n,
    })
    row = pd.DataFrame([f])
    row["league"], row["home"], row["away"], row["country"] = league, home, away, country
    row["tier"] = lg.tier
    row["date"] = pd.Timestamp.today().normalize()
    for name in ("sot", "shots"):
        if not np.isfinite(row[f"{name}_lh"]).all():
            row[[f"{name}_lh", f"{name}_la"]] = np.nan
    for mdl in ("pois", "pxg"):
        p = dc_probs(row[f"{mdl}_lh"].to_numpy(), row[f"{mdl}_la"].to_numpy(), engine.rho)
        for k, v in p.items():
            row[f"{mdl}_{k}"] = v
    if adjustments:
        row = apply_adjustments(row, adjustments, engine.rho)
    return prob.predict_with(models, row)
