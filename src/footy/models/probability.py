"""Map rating features to 1X2 probabilities, trained walk-forward.

Models
------
* ``elo`` / ``pi``  ordered logit on a single rating difference
* ``pois`` / ``pxg`` direct Dixon-Coles probabilities from the online Poisson models
* ``ens``           multinomial logistic regression on all rating features
* ``gbm``           gradient-boosted trees on the same features (cf. Hubacek et al. 2019)
* ``blend``         multinomial LR on rating features + market log-probabilities;
                    learns how much (if anything) the model adds to the bookmaker
* ``mkt``           bookmaker probabilities, margin removed (benchmark)

Walk-forward: for each test year Y the models are fitted on every match played
before 1 July of Y and used to predict matches from 1 July Y to 30 June Y+1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import market

log = logging.getLogger(__name__)

Y_ORDER = {"A": 0, "D": 1, "H": 2}  # ordinal coding for ordered logit
CLASS_COLS = ["ph", "pd", "pa"]


# ----------------------------------------------------------------- features
def _safe_log(x):
    if isinstance(x, pd.Series):
        return np.log(x.astype(float).clip(lower=1e-6))
    return np.log(np.clip(np.asarray(x, dtype=float), 1e-6, None))


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """Model-ready features (no market information)."""
    X = pd.DataFrame(index=d.index)
    X["elo"] = d["elo_diff"] / 100.0
    X["pi"] = d["pi_gd"]
    for m in ("pois", "pxg"):
        X[f"{m}_ratio"] = _safe_log(d[f"{m}_lh"]) - _safe_log(d[f"{m}_la"])
        X[f"{m}_total"] = _safe_log(d[f"{m}_lh"]) + _safe_log(d[f"{m}_la"])
    for k in ("ph", "pd", "pa"):
        X[f"pxg_log{k}"] = _safe_log(d[f"pxg_{k}"])
    for m in ("sot", "shots"):
        has = d[f"{m}_lh"].notna()
        X[f"{m}_ratio"] = (_safe_log(d[f"{m}_lh"]) - _safe_log(d[f"{m}_la"])).where(has, 0.0)
        X[f"{m}_total"] = (_safe_log(d[f"{m}_lh"]) + _safe_log(d[f"{m}_la"])).where(has, 0.0)
    X["has_shots"] = d["sot_lh"].notna().astype(float)
    X["lg_draw"] = d["lg_draw_rate"]
    X["lg_home"] = d["lg_home_rate"]
    X["lg_goals"] = d["lg_goals"]
    for side in ("h", "a"):
        e = d[f"{side}_entry"]
        X[f"{side}_promoted"] = (e == "up").astype(float)
        X[f"{side}_relegated"] = (e == "down").astype(float)
        X[f"{side}_newteam"] = (e == "new").astype(float)
    X["tier"] = d["tier"].astype(float)
    return X


def market_probs(d: pd.DataFrame, book: str = "b365", method: str = "power") -> np.ndarray:
    """De-vigged 1X2 probabilities, falling back to other books when missing."""
    out = np.full((len(d), 3), np.nan)
    for b in (book, "ps", "avg", "b365", "max"):
        cols = [f"{b}_h", f"{b}_d", f"{b}_a"]
        if not set(cols) <= set(d.columns):
            continue
        p = market.devig(d[cols].to_numpy(), method)
        miss = np.isnan(out).any(axis=1)
        out[miss] = p[miss]
    return out


# ----------------------------------------------------------- ordered logit
class OrderedLogit:
    """P(y <= k) = sigmoid(theta_k - x.beta) with y in {A=0, D=1, H=2}."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] != len(y):
            X = X.T
        k = X.shape[1]

        def nll(w):
            beta, t1, dt = w[:k], w[k], np.exp(np.clip(w[k + 1], -8, 4))
            eta = X @ beta
            c1 = expit(t1 - eta)
            c2 = expit(t1 + dt - eta)
            p = np.select([y == 0, y == 1], [c1, c2 - c1], 1 - c2)
            return -np.log(np.clip(p, 1e-12, None)).sum()

        w0 = np.r_[np.full(k, 0.5), -0.5, np.log(1.0)]
        res = minimize(nll, w0, method="L-BFGS-B")
        self.w, self.k = res.x, k
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != self.k:
            X = X.T
        beta, t1, dt = self.w[:self.k], self.w[self.k], np.exp(np.clip(self.w[self.k + 1], -8, 4))
        eta = X @ beta
        c1 = expit(t1 - eta)
        c2 = expit(t1 + dt - eta)
        return np.stack([1 - c2, c2 - c1, c1], axis=1)  # H, D, A


def _y_hda(ftr: pd.Series) -> np.ndarray:
    return ftr.map({"H": 0, "D": 1, "A": 2}).to_numpy()


def _fit_predict_sklearn(est, Xtr, ytr, Xte) -> np.ndarray:
    est.fit(Xtr, ytr)
    proba = est.predict_proba(Xte)
    classes = list(est.classes_) if hasattr(est, "classes_") else list(est[-1].classes_)
    out = np.zeros((len(Xte), 3))
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def make_gbm():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_depth=4, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0)


# -------------------------------------------------------------- walk-forward
def eligible(d: pd.DataFrame, min_matches: int = 10) -> pd.Series:
    return ((d["h_n"] >= min_matches) & (d["a_n"] >= min_matches) & (d["lg_n"] >= 200))


def walk_forward(d: pd.DataFrame, first_year: int = 2004, last_year: int | None = None,
                 use_gbm: bool = True, train_years: int = 12,
                 predict_unplayed: bool = True) -> pd.DataFrame:
    """Add out-of-sample probability columns ``{model}_{ph,pd,pa}`` to ``d``.

    Rows not yet played (fixtures) are predicted with the model trained on all
    history before them.
    """
    d = d.copy()
    X = build_features(d)
    mk = market_probs(d)
    d["mkt_ph"], d["mkt_pd"], d["mkt_pa"] = mk.T
    Xb = X.copy()
    for j, k in enumerate(CLASS_COLS):
        Xb[f"mkt_log{k}"] = _safe_log(mk[:, j])
    has_mkt = np.isfinite(mk).all(axis=1)
    played = d["ftr"].notna()
    elig = eligible(d)
    y = _y_hda(d["ftr"].where(played, "H"))
    yord = d["ftr"].map(Y_ORDER).to_numpy()
    year = np.where(d["date"].dt.month >= 7, d["date"].dt.year, d["date"].dt.year - 1)
    last_year = last_year or int(year.max())
    for m in ("elo", "pi", "ens", "gbm", "blend"):
        for k in CLASS_COLS:
            d[f"{m}_{k}"] = np.nan

    for Y in range(first_year, last_year + 1):
        tr = (year < Y) & (year >= Y - train_years) & played & elig
        te = (year == Y) & elig if not predict_unplayed else (year == Y) & (elig | ~played)
        if te.sum() == 0 or tr.sum() < 5000:
            continue
        log.info("walk-forward %s: train=%d test=%d", Y, tr.sum(), te.sum())
        Xtr, Xte = X[tr].to_numpy(), X[te].to_numpy()
        for name, col in (("elo", "elo"), ("pi", "pi")):
            ol = OrderedLogit().fit(X.loc[tr, [col]].to_numpy(), yord[tr])
            d.loc[te, [f"{name}_{k}" for k in CLASS_COLS]] = ol.predict_proba(X.loc[te, [col]].to_numpy())
        d.loc[te, [f"ens_{k}" for k in CLASS_COLS]] = _fit_predict_sklearn(make_lr(), Xtr, y[tr], Xte)
        if use_gbm:
            d.loc[te, [f"gbm_{k}" for k in CLASS_COLS]] = _fit_predict_sklearn(make_gbm(), Xtr, y[tr], Xte)
        trb, teb = tr & has_mkt, te & has_mkt
        if trb.sum() > 5000 and teb.sum():
            d.loc[teb, [f"blend_{k}" for k in CLASS_COLS]] = _fit_predict_sklearn(
                make_lr(), Xb[trb].to_numpy(), y[trb], Xb[teb].to_numpy())
    return d


def fit_final(d: pd.DataFrame, train_years: int = 12, use_gbm: bool = True) -> dict:
    """Fit every model on the most recent ``train_years`` of played matches.

    Returned dict is used to predict fixtures / arbitrary match-ups.
    """
    X = build_features(d)
    mk = market_probs(d)
    Xb = X.copy()
    for j, k in enumerate(CLASS_COLS):
        Xb[f"mkt_log{k}"] = _safe_log(mk[:, j])
    played = d["ftr"].notna()
    cutoff = d["date"].max() - pd.Timedelta(days=365 * train_years)
    tr = played & eligible(d) & (d["date"] >= cutoff)
    y = _y_hda(d["ftr"].where(played, "H"))
    yord = d["ftr"].map(Y_ORDER).to_numpy()
    out = {
        "elo": OrderedLogit().fit(X.loc[tr, ["elo"]].to_numpy(), yord[tr]),
        "pi": OrderedLogit().fit(X.loc[tr, ["pi"]].to_numpy(), yord[tr]),
        "ens": make_lr().fit(X[tr].to_numpy(), y[tr]),
        "feature_names": list(X.columns),
    }
    if use_gbm:
        out["gbm"] = make_gbm().fit(X[tr].to_numpy(), y[tr])
    trb = tr & np.isfinite(mk).all(axis=1)
    out["blend"] = make_lr().fit(Xb[trb].to_numpy(), y[trb])
    out.update(fit_final_ou(d, train_years))
    return out


def predict_with(models: dict, d: pd.DataFrame) -> pd.DataFrame:
    """Apply models from :func:`fit_final` to feature rows (fixtures or what-ifs)."""
    d = d.copy()
    X = build_features(d)
    mk = market_probs(d)
    d["mkt_ph"], d["mkt_pd"], d["mkt_pa"] = mk.T
    d[["elo_ph", "elo_pd", "elo_pa"]] = models["elo"].predict_proba(X[["elo"]].to_numpy())
    d[["pi_ph", "pi_pd", "pi_pa"]] = models["pi"].predict_proba(X[["pi"]].to_numpy())
    for name in ("ens", "gbm"):
        if name in models:
            est = models[name]
            proba = est.predict_proba(X.to_numpy())
            classes = list(est.classes_) if hasattr(est, "classes_") else list(est[-1].classes_)
            arr = np.zeros((len(d), 3))
            for j, c in enumerate(classes):
                arr[:, int(c)] = proba[:, j]
            d[[f"{name}_ph", f"{name}_pd", f"{name}_pa"]] = arr
    has = np.isfinite(mk).all(axis=1)
    for k in CLASS_COLS:
        d[f"blend_{k}"] = np.nan
    if has.any():
        Xb = X.copy()
        for j, k in enumerate(CLASS_COLS):
            Xb[f"mkt_log{k}"] = _safe_log(mk[:, j])
        est = models["blend"]
        proba = est.predict_proba(Xb[has].to_numpy())
        classes = list(est[-1].classes_)
        arr = np.zeros((has.sum(), 3))
        for j, c in enumerate(classes):
            arr[:, int(c)] = proba[:, j]
        d.loc[has, ["blend_ph", "blend_pd", "blend_pa"]] = arr
    return d


# ---------------------------------------------------------- over/under 2.5
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ou_market(d: pd.DataFrame, book: str = "b365") -> np.ndarray:
    out = np.full(len(d), np.nan)
    for b in (book, "ps", "avg", "b365", "max"):
        cols = [f"{b}_o25", f"{b}_u25"]
        if not set(cols) <= set(d.columns):
            continue
        p = market.devig(d[cols].to_numpy(), "power")[:, 0]
        out = np.where(np.isnan(out), p, out)
    return out


def build_ou_features(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["pxg"] = _logit(d["pxg_po25"])
    X["pois"] = _logit(d["pois_po25"])
    X["pxg_total"] = _safe_log(d["pxg_lh"]) + _safe_log(d["pxg_la"])
    for m in ("sot", "shots"):
        has = d[f"{m}_lh"].notna()
        X[f"{m}_total"] = (_safe_log(d[f"{m}_lh"]) + _safe_log(d[f"{m}_la"])).where(has, 0.0)
    X["has_shots"] = d["sot_lh"].notna().astype(float)
    X["lg_goals"] = d["lg_goals"]
    X["tier"] = d["tier"].astype(float)
    return X


def walk_forward_ou(d: pd.DataFrame, first_year: int = 2004, train_years: int = 12) -> pd.DataFrame:
    """Add ``oumkt_po25`` (market), ``ouens_po25`` and ``oublend_po25`` columns."""
    d = d.copy()
    X = build_ou_features(d)
    pm = ou_market(d)
    d["oumkt_po25"] = pm
    Xb = X.copy()
    Xb["mkt"] = _logit(pm)
    has_mkt = np.isfinite(pm)
    played = d["fthg"].notna()
    y = ((d["fthg"] + d["ftag"]) > 2.5).astype(int).to_numpy()
    elig = eligible(d)
    year = np.where(d["date"].dt.month >= 7, d["date"].dt.year, d["date"].dt.year - 1)
    d["ouens_po25"] = np.nan
    d["oublend_po25"] = np.nan
    for Y in range(first_year, int(year.max()) + 1):
        tr = (year < Y) & (year >= Y - train_years) & played & elig
        te = (year == Y) & (elig | ~played)
        if te.sum() == 0 or tr.sum() < 5000:
            continue
        lr = make_lr().fit(X[tr].to_numpy(), y[tr])
        d.loc[te, "ouens_po25"] = lr.predict_proba(X[te].to_numpy())[:, 1]
        trb, teb = tr & has_mkt, te & has_mkt
        if trb.sum() > 5000 and teb.sum():
            lr = make_lr().fit(Xb[trb].to_numpy(), y[trb])
            d.loc[teb, "oublend_po25"] = lr.predict_proba(Xb[teb].to_numpy())[:, 1]
    return d


def fit_final_ou(d: pd.DataFrame, train_years: int = 12) -> dict:
    """Over/under 2.5 models (ratings only, and ratings + market) on recent history."""
    X = build_ou_features(d)
    pm = ou_market(d)
    Xb = X.copy()
    Xb["mkt"] = _logit(pm)
    played = d["fthg"].notna() & d["ftag"].notna()
    cutoff = d["date"].max() - pd.Timedelta(days=365 * train_years)
    tr = played & eligible(d) & (d["date"] >= cutoff) & X.notna().all(axis=1)
    y = ((d["fthg"] + d["ftag"]) > 2.5).astype(int).to_numpy()
    out = {"ouens": make_lr().fit(X[tr].to_numpy(), y[tr])}
    trb = tr & np.isfinite(pm)
    out["oublend"] = make_lr().fit(Xb[trb].to_numpy(), y[trb])
    return out


def predict_ou_with(models: dict, d: pd.DataFrame) -> pd.DataFrame:
    """Add ``oumkt_po25``, ``ouens_po25`` and ``oublend_po25`` for the rows of ``d``."""
    d = d.copy()
    X = build_ou_features(d)
    pm = ou_market(d)
    d["oumkt_po25"] = pm
    d["ouens_po25"] = np.nan
    d["oublend_po25"] = np.nan
    ok = X.notna().all(axis=1).to_numpy()
    if ok.any():
        d.loc[ok, "ouens_po25"] = models["ouens"].predict_proba(X[ok].to_numpy())[:, 1]
        okb = ok & np.isfinite(pm)
        if okb.any():
            Xb = X.copy()
            Xb["mkt"] = _logit(pm)
            d.loc[okb, "oublend_po25"] = models["oublend"].predict_proba(Xb[okb].to_numpy())[:, 1]
    return d
