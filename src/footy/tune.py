"""Grid-search rating hyper-parameters on an early development window only.

Tuning uses matches before ``dev_end`` so that later seasons remain a clean
out-of-sample test for the backtest.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from . import evaluate as ev
from .models.probability import OrderedLogit, Y_ORDER, eligible
from .ratings.engine import RatingEngine
from .ratings.models import Elo, OnlinePoisson, PiRating

log = logging.getLogger(__name__)

GRIDS = {
    "elo": (lambda k, hfa_lr: Elo(k=k, hfa_lr=hfa_lr), {"k": [15, 20, 25, 30], "hfa_lr": [0.5, 1.5, 3.0]}),
    "pi": (lambda lam, gamma: PiRating(lam=lam, gamma=gamma),
           {"lam": [0.035, 0.05, 0.07, 0.09], "gamma": [0.5, 0.7]}),
    "pois": (lambda eta: OnlinePoisson("goals", eta=eta), {"eta": [0.03, 0.045, 0.06]}),
    "pxg": (lambda eta, w: OnlinePoisson("pxg", eta=eta, pxg_weight=w),
            {"eta": [0.03, 0.045, 0.06], "w": [0.3, 0.5, 0.7]}),
}


def _score(model_name: str, d: pd.DataFrame, f: pd.DataFrame, mask: pd.Series) -> float:
    y = ev.onehot(d.loc[mask, "ftr"])
    if model_name in ("elo", "pi"):
        x = f.loc[mask, "elo_diff" if model_name == "elo" else "pi_gd"].to_numpy()[:, None]
        ol = OrderedLogit().fit(x, d.loc[mask, "ftr"].map(Y_ORDER).to_numpy())
        p = ol.predict_proba(x)
    else:
        p = f.loc[mask, [f"{model_name}_ph", f"{model_name}_pd", f"{model_name}_pa"]].to_numpy()
    return float(ev.rps(p, y).mean())


def tune(matches: pd.DataFrame, which: list[str] | None = None, dev_start: str = "2003-07-01",
         dev_end: str = "2013-07-01", rho_grid=(-0.03, -0.06, -0.09, -0.12)) -> pd.DataFrame:
    d = matches[matches["date"] < dev_end].reset_index(drop=True)
    rows = []
    for name in which or list(GRIDS):
        factory, grid = GRIDS[name]
        keys = list(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, vals))
            eng = RatingEngine(models=[factory(**params)])
            f = eng.run(d)
            f["h_n"], f["a_n"] = f["h_n"], f["a_n"]
            mask = (d["date"] >= dev_start) & d["ftr"].notna() & eligible(f)
            rhos = rho_grid if name in ("pois", "pxg") else (None,)
            for rho in rhos:
                if rho is not None:
                    from .ratings.poisson_math import dc_probs
                    p = dc_probs(f[f"{name}_lh"], f[f"{name}_la"], rho)
                    for k, v in p.items():
                        f[f"{name}_{k}"] = v
                score = _score(name, d, f, mask)
                row = {"model": name, **params, "rho": rho, "rps": score, "n": int(mask.sum())}
                log.info("%s", row)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "rps"])
