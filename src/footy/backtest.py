"""Betting-strategy backtester.

A strategy picks, for each match, at most one bet where the model's expected
value at the available price exceeds a threshold, then stakes it (flat or
fractional Kelly). All probabilities fed in must be out-of-sample (walk-forward).

Honesty rules baked in
----------------------
* Parameters are chosen on a *development* window and reported on a later
  *holdout* window (see :func:`select_and_validate`).
* Results are reported with a bootstrap confidence interval and a one-sided
  p-value; with dozens of leagues x thresholds some "winners" are luck.
* ``max`` odds (best price across ~17 books) are optimistic: you will not get
  them every time and soft books restrict winning accounts. Bet365 odds are
  realistic but higher-margin. Pinnacle / exchange prices are the reference
  for "sharp" lines when you have them (football-data.co.uk direct download).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .models import market
from .ratings.poisson_math import ah_ev, ah_settle, goal_diff_dist, score_matrix

OUT = ("h", "d", "a")


@dataclass
class Strategy:
    name: str = "value"
    market: str = "1x2"          # 1x2 | ou25 | ah
    model: str = "blend"         # probability source prefix (e.g. blend, ens, gbm, mkt, pxg)
    price: str = "max"           # odds to bet at: b365 | max | ps | avg (+ closing: psc ...)
    min_edge: float = 0.03       # bet when p * odds - 1 >= min_edge
    max_edge: float = 1.0        # ignore absurd edges (usually data errors / stale prices)
    min_odds: float = 1.3
    max_odds: float = 8.0
    outcomes: tuple = ("h", "d", "a")
    staking: str = "flat"        # flat | kelly
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.05      # max fraction of bankroll per bet
    leagues: tuple | None = None
    commission: float = 0.0      # e.g. 0.02 for an exchange
    ah_rho: float = -0.06

    def label(self) -> str:
        lg = "all" if not self.leagues else ",".join(self.leagues)
        return (f"{self.market}:{self.model}@{self.price} edge>={self.min_edge:.0%} "
                f"odds {self.min_odds}-{self.max_odds} [{lg}]")


# ------------------------------------------------------------------ selection
def _select_1x2(d: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    p = d[[f"{s.model}_p{o}" for o in OUT]].to_numpy()
    odds = d[[f"{s.price}_{o}" for o in OUT]].to_numpy()
    ev = p * (1 + (odds - 1) * (1 - s.commission)) - 1
    allowed = np.array([o in s.outcomes for o in OUT])
    ok = (np.isfinite(ev) & (ev >= s.min_edge) & (ev <= s.max_edge)
          & (odds >= s.min_odds) & (odds <= s.max_odds) & allowed[None, :])
    ev_m = np.where(ok, ev, -np.inf)
    j = ev_m.argmax(axis=1)
    has = np.isfinite(ev_m[np.arange(len(d)), j])
    rows = np.flatnonzero(has)
    j = j[has]
    sel = d.iloc[rows]
    res = sel["ftr"].str.lower().to_numpy()
    picks = np.array(OUT)[j]
    out = pd.DataFrame({
        "match_id": sel["match_id"].to_numpy(), "date": sel["date"].to_numpy(),
        "league": sel["league"].to_numpy(), "season": sel["season"].to_numpy(),
        "home": sel["home"].to_numpy(), "away": sel["away"].to_numpy(),
        "pick": picks, "p": p[rows, j], "odds": odds[rows, j], "ev": ev[rows, j],
        "win": (res == picks).astype(float),
    })
    out["ret"] = np.where(out["win"] == 1, (out["odds"] - 1) * (1 - s.commission), -1.0)
    # Closing line value vs Pinnacle (or market-average) closing, margin removed.
    for book in ("psc", "avgc", "b365c"):
        cols = [f"{book}_{o}" for o in OUT]
        if set(cols) <= set(d.columns) and d[cols].notna().any().any():
            fair = market.devig(sel[cols].to_numpy(), "power")
            pc = fair[np.arange(len(sel)), j]
            out["clv"] = out["odds"] * pc - 1
            out["clv_book"] = book
            break
    return out


def _select_ou(d: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    po = d[f"{s.model}_po25"].to_numpy()
    p = np.stack([po, 1 - po], axis=1)
    odds = d[[f"{s.price}_o25", f"{s.price}_u25"]].to_numpy()
    ev = p * (1 + (odds - 1) * (1 - s.commission)) - 1
    ok = (np.isfinite(ev) & (ev >= s.min_edge) & (ev <= s.max_edge)
          & (odds >= s.min_odds) & (odds <= s.max_odds))
    ev_m = np.where(ok, ev, -np.inf)
    j = ev_m.argmax(axis=1)
    has = np.isfinite(ev_m[np.arange(len(d)), j])
    rows = np.flatnonzero(has)
    j = j[has]
    sel = d.iloc[rows]
    over = ((sel["fthg"] + sel["ftag"]) > 2.5).to_numpy()
    win = np.where(j == 0, over, ~over).astype(float)
    out = pd.DataFrame({
        "match_id": sel["match_id"].to_numpy(), "date": sel["date"].to_numpy(),
        "league": sel["league"].to_numpy(), "season": sel["season"].to_numpy(),
        "home": sel["home"].to_numpy(), "away": sel["away"].to_numpy(),
        "pick": np.array(["over", "under"])[j], "p": p[rows, j], "odds": odds[rows, j],
        "ev": ev[rows, j], "win": win,
    })
    out["ret"] = np.where(win == 1, (out["odds"] - 1) * (1 - s.commission), -1.0)
    return out


def _select_ah(d: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    cols = [f"{s.price}_ahh", f"{s.price}_aha"]
    need = d["ah_line"].notna() & d[cols].notna().all(axis=1) & d[f"{s.model}_lh"].notna()
    sub = d[need]
    if sub.empty:
        return pd.DataFrame()
    m = score_matrix(sub[f"{s.model}_lh"].to_numpy(), sub[f"{s.model}_la"].to_numpy(), s.ah_rho)
    diffs, pgd = goal_diff_dist(m)
    oh, oa = sub[cols[0]].to_numpy(), sub[cols[1]].to_numpy()
    ev_h, ev_a = ah_ev(diffs, pgd, sub["ah_line"].to_numpy(), oh, oa)
    ev = np.stack([ev_h, ev_a], axis=1)
    odds = np.stack([oh, oa], axis=1)
    ok = ((ev >= s.min_edge) & (ev <= s.max_edge) & (odds >= s.min_odds) & (odds <= s.max_odds))
    ev_m = np.where(ok, ev, -np.inf)
    j = ev_m.argmax(axis=1)
    has = np.isfinite(ev_m[np.arange(len(sub)), j])
    rows = np.flatnonzero(has)
    j = j[has]
    sel = sub.iloc[rows]
    gd = (sel["fthg"] - sel["ftag"]).to_numpy()
    settle = ah_settle(gd, sel["ah_line"].to_numpy())
    settle = np.where(j == 0, settle, -settle)
    o = odds[rows, j]
    ret = np.where(settle > 0, settle * (o - 1) * (1 - s.commission), settle)
    return pd.DataFrame({
        "match_id": sel["match_id"].to_numpy(), "date": sel["date"].to_numpy(),
        "league": sel["league"].to_numpy(), "season": sel["season"].to_numpy(),
        "home": sel["home"].to_numpy(), "away": sel["away"].to_numpy(),
        "pick": np.where(j == 0, "home ", "away ") + sel["ah_line"].map(lambda x: f"{x:+.2f}").to_numpy(),
        "p": np.nan, "odds": o, "ev": ev[rows, j], "win": (settle > 0).astype(float), "ret": ret,
    })


SELECTORS = {"1x2": _select_1x2, "ou25": _select_ou, "ah": _select_ah}


# ------------------------------------------------------------------ staking
def stake(bets: pd.DataFrame, s: Strategy, bank0: float = 100.0) -> pd.DataFrame:
    bets = bets.sort_values(["date", "league", "match_id"]).reset_index(drop=True)
    if s.staking == "flat" or bets.empty:
        bets["stake"] = 1.0
        bets["profit"] = bets["ret"]
        bets["bank"] = bank0 + bets["profit"].cumsum()
        return bets
    # Fractional Kelly, stakes sized from the bankroll at the start of each day.
    b = bets["odds"].to_numpy() - 1
    if "p" in bets and bets["p"].notna().all():
        p = bets["p"].to_numpy()
        f = np.clip((p * b - (1 - p)) / b, 0, None)
    else:  # AH: approximate Kelly with edge / (odds - 1)
        f = np.clip(bets["ev"].to_numpy() / b, 0, None)
    f = np.minimum(f * s.kelly_fraction, s.kelly_cap)
    bank = bank0
    stakes, profits, banks = np.zeros(len(bets)), np.zeros(len(bets)), np.zeros(len(bets))
    for _, idx in bets.groupby("date").groups.items():
        idx = np.asarray(idx)
        st = f[idx] * bank
        pr = st * bets["ret"].to_numpy()[idx]
        stakes[idx], profits[idx] = st, pr
        bank += pr.sum()
        banks[idx] = bank
        if bank <= 0:
            bank = 0
    bets["stake"], bets["profit"], bets["bank"] = stakes, profits, banks
    return bets


def run(d: pd.DataFrame, s: Strategy, bank0: float = 100.0) -> pd.DataFrame:
    if s.leagues:
        d = d[d["league"].isin(s.leagues)]
    d = d[d["ftr"].notna()]
    bets = SELECTORS[s.market](d, s)
    if bets.empty:
        return bets
    return stake(bets, s, bank0)


# ------------------------------------------------------------------ metrics
def summarise(bets: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    if bets is None or bets.empty:
        return {"bets": 0, "roi": np.nan, "profit": 0.0}
    r = bets["ret"].to_numpy()
    n = len(r)
    roi = r.mean()
    sd = r.std(ddof=1) if n > 1 else np.nan
    t = roi / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    pval = float(1 - stats.t.cdf(t, n - 1)) if np.isfinite(t) else np.nan
    rng = np.random.default_rng(seed)
    if n > 1:
        boots = rng.choice(r, size=(n_boot, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = np.nan
    cum = np.cumsum(bets["profit"].to_numpy())
    peak = np.maximum.accumulate(np.r_[0, cum])[1:]
    max_dd = float((peak - cum).max()) if n else 0.0
    losing = (bets["win"].to_numpy() == 0) & (r < 0)
    streak = best = 0
    for x in losing:
        streak = streak + 1 if x else 0
        best = max(best, streak)
    out = {
        "bets": n, "hit_rate": float(bets["win"].mean()), "avg_odds": float(bets["odds"].mean()),
        "avg_ev": float(bets["ev"].mean()), "roi": float(roi), "roi_lo": float(lo),
        "roi_hi": float(hi), "p_value": pval, "profit_units": float(r.sum()),
        "max_drawdown_units": max_dd if bets["stake"].eq(1).all() else np.nan,
        "longest_losing_run": int(best),
        "final_bank": float(bets["bank"].iloc[-1]),
        "staked": float(bets["stake"].sum()), "profit": float(bets["profit"].sum()),
    }
    if "clv" in bets and bets["clv"].notna().any():
        out["avg_clv"] = float(bets["clv"].mean())
        out["beat_close_rate"] = float((bets["clv"] > 0).mean())
    return out


def by(bets: pd.DataFrame, col: str) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    g = bets.groupby(col)
    return pd.DataFrame({
        "bets": g.size(), "roi": g["ret"].mean(), "profit_units": g["ret"].sum(),
        "hit_rate": g["win"].mean(), "avg_odds": g["odds"].mean(),
    }).reset_index()


# ------------------------------------------------------------------ selection
@dataclass
class Window:
    dev: tuple[str, str] = ("2005-07-01", "2016-07-01")
    holdout: tuple[str, str] = ("2016-07-01", "2100-01-01")


def split(d: pd.DataFrame, w: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = d[(d["date"] >= w.dev[0]) & (d["date"] < w.dev[1])]
    hold = d[(d["date"] >= w.holdout[0]) & (d["date"] < w.holdout[1])]
    return dev, hold


def grid(base: Strategy, **axes) -> list[Strategy]:
    import itertools
    keys = list(axes)
    out = []
    for vals in itertools.product(*[axes[k] for k in keys]):
        kw = asdict(base)
        kw.update(dict(zip(keys, vals)))
        out.append(Strategy(**kw))
    return out


def select_and_validate(d: pd.DataFrame, strategies: list[Strategy], w: Window = Window(),
                        min_bets: int = 300) -> pd.DataFrame:
    """Evaluate every strategy on dev and holdout. Pick on dev; judge on holdout."""
    dev, hold = split(d, w)
    rows = []
    for s in strategies:
        sd = summarise(run(dev, s), n_boot=200)
        sh = summarise(run(hold, s), n_boot=200)
        rows.append({"strategy": s.label(), **{f"dev_{k}": v for k, v in sd.items()},
                     **{f"hold_{k}": v for k, v in sh.items()}, "_s": s})
    df = pd.DataFrame(rows)
    df["eligible"] = df["dev_bets"] >= min_bets
    return df.sort_values(["eligible", "dev_roi"], ascending=[False, False]).reset_index(drop=True)


@dataclass
class EdgeReport:
    """Per-league view of where a model gets closest to (or beats) the market."""

    tables: dict = field(default_factory=dict)
