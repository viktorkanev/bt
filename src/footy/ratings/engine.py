"""Run all rating models through history and emit leak-free pre-match features.

Processing is day by day: for every match on day D the features are computed
from the state *before* any match of day D is applied, so nothing from the same
day leaks into a prediction.

Promotion / relegation
----------------------
Ratings are only comparable within a division's pool, so when a team changes
division its rating is re-based:

    new = baseline(new div) + prior(new div, direction)
          + beta * (old rating relative to old div - mean of such movers)

``prior`` is learned online from how earlier movers in the same direction
settled (sampled after ``settle_n`` matches), starting from model defaults.
Teams that disappear for > ``absent_days`` re-enter as newcomers; a shorter
break (> ``offseason_days``) triggers a partial shrink toward the division mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import leagues
from .models import Elo, Model, OnlinePoisson, PiRating
from .poisson_math import dc_probs


@dataclass
class TeamState:
    league: str
    last_date: pd.Timestamp
    n_total: int = 0
    n_div: int = 0
    entry_dir: str = "founder"
    sampled: bool = True
    shrunk: bool = False


@dataclass
class _Running:
    """Running mean with pseudo-count prior."""

    mean: np.ndarray
    n: float = 5.0

    def add(self, x: np.ndarray) -> None:
        self.n += 1
        self.mean = self.mean + (x - self.mean) / self.n


@dataclass
class LeagueCtx:
    home_rate: float = 0.45
    draw_rate: float = 0.27
    goals: float = 2.6
    first_date: pd.Timestamp | None = None
    n: int = 0


def default_models() -> list[Model]:
    """Models with hyper-parameters tuned on 2003-2013 only (see ``footy tune``)."""
    return [
        Elo(k=15.0, hfa_init=60.0, hfa_lr=1.0),
        PiRating(lam=0.05, gamma=0.7),
        OnlinePoisson("goals", eta=0.025),
        OnlinePoisson("pxg", eta=0.03, pxg_weight=0.6),
        OnlinePoisson("sot", eta=0.03),
        OnlinePoisson("shots", eta=0.03),
    ]


@dataclass
class RatingEngine:
    models: list[Model] = field(default_factory=default_models)
    offseason_days: int = 60
    absent_days: int = 300
    settle_n: int = 20
    beta: float = 0.5
    base_alpha: float = 0.01
    ctx_alpha: float = 0.004
    rho: float = -0.07

    def __post_init__(self):
        self.teams: dict[str, TeamState] = {}
        self.base: dict[tuple[str, str], np.ndarray] = {}
        self.prior: dict[tuple[str, str, str], _Running] = {}
        self.exit: dict[tuple[str, str, str], _Running] = {}
        self.ctx: dict[str, LeagueCtx] = {}

    # ----------------------------------------------------------- helpers
    def _baseline(self, m: Model, league: str) -> np.ndarray:
        return self.base.get((m.name, league), m.neutral)

    def _prior(self, m: Model, league: str, direction: str) -> _Running:
        key = (m.name, league, direction)
        if key not in self.prior:
            self.prior[key] = _Running(m.default_prior[direction].astype(float).copy())
        return self.prior[key]

    def _exit(self, m: Model, league: str, direction: str) -> _Running:
        key = (m.name, league, direction)
        if key not in self.exit:
            self.exit[key] = _Running(np.zeros_like(m.neutral, dtype=float), n=1.0)
        return self.exit[key]

    def _enter(self, team: str, league: str, date, direction: str, old_league: str | None):
        for m in self.models:
            rel_adj = 0.0
            if old_league is not None and m.has(team):
                exit_rel = m.vec(team) - self._baseline(m, old_league)
                ex = self._exit(m, old_league, direction)
                rel_adj = self.beta * (exit_rel - ex.mean)
                ex.add(exit_rel)
            init = self._baseline(m, league) + self._prior(m, league, direction).mean + rel_adj
            m.set_vec(team, init)
        st = self.teams.get(team)
        n_total = st.n_total if st else 0
        self.teams[team] = TeamState(league, date, n_total, 0, direction,
                                     sampled=(direction == "founder"))

    def _prepare(self, team: str, league: str, date) -> None:
        lg_ctx = self.ctx.setdefault(league, LeagueCtx(first_date=date))
        st = self.teams.get(team)
        if st is None:
            founder = (date - lg_ctx.first_date).days <= 60
            self._enter(team, league, date, "founder" if founder else "new", None)
            return
        gap = (date - st.last_date).days
        if st.league != league:
            t_old, t_new = leagues.get(st.league).tier, leagues.get(league).tier
            direction = "up" if t_new < t_old else "down" if t_new > t_old else "new"
            self._enter(team, league, date, direction, st.league if direction != "new" else None)
        elif gap > self.absent_days:
            self._enter(team, league, date, "up", None)
        elif gap > self.offseason_days and not st.shrunk:
            st.shrunk = True
            for m in self.models:
                b = self._baseline(m, league)
                m.set_vec(team, b + (m.vec(team) - b) * (1 - m.shrink))

    def _after(self, team: str, league: str, date) -> None:
        st = self.teams[team]
        st.last_date = date
        st.shrunk = False
        st.n_total += 1
        st.n_div += 1
        for m in self.models:
            key = (m.name, league)
            b = self.base.get(key, m.neutral)
            self.base[key] = b + self.base_alpha * (m.vec(team) - b)
        if not st.sampled and st.n_div >= self.settle_n:
            for m in self.models:
                self._prior(m, league, st.entry_dir).add(m.vec(team) - self._baseline(m, league))
            st.sampled = True

    # --------------------------------------------------------------- main
    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return a feature frame indexed like ``matches`` (must be date-sorted)."""
        cols = ["league", "date", "home", "away", "country", "fthg", "ftag", "hst", "ast",
                "hs", "as_"]
        df = matches[cols].copy()
        df["hkey"] = df["country"] + "|" + df["home"]
        df["akey"] = df["country"] + "|" + df["away"]
        df["played"] = df["fthg"].notna() & df["ftag"].notna()
        recs = df.to_dict("records")
        dates = df["date"].to_numpy()
        feats: list[dict] = [None] * len(recs)  # type: ignore[list-item]
        # group indices by date
        order = np.argsort(dates, kind="stable")
        boundaries = np.flatnonzero(np.diff(dates[order].astype("datetime64[D]").astype(np.int64))) + 1
        groups = np.split(order, boundaries)
        for g in groups:
            day = [recs[i] for i in g]
            date = pd.Timestamp(day[0]["date"])
            for r in day:
                self._prepare(r["hkey"], r["league"], date)
                self._prepare(r["akey"], r["league"], date)
            for i, r in zip(g, day):
                f: dict = {}
                h, a, lg = r["hkey"], r["akey"], r["league"]
                for m in self.models:
                    f.update(m.pre(h, a, lg))
                sh, sa = self.teams[h], self.teams[a]
                c = self.ctx[lg]
                f.update({
                    "h_n": sh.n_total, "a_n": sa.n_total,
                    "h_ndiv": sh.n_div, "a_ndiv": sa.n_div,
                    "h_entry": sh.entry_dir if sh.n_div < self.settle_n else "settled",
                    "a_entry": sa.entry_dir if sa.n_div < self.settle_n else "settled",
                    "h_rest": min((date - sh.last_date).days, 60) if sh.n_total else 60,
                    "a_rest": min((date - sa.last_date).days, 60) if sa.n_total else 60,
                    "lg_home_rate": c.home_rate, "lg_draw_rate": c.draw_rate,
                    "lg_goals": c.goals, "lg_n": c.n,
                })
                feats[i] = f
            for r in day:
                if not r["played"]:
                    continue
                h, a, lg = r["hkey"], r["akey"], r["league"]
                for m in self.models:
                    if isinstance(m, OnlinePoisson) and not m.available(r):
                        continue
                    m.post(h, a, lg, r)
                self._after(h, lg, date)
                self._after(a, lg, date)
                c = self.ctx[lg]
                res_h = float(r["fthg"] > r["ftag"])
                res_d = float(r["fthg"] == r["ftag"])
                c.home_rate += self.ctx_alpha * (res_h - c.home_rate)
                c.draw_rate += self.ctx_alpha * (res_d - c.draw_rate)
                c.goals += self.ctx_alpha * (r["fthg"] + r["ftag"] - c.goals)
                c.n += 1
        out = pd.DataFrame(feats, index=matches.index)
        # Shot-based features only make sense once the league has shot data.
        for name in ("sot", "shots"):
            if f"{name}_lh" in out.columns:
                has = self._shot_coverage(matches, name)
                out.loc[~has, [f"{name}_lh", f"{name}_la"]] = np.nan
        for name in ("pois", "pxg"):
            if f"{name}_lh" in out.columns:
                p = dc_probs(out[f"{name}_lh"].to_numpy(), out[f"{name}_la"].to_numpy(), self.rho)
                for k, v in p.items():
                    out[f"{name}_{k}"] = v
        return out

    @staticmethod
    def _shot_coverage(matches: pd.DataFrame, name: str) -> pd.Series:
        """True where the team-league has had shot data recently (previous 30 matches)."""
        col = "hst" if name == "sot" else "hs"
        avail = matches[col].notna().astype(float)
        cov = avail.groupby(matches["league"]).transform(
            lambda s: s.shift(1).rolling(300, min_periods=50).mean())
        return cov.fillna(0) > 0.8

    # ------------------------------------------------------------ tables
    def ratings_table(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Current ratings for every team, with the division they last played in."""
        rows = []
        for key, st in self.teams.items():
            country, name = key.split("|", 1)
            row = {"team": name, "country": country, "league": st.league,
                   "last_match": st.last_date, "matches": st.n_total}
            for m in self.models:
                v = m.vec(key)
                if m.name == "elo":
                    row["elo"] = v[0]
                elif m.name == "pi":
                    row["pi_home"], row["pi_away"] = v
                else:
                    b = self._baseline(m, st.league)
                    row[f"{m.name}_att"], row[f"{m.name}_def"] = v - b
            rows.append(row)
        df = pd.DataFrame(rows)
        if as_of is not None:
            df = df[df["last_match"] >= as_of - pd.Timedelta(days=400)]
        return df
