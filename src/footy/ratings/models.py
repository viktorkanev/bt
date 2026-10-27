"""Sequential (online) team-strength models.

Each model keeps one parameter vector per team and exposes

* ``pre(h, a, league)``  -> dict of pre-match features (no look-ahead)
* ``post(h, a, league, m)`` -> update after the result is known
* ``vec/set_vec``        -> used by the engine for promotion/relegation handling

All models are cheap enough to run over ~250k matches in pure Python.
"""

from __future__ import annotations

import math

import numpy as np


class Model:
    name = "model"
    neutral: np.ndarray = np.zeros(1)
    # Default relative strength for a team entering a division, by direction.
    default_prior: dict[str, np.ndarray] = {}
    shrink = 0.2  # fraction pulled toward the division baseline in the off-season

    def __init__(self):
        self.teams: dict[str, np.ndarray] = {}

    def vec(self, team: str) -> np.ndarray:
        return self.teams.get(team, self.neutral).copy()

    def set_vec(self, team: str, v: np.ndarray) -> None:
        self.teams[team] = np.asarray(v, dtype=float)

    def has(self, team: str) -> bool:
        return team in self.teams


# --------------------------------------------------------------------------- Elo
class Elo(Model):
    """Goal-difference weighted Elo with a per-league, self-adjusting home advantage.

    The home advantage parameter learns online, so it automatically tracked the
    collapse of home advantage during the 2020-21 empty-stadium period.
    """

    name = "elo"
    neutral = np.array([1500.0])
    default_prior = {"up": np.array([-70.0]), "down": np.array([70.0]),
                     "new": np.array([-90.0]), "founder": np.array([0.0])}
    shrink = 0.15

    def __init__(self, k: float = 20.0, hfa_init: float = 60.0, hfa_lr: float = 1.5):
        super().__init__()
        self.k, self.hfa_init, self.hfa_lr = k, hfa_init, hfa_lr
        self.hfa: dict[str, float] = {}

    def _expect(self, h, a, league):
        hfa = self.hfa.get(league, self.hfa_init)
        dr = self.teams[h][0] + hfa - self.teams[a][0]
        return dr, 1.0 / (1.0 + 10 ** (-dr / 400.0))

    def pre(self, h, a, league):
        dr, e = self._expect(h, a, league)
        return {"elo_diff": dr, "elo_exp": e, "elo_hfa": self.hfa.get(league, self.hfa_init)}

    def post(self, h, a, league, m):
        _, e = self._expect(h, a, league)
        gd = abs(m["fthg"] - m["ftag"])
        s = 1.0 if m["fthg"] > m["ftag"] else 0.0 if m["fthg"] < m["ftag"] else 0.5
        g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0
        delta = self.k * g * (s - e)
        self.teams[h] = self.teams[h] + delta
        self.teams[a] = self.teams[a] - delta
        self.hfa[league] = self.hfa.get(league, self.hfa_init) + self.hfa_lr * (s - e)


# --------------------------------------------------------------------- pi-ratings
class PiRating(Model):
    """Pi-ratings (Constantinou & Fenton, 2013).

    Each team has a home and an away rating on a goal-difference scale. The
    expected goal difference is derived from the home team's home rating and
    the away team's away rating; errors are dampened logarithmically so big
    wins do not dominate.
    """

    name = "pi"
    neutral = np.array([0.0, 0.0])
    default_prior = {"up": np.array([-0.25, -0.25]), "down": np.array([0.25, 0.25]),
                     "new": np.array([-0.3, -0.3]), "founder": np.array([0.0, 0.0])}
    shrink = 0.15

    def __init__(self, lam: float = 0.06, gamma: float = 0.7, c: float = 3.0):
        super().__init__()
        self.lam, self.gamma, self.c = lam, gamma, c

    def _g(self, r):
        return math.copysign(10 ** (abs(r) / self.c) - 1.0, r)

    def pre(self, h, a, league):
        rh, ra = self.teams[h][0], self.teams[a][1]
        return {"pi_gd": self._g(rh) - self._g(ra), "pi_h": rh, "pi_a": ra}

    def post(self, h, a, league, m):
        th, ta = self.teams[h].copy(), self.teams[a].copy()
        exp_gd = self._g(th[0]) - self._g(ta[1])
        obs_gd = m["fthg"] - m["ftag"]
        err = abs(obs_gd - exp_gd)
        psi = self.c * math.log10(1.0 + err)
        psi_h = psi if exp_gd < obs_gd else -psi
        psi_a = -psi_h
        d_h = psi_h * self.lam
        d_a = psi_a * self.lam
        th[0] += d_h
        th[1] += d_h * self.gamma
        ta[1] += d_a
        ta[0] += d_a * self.gamma
        self.teams[h], self.teams[a] = th, ta


# --------------------------------------------------------------- online Poisson
class OnlinePoisson(Model):
    """Dynamic attack/defence Poisson model (online Dixon-Coles).

    log(lambda_home) = mu_L + home_L + att_h - def_a
    log(lambda_away) = mu_L + att_a - def_h

    Parameters move by a stochastic-gradient step on the Poisson log-likelihood
    after every match, which behaves like a time-decayed Dixon-Coles fit but
    costs O(1) per match. ``target`` selects what is being modelled:

    * ``goals``  - full-time goals
    * ``pxg``    - "pseudo-xG": goals blended with shots-on-target x league
                   conversion rate (less noisy than goals alone)
    * ``sot``    - shots on target
    * ``shots``  - total shots
    """

    neutral = np.array([0.0, 0.0])
    default_prior = {"up": np.array([-0.10, -0.10]), "down": np.array([0.10, 0.10]),
                     "new": np.array([-0.12, -0.12]), "founder": np.array([0.0, 0.0])}
    shrink = 0.2

    def __init__(self, target: str = "goals", eta: float = 0.04, eta_league: float = 0.004,
                 pxg_weight: float = 0.5, cap: float = 6.0):
        super().__init__()
        self.target, self.eta, self.eta_league = target, eta, eta_league
        self.pxg_weight, self.cap = pxg_weight, cap
        self.name = {"goals": "pois", "pxg": "pxg", "sot": "sot", "shots": "shots"}[target]
        init_mu = {"goals": math.log(1.3), "pxg": math.log(1.3),
                   "sot": math.log(4.3), "shots": math.log(11.5)}[target]
        self.init_mu = init_mu
        self.mu: dict[str, float] = {}
        self.home: dict[str, float] = {}
        self.conv: dict[str, float] = {}  # league goals per shot on target (pxg)

    def lambdas(self, h, a, league):
        mu = self.mu.get(league, self.init_mu)
        hm = self.home.get(league, 0.2)
        th, ta = self.teams[h], self.teams[a]
        return math.exp(mu + hm + th[0] - ta[1]), math.exp(mu + ta[0] - th[1])

    def _observed(self, m):
        if self.target == "goals":
            return m["fthg"], m["ftag"]
        if self.target == "sot":
            return m["hst"], m["ast"]
        if self.target == "shots":
            return m["hs"], m["as_"]
        # pxg
        gh, ga, sh, sa = m["fthg"], m["ftag"], m["hst"], m["ast"]
        if sh != sh or sa != sa:  # NaN -> fall back to goals
            return gh, ga
        conv = self.conv.get(m["league"], 0.30)
        w = self.pxg_weight
        return w * gh + (1 - w) * conv * sh, w * ga + (1 - w) * conv * sa

    def available(self, m) -> bool:
        if self.target in ("goals", "pxg"):
            return True
        a, b = self._observed(m)
        return a == a and b == b

    def pre(self, h, a, league):
        # Outcome probabilities are derived later, vectorised (see poisson_math).
        lh, la = self.lambdas(h, a, league)
        return {f"{self.name}_lh": lh, f"{self.name}_la": la}

    def post(self, h, a, league, m):
        oh, oa = self._observed(m)
        if oh != oh or oa != oa:
            return
        if self.target == "pxg" and m["hst"] == m["hst"] and m["ast"] == m["ast"]:
            tot_sot = m["hst"] + m["ast"]
            if tot_sot > 0:
                c = self.conv.get(league, 0.30)
                self.conv[league] = c + 0.002 * ((m["fthg"] + m["ftag"]) / tot_sot - c) * min(tot_sot, 12) / 6
        cap = self.cap * (4 if self.target in ("sot", "shots") else 1)
        oh, oa = min(oh, cap), min(oa, cap)
        lh, la = self.lambdas(h, a, league)
        # Scale residuals so that eta means the same "memory" for every target.
        scale = 1.3 / math.exp(self.init_mu)
        rh, ra = (oh - lh) * scale, (oa - la) * scale
        th, ta = self.teams[h].copy(), self.teams[a].copy()
        th[0] += self.eta * rh
        ta[1] -= self.eta * rh
        ta[0] += self.eta * ra
        th[1] -= self.eta * ra
        self.teams[h], self.teams[a] = th, ta
        self.mu[league] = self.mu.get(league, self.init_mu) + self.eta_league * (rh + ra) / 2
        self.home[league] = self.home.get(league, 0.2) + self.eta_league * (rh - ra) / 2
