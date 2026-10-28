"""Upcoming fixtures: probabilities from the models plus the research layer.

``footy upcoming`` does, in about a minute:

1. refresh this season's results and download fixtures with current odds
   (football-data.co.uk publishes them around Tuesday and Friday);
2. run the rating engine over history + fixtures (ratings as of today);
3. apply the saved models (``footy build``) and team-news adjustments;
4. add the research layer:
   * **research-adjusted probabilities**: the bookmaker's no-margin
     probabilities corrected for the confirmed luck over-reaction
     (frozen coefficients in ``config/forward_rules.json``);
   * **F1** lower-division over/under 2.5 picks and **F2** luck-fade Asian
     handicap picks (rules frozen on 2026-09-03, still being forward-tested);
5. log every F1/F2 pick with the price available now to a paper-trading
   ledger, and settle earlier picks whose results have arrived (ROI and
   closing-line value against Pinnacle).

Match-result value bets from the plain model are shown for reference only;
the research found no 1X2 edge at prices you can actually get.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import leagues as L
from . import pipeline
from .data import sources, store
from .models import market
from .models import probability as prob
from .ratings.engine import RatingEngine
from .ratings.poisson_math import ah_ev, ah_settle, goal_diff_dist, outcome_probs, score_matrix
from .research import confirm, forward, signals

log = logging.getLogger(__name__)

OUT_DIR = store.ROOT / "reports" / "upcoming"
LEDGER = store.ROOT / "reports" / "research" / "forward" / "ledger.csv"
BOOK_NAMES = {"ps": "Pinnacle", "b365": "Bet365", "max": "Best price", "avg": "Market average",
              "bfe": "Betfair Exchange"}
# Live picks prefer the exchange when it has a price (no margin, no account limits).
F2_VENUES = ("bfe", "ps", "b365", "avg", "max")


# ----------------------------------------------------------------- inputs
def load_inputs(fetch: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = store.load_store()
    meta = store.load_meta()
    if fetch:
        if meta.get("source") == "football-data":
            try:
                matches = sources.refresh_current_season(matches, store.RAW_DIR)
                store.save_store(matches, "football-data")
            except Exception as e:  # noqa: BLE001
                log.warning("could not refresh results (%s); using stored results", e)
        else:
            log.warning("results come from the GitHub mirror; run `footy fetch --source "
                        "football-data` once to get live results and fixtures")
        fixtures = sources.fetch_fixtures(store.RAW_DIR)
        if not fixtures.empty:
            fixtures.to_parquet(store.FIXTURES, index=False)
    elif store.FIXTURES.exists():
        fixtures = pd.read_parquet(store.FIXTURES)
    else:
        fixtures = pd.DataFrame()
    return matches, fixtures


def _ensure_models() -> dict:
    models = pipeline.load_models()
    if "oublend" not in models:
        log.info("saved models predate the over/under models; fitting them now")
        models.update(prob.fit_final_ou(pipeline.load_predictions()))
        with open(pipeline.MODELS_PKL, "wb") as fh:
            import pickle
            pickle.dump(models, fh)
    return models


# ----------------------------------------------------------------- scoring
def score(matches: pd.DataFrame, fixtures: pd.DataFrame, models: dict | None = None,
          spec: dict | None = None, adjustments: list[dict] | None = None) -> pd.DataFrame:
    """Probabilities, prices, research layer and picks for each fixture."""
    if fixtures is None or fixtures.empty:
        return pd.DataFrame()
    last_played = matches.loc[matches["ftr"].notna(), "date"].max()
    fx = fixtures[~fixtures["match_id"].isin(set(matches["match_id"]))]
    fx = fx[fx["date"] >= last_played - pd.Timedelta(days=3)].copy()
    if fx.empty:
        return pd.DataFrame()
    fx[["fthg", "ftag"]] = np.nan
    fx["ftr"] = pd.Series(np.nan, index=fx.index, dtype=object)
    data = pd.concat([matches, fx], ignore_index=True)
    data = data.sort_values(["date", "league", "home"]).reset_index(drop=True)
    log.info("rating %d matches + %d fixtures", len(matches), len(fx))
    eng = RatingEngine()
    d = pd.concat([data, eng.run(data)], axis=1)
    mk = prob.market_probs(d)
    d["mkt_ph"], d["mkt_pd"], d["mkt_pa"] = mk.T
    d = signals.add_all(d)
    d["low"] = (d["tier"] >= 2).astype(float)
    for c in confirm.FEATS:
        d[f"low_{c}"] = d[c] * d["low"]
    out = d[d["match_id"].isin(set(fx["match_id"]))].copy()

    models = models or _ensure_models()
    spec = spec or json.loads(forward.FROZEN.read_text(encoding="utf-8"))
    adjustments = pipeline.load_adjustments() if adjustments is None else adjustments
    out = pipeline.apply_adjustments(out, adjustments, eng.rho)
    out = prob.predict_with(models, out)
    out = prob.predict_ou_with(models, out)
    out = pipeline.add_value_columns(out)
    out = research_layer(out, spec)
    out["low_data"] = (out["h_n"] < 10) | (out["a_n"] < 10)
    return out.sort_values(["date", "time", "league", "home"]).reset_index(drop=True)


def _first_book(df: pd.DataFrame, cols: tuple[str, str], books) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    book = pd.Series("", index=df.index)
    a = np.full(len(df), np.nan)
    b = np.full(len(df), np.nan)
    for bk in books:
        ca, cb = f"{bk}_{cols[0]}", f"{bk}_{cols[1]}"
        if ca not in df or cb not in df:
            continue
        has = df[[ca, cb]].notna().all(axis=1).to_numpy() & np.isnan(a)
        a[has], b[has] = df.loc[has, ca], df.loc[has, cb]
        book[has] = bk
    return book, a, b


def research_layer(fx: pd.DataFrame, spec: dict) -> pd.DataFrame:
    fx = fx.copy()
    f2 = spec["F2"]
    coef = np.array([f2["coef"][c] for c in forward.F2_COLS])
    feats = fx[forward.F2_COLS].to_numpy(dtype=float)
    has = np.isfinite(feats).all(axis=1) & fx[["x_lh", "x_la"]].notna().all(axis=1).to_numpy()
    delta = np.where(has, np.nan_to_num(feats) @ coef, np.nan)
    fx["luck_shift_goals"] = delta  # expected goal-difference correction for the home side
    for k in ("ph", "pd", "pa"):
        fx[f"res_{k}"] = np.nan
    fx["f2_pick"], fx["f2_side"] = "", ""
    for k in ("f2_book", "f2_odds", "f2_signal", "f2_ev", "f2_line"):
        fx[k] = np.nan if k != "f2_book" else ""
    if has.any():
        idx = np.flatnonzero(has)
        s = fx.iloc[idx]
        lh0, la0 = s["x_lh"].to_numpy(), s["x_la"].to_numpy()
        lh1 = np.clip(lh0 + delta[idx] / 2, 0.05, None)
        la1 = np.clip(la0 - delta[idx] / 2, 0.05, None)
        m0, m1 = score_matrix(lh0, la0, f2["rho"]), score_matrix(lh1, la1, f2["rho"])
        p0 = np.stack(outcome_probs(m0), 1)
        p1 = np.stack(outcome_probs(m1), 1)
        base = s[["mkt_ph", "mkt_pd", "mkt_pa"]].to_numpy()
        res = np.clip(base + (p1 - p0), 0.005, None)
        res /= res.sum(axis=1, keepdims=True)
        fx.iloc[idx, [fx.columns.get_loc(f"res_{k}") for k in ("ph", "pd", "pa")]] = res
        # F2: luck-fade Asian handicap at the first available book in the frozen order
        book, oh, oa = _first_book(s, ("ahh", "aha"), F2_VENUES)
        oh_net = np.where(book.to_numpy() == "bfe", forward.effective_odds("bfe", oh), oh)
        oa_net = np.where(book.to_numpy() == "bfe", forward.effective_odds("bfe", oa), oa)
        ok = s["ah_line"].notna().to_numpy() & np.isfinite(oh)
        if ok.any():
            diffs, pgd1 = goal_diff_dist(m1[ok])
            _, pgd0 = goal_diff_dist(m0[ok])
            line = s["ah_line"].to_numpy()[ok]
            eh, ea = ah_ev(diffs, pgd1, line, oh_net[ok], oa_net[ok])
            eh0, ea0 = ah_ev(diffs, pgd0, line, oh_net[ok], oa_net[ok])
            sh, sa = eh - eh0, ea - ea0
            side = np.where(sh >= sa, 0, 1)
            sig = np.maximum(sh, sa)
            rows = idx[ok]
            picks = sig >= f2["min_signal"]
            for r, sd, sg, e1, e2, ln, bk, o1, o2, pk in zip(rows, side, sig, eh, ea, line,
                                                              book.to_numpy()[ok], oh[ok], oa[ok], picks):
                fx.iat[r, fx.columns.get_loc("f2_signal")] = sg
                fx.iat[r, fx.columns.get_loc("f2_line")] = ln
                fx.iat[r, fx.columns.get_loc("f2_book")] = bk
                fx.iat[r, fx.columns.get_loc("f2_odds")] = o1 if sd == 0 else o2
                fx.iat[r, fx.columns.get_loc("f2_ev")] = e1 if sd == 0 else e2
                if pk:
                    fx.iat[r, fx.columns.get_loc("f2_side")] = "home" if sd == 0 else "away"
                    team_line = ln if sd == 0 else -ln
                    team_line = 0.0 if abs(team_line) < 1e-9 else team_line
                    who = fx.iat[r, fx.columns.get_loc("home" if sd == 0 else "away")]
                    fx.iat[r, fx.columns.get_loc("f2_pick")] = f"{who} {team_line:+.2f}"
    # F1: lower-division over/under 2.5 at the best qualifying price
    f1 = spec["F1"]
    fx["f1_pick"], fx["f1_book"] = "", ""
    fx["f1_odds"] = fx["f1_ev"] = fx["f1_prob"] = np.nan
    po = fx[f"{f1['model']}_po25"].to_numpy()
    elig = fx["tier"].isin(f1["tiers"]).to_numpy() & np.isfinite(po)
    for i in np.flatnonzero(elig):
        best = None
        for bk in forward.venues(f1["prices"]):
            for side, col, p in (("Over 2.5", f"{bk}_o25", po[i]), ("Under 2.5", f"{bk}_u25", 1 - po[i])):
                o = fx.iat[i, fx.columns.get_loc(col)] if col in fx else np.nan
                if not np.isfinite(o) or o > f1["max_odds"]:
                    continue
                ev = p * float(forward.effective_odds(bk, o)) - 1
                if f1["min_edge"] <= ev <= f1["max_edge"] and (best is None or ev > best[3]):
                    best = (side, bk, o, ev, p)
        if best:
            for k, v in zip(("f1_pick", "f1_book", "f1_odds", "f1_ev", "f1_prob"), best):
                fx.iat[i, fx.columns.get_loc(k)] = v
    return fx


# ------------------------------------------------------------- pick details
AH_OUTCOMES = (("win", 1.0), ("half_win", 0.5), ("push", 0.0), ("half_loss", -0.5), ("loss", -1.0))


def _ah_side_outcomes(lh, la, line, side, rho) -> dict[str, np.ndarray]:
    """Probability of each handicap result (full/half win, push, half/full loss) for a side."""
    lh = np.clip(np.asarray(lh, float), 0.05, None)
    la = np.clip(np.asarray(la, float), 0.05, None)
    diffs, pgd = goal_diff_dist(score_matrix(lh, la, rho))
    res = ah_settle(diffs[None, :], np.asarray(line, float)[:, None])
    res = np.where(np.asarray(side)[:, None] == 0, res, -res)
    return {k: (pgd * np.isclose(res, v)).sum(axis=1) for k, v in AH_OUTCOMES}


def _break_even(o: dict) -> np.ndarray:
    """Decimal odds at which the bet has zero expected value."""
    return 1 + (o["loss"] + 0.5 * o["half_loss"]) / np.maximum(o["win"] + 0.5 * o["half_win"], 1e-9)


def _ah_ev(o: dict, odds) -> np.ndarray:
    odds = np.asarray(odds, float)
    return (odds - 1) * (o["win"] + 0.5 * o["half_win"]) - (o["loss"] + 0.5 * o["half_loss"])


def _calibrate_supremacy(lh, la, line, target_home_be, rho, iters: int = 40) -> np.ndarray:
    """Goal-supremacy shift that makes the model reproduce the market's handicap price."""
    lo = np.full(len(line), -4.0)
    hi = np.full(len(line), 4.0)
    zeros = np.zeros(len(line), dtype=int)
    for _ in range(iters):
        mid = (lo + hi) / 2
        be = _break_even(_ah_side_outcomes(lh + mid / 2, la - mid / 2, line, zeros, rho))
        short = be > target_home_be  # home not strong enough yet
        lo = np.where(short, mid, lo)
        hi = np.where(short, hi, mid)
    return (lo + hi) / 2


def _commission_needed(book_odds) -> np.ndarray:
    """Exchange price needed to match a bookmaker price after commission."""
    c = forward.COMMISSION["bfe"]
    return 1 + (np.asarray(book_odds, float) - 1) / (1 - c)


def _luck_reason(r: pd.Series) -> str:
    delta = r["luck_shift_goals"]
    side = "h" if delta < 0 else "a"
    team = r["home"] if side == "h" else r["away"]
    fin, gk = r.get(f"{side}_fin_luck", np.nan), r.get(f"{side}_gk_luck", np.nan)
    bits = []
    if np.isfinite(fin) and fin > 0.05:
        bits.append(f"scored {fin:.2f} goals a game more than their shots on target justify")
    if np.isfinite(gk) and gk > 0.05:
        bits.append(f"conceded {gk:.2f} a game fewer than the shots they faced justify")
    why = " and ".join(bits) if bits else "have had better luck than their opponents"
    return (f"Recently {team} {why}. Bookmakers over-rate that kind of luck; "
            f"the correction is worth about {abs(delta):.2f} goals.")


def pick_details(fx: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """One row per research pick (and near-miss) with plain-language fields.

    Probabilities are ours (after the research correction) and the market's
    (margin removed). ``breakeven_odds`` is the lowest price at which the bet is
    profitable on average; ``min_odds`` adds the rule's required edge.
    """
    rows = []
    f1 = spec["F1"]
    rho = spec["F2"]["rho"]
    if fx.empty or "f2_pick" not in fx:
        fx = pd.DataFrame(columns=["f2_pick", "tier", f"{f1['model']}_po25"])
    # ---- F2 luck-fade handicap picks
    pk = fx[fx["f2_pick"].fillna("") != ""]
    if len(pk):
        side = (pk["f2_side"] == "away").astype(int).to_numpy()
        line = pk["f2_line"].to_numpy(float)
        bk = pk["f2_book"].to_numpy()
        oh = np.array([r[f"{b}_ahh"] for (_, r), b in zip(pk.iterrows(), bk)], float)
        oa = np.array([r[f"{b}_aha"] for (_, r), b in zip(pk.iterrows(), bk)], float)
        fair = market.devig(np.c_[oh, oa], "power")
        s0 = _calibrate_supremacy(pk["x_lh"].to_numpy(), pk["x_la"].to_numpy(), line, 1 / fair[:, 0], rho)
        s1 = s0 + pk["luck_shift_goals"].to_numpy()
        lh, la = pk["x_lh"].to_numpy(), pk["x_la"].to_numpy()
        mk = _ah_side_outcomes(lh + s0 / 2, la - s0 / 2, line, side, rho)
        rs = _ah_side_outcomes(lh + s1 / 2, la - s1 / 2, line, side, rho)
        be, be_mkt = _break_even(rs), _break_even(mk)
        for i, (_, r) in enumerate(pk.iterrows()):
            col = "ahh" if side[i] == 0 else "aha"
            prices = {v: r.get(f"{v}_{col}", np.nan) for v in F2_VENUES}
            prices = {v: o for v, o in prices.items() if np.isfinite(o)}
            best_v = max(prices, key=lambda v: float(forward.effective_odds(v, prices[v])))
            best_net = float(forward.effective_odds(best_v, prices[best_v]))
            team_line = r["f2_pick"].rsplit(" ", 1)[-1]
            edge = float(_ah_ev({k: rs[k][[i]] for k in rs}, [best_net])[0])
            rows.append({
                "match_id": r["match_id"], "date": r["date"], "time": _txt(r["time"]),
                "league": r["league"], "league_name": L.league_name(r["league"]),
                "home": r["home"], "away": r["away"], "rule": "F2", "rule_name": "Luck fade",
                "market": "Asian handicap",
                "selection": f"{r['f2_pick'].rsplit(' ', 1)[0]} {team_line} (Asian handicap)",
                "our_prob": float(rs["win"][i] + rs["half_win"][i]),
                "market_prob": float(mk["win"][i] + mk["half_win"][i]),
                "push_prob": float(rs["push"][i]),
                "lose_prob": float(rs["loss"][i] + rs["half_loss"][i]),
                "half_result_prob": float(rs["half_win"][i] + rs["half_loss"][i]),
                "breakeven_odds": float(be[i]), "market_fair_odds": float(be_mkt[i]),
                "min_odds": float(be[i]), "min_odds_exchange": float(_commission_needed(be[i])),
                "best_odds": float(prices[best_v]), "best_book": BOOK_NAMES.get(best_v, best_v),
                "edge_at_best": edge, "status": "Bet" if edge > 0 else "Wait",
                "reason": _luck_reason(r),
            })
    # ---- F1 lower-division over/under picks and near misses
    po = fx[f"{f1['model']}_po25"]
    for _, r in fx[fx["tier"].isin(f1["tiers"]) & po.notna()].iterrows():
        p_over = float(r[f"{f1['model']}_po25"])
        m_over = float(r["oumkt_po25"]) if np.isfinite(r.get("oumkt_po25", np.nan)) else np.nan
        cands = []
        for label, p, m, col in (("Over 2.5 goals", p_over, m_over, "o25"),
                                 ("Under 2.5 goals", 1 - p_over, 1 - m_over, "u25")):
            prices = {v: r.get(f"{v}_{col}", np.nan) for v in forward.venues(f1["prices"])}
            prices = {v: o for v, o in prices.items() if np.isfinite(o) and o <= f1["max_odds"]}
            if not prices:
                continue
            best_v = max(prices, key=lambda v: float(forward.effective_odds(v, prices[v])))
            ev = p * float(forward.effective_odds(best_v, prices[best_v])) - 1
            cands.append((label, p, m, best_v, prices[best_v], ev))
        if not cands:
            continue
        label, p, m, best_v, best_o, ev = max(cands, key=lambda c: c[5])
        picked = bool(r.get("f1_pick"))
        near = (not picked and -0.04 <= ev < f1["min_edge"] and np.isfinite(m) and p - m >= 0.02)
        if not (picked or near):
            continue
        if picked:  # use the price the rule actually picked
            label = "Over 2.5 goals" if r["f1_pick"].startswith("Over") else "Under 2.5 goals"
            p = float(r["f1_prob"])
            m = m_over if label.startswith("Over") else 1 - m_over
            best_v, best_o, ev = r["f1_book"], float(r["f1_odds"]), float(r["f1_ev"])
        more = label.startswith("Over")
        rows.append({
            "match_id": r["match_id"], "date": r["date"], "time": _txt(r["time"]),
            "league": r["league"], "league_name": L.league_name(r["league"]),
            "home": r["home"], "away": r["away"], "rule": "F1", "rule_name": "Lower-league goals",
            "market": "Over/under 2.5", "selection": label, "our_prob": p, "market_prob": m,
            "push_prob": 0.0, "lose_prob": 1 - p, "half_result_prob": 0.0,
            "breakeven_odds": 1 / p, "market_fair_odds": 1 / m if np.isfinite(m) and m > 0 else np.nan,
            "min_odds": (1 + f1["min_edge"]) / p,
            "min_odds_exchange": float(_commission_needed((1 + f1["min_edge"]) / p)),
            "best_odds": float(best_o), "best_book": BOOK_NAMES.get(best_v, best_v),
            "edge_at_best": ev, "status": "Bet" if picked else "Wait",
            "reason": (f"Our goals model gives a {p_over:.0%} chance of 3+ goals against the market's "
                       f"{m_over:.0%}. It expects {'more' if more else 'fewer'} goals than priced; "
                       f"this rule has only been profitable below the top flight."),
        })
    cols = ["match_id", "date", "time", "league", "league_name", "home", "away", "rule", "rule_name",
            "market", "selection", "our_prob", "market_prob", "push_prob", "lose_prob",
            "half_result_prob", "breakeven_odds", "market_fair_odds", "min_odds", "min_odds_exchange",
            "best_odds", "best_book", "edge_at_best", "status", "reason"]
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values(["status", "date", "time", "league"],
                              key=lambda c: c.map({"Bet": 0, "Wait": 1}) if c.name == "status" else c)
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ ledger
LEDGER_COLS = ["logged_at", "match_id", "date", "league", "home", "away", "rule", "pick", "side",
               "line", "book", "odds", "prob", "ev", "signal"]


def picks_from(fx: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = []
    for _, r in fx.iterrows():
        if r.get("f1_pick"):
            rows.append({"logged_at": now, "match_id": r["match_id"], "date": r["date"],
                         "league": r["league"], "home": r["home"], "away": r["away"], "rule": "F1",
                         "pick": r["f1_pick"], "side": "over" if r["f1_pick"].startswith("Over") else "under",
                         "line": 2.5, "book": r["f1_book"], "odds": r["f1_odds"], "prob": r["f1_prob"],
                         "ev": r["f1_ev"], "signal": np.nan})
        if r.get("f2_pick"):
            rows.append({"logged_at": now, "match_id": r["match_id"], "date": r["date"],
                         "league": r["league"], "home": r["home"], "away": r["away"], "rule": "F2",
                         "pick": r["f2_pick"], "side": r["f2_side"],
                         "line": r["f2_line"], "book": r["f2_book"], "odds": r["f2_odds"],
                         "prob": np.nan, "ev": r["f2_ev"], "signal": r["f2_signal"]})
    return pd.DataFrame(rows, columns=LEDGER_COLS)


def update_ledger(fx: pd.DataFrame, matches: pd.DataFrame, path: Path = LEDGER) -> tuple[pd.DataFrame, dict]:
    """Append new picks (first price seen is kept), settle finished ones, summarise."""
    path = Path(path)
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=LEDGER_COLS)
    new = picks_from(fx) if not fx.empty else pd.DataFrame(columns=LEDGER_COLS)
    led = pd.concat([old[LEDGER_COLS], new], ignore_index=True) if len(old) else new
    led = led.drop_duplicates(["match_id", "rule"], keep="first").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(path, index=False)
    return led, settle(led, matches)


def settle_rows(led: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Ledger rows with result, return per unit and closing-line value where known."""
    close = [f"{bk}_{o}" for bk in forward.CLOSING_BOOKS for o in ("o25", "u25", "ahh", "aha")]
    cols = ["match_id", "fthg", "ftag"] + [c for c in close + ["ahc_line"] if c in matches.columns]
    m = led.merge(matches[cols], on="match_id", how="left")
    done = m["fthg"].notna() & m["ftag"].notna()
    m["odds"] = m["odds"].astype(float)
    net = np.array([float(forward.effective_odds(b, o)) for b, o in zip(m["book"], m["odds"])])
    ret = np.full(len(m), np.nan)
    goals = (m["fthg"] + m["ftag"]).to_numpy()
    gd = (m["fthg"] - m["ftag"]).to_numpy()
    f1 = (m["rule"] == "F1").to_numpy()
    over = m["side"].eq("over").to_numpy()
    win1 = np.where(over, goals > 2.5, goals < 2.5)
    ret = np.where(f1 & done, np.where(win1, net - 1, -1.0), ret)
    f2 = (m["rule"] == "F2").to_numpy()
    home = m["side"].eq("home").to_numpy()
    stl = ah_settle(np.nan_to_num(gd), m["line"].astype(float).to_numpy())
    stl = np.where(home, stl, -stl)
    ret = np.where(f2 & done, np.where(stl > 0, stl * (net - 1), stl), ret)
    side_idx = np.where(over | home, 0, 1)
    cl1, bk1 = forward.closing_value(m, ("o25", "u25"), side_idx)
    same = (m["ahc_line"] == m["line"]).to_numpy() if "ahc_line" in m else np.zeros(len(m), bool)
    cl2, bk2 = forward.closing_value(m, ("ahh", "aha"), side_idx, same)
    clv = np.where(f1, cl1, np.where(f2, cl2, np.nan))
    m["clv_book"] = np.where(f1, bk1, bk2)
    m["ret"], m["clv"] = ret, clv
    m["settled"] = done
    m["result"] = np.where(~done, "pending",
                           np.where(m["ret"] > 0, "won", np.where(m["ret"] < 0, "lost", "void")))
    m["score"] = np.where(done, m["fthg"].fillna(0).astype(int).astype(str) + "-"
                          + m["ftag"].fillna(0).astype(int).astype(str), "")
    return m


def settle(led: pd.DataFrame, matches: pd.DataFrame) -> dict:
    if led.empty:
        return {"logged": 0}
    m = settle_rows(led, matches)
    done = m["settled"]
    out = {"logged": int(len(m)), "settled": int(done.sum()), "pending": int((~done).sum())}
    for rule in ("F1", "F2"):
        g = m[(m["rule"] == rule) & done]
        r = {"settled": int(len(g)), "pending": int(((m["rule"] == rule) & ~done).sum())}
        if len(g):
            r.update({"roi": float(g["ret"].mean()), "profit_units": float(g["ret"].sum()),
                      "hit_rate": float((g["ret"] > 0).mean())})
        c = g["clv"].dropna()
        if len(c):
            r.update({"clv_bets": int(len(c)), "avg_clv": float(c.mean()),
                      "beat_close": float((c > 0).mean())})
        out[rule] = r
    return out


# ------------------------------------------------------------------ output
SHOW = ["date", "time", "league", "home", "away", "low_data",
        "mkt_ph", "mkt_pd", "mkt_pa", "blend_ph", "blend_pd", "blend_pa", "ens_ph", "ens_pd", "ens_pa",
        "res_ph", "res_pd", "res_pa", "luck_shift_goals", "oumkt_po25", "oublend_po25",
        "best_h", "best_d", "best_a", "ev_h", "ev_d", "ev_a", "ah_line",
        "f1_pick", "f1_book", "f1_odds", "f1_ev", "f2_pick", "f2_side", "f2_book", "f2_odds", "f2_signal",
        "adjusted", "match_id"]


def run(fetch: bool = True, out_dir: Path = OUT_DIR) -> dict:
    matches, fixtures = load_inputs(fetch)
    fx = score(matches, fixtures)
    led, summary = update_ledger(fx, matches)
    spec = json.loads(forward.FROZEN.read_text(encoding="utf-8"))
    picks = pick_details(fx, spec) if not fx.empty else pick_details(pd.DataFrame(), spec)
    ledger_rows = settle_rows(led, matches) if len(led) else pd.DataFrame()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = fx[[c for c in SHOW if c in fx.columns]] if not fx.empty else pd.DataFrame(columns=SHOW)
    table.to_csv(out_dir / "fixtures.csv", index=False)
    picks.to_csv(out_dir / "picks.csv", index=False)
    meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "results_through": str(matches.loc[matches["ftr"].notna(), "date"].max().date()),
            "fixtures": int(len(table)), "bets": int((picks["status"] == "Bet").sum()),
            "watch": int((picks["status"] == "Wait").sum()),
            "f1_picks": int((table["f1_pick"] != "").sum()) if len(table) else 0,
            "f2_picks": int((table["f2_pick"] != "").sum()) if len(table) else 0, "ledger": summary}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(table, meta, picks, ledger_rows), encoding="utf-8")
    return meta


def _txt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    x = str(x)
    return x[:5] if len(x) == 8 and x[2] == ":" and x[5] == ":" else x  # 19:30:00 -> 19:30


def _pct(x) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def _signed(x, d=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def _odds(x) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"


def _when(date, time) -> str:
    d = pd.Timestamp(date)
    return f"{d.strftime('%a %d %b')} {_txt(time)}".strip()


CSS = """
:root { --bg:#f4f6f5; --surface:#fff; --surface-2:#eef2f0; --ink:#13201a; --ink-2:#46544c; --muted:#6f7d75;
  --rule:#dbe3de; --accent:#1c5cab; --ours:#2a78d6; --mkt:#9aa7a0; --good:#0a7d0a; --good-bg:#e3f4e3;
  --wait:#8a5a00; --wait-bg:#fbf0d9; --bad:#b42828;
  --display:"Barlow Condensed","Arial Narrow",sans-serif; --body:"IBM Plex Sans",system-ui,"Segoe UI",sans-serif; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#111513; --surface:#181d1a; --surface-2:#1f2522; --ink:#eef2ef;
  --ink-2:#b8c3bd; --muted:#8b9890; --rule:#2b3330; --accent:#6da7ec; --ours:#3987e5; --mkt:#5d6a63;
  --good:#5fd35f; --good-bg:#16301a; --wait:#f0b84a; --wait-bg:#33290f; --bad:#ef7070; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#111513; --surface:#181d1a; --surface-2:#1f2522; --ink:#eef2ef;
  --ink-2:#b8c3bd; --muted:#8b9890; --rule:#2b3330; --accent:#6da7ec; --ours:#3987e5; --mkt:#5d6a63;
  --good:#5fd35f; --good-bg:#16301a; --wait:#f0b84a; --wait-bg:#33290f; --bad:#ef7070; color-scheme:dark; }
* { box-sizing:border-box; }
[hidden] { display:none !important; }
body { background:var(--bg); color:var(--ink); font:15px/1.5 var(--body); margin:0; }
.wrap { max-width:1000px; margin:0 auto; padding:20px max(16px, env(safe-area-inset-right, 0px)) 56px max(16px, env(safe-area-inset-left, 0px));
  display:grid; gap:16px; }
h1,h2,h3 { font-family:var(--display); margin:0; line-height:1.1; text-wrap:balance; }
h1 { font-size:clamp(30px,6vw,44px); } h2 { font-size:24px; } h3 { font-size:18px; font-weight:600; }
.sub { color:var(--ink-2); margin:0; max-width:70ch; } .muted { color:var(--muted); } .small { font-size:12.5px; }
.eyebrow { font:600 12px var(--body); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.tabs { display:flex; gap:4px; border-bottom:1px solid var(--rule); position:sticky; top:env(safe-area-inset-top, 0px); background:var(--bg); z-index:2; padding-top:4px; overflow-x:auto; }
body::before { content:""; position:fixed; inset:0 0 auto; height:env(safe-area-inset-top, 0px); z-index:10; background:var(--bg); }
.tabs button { font:600 14px var(--body); color:var(--ink-2); background:none; border:0; padding:10px 14px; cursor:pointer;
  border-bottom:3px solid transparent; }
.tabs button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--accent); }
.tabs button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.count { font:600 12px var(--body); background:var(--surface-2); border-radius:999px; padding:1px 7px; margin-left:4px; }
.cards { display:grid; gap:12px; }
.card { background:var(--surface); border:1px solid var(--rule); border-radius:12px; padding:14px 16px; display:grid; gap:10px; }
.top { display:flex; flex-wrap:wrap; justify-content:space-between; gap:6px 12px; align-items:center; }
.meta { font-size:12.5px; color:var(--muted); }
.teams { font:700 22px var(--display); }
.selection { font:600 17px var(--body); }
.status { display:inline-flex; align-items:center; gap:6px; font:600 13px var(--body); border-radius:999px; padding:3px 10px; }
.status.bet { color:var(--good); background:var(--good-bg); } .status.wait { color:var(--wait); background:var(--wait-bg); }
.rule { font-size:12px; font-weight:600; color:var(--ink-2); background:var(--surface-2); border-radius:6px; padding:2px 8px; }
.probs { display:grid; gap:6px; }
.bar { display:grid; grid-template-columns:72px minmax(0,1fr) 44px; align-items:center; gap:8px; font-size:13px; }
.track { height:8px; background:var(--surface-2); border-radius:4px; overflow:hidden; }
.fill { height:100%; border-radius:4px; } .fill.ours { background:var(--ours); } .fill.mkt { background:var(--mkt); }
.stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
@media (max-width:600px) { .stats { grid-template-columns:repeat(2,minmax(0,1fr)); } }
.stat { background:var(--surface-2); border-radius:8px; padding:8px 10px; display:grid; gap:2px; }
.stat .label { font-size:12px; color:var(--muted); } .stat .value { font:600 20px var(--display); font-variant-numeric:tabular-nums; }
.stat .note { font-size:11.5px; color:var(--ink-2); }
.why { font-size:13.5px; color:var(--ink-2); margin:0; }
.tablewrap { overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th,td { padding:7px 10px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--rule); vertical-align:top; }
th { font:600 12px var(--body); color:var(--muted); background:var(--surface-2); }
th.l,td.l { text-align:left; } tbody tr:last-child td { border-bottom:0; }
.won { color:var(--good); font-weight:600; } .lost { color:var(--bad); font-weight:600; }
details summary { cursor:pointer; color:var(--ink-2); font-size:13.5px; }
code { background:var(--surface-2); padding:1px 5px; border-radius:4px; }
"""


def _pick_card(r: pd.Series) -> str:
    e = html.escape
    bet = r["status"] == "Bet"
    status = ('<span class="status bet">✓ Bet</span>' if bet else
              f'<span class="status wait">◷ Wait for {_odds(r["min_odds"])}+</span>')
    probs = (f'<div class="bar"><span>Our view</span><div class="track"><div class="fill ours" style="width:{r["our_prob"] * 100:.0f}%"></div></div><b>{_pct(r["our_prob"])}</b></div>'
             f'<div class="bar"><span class="muted">Market</span><div class="track"><div class="fill mkt" style="width:{r["market_prob"] * 100:.0f}%"></div></div><span>{_pct(r["market_prob"])}</span></div>')
    extra = ""
    if r["push_prob"] > 0.005:
        extra += f"Stake refunded {_pct(r['push_prob'])} of the time. "
    if r["half_result_prob"] > 0.005:
        extra += f"Half win or half loss {_pct(r['half_result_prob'])}. "
    win_word = "Chance the bet wins" if r["market"] == "Asian handicap" else "Chance it lands"
    min_note = "break-even" if r["rule"] == "F2" else "break-even + 3% edge"
    edge_cls = "won" if r["edge_at_best"] > 0 else "lost"
    return f"""<article class="card">
  <div class="top"><span class="meta">{e(_when(r['date'], r['time']))} · {e(r['league_name'])}</span>
    <span><span class="rule">{e(r['rule_name'])}</span> {status}</span></div>
  <div class="teams">{e(r['home'])} – {e(r['away'])}</div>
  <div class="selection">{e(r['selection'])}</div>
  <div class="probs" aria-label="{win_word}">{probs}<div class="small muted">{e(win_word)}, ours vs the bookmaker's price with its margin removed. {e(extra)}</div></div>
  <div class="stats">
    <div class="stat"><span class="label">Break-even odds</span><span class="value">{_odds(r['breakeven_odds'])}</span><span class="note">market fair {_odds(r['market_fair_odds'])}</span></div>
    <div class="stat"><span class="label">Bet at or above</span><span class="value">{_odds(r['min_odds'])}</span><span class="note">{min_note} · Betfair {_odds(r['min_odds_exchange'])}+</span></div>
    <div class="stat"><span class="label">Best price now</span><span class="value">{_odds(r['best_odds'])}</span><span class="note">{e(str(r['best_book']))}</span></div>
    <div class="stat"><span class="label">Edge at that price</span><span class="value {edge_cls}">{_signed(r['edge_at_best'])}</span><span class="note">expected profit per £1</span></div>
  </div>
  <p class="why">{e(r['reason'])}</p>
</article>"""


def render_html(t: pd.DataFrame, meta: dict, picks: pd.DataFrame | None = None,
                ledger_rows: pd.DataFrame | None = None) -> str:
    e = html.escape
    picks = picks if picks is not None else pd.DataFrame(columns=["status"])
    bets = picks[picks["status"] == "Bet"] if len(picks) else picks
    watch = picks[picks["status"] == "Wait"] if len(picks) else picks
    led = meta.get("ledger", {})

    picks_html = (f"<h2>Bets now <span class='count'>{len(bets)}</span></h2>"
                  "<p class='sub small'>Edges are the model's estimates. Track record so far: Luck fade returned "
                  "about +1% at Bet365 and Pinnacle prices over 4,000+ bets (2019-2026, not statistically proven); "
                  "Lower-league goals about +5% at the best price across bookmakers (3,660 bets, 2016-2026). "
                  "Betfair Exchange prices are shown before its 5% commission; the minimum it needs is after.</p>"
                  + ("<div class='cards'>" + "".join(_pick_card(r) for _, r in bets.iterrows()) + "</div>"
                     if len(bets) else "<p class='sub'>No research picks at current prices.</p>"))
    if len(watch):
        picks_html += (f"<details><summary>Watchlist: {len(watch)} goals plays that need a better price</summary>"
                       "<div class='cards' style='margin-top:12px'>"
                       + "".join(_pick_card(r) for _, r in watch.iterrows()) + "</div></details>")

    groups = []
    if len(t):
        for (day, lg), g in t.groupby([t["date"].astype(str).str[:10], "league"], sort=True):
            rows = []
            for _, r in g.iterrows():
                dot = "● " if (r.get("f1_pick") or r.get("f2_pick")) else ""
                shift = r["luck_shift_goals"]
                rows.append(
                    f"<tr><td class='l'>{dot}{e(r['home'])} – {e(r['away'])}<div class='muted small'>{e(_txt(r['time']))}</div></td>"
                    f"<td>{_pct(r['mkt_ph'])} / {_pct(r['mkt_pd'])} / {_pct(r['mkt_pa'])}</td>"
                    f"<td><b>{_pct(r['res_ph'])} / {_pct(r['res_pd'])} / {_pct(r['res_pa'])}</b></td>"
                    f"<td>{'–' if not np.isfinite(shift) else f'{shift:+.2f}'}</td>"
                    f"<td>{_pct(r['oumkt_po25'])} → <b>{_pct(r['oublend_po25'])}</b></td></tr>")
            groups.append(f"<h3>{e(pd.Timestamp(day).strftime('%a %d %b'))} · {e(L.league_name(lg))}</h3>"
                          "<div class='tablewrap'><table><thead><tr><th class='l'>Match</th><th>Market H/D/A</th>"
                          "<th>Research H/D/A</th><th>Luck shift</th><th>Over 2.5: market → model</th></tr></thead>"
                          f"<tbody>{''.join(rows)}</tbody></table></div>")
    fixtures_html = ("<p class='sub'>Research H/D/A is the market corrected for luck; a negative luck shift means "
                     "the home side is over-rated. ● marks a match with a research pick.</p>" + "".join(groups)
                     if groups else
                     "<p class='sub'>No fixtures found. football-data.co.uk publishes fixtures with odds around "
                     "Tuesday (midweek) and Friday (weekend). Run <code>footy upcoming</code> again then.</p>")

    def led_row(rule, name):
        r = led.get(rule, {})
        return (f"<tr><td class='l'>{name}</td><td>{r.get('settled', 0)}</td><td>{r.get('pending', 0)}</td>"
                f"<td>{_signed(r.get('roi', np.nan))}</td><td>{_signed(r.get('avg_clv', np.nan))}</td></tr>")
    ledger_html = ("<p class='sub'>Every pick is logged at the first price seen and settled when the result "
                   "arrives. Judge the rules on closing-line value (CLV) over 500+ bets, not on a few weeks of ROI.</p>"
                   "<div class='tablewrap'><table><thead><tr><th class='l'>Rule</th><th>Settled</th><th>Pending</th>"
                   "<th>ROI</th><th>Avg CLV</th></tr></thead><tbody>"
                   + led_row("F1", "Lower-league goals (F1)") + led_row("F2", "Luck fade (F2)") + "</tbody></table></div>")
    if ledger_rows is not None and len(ledger_rows):
        lr = ledger_rows.sort_values("date", ascending=False).head(200)
        body = "".join(
            f"<tr><td class='l'>{e(pd.Timestamp(r['date']).strftime('%d %b'))}</td><td class='l'>{e(r['home'])} – {e(r['away'])}</td>"
            f"<td class='l'>{e(str(r['pick']))}</td><td>{_odds(float(r['odds']))} <span class='muted small'>{e(BOOK_NAMES.get(r['book'], r['book']))}</span></td>"
            f"<td>{e(r['score'])}</td><td class='l {r['result']}'>{e(r['result'])}</td><td>{_signed(r['ret'])}</td><td>{_signed(r['clv'])}</td></tr>"
            for _, r in lr.iterrows())
        ledger_html += ("<div class='tablewrap'><table><thead><tr><th class='l'>Date</th><th class='l'>Match</th>"
                        "<th class='l'>Pick</th><th>Odds</th><th>Score</th><th class='l'>Result</th><th>Return</th>"
                        f"<th>CLV</th></tr></thead><tbody>{body}</tbody></table></div>")

    n_bets = len(bets)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Matchday Picks</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body><div class="wrap">
<div class="eyebrow">Updated {e(meta['generated'])} · results through {e(meta['results_through'])}</div>
<h1>Matchday picks</h1>
<p class="sub">Picks from the two research rules: <b>Luck fade</b> (back the team the market over-rates because of lucky finishing
or goalkeeping, Asian handicap) and <b>Lower-league goals</b> (over/under 2.5 below the top flight). Both are still being
forward-tested, so stake small or paper-trade until the ledger shows positive closing-line value over 500 bets.</p>
<nav class="tabs" role="tablist">
  <button role="tab" id="tab-picks" aria-selected="true" aria-controls="panel-picks">Picks <span class="count">{n_bets}</span></button>
  <button role="tab" id="tab-fixtures" aria-selected="false" aria-controls="panel-fixtures">All fixtures <span class="count">{len(t)}</span></button>
  <button role="tab" id="tab-ledger" aria-selected="false" aria-controls="panel-ledger">Ledger</button>
</nav>
<section id="panel-picks" role="tabpanel" aria-labelledby="tab-picks" class="cards">{picks_html}</section>
<section id="panel-fixtures" role="tabpanel" aria-labelledby="tab-fixtures" hidden class="cards">{fixtures_html}</section>
<section id="panel-ledger" role="tabpanel" aria-labelledby="tab-ledger" hidden class="cards">{ledger_html}</section>
</div>
<script>
document.querySelectorAll('.tabs button').forEach(function (b) {{
  b.addEventListener('click', function () {{
    document.querySelectorAll('.tabs button').forEach(function (x) {{
      var on = x === b; x.setAttribute('aria-selected', on);
      document.getElementById(x.getAttribute('aria-controls')).hidden = !on;
    }});
  }});
}});
</script></body></html>"""
