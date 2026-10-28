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


def settle(led: pd.DataFrame, matches: pd.DataFrame) -> dict:
    if led.empty:
        return {"logged": 0}
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = fx[[c for c in SHOW if c in fx.columns]] if not fx.empty else pd.DataFrame(columns=SHOW)
    table.to_csv(out_dir / "fixtures.csv", index=False)
    meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "results_through": str(matches.loc[matches["ftr"].notna(), "date"].max().date()),
            "fixtures": int(len(table)), "f1_picks": int((table["f1_pick"] != "").sum()) if len(table) else 0,
            "f2_picks": int((table["f2_pick"] != "").sum()) if len(table) else 0, "ledger": summary}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(table, meta), encoding="utf-8")
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


def render_html(t: pd.DataFrame, meta: dict) -> str:
    e = html.escape
    led = meta.get("ledger", {})

    def ledger_row(rule, name):
        r = led.get(rule, {})
        if not r:
            return f"<tr><td class='l'>{name}</td><td>0</td><td>0</td><td>–</td><td>–</td></tr>"
        return (f"<tr><td class='l'>{name}</td><td>{r.get('settled', 0)}</td><td>{r.get('pending', 0)}</td>"
                f"<td>{_signed(r.get('roi', np.nan))}</td><td>{_signed(r.get('avg_clv', np.nan))}"
                f"{' (' + str(r['clv_bets']) + ')' if r.get('clv_bets') else ''}</td></tr>")

    picks = t[(t["f1_pick"] != "") | (t["f2_pick"] != "")] if len(t) else t
    pick_cards = []
    for _, r in picks.iterrows():
        chips = []
        if r["f1_pick"]:
            chips.append(f"<span class='chip f1'>F1 · {e(r['f1_pick'])} @ {r['f1_odds']:.2f} "
                         f"{e(BOOK_NAMES.get(r['f1_book'], r['f1_book']))} · edge {_signed(r['f1_ev'])}</span>")
        if r["f2_pick"]:
            chips.append(f"<span class='chip f2'>F2 · {e(r['f2_pick'])} @ {r['f2_odds']:.2f} "
                         f"{e(BOOK_NAMES.get(r['f2_book'], r['f2_book']))} · luck signal {_signed(r['f2_signal'])}</span>")
        pick_cards.append(f"<div class='pick'><div class='when'>{e(str(r['date'])[:10])} {e(_txt(r['time']))} · "
                          f"{e(L.league_name(r['league']))}</div><div class='teams'>{e(r['home'])} – {e(r['away'])}</div>"
                          f"<div class='chips'>{''.join(chips)}</div></div>")
    groups = []
    if len(t):
        for (day, lg), g in t.groupby([t["date"].astype(str).str[:10], "league"], sort=True):
            rows = []
            for _, r in g.iterrows():
                tag = ""
                if r["f1_pick"] or r["f2_pick"]:
                    tag = "<span class='dot' title='research pick'></span>"
                lowd = " <span class='muted'>(little data)</span>" if r.get("low_data") else ""
                shift = r["luck_shift_goals"]
                shift_txt = "–" if not np.isfinite(shift) else f"{shift:+.2f}"
                rows.append(
                    f"<tr><td class='l'>{tag}{e(r['home'])} – {e(r['away'])}{lowd}<div class='muted small'>{e(_txt(r['time']))}</div></td>"
                    f"<td>{_pct(r['mkt_ph'])} / {_pct(r['mkt_pd'])} / {_pct(r['mkt_pa'])}</td>"
                    f"<td><b>{_pct(r['res_ph'])} / {_pct(r['res_pd'])} / {_pct(r['res_pa'])}</b><div class='muted small'>luck shift {shift_txt}</div></td>"
                    f"<td>{_pct(r['blend_ph'])} / {_pct(r['blend_pd'])} / {_pct(r['blend_pa'])}</td>"
                    f"<td>{_pct(r['oumkt_po25'])} → <b>{_pct(r['oublend_po25'])}</b></td></tr>")
            groups.append(f"<h3>{e(day)} · {e(L.league_name(lg))} <span class='muted'>{e(lg)}</span></h3>"
                          "<div class='tablewrap'><table><thead><tr><th class='l'>Match</th><th>Market H/D/A</th>"
                          "<th>Research H/D/A</th><th>Model H/D/A</th><th>Over 2.5: market → model</th></tr></thead>"
                          f"<tbody>{''.join(rows)}</tbody></table></div>")
    empty = ("<p class='sub'>No fixtures found. football-data.co.uk publishes upcoming fixtures with odds "
             "around Tuesday (midweek games) and Friday (weekend). Run <code>footy upcoming</code> again then.</p>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Upcoming Fixtures</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{ --bg:#f4f6f5; --surface:#fff; --surface-2:#eef2f0; --ink:#13201a; --ink-2:#46544c; --muted:#6f7d75; --rule:#dbe3de;
  --f1:#2a78d6; --f2:#eb6834; --display:"Barlow Condensed","Arial Narrow",sans-serif; --body:"IBM Plex Sans",system-ui,"Segoe UI",sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111513; --surface:#181d1a; --surface-2:#1f2522; --ink:#eef2ef; --ink-2:#b8c3bd;
  --muted:#8b9890; --rule:#2b3330; --f1:#3987e5; --f2:#d95926; color-scheme: dark; }} }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 var(--body); margin:0; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:24px max(16px, env(safe-area-inset-right, 0px)) 64px max(16px, env(safe-area-inset-left, 0px)); display:grid; gap:18px; }}
h1,h2,h3 {{ font-family:var(--display); margin:0; line-height:1.1; }}
h1 {{ font-size:clamp(32px,6vw,46px); }} h2 {{ font-size:26px; }} h3 {{ font-size:19px; font-weight:600; margin-top:8px; }}
.sub {{ color:var(--ink-2); max-width:70ch; margin:0; }} .muted {{ color:var(--muted); }} .small {{ font-size:12px; }}
.eyebrow {{ font:600 12px var(--body); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
.picks {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }}
.pick {{ background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:12px 14px; display:grid; gap:6px; }}
.pick .when {{ font-size:12.5px; color:var(--muted); }} .pick .teams {{ font:600 20px var(--display); }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ font-size:12.5px; padding:4px 8px; border-radius:6px; background:var(--surface-2); border-left:3px solid var(--f1); }}
.chip.f2 {{ border-left-color:var(--f2); }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--f2); margin-right:6px; vertical-align:middle; }}
.tablewrap {{ overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:7px 10px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font:600 12px var(--body); color:var(--muted); background:var(--surface-2); }}
th.l,td.l {{ text-align:left; }} tbody tr:last-child td {{ border-bottom:0; }}
code {{ background:var(--surface-2); padding:1px 5px; border-radius:4px; }}
</style></head><body><div class="wrap">
<div class="eyebrow">Generated {e(meta['generated'])} · results through {e(meta['results_through'])}</div>
<h1>Upcoming fixtures</h1>
<p class="sub"><b>Market</b> is the bookmaker's probability with the margin removed. <b>Research</b> corrects it for the
confirmed luck over-reaction (a positive luck shift favours the home side). <b>Model</b> is our ratings + market blend. Exchange prices are shown before Betfair's 5% commission; edges are after it.
Picks come from the two frozen research rules, which are still being forward-tested: F1 lower-division over/under 2.5,
F2 luck-fade Asian handicap. Treat them as paper trades until the ledger shows positive closing-line value over 500 bets.</p>
<h2>Research picks</h2>
{('<div class="picks">' + ''.join(pick_cards) + '</div>') if pick_cards else '<p class="sub">No picks for these fixtures.</p>'}
<h2>Paper-trading ledger</h2>
<div class="tablewrap"><table><thead><tr><th class="l">Rule</th><th>Settled</th><th>Pending</th><th>ROI</th><th>Avg CLV vs closing price (bets)</th></tr></thead>
<tbody>{ledger_row('F1', 'F1 lower-division over/under 2.5')}{ledger_row('F2', 'F2 luck-fade Asian handicap')}</tbody></table></div>
<h2>All fixtures</h2>
{''.join(groups) if groups else empty}
</div></body></html>"""
