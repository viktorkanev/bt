"""Interactive dashboard: ``streamlit run app/dashboard.py``.

Requires ``footy build`` (and ideally ``footy report``) to have been run.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from footy import backtest as bt
from footy import evaluate as ev
from footy import leagues as L
from footy import pipeline
from footy.data import store
from footy.models.probability import eligible
from footy.ratings.poisson_math import goal_diff_dist, over_prob, score_matrix

st.set_page_config(page_title="Football Edge", layout="wide")

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REPORT_DIR = store.ROOT / "reports" / "latest"
MODEL_NAMES = {
    "mkt": "Market (Bet365, margin removed)", "elo": "Elo", "pi": "Pi-ratings",
    "pois": "Dixon-Coles goals", "pxg": "Dixon-Coles pseudo-xG", "ens": "Ensemble (ratings only)",
    "gbm": "Gradient boosting (ratings only)", "blend": "Ensemble + market",
    "oumkt": "Market O/U", "ouens": "O/U ensemble", "oublend": "O/U ensemble + market",
}


@st.cache_data(show_spinner="Loading predictions...")
def load_predictions() -> pd.DataFrame:
    return pipeline.load_predictions()


@st.cache_resource(show_spinner="Loading models...")
def load_engine_models():
    return pipeline.load_engine(), pipeline.load_models()


def league_label(code: str) -> str:
    lg = L.get(code)
    return f"{lg.name} ({code}, tier {lg.tier})"


def fig_layout(fig: go.Figure, height: int = 360, ytitle: str = "") -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=ytitle,
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


try:
    D = load_predictions()
except FileNotFoundError:
    st.error("No predictions found. Run `footy fetch` then `footy build` first.")
    st.stop()

meta = store.load_meta()
played = D[D["ftr"].notna()]
st.sidebar.title("Football Edge")
st.sidebar.caption(f"{meta.get('source', '?')} · {len(played):,} matches · last result "
                   f"{played['date'].max().date()}")
page = st.sidebar.radio("Page", ["Overview", "Backtest lab", "Model accuracy", "Match predictor",
                                 "Upcoming & value", "Ratings", "Team news"])
all_leagues = sorted(D["league"].unique(), key=lambda c: (L.get(c).tier, c))


# ------------------------------------------------------------------ overview
if page == "Overview":
    st.title("Football Edge")
    st.write("Rating models and walk-forward predictions for "
             f"{D['league'].nunique()} leagues, benchmarked against bookmaker prices.")
    summ = REPORT_DIR / "strategies.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        tiers = pd.read_csv(REPORT_DIR / "strategies_by_tier.csv")
        ou = tiers[(tiers.strategy == "ou_blend_max") & (tiers.tier_grp == 0)].set_index("period")
        c = st.columns(4)
        if "holdout" in ou.index:
            c[0].metric("O/U 2.5, tiers 2-5, holdout ROI", f"{ou.loc['holdout', 'roi']:+.1%}",
                        f"{int(ou.loc['holdout', 'bets'])} bets")
        acc = pd.read_csv(REPORT_DIR / "accuracy_overall.csv")
        h = acc[acc.period == "holdout"].set_index("model")["rps"]
        c[1].metric("Market RPS (holdout)", f"{h['mkt']:.4f}")
        c[2].metric("Best ratings-only RPS", f"{h[['ens', 'gbm', 'pxg']].min():.4f}")
        c[3].metric("Model+market RPS", f"{h['blend']:.4f}")
        st.subheader("Fixed strategies")
        cols = ["name", "dev_bets", "dev_roi", "hold_bets", "hold_roi", "hold_roi_lo", "hold_roi_hi",
                "hold_p_value"]
        st.dataframe(s[cols].style.format({c: "{:+.2%}" for c in cols if "roi" in c}
                                          | {"hold_p_value": "{:.3f}"}), hide_index=True,
                     use_container_width=True)
        st.subheader("Lower leagues vs top flights")
        st.dataframe(tiers.round(4), hide_index=True, use_container_width=True)
        html = REPORT_DIR / "index.html"
        if html.exists():
            st.caption(f"Full static report: {html}")
    else:
        st.info("Run `footy report` to generate the summary tables.")


# ------------------------------------------------------------------ backtest lab
elif page == "Backtest lab":
    st.title("Backtest lab")
    st.caption("Probabilities are out-of-sample (walk-forward). Pick a rule and see how it would have done.")
    c1, c2, c3 = st.columns(3)
    mkt = c1.selectbox("Market", ["1x2", "ou25", "ah"],
                       format_func={"1x2": "Match result (1X2)", "ou25": "Over/Under 2.5",
                                    "ah": "Asian handicap"}.get)
    model_opts = {"1x2": ["blend", "ens", "gbm", "pxg", "pois", "elo", "pi", "mkt"],
                  "ou25": ["oublend", "ouens", "pxg", "pois"], "ah": ["pxg", "pois"]}[mkt]
    model = c2.selectbox("Probability model", model_opts, format_func=lambda m: MODEL_NAMES.get(m, m))
    prices = [p for p in ("max", "b365", "ps", "avg", "psc", "maxc", "avgc", "b365c")
              if D.get(f"{p}_{'h' if mkt == '1x2' else 'o25' if mkt == 'ou25' else 'ahh'}",
                       pd.Series(dtype=float)).notna().any()]
    price = c3.selectbox("Bet at price", prices,
                         format_func=lambda p: {"max": "Best price (market max)", "b365": "Bet365",
                                                "ps": "Pinnacle", "avg": "Market average"}.get(p, p))
    c4, c5, c6 = st.columns(3)
    edge = c4.slider("Minimum edge (EV)", 0.0, 0.20, 0.03, 0.005, format="%.3f")
    max_edge = c5.slider("Ignore edges above (stale-price filter)", 0.05, 1.0, 1.0, 0.01)
    odds_rng = c6.slider("Odds range", 1.01, 15.0, (1.3, 8.0 if mkt == "1x2" else 4.0), 0.05)
    c7, c8, c9 = st.columns(3)
    tier_pick = c7.multiselect("Tiers", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    lg_opts = [c for c in all_leagues if L.get(c).tier in tier_pick]
    lg_pick = c8.multiselect("Leagues (empty = all in chosen tiers)", lg_opts, format_func=league_label)
    outcomes = ("h", "d", "a")
    if mkt == "1x2":
        outcomes = tuple(c9.multiselect("Outcomes", ["h", "d", "a"], default=["h", "d", "a"],
                                        format_func={"h": "Home", "d": "Draw", "a": "Away"}.get))
    c10, c11, c12 = st.columns(3)
    staking = c10.radio("Staking", ["flat", "kelly"], horizontal=True)
    kf = c11.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05, disabled=staking == "flat")
    yrs = c12.slider("Seasons from", 2004, int(D["date"].dt.year.max()), (2016, int(D["date"].dt.year.max())))
    s = bt.Strategy(market=mkt, model=model, price=price, min_edge=edge, max_edge=max_edge,
                    min_odds=odds_rng[0], max_odds=odds_rng[1], outcomes=outcomes, staking=staking,
                    kelly_fraction=kf, leagues=tuple(lg_pick or lg_opts))
    sub = D[(D["date"] >= f"{yrs[0]}-07-01") & (D["date"] < f"{yrs[1] + 1}-07-01") & eligible(D)]
    bets = bt.run(sub, s)
    if bets.empty:
        st.warning("No bets match this rule.")
        st.stop()
    sm = bt.summarise(bets)
    m = st.columns(6)
    m[0].metric("Bets", f"{sm['bets']:,}")
    m[1].metric("ROI (yield)", f"{sm['roi']:+.2%}", f"95%: {sm['roi_lo']:+.1%} to {sm['roi_hi']:+.1%}",
                delta_color="off")
    m[2].metric("p-value (ROI > 0)", f"{sm['p_value']:.3f}")
    m[3].metric("Hit rate", f"{sm['hit_rate']:.1%}")
    m[4].metric("Avg odds", f"{sm['avg_odds']:.2f}")
    if staking == "kelly":
        m[5].metric("Bankroll (start 100)", f"{sm['final_bank']:.1f}")
    else:
        m[5].metric("Profit (units)", f"{sm['profit_units']:+.1f}")
    fig = go.Figure()
    y = bets["bank"] if staking == "kelly" else bets["profit"].cumsum()
    fig.add_trace(go.Scatter(x=bets["date"], y=y, mode="lines", line=dict(color=PALETTE[0], width=2),
                             name="Bankroll" if staking == "kelly" else "Cumulative units"))
    st.plotly_chart(fig_layout(fig, 380, "Bankroll" if staking == "kelly" else "Units"),
                    use_container_width=True)
    if "clv" in bets and bets["clv"].notna().any():
        st.info(f"Average closing line value: {bets['clv'].mean():+.2%} "
                f"(beat the close on {(bets['clv'] > 0).mean():.0%} of bets).")
    a, b = st.columns(2)
    a.subheader("By league")
    bl = bt.by(bets, "league").sort_values("bets", ascending=False)
    bl["league"] = bl["league"].map(league_label)
    a.dataframe(bl.style.format({"roi": "{:+.2%}", "hit_rate": "{:.1%}", "avg_odds": "{:.2f}",
                                 "profit_units": "{:+.1f}"}), hide_index=True, use_container_width=True)
    b.subheader("By season")
    bets["year"] = np.where(bets["date"].dt.month >= 7, bets["date"].dt.year, bets["date"].dt.year - 1)
    b.dataframe(bt.by(bets, "year").style.format({"roi": "{:+.2%}", "hit_rate": "{:.1%}",
                                                   "avg_odds": "{:.2f}", "profit_units": "{:+.1f}"}),
                hide_index=True, use_container_width=True)
    with st.expander("ROI by claimed edge (stale-price check)"):
        bets["band"] = pd.cut(bets["ev"], [0, 0.03, 0.05, 0.08, 0.12, 0.2, 10]).astype(str)
        st.dataframe(bt.by(bets, "band"), hide_index=True)
    with st.expander("All bets"):
        st.dataframe(bets, hide_index=True)
        st.download_button("Download bets CSV", bets.to_csv(index=False), "bets.csv", "text/csv")
    st.caption("Strategy: " + s.label())


# ------------------------------------------------------------------ accuracy
elif page == "Model accuracy":
    st.title("Model accuracy vs the market")
    models = st.multiselect("Models", ["mkt", "elo", "pi", "pois", "pxg", "ens", "gbm", "blend"],
                            default=["mkt", "pxg", "ens", "gbm", "blend"],
                            format_func=lambda m: MODEL_NAMES[m])
    c1, c2 = st.columns(2)
    lgs = c1.multiselect("Leagues (empty = all)", all_leagues, format_func=league_label)
    yrs = c2.slider("Seasons", 2005, int(D["date"].dt.year.max()), (2016, int(D["date"].dt.year.max())))
    e = D[D["ftr"].notna() & eligible(D) & (D["date"] >= f"{yrs[0]}-07-01")
          & (D["date"] < f"{yrs[1] + 1}-07-01")]
    if lgs:
        e = e[e["league"].isin(lgs)]
    if not models:
        st.stop()
    tbl = ev.score_table(e, models)
    tbl["model"] = tbl["model"].map(MODEL_NAMES)
    st.dataframe(tbl.style.format({"rps": "{:.4f}", "logloss": "{:.4f}", "brier": "{:.4f}",
                                   "accuracy": "{:.1%}"}), hide_index=True, use_container_width=True)
    bl = ev.score_table(e, models, by=["league"]).pivot(index="league", columns="model", values="rps")
    if "mkt" in bl:
        for m_ in models:
            if m_ != "mkt":
                bl[f"{m_} - mkt"] = bl[m_] - bl["mkt"]
    bl.index = bl.index.map(league_label)
    st.subheader("RPS by league (lower is better)")
    st.dataframe(bl.style.format("{:.4f}"), use_container_width=True)
    st.subheader("Calibration")
    oc = st.radio("Outcome", ["h", "d", "a"], horizontal=True,
                  format_func={"h": "Home win", "d": "Draw", "a": "Away win"}.get)
    y = ev.onehot(e["ftr"])[:, "hda".index(oc)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color="gray", width=1)))
    for i, m_ in enumerate(models):
        cal = ev.calibration(e[f"{m_}_p{oc}"].to_numpy(), y)
        cal = cal[cal["n"] >= 30]
        fig.add_trace(go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers",
                                 name=MODEL_NAMES[m_], line=dict(color=PALETTE[i % 8], width=2)))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Observed", hovermode="closest")
    st.plotly_chart(fig_layout(fig, 420), use_container_width=True)


# ------------------------------------------------------------------ predictor
elif page == "Match predictor":
    st.title("Match predictor")
    engine, models = load_engine_models()
    rt = engine.ratings_table()
    last = rt["last_match"].max()
    rt = rt[rt["last_match"] >= last - pd.Timedelta(days=400)]
    c1, c2, c3 = st.columns(3)
    opts = [c for c in all_leagues if (rt["league"] == c).any()]
    lg = c1.selectbox("League", opts, index=opts.index("E0") if "E0" in opts else 0,
                      format_func=league_label)
    teams = sorted(rt.loc[rt["league"] == lg, "team"])
    home = c2.selectbox("Home", teams, index=0)
    away = c3.selectbox("Away", teams, index=min(1, len(teams) - 1))
    st.caption("Team news: shift expected goals for injuries, suspensions or rotation.")
    a1, a2, a3, a4 = st.columns(4)
    ha = a1.slider(f"{home} attack", -0.4, 0.4, 0.0, 0.01, help="-0.10 = 10% fewer goals scored")
    hd = a2.slider(f"{home} defence", -0.4, 0.4, 0.0, 0.01, help="+0.10 = concede 10% more")
    aa = a3.slider(f"{away} attack", -0.4, 0.4, 0.0, 0.01)
    ad = a4.slider(f"{away} defence", -0.4, 0.4, 0.0, 0.01)
    adj = [x for x in (
        {"team": home, "attack": ha, "defence": hd, "note": "manual"},
        {"team": away, "attack": aa, "defence": ad, "note": "manual"}) if x["attack"] or x["defence"]]
    if home == away:
        st.warning("Pick two different teams.")
        st.stop()
    row = pipeline.what_if(engine, models, lg, home, away, adjustments=adj or None)
    probs = pd.DataFrame({
        MODEL_NAMES[m_]: [row[f"{m_}_p{o}"].iloc[0] for o in "hda"]
        for m_ in ("elo", "pi", "pois", "pxg", "ens", "gbm") if f"{m_}_ph" in row
    }, index=["Home", "Draw", "Away"]).T
    probs["Fair odds H/D/A"] = probs.apply(lambda r: " / ".join(f"{1 / v:.2f}" for v in r), axis=1)
    st.dataframe(probs.style.format({"Home": "{:.1%}", "Draw": "{:.1%}", "Away": "{:.1%}"}),
                 use_container_width=True)
    lh, la = float(row["pxg_lh"].iloc[0]), float(row["pxg_la"].iloc[0])
    m = score_matrix([lh], [la], engine.rho)
    c1, c2 = st.columns([3, 2])
    fig = go.Figure(go.Heatmap(z=m[0, :6, :6] * 100, x=list(range(6)), y=list(range(6)),
                               colorscale="Blues", text=np.round(m[0, :6, :6] * 100, 1),
                               texttemplate="%{text}%", showscale=False))
    fig.update_layout(xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
                      title=f"Expected goals {lh:.2f} - {la:.2f}")
    c1.plotly_chart(fig_layout(fig, 420), use_container_width=True)
    diffs, pgd = goal_diff_dist(m)
    lines = np.arange(-2.5, 2.75, 0.5)
    ah = pd.DataFrame({"Home line": lines, "P(home covers)": [
        float(pgd[0][(diffs + ln) > 0].sum() / max(1e-9, 1 - pgd[0][(diffs + ln) == 0].sum()))
        for ln in lines]})
    ah["Fair odds"] = 1 / ah["P(home covers)"]
    c2.metric("P(over 2.5)", f"{float(over_prob(m)[0]):.1%}")
    c2.metric("P(both teams score)", f"{float(1 - m[0, 0, :].sum() - m[0, :, 0].sum() + m[0, 0, 0]):.1%}")
    c2.dataframe(ah.style.format({"Home line": "{:+.1f}", "P(home covers)": "{:.1%}",
                                  "Fair odds": "{:.2f}"}), hide_index=True)


# ------------------------------------------------------------------ upcoming
elif page == "Upcoming & value":
    st.title("Upcoming fixtures")
    fx = pipeline.upcoming(D)
    if fx.empty:
        st.info("No fixtures loaded. Run `footy fetch` on a machine that can reach football-data.co.uk "
                "(it downloads fixtures.csv with current odds), then `footy build`.")
        st.stop()
    min_ev = st.slider("Flag value at EV of at least", 0.0, 0.15, 0.03, 0.005)
    show = fx[["date", "league", "home", "away", "blend_ph", "blend_pd", "blend_pa", "pxg_po25",
               "best_h", "best_d", "best_a", "ev_h", "ev_d", "ev_a"]].copy()
    show["value"] = show[["ev_h", "ev_d", "ev_a"]].max(axis=1)
    show = show.sort_values(["date", "league"])
    st.dataframe(show.style.format({c: "{:.1%}" for c in ("blend_ph", "blend_pd", "blend_pa",
                                                          "pxg_po25")}
                                   | {c: "{:+.1%}" for c in ("ev_h", "ev_d", "ev_a", "value")})
                 .highlight_between(subset=["value"], left=min_ev, right=10, color="#cde2fb"),
                 hide_index=True, use_container_width=True)
    if "adjusted" in fx and fx["adjusted"].str.len().gt(0).any():
        st.caption("Team-news adjustments applied: " +
                   "; ".join(sorted(set(fx.loc[fx['adjusted'] != '', 'adjusted']))))


# ------------------------------------------------------------------ ratings
elif page == "Ratings":
    st.title("Current ratings")
    engine, _ = load_engine_models()
    rt = engine.ratings_table()
    last = rt["last_match"].max()
    rt = rt[rt["last_match"] >= last - pd.Timedelta(days=400)]
    opts = [c for c in all_leagues if (rt["league"] == c).any()]
    lg = st.selectbox("League", opts, index=opts.index("E0") if "E0" in opts else 0,
                      format_func=league_label)
    t = rt[rt["league"] == lg].sort_values("elo", ascending=False)
    t["attack"] = np.exp(t["pxg_att"]) - 1
    t["defence"] = np.exp(t["pxg_def"]) - 1
    fig = go.Figure(go.Bar(x=t["elo"], y=t["team"], orientation="h", marker_color=PALETTE[0]))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[t["elo"].min() - 50, t["elo"].max() + 20])
    st.plotly_chart(fig_layout(fig, max(300, 24 * len(t)), ""), use_container_width=True)
    st.dataframe(t[["team", "elo", "pi_home", "pi_away", "attack", "defence", "matches", "last_match"]]
                 .style.format({"elo": "{:.0f}", "pi_home": "{:+.2f}", "pi_away": "{:+.2f}",
                                "attack": "{:+.0%}", "defence": "{:+.0%}"}),
                 hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ team news
elif page == "Team news":
    st.title("Team news adjustments")
    st.write("Ratings only see results and shots. Injuries, suspensions, rotation and manager changes "
             "are entered by hand in `config/adjustments.yaml` and applied to upcoming predictions.")
    adj = pipeline.load_adjustments()
    if adj:
        st.dataframe(pd.DataFrame(adj), hide_index=True, use_container_width=True)
    else:
        st.info("No adjustments set.")
    st.code(pipeline.ADJUSTMENTS.read_text(encoding="utf-8") if pipeline.ADJUSTMENTS.exists() else "", language="yaml")
    st.markdown("See `docs/RESEARCH.md` for team-news sources (accounts and sites) and rules of thumb "
                "for sizing an adjustment.")
