"""Command line interface: ``footy <command>``.

    footy fetch      download results + odds (football-data.co.uk, falls back to GitHub mirror)
    footy build      run ratings + walk-forward models, save predictions
    footy report     backtests + accuracy tables + static HTML dashboard
    footy backtest   run one strategy and print its summary
    footy upcoming   refresh results + fixtures, probabilities and research picks, paper-trading ledger
    footy predict    same as `upcoming --no-fetch`
    footy matchup    predict any match-up from the latest ratings
    footy tune       grid-search rating hyper-parameters on the development window
    footy confirm    re-run the pre-registered market-surprise test (docs/PREREGISTRATION.md)
    footy forward    score the frozen rules on matches after 2026-09-03 (forward test)
    footy clv        closing-line value of the frozen rules on 2016-2026
    footy push-picks send the latest picks to the live odds watcher (Raspberry Pi)
    footy all        fetch -> build -> report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import leagues


def _fetch(args) -> None:
    from .data import sources, store

    codes = args.leagues.split(",") if args.leagues else None
    kw = {"league_codes": codes} if args.source != "mirror" else {}
    matches, used = sources.load_matches(store.RAW_DIR, source=args.source, **kw)
    if codes:
        matches = matches[matches["league"].isin(codes)]
    store.save_store(matches, used)
    print(f"saved {len(matches):,} matches from {used} "
          f"({matches['date'].min().date()} -> {matches['date'].max().date()})")
    if not args.no_fixtures:
        fx = sources.fetch_fixtures(store.RAW_DIR)
        if not fx.empty:
            fx.to_parquet(store.FIXTURES, index=False)
            print(f"saved {len(fx)} upcoming fixtures")
        else:
            print("no fixtures downloaded (football-data.co.uk unreachable?)")


def _build(args) -> None:
    from .data import store
    from .pipeline import build

    matches = store.load_store()
    fixtures = pd.read_parquet(store.FIXTURES) if store.FIXTURES.exists() else None
    d = build(matches, fixtures, use_gbm=not args.no_gbm)
    print(f"predictions saved for {len(d):,} matches")


def _report(args) -> None:
    from .report import make_report

    out = make_report()
    print(f"report written to {out}")


def _backtest(args) -> None:
    from . import backtest as bt
    from .pipeline import load_predictions

    d = load_predictions()
    s = bt.Strategy(market=args.market, model=args.model, price=args.price,
                    min_edge=args.edge, min_odds=args.min_odds, max_odds=args.max_odds,
                    staking=args.staking, kelly_fraction=args.kelly,
                    leagues=tuple(args.leagues.split(",")) if args.leagues else None)
    if args.start:
        d = d[d["date"] >= args.start]
    if args.end:
        d = d[d["date"] < args.end]
    bets = bt.run(d, s)
    print(s.label())
    print(json.dumps(bt.summarise(bets), indent=2, default=float))
    if not bets.empty:
        print(bt.by(bets, "league").sort_values("bets", ascending=False).to_string(index=False))
        if args.out:
            bets.to_csv(args.out, index=False)


def _upcoming(args) -> None:
    from . import upcoming

    meta = upcoming.run(fetch=not args.no_fetch)
    print(f"{meta['fixtures']} fixtures scored (results through {meta['results_through']}): "
          f"{meta['bets']} bets, {meta['watch']} on the watchlist")
    pf = upcoming.OUT_DIR / "picks.csv"
    picks = pd.read_csv(pf, parse_dates=["date"]) if pf.exists() else pd.DataFrame()
    for status, title in (("Bet", "BETS"), ("Wait", "WATCHLIST (needs a better price)")):
        g = picks[picks["status"] == status] if len(picks) else picks
        if g.empty:
            continue
        print(f"\n{title}")
        for _, r in g.iterrows():
            when = f"{r['date']:%a %d %b} {'' if pd.isna(r['time']) else r['time']}".strip()
            print(f"  {when} · {r['league_name']} · {r['home']} - {r['away']}")
            print(f"    {r['selection']}  [{r['rule_name']}]")
            print(f"    our chance {r['our_prob']:.0%} vs market {r['market_prob']:.0%} · "
                  f"break-even {r['breakeven_odds']:.2f} · bet at {r['min_odds']:.2f}+ "
                  f"(Betfair {r['min_odds_exchange']:.2f}+)")
            print(f"    best price now {r['best_odds']:.2f} at {r['best_book']} · edge {r['edge_at_best']:+.1%}")
    led = meta["ledger"]
    for rule in ("F1", "F2"):
        r = led.get(rule)
        if r:
            print(f"ledger {rule}: settled {r.get('settled', 0)}, pending {r.get('pending', 0)}"
                  + (f", ROI {r['roi']:+.1%}" if "roi" in r else "")
                  + (f", avg CLV {r['avg_clv']:+.1%} over {r['clv_bets']}" if "avg_clv" in r else ""))
    print(f"\nopen {upcoming.OUT_DIR / 'index.html'}")


def _matchup(args) -> None:
    from .pipeline import load_adjustments, load_engine, load_models, what_if

    row = what_if(load_engine(), load_models(), args.league, args.home, args.away,
                  adjustments=load_adjustments())
    print(f"{args.home} vs {args.away} ({leagues.league_name(args.league)})")
    print(f"{'model':<6} {'home':>6} {'draw':>6} {'away':>6}   fair odds")
    for m in ("elo", "pi", "pxg", "ens", "gbm"):
        if f"{m}_ph" in row:
            p = [float(row[f"{m}_p{o}"].iloc[0]) for o in ("h", "d", "a")]
            print(f"{m:<6} {p[0]:>6.1%} {p[1]:>6.1%} {p[2]:>6.1%}   "
                  + " / ".join(f"{1 / x:.2f}" for x in p))
    print(f"expected goals {float(row['pxg_lh'].iloc[0]):.2f} - {float(row['pxg_la'].iloc[0]):.2f}, "
          f"P(over 2.5) {float(row['pxg_po25'].iloc[0]):.1%}")


def _tune(args) -> None:
    from .data import store
    from .tune import tune

    r = tune(store.load_store(), which=args.models.split(",") if args.models else None)
    print(r.groupby("model").head(3).to_string(index=False))


def _confirm(args) -> None:
    from .research import confirm

    r = confirm.run(args.window)
    print(json.dumps({k: r[k] for k in ("H1", "H2", "H3", "H4", "holm_adjusted_p")},
                     indent=2, default=float))


def _push_picks(args) -> None:
    from .upcoming import OUT_DIR
    from .watch.__main__ import main as watch_main

    path = args.file or str(OUT_DIR / "picks.json")
    watch_main(["push", path, "--url", args.url, "--token", args.token])


def _clv(args) -> None:
    from .research import clv

    r = clv.run()
    for k, v in r.items():
        if k != "by_season":
            print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                 for kk, vv in v.items()}, default=str))
    print(f"details in {clv.OUT_DIR / 'results.json'}")


def _forward(args) -> None:
    from .research import forward

    if args.freeze:
        if forward.FROZEN.exists() and not args.force:
            print(f"{forward.FROZEN} already exists; frozen rules must not change (use --force)")
            return
        forward.freeze()
    r = forward.evaluate()
    print(json.dumps(r, indent=2, default=float))
    print(f"bets written to {forward.OUT_DIR}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="footy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download data")
    f.add_argument("--source", default="auto", choices=["auto", "football-data", "mirror"])
    f.add_argument("--leagues", help="comma separated league codes (default: all)")
    f.add_argument("--no-fixtures", action="store_true")
    f.set_defaults(func=_fetch)

    b = sub.add_parser("build", help="ratings + walk-forward predictions")
    b.add_argument("--no-gbm", action="store_true", help="skip gradient boosting (faster)")
    b.set_defaults(func=_build)

    r = sub.add_parser("report", help="accuracy + backtest report and HTML dashboard")
    r.set_defaults(func=_report)

    t = sub.add_parser("backtest", help="backtest one strategy")
    t.add_argument("--market", default="1x2", choices=["1x2", "ou25", "ah"])
    t.add_argument("--model", default="blend")
    t.add_argument("--price", default="max")
    t.add_argument("--edge", type=float, default=0.03)
    t.add_argument("--min-odds", type=float, default=1.3)
    t.add_argument("--max-odds", type=float, default=8.0)
    t.add_argument("--staking", default="flat", choices=["flat", "kelly"])
    t.add_argument("--kelly", type=float, default=0.25)
    t.add_argument("--leagues")
    t.add_argument("--start")
    t.add_argument("--end")
    t.add_argument("--out", help="write bets to CSV")
    t.set_defaults(func=_backtest)

    up = sub.add_parser("upcoming", help="refresh results, pull fixtures, score them with the "
                                         "models and research rules, update the paper-trading ledger")
    up.add_argument("--no-fetch", action="store_true", help="use stored results and fixtures")
    up.set_defaults(func=_upcoming)

    pr = sub.add_parser("predict", help="same as `upcoming --no-fetch`")
    pr.set_defaults(func=lambda x: _upcoming(argparse.Namespace(no_fetch=True)))

    mu = sub.add_parser("matchup", help="predict any match-up")
    mu.add_argument("league", choices=sorted(leagues.LEAGUES))
    mu.add_argument("home")
    mu.add_argument("away")
    mu.set_defaults(func=_matchup)

    tu = sub.add_parser("tune", help="tune rating hyper-parameters")
    tu.add_argument("--models", help="subset, e.g. elo,pi")
    tu.set_defaults(func=_tune)

    c = sub.add_parser("confirm", help="re-run the pre-registered market-surprise test")
    c.add_argument("--window", default="holdout", choices=["dev", "holdout"])
    c.set_defaults(func=_confirm)

    pp = sub.add_parser("push-picks", help="send the latest picks to the odds watcher (e.g. on a Raspberry Pi)")
    pp.add_argument("url", help="watcher address, e.g. http://raspberrypi.local:8080")
    pp.add_argument("--token", default="", help="shared secret if the watcher has one")
    pp.add_argument("--file", help="default: reports/upcoming/picks.json")
    pp.set_defaults(func=_push_picks)

    cl = sub.add_parser("clv", help="closing-line value of the frozen rules (needs closing odds "
                                    "from football-data.co.uk)")
    cl.set_defaults(func=_clv)

    fw = sub.add_parser("forward", help="score frozen rules on matches after the freeze date")
    fw.add_argument("--freeze", action="store_true", help="(re)fit and write frozen rules")
    fw.add_argument("--force", action="store_true")
    fw.set_defaults(func=_forward)

    a = sub.add_parser("all", help="fetch -> build -> report")
    a.add_argument("--source", default="auto", choices=["auto", "football-data", "mirror"])
    a.add_argument("--leagues")
    a.add_argument("--no-fixtures", action="store_true")
    a.add_argument("--no-gbm", action="store_true")
    a.set_defaults(func=lambda x: (_fetch(x), _build(x), _report(x)))

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
