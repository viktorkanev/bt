"""python -m footy.watch {run,check,feeds,scan,sports,polymarket,oddspapi,notify-test,init-config,push,paper,paper-reset}"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

from . import config as C


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="footy-watch", description="Live odds watcher")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("run", "poll, alert and serve the status page"),
                      ("check", "poll every provider once and print what it found (no notifications)"),
                      ("sports", "list the-odds-api sport keys (all sports, with their group)"),
                      ("polymarket", "test the Polymarket feed: which Odds API matches it also lists, and prices"),
                      ("notify-test", "send a test notification"),
                      ("paper", "paper-trading results by place and strategy"),
                      ("paper-reset", "delete all paper bets and start the imaginary bankrolls again")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--config", default="watch.toml")
        if name == "paper-reset":
            p.add_argument("--yes", action="store_true", help="really delete")
    p = sub.add_parser("oddspapi", help="test the OddsPapi feed: requests left, leagues, markets, sample prices")
    p.add_argument("--config", default="watch.toml")
    p.add_argument("--sport", default="soccer", help="soccer, baseball, basketball, icehockey, tennis, volleyball, "
                                                    "rugbyunion, rugbyleague, americanfootball")
    p.add_argument("--save", default="", help="folder to save the raw responses in (send them if parsing fails)")
    p.add_argument("--bookmakers", action="store_true", help="also list bookmaker slugs (1 request)")
    p = sub.add_parser("feeds", help="compare what each odds feed delivered (from the feed recorder)")
    p.add_argument("--config", default="watch.toml")
    p.add_argument("--days", type=float, default=7.0)
    p = sub.add_parser("scan", help="run the prediction-market strategies once and print what they find")
    p.add_argument("--config", default="watch.toml")
    p.add_argument("--strategy", choices=["consistency", "weather"], default=None)
    p.add_argument("--record", action="store_true", help="also record the positions (no notifications)")
    p = sub.add_parser("init-config", help="write an example watch.toml")
    p.add_argument("path", nargs="?", default="watch.toml")
    p = sub.add_parser("push", help="send picks.json (from `footy upcoming`) to a running watcher")
    p.add_argument("picks", help="path to reports/upcoming/picks.json")
    p.add_argument("--url", required=True, help="e.g. http://raspberrypi.local:8080")
    p.add_argument("--token", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "init-config":
        src = Path(__file__).resolve().parents[3] / "deploy" / "raspberry-pi" / "watch.example.toml"
        if Path(args.path).exists():
            print(f"{args.path} exists; not overwriting")
            return 1
        shutil.copy(src, args.path)
        print(f"wrote {args.path}; add your API keys and notification settings")
        return 0
    if args.cmd == "push":
        data = Path(args.picks).read_bytes()
        req = urllib.request.Request(args.url.rstrip("/") + "/api/picks", data=data,
                                     headers={"Content-Type": "application/json", "X-Token": args.token})
        with urllib.request.urlopen(req, timeout=30) as r:
            print(r.read().decode())
        return 0

    cfg = C.load(args.config if Path(args.config).exists() else None)
    if args.cmd == "sports":
        from .providers import OddsAPIProvider
        prov = OddsAPIProvider(cfg["oddsapi"])
        for s in prov.list_sports():
            mapped = "watched" if s["key"] in cfg["oddsapi"]["sports"] or s.get("group") in cfg["oddsapi"]["groups"] \
                else ""
            print(f"{s.get('group', ''):<16} {s['key']:<45} {s.get('title', ''):<35} "
                  f"{'active' if s.get('active') else '':<7} {mapped}")
        print(f"credits remaining: {prov.remaining}")
        return 0
    if args.cmd == "feeds":
        from .core import utcnow
        from .paper import PaperBook
        from .recorder import Recorder, minutes_between, monthly_requests
        rec = Recorder(cfg)
        paper = PaperBook(cfg["server"]["db"], cfg) if cfg["paper"].get("enabled") else None
        rep = rec.report(utcnow(), args.days, paper)
        print(f"last {args.days:g} days, recorder file {rep['db_mb']} MB")
        for name, d in sorted(rep["providers"].items()):
            if name.startswith("pm:"):
                print(f"\n{name}: {d['polls']} scans, {d['avg_seconds']} s each, {d['errors']} errors")
                continue
            print(f"\n{name}: {d['polls']} checks ({d['errors']} errors), {d['avg_seconds']} s each")
            print(f"  matches {d['matches_total']} ({', '.join(f'{k} {v}' for k, v in d['matches'].items())}), "
                  f"only here {d['only_here']}")
            print(f"  books {len(d['books'])}: {', '.join(b['book'] for b in d['books'][:12])}")
            print(f"  only from it: {', '.join(d['exclusive_books']) or '-'}; "
                  f"price changes per selection-hour {d['changes_per_selection_hour']}")
            for other, x in d["lead"].items():
                print(f"  vs {other}: {x['median_minutes_ahead']:+.1f} min ahead (median of {x['samples']})")
            if d.get("paper_bets_only_here") is not None:
                print(f"  paper bets only possible through it: {d['paper_bets_only_here']}, expected "
                      f"{d['paper_ev_only_here']}, profit {d['paper_profit_only_here']}")
            if d.get("requests_per_poll"):
                r = d["requests_per_poll"]
                print(f"  {r:g} requests per check -> a month: " +
                      ", ".join(f"every {m} min {monthly_requests(r, m):,}" for m in (5, 15, 30, 60)))
                print("  allowance -> check every: " +
                      ", ".join(f"{a:,}/mo {minutes_between(r, a, 10):.0f} min" for a in (250, 5000, 20000, 100000)))
        return 0
    if args.cmd == "scan":
        from .core import utcnow
        from .pm import build_scanners
        for name in ("consistency", "weather"):
            if args.strategy in (None, name):
                cfg[name]["enabled"] = True
            elif args.strategy:
                cfg[name]["enabled"] = False
        now = utcnow()
        book = None
        if args.record:
            from .pm.ledger import PMBook
            book = PMBook(cfg["server"]["db"], cfg)
        for sc in build_scanners(cfg):
            import time as _t
            t0 = _t.time()
            sigs = sc.run(now)
            print(f"\n{sc.name}: {sc.last_count} events scanned in {_t.time() - t0:.0f} s, {len(sigs)} opportunities")
            for sg in sigs:
                print(f"  {100 * sg.edge:+6.1f}%  {(100 * sg.apy if sg.apy else 0):6.0f}%/yr  {sg.title[:70]}")
                print(f"           {sg.detail[:200]}")
                if sg.link:
                    print(f"           {sg.link}")
                if book:
                    book.place(sg, now)
        return 0
    if args.cmd == "oddspapi":
        from collections import Counter
        from .core import EventIndex, utcnow
        from .oddspapi import OddsPapiProvider
        prov = OddsPapiProvider(cfg["oddspapi"])
        if args.save:
            prov.dump_dir = Path(args.save)
        now = utcnow()
        print(f"requests used this month: {prov.used} of {prov.monthly}")
        ids = prov.sport_ids()
        print("sports found:", ", ".join(f"{k}={v}" for k, v in sorted(ids.items())) or "none")
        if args.bookmakers:
            bk = prov.get("/bookmakers")
            items = bk if isinstance(bk, list) else []
            print("bookmakers:", ", ".join(str(b.get("slug") or b.get("bookmaker") or b) for b in items[:400]))
        sid = ids.get(args.sport)
        if sid is None:
            print(f"no sport called {args.sport}")
            return 1
        mmap = prov.market_map(sid)
        total = prov.cache["markets"][str(sid)].get("total")
        print(f"markets understood: {len(mmap)} of {total}: " +
              ", ".join(f"{k} {v}" for k, v in Counter(m["market"] for m in mmap.values()).items()))
        fx = prov.fixtures(args.sport, sid, now)
        tids = prov.tournaments(args.sport, fx, now)
        names = {f["tournamentId"]: f["tournament"] for f in fx.values()}
        per = Counter(f["tournamentId"] for f in fx.values())
        print(f"{len(fx)} upcoming fixtures; tournaments polled ({'configured' if prov.sports.get(args.sport) else 'busiest'}):")
        for t in tids[:25]:
            print(f"  {t:>8}  {per.get(t, 0):>3} fixtures  {names.get(t, '')}")
        if tids:
            data = []  # one request per bookmaker: the API takes exactly one
            for bk in prov.bookmakers:
                part = prov.get("/odds-by-tournaments", tournamentIds=",".join(tids[:prov.per_request]),
                                bookmaker=bk, verbosity=prov.verbosity)
                data += part if isinstance(part, list) else list(part.values()) if isinstance(part, dict) else []
            idx = EventIndex()
            qs = OddsPapiProvider.parse(data, fx, mmap, idx, args.sport, now, prov.book_names, prov.lookahead_h)
            print(f"\n{len(idx.events)} matches, {len(qs)} prices: " +
                  ", ".join(f"{k} {v}" for k, v in Counter((q.book, q.market) for q in qs).most_common(12)))
            for ev in sorted(idx.events.values(), key=lambda e: e.kickoff)[:10]:
                win = sorted((q for q in qs if q.event == ev.id and q.market in ("1x2", "ml")),
                             key=lambda q: (q.book, q.outcome))
                print(f"  {ev.kickoff:%a %d %b %H:%M} {ev.home} v {ev.away}: " +
                      "  ".join(f"{q.book} {q.outcome} {q.price:.2f}" for q in win[:6]))
        print(f"\nrequests used this month: {prov.used} of {prov.monthly}")
        return 0
    if args.cmd == "polymarket":
        from .core import EventIndex, parse_time, utcnow
        from .providers import OddsAPIProvider, PolymarketProvider, oddsapi_sport_kind
        prov = PolymarketProvider(cfg["polymarket"])
        idx, now = EventIndex(), utcnow()
        if cfg["oddsapi"].get("enabled") and cfg["oddsapi"].get("api_key"):
            odds = OddsAPIProvider(cfg["oddsapi"])
            odds.refresh_events(force=True)  # match lists are free: no odds credits used
            for sport, evs in odds._events.items():
                for e in evs:
                    ko = parse_time(e["commence_time"])
                    if now <= ko <= now + timedelta(hours=prov.lookahead_h):
                        idx.add("oddsapi", e["id"], e["home_team"], e["away_team"], ko, odds.sports.get(sport),
                                sport=oddsapi_sport_kind(sport))
            print(f"Odds API: {len(idx.events)} matches in the next {prov.lookahead_h:g} h")
        else:
            prov.only_known = False
            print("Odds API off: showing every Polymarket football match")
        t0 = time.time()
        qs = prov.poll(idx, now)
        found = {q.event for q in qs}
        print(f"Polymarket: {prov.listed()} matches listed, {len(found)} of them priced, {len(qs)} prices "
              f"in {time.time() - t0:.1f}s" + (f" ({prov.last_error})" if prov.last_error else ""))
        try:
            codes = sorted({str(s.get("sport")) for s in prov.list_sports() if prov.kinds.get(str(s.get("sport")))})
            print("other sports it lists (codes for [polymarket] sports): " + ", ".join(codes))
        except Exception as e:  # noqa: BLE001
            print(f"sports list unavailable: {e}")
        for ev in sorted((idx.events[e] for e in found), key=lambda x: x.kickoff)[:20]:
            mine = sorted((q for q in qs if q.event == ev.id and q.market == "1x2"), key=lambda q: q.outcome)
            print(f"  {ev.kickoff:%a %d %b %H:%M}  {ev.home} v {ev.away}  " +
                  "  ".join(f"{q.outcome} {1 / q.price:.2f}" for q in mine))
        return 0
    if args.cmd in ("paper", "paper-reset"):
        from .paper import VENUES, PaperBook, summarize
        book = PaperBook(cfg["server"]["db"], cfg)
        if args.cmd == "paper-reset":
            if not args.yes:
                print("this deletes every paper bet; run again with --yes")
                return 1
            print(f"deleted {book.reset()} paper bets")
            return 0
        start, cur = float(cfg["paper"]["bankroll"]), cfg["paper"].get("currency", "")
        fmt = lambda x, pct=False: "-" if x is None else (f"{x:+.1%}" if pct else f"{x:+,.2f}")  # noqa: E731
        print(f"{'place':<18}{'strategy':<26}{'settled':>8}{'open':>6}{'profit':>11}{'ROI':>8}"
              f"{'avg EV':>8}{'avg CLV':>9}{'bankroll':>11}")
        for v in cfg["paper"].get("venues", list(VENUES)):
            for code, name in [("all", "All")] + book.strategies():
                s = summarize(book.bets(venue=v, strategy=code), start)
                if s["bets"] == 0 and code != "all":
                    continue
                bank = f"{cur}{s['bankroll']:,.2f}" if code == "all" else ""
                print(f"{VENUES.get(v, v) if code == 'all' else '':<18}{name[:25]:<26}{s['settled']:>8}{s['open']:>6}"
                      f"{fmt(s['profit']):>11}{fmt(s['roi'], True):>8}{fmt(s['avg_ev'], True):>8}"
                      f"{fmt(s['avg_clv'], True):>9}{bank:>11}")
        return 0
    if args.cmd == "notify-test":
        from .notify import Notifier
        Notifier(cfg["notify"]).send("footy-watch test", "If you can read this, notifications work.")
        return 0

    from .server import Watcher, serve
    w = Watcher(cfg)
    if not w.providers and not w.scanners:
        print("nothing enabled; set enabled = true for oddsapi, betfair, oddspapi, polymarket, file, "
              "consistency or weather in the config")
        return 1
    if args.cmd == "check":
        w.notifier.cfg = {"console": False}
        sent = w.poll_once()
        st = w.status()
        print(json.dumps(st, indent=2, default=str))
        for a in sent:
            ev = w.index.events[a.event_id]
            print(f"{a.kind:5} {a.match:<40} {a.selection(ev.home, ev.away):<40} {a.price:.2f} "
                  f"{a.book:<15} edge {a.edge:+.1%}")
        return 0
    serve(w)
    try:
        w.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
