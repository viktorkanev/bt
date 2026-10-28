"""python -m footy.watch {run,check,sports,notify-test,init-config,push}"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

from . import config as C


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="footy-watch", description="Live odds watcher")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("run", "poll, alert and serve the status page"),
                      ("check", "poll every provider once and print what it found (no notifications)"),
                      ("sports", "list the-odds-api soccer sport keys available to your key"),
                      ("notify-test", "send a test notification")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--config", default="watch.toml")
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
            mapped = cfg["oddsapi"]["sports"].get(s["key"], "")
            print(f"{s['key']:<45} {s.get('title', ''):<35} {'active' if s.get('active') else '':<7} {mapped}")
        print(f"credits remaining: {prov.remaining}")
        return 0
    if args.cmd == "notify-test":
        from .notify import Notifier
        Notifier(cfg["notify"]).send("footy-watch test", "If you can read this, notifications work.")
        return 0

    from .server import Watcher, serve
    w = Watcher(cfg)
    if not w.providers:
        print("no providers enabled; set enabled = true for oddsapi, betfair or file in the config")
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
