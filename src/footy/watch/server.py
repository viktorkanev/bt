"""The watcher service: polls providers, raises alerts, serves a status page.

Runs comfortably on a Raspberry Pi Zero 2 W (standard library only, ~30 MB RAM).
"""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import providers as P
from .core import EventIndex, Quote, utcnow
from .notify import Notifier, book_name, format_alert, local_time
from .rules import Alert, pick_alerts, value_alerts
from .state import State

log = logging.getLogger(__name__)


class Watcher:
    def __init__(self, cfg: dict, provider_list: list | None = None):
        self.cfg = cfg
        self.providers = provider_list if provider_list is not None else P.build(cfg)
        self.index = EventIndex()
        self.quotes: dict[tuple, Quote] = {}
        self.state = State(cfg["server"]["db"])
        self.notifier = Notifier(cfg["notify"])
        self.picks: list[dict] = []
        self.pick_status: list[dict] = []
        self.next_poll = {p.name: 0.0 for p in self.providers}
        self.last_poll: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.started = utcnow()
        self.lock = threading.Lock()
        self.load_picks()

    # ---------------------------------------------------------- picks
    def load_picks(self) -> None:
        f = Path(self.cfg["server"]["picks_file"])
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.picks = data.get("picks", data) if isinstance(data, dict) else data
                log.info("loaded %d research picks from %s", len(self.picks), f)
            except Exception as e:  # noqa: BLE001
                log.warning("could not read picks file %s: %s", f, e)

    def save_picks(self, payload: dict | list) -> int:
        f = Path(self.cfg["server"]["picks_file"])
        f.write_text(json.dumps(payload), encoding="utf-8")
        self.load_picks()
        return len(self.picks)

    # ----------------------------------------------------------- polling
    def poll_once(self, now=None) -> list[Alert]:
        """Poll every provider that is due, evaluate rules, send alerts."""
        now = now or utcnow()
        polled = False
        for prov in self.providers:
            if time.time() < self.next_poll.get(prov.name, 0):
                continue
            try:
                fresh = prov.poll(self.index, now)
                with self.lock:
                    for q in fresh:
                        self.quotes[(q.event, q.market, q.line, q.book, q.outcome)] = q
                self.last_poll[prov.name] = now.isoformat(timespec="seconds")
                self.errors[prov.name] = getattr(prov, "last_error", "")
                polled = True
                log.info("%s: %d quotes", prov.name, len(fresh))
            except Exception as e:  # noqa: BLE001
                self.errors[prov.name] = str(e)
                log.warning("%s poll failed: %s", prov.name, e)
            n_active = len(prov.active_sports(now)) if hasattr(prov, "active_sports") else 0
            self.next_poll[prov.name] = time.time() + prov.interval(n_active, now)
        if not polled:
            return []
        return self.evaluate(now)

    def evaluate(self, now) -> list[Alert]:
        a = self.cfg["alerts"]
        cutoff = now - timedelta(hours=3)
        with self.lock:
            for k in [k for k, q in self.quotes.items()
                      if (ev := self.index.events.get(q.event)) is None or ev.kickoff < cutoff]:
                self.quotes.pop(k)
            self.index.prune(cutoff)
            quotes = list(self.quotes.values())
        alerts: list[Alert] = []
        if a.get("value_alerts", True):
            alerts += value_alerts(self.index, quotes, self.cfg, now)
        if a.get("pick_alerts", True) and self.picks:
            pa, self.pick_status = pick_alerts(self.picks, self.index, quotes, self.cfg, now)
            alerts += pa
        sent = []
        tz = self.cfg["server"].get("timezone", "UTC")
        for al in sorted(alerts, key=lambda x: -x.edge):
            if self.state.should_send(al, now, a.get("cooldown_minutes", 240), a.get("reprice_step", 0.01)):
                ev = self.index.events[al.event_id]
                title, body = format_alert(al, ev.home, ev.away, tz)
                self.notifier.send(title, body, "high" if al.kind == "pick" else "default")
                self.state.record(al, now)
                sent.append(al)
        self.state.prune(now - timedelta(days=7))
        return sent

    def run_forever(self) -> None:
        loop = float(self.cfg["server"].get("loop_seconds", 5))
        while True:
            try:
                self.poll_once()
            except Exception as e:  # noqa: BLE001
                log.exception("loop error: %s", e)
            time.sleep(loop)

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        now = utcnow()
        provs = []
        for p in self.providers:
            provs.append({"name": p.name, "last_poll": self.last_poll.get(p.name),
                          "next_poll_in_s": max(0, int(self.next_poll.get(p.name, 0) - time.time())),
                          "error": self.errors.get(p.name, ""),
                          "credits_remaining": getattr(p, "remaining", None)})
        return {"now": now.isoformat(timespec="seconds"), "started": self.started.isoformat(timespec="seconds"),
                "providers": provs, "events": len(self.index.events), "quotes": len(self.quotes),
                "picks": len(self.picks), "alerts_sent": self.notifier.sent,
                "notify_error": self.notifier.last_error}


# ------------------------------------------------------------------- web
CSS = """
:root{--bg:#f4f6f5;--surface:#fff;--surface-2:#eef2f0;--ink:#13201a;--ink-2:#46544c;--muted:#6f7d75;--rule:#dbe3de;
--good:#0a7d0a;--good-bg:#e3f4e3;--wait:#8a5a00;--wait-bg:#fbf0d9;--bad:#b42828}
@media (prefers-color-scheme:dark){:root{--bg:#111513;--surface:#181d1a;--surface-2:#1f2522;--ink:#eef2ef;--ink-2:#b8c3bd;
--muted:#8b9890;--rule:#2b3330;--good:#5fd35f;--good-bg:#16301a;--wait:#f0b84a;--wait-bg:#33290f;--bad:#ef7070;color-scheme:dark}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
.wrap{max-width:980px;margin:0 auto;padding:16px max(14px,env(safe-area-inset-right,0px)) 40px max(14px,env(safe-area-inset-left,0px));display:grid;gap:14px}
h1{font-size:28px;margin:0}h2{font-size:19px;margin:8px 0 0}.muted{color:var(--muted)}.small{font-size:12.5px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:8px 12px;font-size:13px}
.chip b{font-size:16px}.err{color:var(--bad)}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-size:12px;color:var(--muted);background:var(--surface-2);font-weight:600}th.l,td.l{text-align:left}
tbody tr:last-child td{border-bottom:0}.pill{border-radius:999px;padding:2px 9px;font-weight:600;font-size:12.5px}
.ok{color:var(--good);background:var(--good-bg)}.wait{color:var(--wait);background:var(--wait-bg)}
"""


def _pct(x) -> str:
    return "–" if x is None else f"{x * 100:+.1f}%"


def render(w: Watcher) -> str:
    e = html.escape
    st = w.status()
    tz = w.cfg["server"].get("timezone", "UTC")
    chips = []
    for p in st["providers"]:
        cred = f" · {p['credits_remaining']} credits left" if p["credits_remaining"] is not None else ""
        err = f"<div class='err small'>{e(p['error'])}</div>" if p["error"] else ""
        last = local_time(datetime.fromisoformat(p["last_poll"]), tz)[-5:] if p["last_poll"] else "never"
        chips.append(f"<div class='chip'><b>{e(p['name'])}</b> · last {e(last)}"
                     f" · next in {p['next_poll_in_s']}s{cred}{err}</div>")
    chips.append(f"<div class='chip'><b>{st['events']}</b> matches · <b>{st['quotes']}</b> prices · "
                 f"<b>{st['picks']}</b> picks · <b>{st['alerts_sent']}</b> alerts sent</div>")
    if st["notify_error"]:
        chips.append(f"<div class='chip err'>Notification error: {e(st['notify_error'])}</div>")
    prow = []
    for row in w.pick_status:
        pk, b = row["pick"], row["best"]
        if not row["matched"]:
            live, status = "<span class='muted'>not found in odds feed</span>", ""
        elif b is None:
            live, status = "<span class='muted'>no price for this market</span>", ""
        else:
            line = "" if pk["market"] == "ou" else f" @ line {b['line'] if pk['side'] == 'home' else -b['line']:+.2f}"
            live = f"{b['price']:.2f} {e(book_name(b['book']))}{line}"
            ok = b["edge"] >= float(pk.get("min_edge", 0))
            status = (f"<span class='pill ok'>Bet · {_pct(b['edge'])}</span>" if ok else
                      f"<span class='pill wait'>Wait for {b['need']:.2f}+</span>")
        prow.append(f"<tr><td class='l'>{e(pk['home'])} – {e(pk['away'])}<div class='muted small'>"
                    f"{e(pk.get('league_name', ''))}</div></td><td class='l'>{e(pk.get('selection', ''))}</td>"
                    f"<td>{live}</td><td>{status}</td></tr>")
    since = utcnow() - timedelta(hours=24)
    arow = []
    for a in w.state.recent(since):
        when = local_time(datetime.fromisoformat(a["last_sent"]), tz)
        ko = local_time(datetime.fromisoformat(a["kickoff"]), tz)
        al = Alert(a["kind"], "", a["match"], datetime.fromisoformat(a["kickoff"]), a["league"], a["market"],
                   a["line"], a["outcome"], a["book"], a["price"], a["fair_odds"], a["edge"])
        home, away = (a["match"].split(" v ", 1) + [""])[:2]
        arow.append(f"<tr><td class='l'>{e(when)}</td><td class='l'>{'Pick' if a['kind'] == 'pick' else 'Wrong odds'}</td>"
                    f"<td class='l'>{e(a['match'])}<div class='muted small'>{e(ko)}</div></td>"
                    f"<td class='l'>{e(al.selection(home, away))}</td><td>{a['price']:.2f} {e(book_name(a['book']))}</td>"
                    f"<td>{a['fair_odds']:.2f}</td><td><b>{_pct(a['edge'])}</b></td></tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta http-equiv="refresh" content="30">
<title>Odds Watch</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Odds watch</h1><div class="muted small">Updated {e(local_time(utcnow(), tz))} · page refreshes every 30 s</div>
<div class="chips">{''.join(chips)}</div>
<h2>Research picks, live</h2>
<div class="tablewrap"><table><thead><tr><th class="l">Match</th><th class="l">Pick</th><th>Best live price</th><th>Status</th></tr></thead>
<tbody>{''.join(prow) or "<tr><td class='l muted' colspan='4'>No picks loaded. Run footy upcoming, then footy push-picks on your PC.</td></tr>"}</tbody></table></div>
<h2>Alerts, last 24 hours</h2>
<div class="tablewrap"><table><thead><tr><th class="l">Sent</th><th class="l">Type</th><th class="l">Match</th><th class="l">Selection</th>
<th>Price</th><th>Fair / need</th><th>Edge</th></tr></thead>
<tbody>{''.join(arow) or "<tr><td class='l muted' colspan='7'>No alerts yet.</td></tr>"}</tbody></table></div>
</div></body></html>"""


def make_handler(w: Watcher):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            log.debug(fmt, *args)

        def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, render(w), "text/html")
            elif self.path == "/api/status":
                self._send(200, json.dumps({**w.status(), "pick_status": w.pick_status}, default=str))
            elif self.path == "/api/alerts":
                self._send(200, json.dumps(w.state.recent(utcnow() - timedelta(hours=48)), default=str))
            elif self.path == "/health":
                self._send(200, '{"ok": true}')
            else:
                self._send(404, '{"error": "not found"}')

        def do_POST(self):  # noqa: N802
            if self.path != "/api/picks":
                return self._send(404, '{"error": "not found"}')
            token = w.cfg["server"].get("token")
            if token and self.headers.get("X-Token") != token:
                return self._send(403, '{"error": "bad token"}')
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                count = w.save_picks(payload)
            except Exception as ex:  # noqa: BLE001
                return self._send(400, json.dumps({"error": str(ex)}))
            self._send(200, json.dumps({"picks": count}))
    return Handler


def serve(w: Watcher) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((w.cfg["server"]["host"], int(w.cfg["server"]["port"])), make_handler(w))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("status page on http://%s:%s/", w.cfg["server"]["host"], w.cfg["server"]["port"])
    return srv
