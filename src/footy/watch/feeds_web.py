"""The feed comparison page (/feeds): what each odds feed delivered, to decide which to pay for."""

from __future__ import annotations

import html
from datetime import datetime

from .core import sport_name, utcnow
from .notify import book_name, local_time
from .paper_web import PAPER_CSS
from .recorder import minutes_between, monthly_requests

FEED_NAMES = {"oddsapi": "The Odds API", "oddspapi": "OddsPapi", "betfair": "Betfair API", "polymarket": "Polymarket",
              "file": "File feed"}
ALLOWANCES = (250, 5_000, 20_000, 100_000)
CSS = """
.wrap>*{min-width:0} td.wraptext{white-space:normal;min-width:200px;max-width:420px}
.days a{margin-right:6px}
"""


def _n(x, digits: int = 0) -> str:
    return "–" if x is None else f"{x:,.{digits}f}"


def _every(minutes: float) -> str:
    if minutes < 1:
        return "under a minute"
    if minutes < 90:
        return f"{minutes:.0f} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / 1440:.1f} days"


def render_feeds(w, base_css: str, days: float = 7.0) -> str:
    e = html.escape
    tz = w.cfg["server"].get("timezone", "UTC")
    rep = w.recorder.report(utcnow(), days, w.paper)
    provs = {k: v for k, v in rep["providers"].items() if not k.startswith("pm:")}
    scans = {k[3:]: v for k, v in rep["providers"].items() if k.startswith("pm:")}

    cov = []
    for p, d in sorted(provs.items(), key=lambda kv: -kv[1]["matches_total"]):
        sports = ", ".join(f"{sport_name(s)} {n}" for s, n in sorted(d["matches"].items(), key=lambda kv: -kv[1]))
        books = ", ".join(book_name(b["book"]) for b in d["books"][:8]) + (
            f" +{len(d['books']) - 8}" if len(d["books"]) > 8 else "")
        excl = ", ".join(book_name(b) for b in d["exclusive_books"][:8]) or "–"
        cov.append(f"<tr><td class='l'><b>{e(FEED_NAMES.get(p, p))}</b></td><td>{_n(d['matches_total'])}"
                   f"<div class='muted small'>{e(sports)}</div></td><td>{_n(d['only_here'])}</td>"
                   f"<td class='l wraptext small'>{e(books) or '–'}</td><td class='l wraptext small'>{e(excl)}</td>"
                   f"<td>{_n(d['changes_per_selection_hour'], 2)}</td></tr>")
    ops = []
    for p, d in sorted(provs.items()):
        last = local_time(datetime.fromisoformat(d["last"]), tz) if d.get("last") else "–"
        ops.append(f"<tr><td class='l'>{e(FEED_NAMES.get(p, p))}</td><td>{_n(d['polls'])}</td>"
                   f"<td>{_n(d['avg_seconds'], 1)}</td><td>{_n(d['requests_per_poll'], 1)}</td>"
                   f"<td>{_n(d['requests_per_day'])}</td><td>{_n(d['errors'])}</td><td class='l'>{e(last)}</td></tr>")
    lead = []
    for p, d in sorted(provs.items()):
        for other, x in sorted(d["lead"].items()):
            if p < other:
                m = x["median_minutes_ahead"]
                who = FEED_NAMES.get(p, p) if m > 0 else FEED_NAMES.get(other, other)
                lead.append(f"<tr><td class='l'>{e(FEED_NAMES.get(p, p))} vs {e(FEED_NAMES.get(other, other))}</td>"
                            f"<td class='l'>{e(who) if abs(m) >= 0.5 else 'same time'}</td><td>{abs(m):.1f}</td>"
                            f"<td>{_n(x['samples'])}</td></tr>")
    need = []
    for p, d in sorted(provs.items()):
        if d.get("paper_bets_only_here") is None:
            continue
        clv = d["paper_avg_clv_only_here"]
        need.append(f"<tr><td class='l'>{e(FEED_NAMES.get(p, p))}</td><td>{_n(d['paper_bets_only_here'])}</td>"
                    f"<td>{_n(d['paper_ev_only_here'], 2)}</td><td>{_n(d['paper_profit_only_here'], 2)}</td>"
                    f"<td>{'–' if clv is None else f'{100 * clv:+.1f}%'}</td></tr>")
    plans = []
    for p, d in sorted(provs.items()):
        rpp = d.get("requests_per_poll")
        if not rpp:
            continue
        need_cells = "".join(f"<td>{_n(monthly_requests(rpp, m))}</td>" for m in (5, 15, 30, 60))
        every_cells = "".join(f"<td>{e(_every(minutes_between(rpp, a, 10)))}</td>" for a in ALLOWANCES)
        plans.append(f"<tr><td class='l'>{e(FEED_NAMES.get(p, p))}</td><td>{rpp:g}</td>{need_cells}{every_cells}</tr>")
    scan_rows = "".join(
        f"<tr><td class='l'>{e(n)}</td><td>{_n(d['polls'])}</td><td>{_n(d['avg_seconds'], 1)}</td>"
        f"<td>{_n(d['errors'])}</td></tr>" for n, d in sorted(scans.items()))
    day_links = " ".join(f"<a href='/feeds?days={x}'>{'<b>' if x == days else ''}{x:g} days{'</b>' if x == days else ''}</a>"
                         for x in (1, 7, 30))

    def table(head: str, rows: list, empty: str, cols: int) -> str:
        return (f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>"
                f"{''.join(rows) or f'<tr><td class=l colspan={cols}>{empty}</td></tr>'}</tbody></table></div>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Feeds Compared</title><style>{base_css}{PAPER_CSS}{CSS}</style></head><body><div class="wrap">
<a class="back" href="/">← Odds watch</a>
<h1>Feeds compared</h1>
<div class="muted small">What each odds feed delivered over the last {days:g} days, recorded on this Pi
({rep['db_mb']} MB kept {w.recorder.keep_days:g} days). <span class="days">{day_links}</span></div>
<h2>Coverage</h2>
<div class="muted small">Matches it listed; "only here" = no other feed had them. Books it supplied, and the ones no
other feed supplied. Price changes per selection per hour: how lively its prices are at your polling rate.</div>
{table("<th class='l'>Feed</th><th>Matches</th><th>Only here</th><th class='l'>Books</th><th class='l'>Only from it</th><th>Changes/h</th>", cov, "Nothing recorded yet.", 6)}
<h2>Who shows a price move first</h2>
<div class="muted small">Where two feeds carry the same bookmaker (Pinnacle through The Odds API and OddsPapi, say):
the median minutes one had a new price before the other. It reflects how often each one polls on your plan.</div>
{table("<th class='l'>Feeds</th><th class='l'>First</th><th>Minutes ahead</th><th>Prices compared</th>", lead, "Needs two feeds with the same bookmaker.", 4)}
<h2>Paper bets that needed it</h2>
<div class="muted small">Bets placed at books only this feed supplied: what you'd lose without it.</div>
{table("<th class='l'>Feed</th><th>Bets</th><th>Expected €</th><th>Profit €</th><th>Avg CLV</th>", need, "No paper bets yet.", 5)}
<h2>Requests and plans</h2>
<div class="muted small">Measured requests (or credits) per check with your current settings: what a month needs at
each rate, and how often each monthly allowance lets it check (10 kept in reserve).</div>
{table("<th class='l'>Feed</th><th>Per check</th><th>Every 5 min</th><th>15 min</th><th>30 min</th><th>60 min</th>" + "".join(f"<th>{a:,}/mo</th>" for a in ALLOWANCES), plans, "No paid feed polled yet.", 10)}
<h2>Polling</h2>
{table("<th class='l'>Feed</th><th>Checks</th><th>Seconds each</th><th>Per check</th><th>Per day</th><th>Errors</th><th class='l'>Last</th>", ops, "Nothing recorded yet.", 7)}
{f"<h2>Prediction-market scans</h2>" + table("<th class='l'>Strategy</th><th>Scans</th><th>Seconds each</th><th>Errors</th>", [scan_rows], "", 4) if scans else ""}
<div class="note"><a href="/api/feeds?days={days:g}">This report as JSON</a></div>
</div></body></html>"""
