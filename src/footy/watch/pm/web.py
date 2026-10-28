"""The prediction-market page (/pm) and its CSV."""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime

from ..core import utcnow
from ..notify import local_time
from ..paper_web import CHART_JS, PAPER_CSS, _cls, _pct
from .ledger import STRATEGIES, summarize

PM_CSS = """
.wrap>*{min-width:0}
td.wraptext{white-space:normal;min-width:220px;max-width:440px}
td.legs{white-space:normal;min-width:170px}
"""
SERIES = [{"k": "actual", "name": "Actual profit", "c": "var(--s1)", "d": ""},
          {"k": "ev", "name": "Expected when placed", "c": "var(--s2)", "d": "6 4"}]


def _usd(x, signed: bool = True) -> str:
    if x is None:
        return "–"
    sign = ("+" if x >= 0 else "−") if signed else ("−" if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def render_pm(w, base_css: str, strategy: str = "") -> str:
    e = html.escape
    tz = w.cfg["server"].get("timezone", "UTC")
    start = float(w.cfg.get("pm", {}).get("bankroll", 1000))
    names = [sc.name for sc in w.scanners]
    strategy = strategy if strategy in names else (names[0] if names else "")
    tabs = []
    for n in names:
        s = summarize(w.pm.positions(strategy=n), start)
        tabs.append(f"<a class='{'on' if n == strategy else ''}' href='/pm?strategy={e(n)}'>{e(STRATEGIES.get(n, n))}"
                    f"<small class='{_cls(s['profit'])}'>{_usd(s['profit'])} · {s['positions']} "
                    f"position{'' if s['positions'] == 1 else 's'}</small></a>")
    rows = w.pm.positions(strategy=strategy) if strategy else []
    s = summarize(rows, start)
    info = w.scan_info.get(strategy, {})
    last = (f"Last scan {e(local_time(datetime.fromisoformat(info['last_run']), tz))}: {info.get('events', 0)} events, "
            f"{info.get('signals', 0)} opportunities, {info.get('new', 0)} new"
            + (f" · <span class='err'>{e(info['error'])}</span>" if info.get("error") else "")) if info else "not run yet"
    tiles = f"""
<div class="tile hero"><div class="lab">Bankroll, {e(STRATEGIES.get(strategy, strategy))}</div>
<div class="val">{_usd(s['bankroll'], False)}</div><div class="sub">started at {_usd(start, False)}</div></div>
<div class="tile"><div class="lab">Profit</div><div class="val {_cls(s['profit'])}">{_usd(s['profit'])}</div>
<div class="sub">ROI {_pct(s['roi'])} · {s['settled']} settled · {s['won']} won</div></div>
<div class="tile"><div class="lab">Average edge when placed</div><div class="val">{_pct(s['avg_edge'])}</div>
<div class="sub">expected {_usd(s['expected_profit'])} on settled</div></div>
<div class="tile"><div class="lab">Open</div><div class="val">{s['open']}</div>
<div class="sub">{_usd(s['open_cost'], False)} tied up</div></div>"""
    series = [{**p, "when": local_time(datetime.fromisoformat(p["kickoff"]), tz)} for p in s["series"]]
    if series:
        data = json.dumps(series).replace("</", "<\\/")
        chart = (f"<div id='pchart' data-cur='$' data-series='{e(json.dumps(SERIES))}'></div>"
                 f"<script type='application/json' id='pchart-data'>{data}</script><script>{CHART_JS}</script>")
    else:
        chart = "<p class='muted'>Nothing settled yet. Positions settle when Polymarket resolves the market.</p>"

    def legs_txt(r):
        return "<br>".join(e(f"{lg['side']} {lg['label']} @ {100 * lg['price']:.1f}¢ × {lg['shares']:.0f}")
                           for lg in r["legs"][:6]) + (f"<br>+{len(r['legs']) - 6} more" if len(r["legs"]) > 6 else "")

    def title_cell(r):
        t = e(r["title"])
        return f"<a href='https://polymarket.com/event/{e(r['slug'])}'>{t}</a>" if r.get("slug") else t
    open_rows, done_rows = [], []
    for r in sorted(rows, key=lambda r: r["end_date"] or ""):
        ends = local_time(datetime.fromisoformat(r["end_date"]), tz) if r["end_date"] else "–"
        cells = (f"<td class='l wraptext'>{title_cell(r)}<div class='muted small'>{e(r['detail'][:200])}</div></td>"
                 f"<td class='l small legs'>{legs_txt(r)}</td><td>{_usd(r['cost'], False)}</td>"
                 f"<td>{_pct(r['edge'])}</td>")
        if r["status"] == "open":
            apy = f"{100 * r['apy']:.0f}%" if r["apy"] is not None else "–"
            open_rows.append(f"<tr>{cells}<td>{apy}</td><td class='l'>{e(ends)}</td></tr>")
        else:
            done_rows.append(f"<tr>{cells}<td>{_usd(r['payout'], False)}</td>"
                             f"<td class='{_cls(r['profit'])}'>{_usd(r['profit'])}</td></tr>")
    head = "<th class='l'>Event</th><th class='l'>Legs</th><th>Cost</th><th>Edge</th>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta http-equiv="refresh" content="300">
<title>Prediction Markets</title><style>{base_css}{PAPER_CSS}{PM_CSS}</style></head><body><div class="wrap">
<a class="back" href="/">← Odds watch</a>
<h1>Prediction markets</h1>
<div class="muted small">Imaginary positions on Polymarket, taken automatically at the order book's best prices
(taker fee included) and settled from Polymarket's resolution. {last}. Updated {e(local_time(utcnow(), tz))}.</div>
<nav class="seg" aria-label="Strategy">{''.join(tabs)}</nav>
<div class="tiles">{tiles}</div>
<div class="card"><h2>Profit: actual vs expected</h2>
<div class="muted small">Settled positions in order. Arbitrage should track the expected line closely; weather
bets swing around it, inside the shaded luck range.</div>
<div class="legend"><span><i class="key k1"></i>Actual profit</span><span><i class="key k2"></i>Expected when placed</span>
<span><i class="key band"></i>Luck range</span></div>{chart}</div>
<h2>Open positions</h2>
<div class="tablewrap"><table><thead><tr>{head}<th>Per year</th><th class="l">Resolves</th></tr></thead>
<tbody>{''.join(open_rows) or "<tr><td class='l muted' colspan='6'>None right now.</td></tr>"}</tbody></table></div>
<details{' open' if len(done_rows) <= 10 else ''}><summary><h2>Settled ({len(done_rows)})</h2></summary>
<div class="tablewrap"><table><thead><tr>{head}<th>Paid</th><th>Profit</th></tr></thead>
<tbody>{''.join(done_rows[::-1][:200]) or "<tr><td class='l muted' colspan='6'>None yet.</td></tr>"}</tbody></table></div>
</details>
<div class="note"><a href="/api/pm.csv">Download all positions (CSV)</a></div>
</div></body></html>"""


def pm_csv(w) -> str:
    cols = ["id", "placed", "strategy", "kind", "event_id", "title", "slug", "end_date", "cost", "payout_min",
            "expected", "edge", "apy", "prob", "status", "payout", "profit", "settled", "detail", "legs"]
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(cols)
    for r in w.pm.positions():
        wr.writerow([json.dumps(r[c]) if c == "legs" else r.get(c) for c in cols])
    return buf.getvalue()
