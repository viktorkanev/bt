"""Render the static HTML dashboard from the report tables.

The page is self-contained (data embedded as JSON, charts drawn as inline SVG
by a few hundred lines of vanilla JS) so it can be opened from disk, hosted
anywhere, or shared as a single file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.replace({np.nan: None})
    return df.to_dict("records")


def _clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def render(out_dir: Path, standalone: bool = True) -> str:
    out_dir = Path(out_dir)
    data = {
        "summary": json.loads((out_dir / "summary.json").read_text()),
        "curves": json.loads((out_dir / "curves.json").read_text()),
        "overall": _records(out_dir / "accuracy_overall.csv"),
        "byLeague": _records(out_dir / "league_edge.csv"),
        "byYear": _records(out_dir / "accuracy_by_year.csv"),
        "byTier": _records(out_dir / "strategies_by_tier.csv"),
        "ou": _records(out_dir / "accuracy_ou.csv"),
        "calibration": _records(out_dir / "accuracy_calibration.csv"),
        "strategies": _records(out_dir / "strategies.csv"),
        "grid": _records(out_dir / "strategy_grid.csv"),
        "ratings": _records(out_dir / "ratings.csv"),
        "bands": _records(out_dir / "edge_bands.csv"),
        "capped": _records(out_dir / "capped_strategies.csv"),
        "upcoming": _records(out_dir / "upcoming.csv")[:300],
    }
    research = {}
    rdir = out_dir.parent / "research"
    for w in ("dev", "holdout"):
        f = rdir / w / "results.json"
        if f.exists():
            research[w] = json.loads(f.read_text())
    ff = rdir / "forward" / "summary.json"
    if ff.exists():
        research["forward"] = json.loads(ff.read_text())
    data["research"] = research
    payload = json.dumps(_clean(data), default=str).replace("</", "<\\/")
    body = TEMPLATE.replace("__DATA__", payload)
    if not standalone:
        return body.replace("<!--BODY-->\n", "", 1)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + body.replace("<!--BODY-->", "</head>\n<body>", 1) + "\n</body>\n</html>\n")


TEMPLATE = r"""<title>Football Edge Scorecard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column like a matchday programme's stats page; sticky chip nav;
   tables scroll inside their own frames. */
:root {
  --bg: #f4f6f5; --surface: #ffffff; --surface-2: #eef2f0; --ink: #13201a; --ink-2: #46544c;
  --muted: #6f7d75; --rule: #dbe3de; --accent: #1c5cab;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
  --pos: #2a78d6; --neg: #e34948; --mid: #f0efec; --good: #0ca30c; --bad: #d03b3b;
  --display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111513; --surface: #181d1a; --surface-2: #1f2522; --ink: #eef2ef; --ink-2: #b8c3bd;
  --muted: #8b9890; --rule: #2b3330; --accent: #6da7ec;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  --pos: #3987e5; --neg: #e66767; --mid: #383835; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111513; --surface: #181d1a; --surface-2: #1f2522; --ink: #eef2ef; --ink-2: #b8c3bd;
  --muted: #8b9890; --rule: #2b3330; --accent: #6da7ec;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  --pos: #3987e5; --neg: #e66767; --mid: #383835; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); margin: 0; }
.wrap { max-width: 1080px; margin: 0 auto; padding-block: 24px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header.top { display: grid; gap: 6px; padding-block: 8px 20px; border-bottom: 1px solid var(--rule); }
.eyebrow { font: 600 12px/1.2 var(--body); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1, h2, h3 { font-family: var(--display); font-weight: 700; line-height: 1.05; text-wrap: balance; margin: 0; }
h1 { font-size: clamp(34px, 6vw, 52px); letter-spacing: .01em; }
h2 { font-size: 30px; }
h3 { font-size: 20px; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
.lede { color: var(--ink-2); font-size: 16px; }
nav.chips { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  display: flex; gap: 8px; overflow-x: auto; padding-block: 10px; border-bottom: 1px solid var(--rule); scrollbar-width: none; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }
nav.chips a { flex: none; font: 500 13px/1 var(--body); color: var(--ink-2); text-decoration: none;
  padding: 8px 12px; border: 1px solid var(--rule); border-radius: 999px; background: var(--surface); }
nav.chips a:hover, nav.chips a:focus-visible { color: var(--accent); border-color: var(--accent); outline: none; }
section { display: grid; gap: 16px; padding-block: 36px 8px; scroll-margin-top: 64px; }
.sub { color: var(--ink-2); }
.verdict { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 24px; align-items: start; }
@media (max-width: 760px) { .verdict { grid-template-columns: minmax(0, 1fr); } }
.hero { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 20px; display: grid; gap: 6px; }
.hero .value { font: 700 64px/1 var(--display); }
.hero .ci { font: 500 14px/1.3 var(--mono); color: var(--ink-2); }
.tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.tile { border-top: 2px solid var(--rule); padding-top: 8px; display: grid; gap: 2px; }
.tile .label { font-size: 13px; color: var(--muted); }
.tile .value { font: 600 30px/1.1 var(--display); }
.tile .note { font-size: 12.5px; color: var(--ink-2); }
.findings { display: grid; gap: 10px; padding: 0; margin: 0; list-style: none; }
.findings li { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 8px; align-items: start; }
.pill { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%;
  font: 700 12px/1 var(--body); color: #fff; margin-top: 2px; }
.pill.good { background: var(--good); } .pill.bad { background: var(--bad); } .pill.info { background: var(--s1); }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px; display: grid; gap: 10px; min-width: 0; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.legend i.dot { width: 9px; height: 9px; border-radius: 50%; }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
svg text { fill: var(--muted); font: 11px var(--body); }
.tablewrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th { font: 600 12px/1.2 var(--body); color: var(--muted); background: var(--surface-2); position: sticky; top: 0; }
th:first-child, td:first-child, td.l, th.l { text-align: left; }
tbody tr:last-child td { border-bottom: 0; }
td.wraptext { white-space: normal; min-width: 150px; }
td .pill { vertical-align: middle; margin-top: 0; }
td.code { font-family: var(--mono); font-size: 12px; color: var(--ink-2); }
td.cell { position: relative; }
.bar-pos, .bar-neg { display: inline-block; height: 8px; border-radius: 2px; vertical-align: middle; margin-left: 6px; }
.bar-pos { background: var(--pos); } .bar-neg { background: var(--neg); }
.muted { color: var(--muted); }
.tag { font: 600 11px/1 var(--body); letter-spacing: .04em; text-transform: uppercase; padding: 3px 6px; border-radius: 4px; background: var(--surface-2); color: var(--ink-2); }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
select { font: 14px var(--body); padding: 7px 10px; border-radius: 8px; border: 1px solid var(--rule); background: var(--surface); color: var(--ink); max-width: 100%; }
select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
#tip { position: fixed; pointer-events: none; z-index: 20; background: var(--surface); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 8px; padding: 8px 10px; font: 12.5px/1.45 var(--body); box-shadow: 0 6px 20px rgba(0,0,0,.12); max-width: 260px; }
#tip b { font-weight: 600; }
.method { display: grid; gap: 12px; }
.method dl { display: grid; grid-template-columns: minmax(0, 180px) minmax(0, 1fr); gap: 8px 16px; margin: 0; }
.method dt { font-weight: 600; } .method dd { margin: 0; color: var(--ink-2); }
@media (max-width: 600px) { .method dl { grid-template-columns: minmax(0, 1fr); } .hero .value { font-size: 52px; } }
code { font-family: var(--mono); font-size: 13px; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 12px; overflow-x: auto; font: 12.5px/1.6 var(--mono); margin: 0; }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 12.5px; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
</style>
<!--BODY-->
<div class="wrap">
  <header class="top">
    <div class="eyebrow" id="meta-line">Football results engine</div>
    <h1>Football Edge Scorecard</h1>
    <p class="lede">Rating models, walk-forward predictions and betting backtests across <span id="lede-leagues">38</span> leagues.
      Parameters were chosen on the development window and judged on later seasons the models never saw.</p>
  </header>
  <nav class="chips" aria-label="Sections">
    <a href="#verdict">Verdict</a><a href="#idea">Skill vs luck test</a><a href="#accuracy">Model vs market</a><a href="#leagues">Lower leagues</a>
    <a href="#strategies">Strategies</a><a href="#grid">Overfitting check</a><a href="#ratings">Ratings</a>
    <a href="#upcoming">Upcoming</a><a href="#method">Method</a>
  </nav>

  <section id="verdict">
    <div class="eyebrow">Verdict</div>
    <div class="verdict">
      <div class="hero" id="hero"></div>
      <div class="tiles" id="tiles"></div>
    </div>
    <ul class="findings" id="findings"></ul>
  </section>

  <section id="idea">
    <h2>New idea: does the market confuse luck with skill?</h2>
    <p class="sub">For every team we track how it has done against what the bookmaker expected, then split that surprise into results, shot dominance,
      finishing luck (goals beyond what shots on target justify) and goalkeeper luck. The rules were written down and committed before the 2016–2026 holdout was run once.</p>
    <div class="grid2">
      <div class="card"><h3>Effect on the next match, by part</h3>
        <p class="sub" style="font-size:13px">t-statistic of each part in predicting goal difference beyond the market. Beyond ±2 is significant. Negative means the market over-reacts.</p>
        <div class="legend" id="h1Legend"></div><div id="h1Chart"></div></div>
      <div class="card"><h3>What the luck effect is worth</h3>
        <p class="sub" style="font-size:13px">Handicap bets on the holdout, re-priced at different bookmaker margins. A projection, not a test.</p>
        <div class="tablewrap"><table id="projTable"></table></div>
        <p class="sub" style="font-size:13px" id="fwdNote"></p></div>
    </div>
    <div class="tablewrap"><table id="hypTable"></table></div>
  </section>

  <section id="accuracy">
    <h2>Model vs market</h2>
    <p class="sub">Ranked probability score (RPS) is the standard accuracy measure for 1X2 forecasts; lower is better.
      Bookmaker prices with the margin removed are the benchmark. The chart shows each model's gap to that benchmark on the holdout seasons.</p>
    <div class="grid2">
      <div class="card"><h3>Gap to the market, holdout</h3><p class="sub" style="font-size:13px">RPS difference × 1000. Left of zero means more accurate than Bet365.</p><div id="gapChart"></div></div>
      <div class="card"><h3>Accuracy by season</h3><div class="legend" id="yearLegend"></div><div id="yearChart"></div></div>
    </div>
    <div class="grid2">
      <div class="card"><h3>Bookmaker margin by season</h3><p class="sub" style="font-size:13px">Average Bet365 overround on 1X2.</p><div id="marginChart"></div></div>
      <div class="card"><h3>Calibration, home wins (holdout)</h3><div class="legend" id="calLegend"></div><div id="calChart"></div></div>
    </div>
    <div class="tablewrap"><table id="ouTable"></table></div>
  </section>

  <section id="leagues">
    <h2>Where is the edge? Lower leagues vs top flights</h2>
    <p class="sub">Each league's market margin, how close the models get to the market, and the return of three fixed strategies in the development and holdout windows.
      Lower divisions carry bigger margins but the gap between model and market is also where any edge would have to come from.</p>
    <div class="card"><h3>Return by tier</h3><div class="legend" id="tierLegend"></div><div id="tierChart"></div></div>
    <div class="controls"><label for="edgeStrat" class="muted">Strategy in table</label>
      <select id="edgeStrat"><option value="ou_blend_max">Over/Under 2.5 model+market @ best price</option><option value="blend_max">1X2 model+market @ best price</option><option value="consensus_max">1X2 consensus vs best price</option><option value="blend_b365">1X2 model+market @ Bet365</option></select></div>
    <div class="tablewrap"><table id="leagueTable"></table></div>
  </section>

  <section id="strategies">
    <h2>Strategies</h2>
    <p class="sub">Six strategies fixed before looking at results. Flat 1-unit stakes. ROI is profit per unit staked; the range is a 95% bootstrap interval.</p>
    <div class="card"><h3>Cumulative profit, holdout seasons</h3><div class="legend" id="eqLegend"></div><div id="eqChart"></div></div>
    <div class="tablewrap"><table id="stratTable"></table></div>
    <div class="card"><h3>Where the profit comes from</h3>
      <p class="sub" style="font-size:13px">Holdout ROI by the edge the strategy claimed. When profit sits only in the largest edges, the best price was usually a stale or outlier quote that is hard to get in practice.</p>
      <div class="controls"><label for="bandStrat" class="muted">Strategy</label>
        <select id="bandStrat"><option value="blend_max">Model+market @ best price</option><option value="consensus_max">Consensus vs best price</option><option value="ou_blend_max">Over/Under 2.5 model+market</option></select></div>
      <div class="legend" id="bandLegend"></div><div id="bandChart"></div></div>
    <div class="tablewrap"><table id="capTable"></table></div>
  </section>

  <section id="grid">
    <h2>Overfitting check</h2>
    <p class="sub">Every combination of model, price, edge threshold and odds cap, scored on the development window and then on the holdout.
      If development results predicted the future, the dots would rise to the right.</p>
    <div class="card"><div class="legend" id="gridLegend"></div><div id="gridChart"></div></div>
    <div class="tablewrap"><table id="gridTable"></table></div>
  </section>

  <section id="ratings">
    <h2>Current ratings</h2>
    <p class="sub">Latest Elo and pseudo-xG attack/defence for teams active in the last year, as percent above or below the division average. A higher defence figure means fewer goals conceded.</p>
    <div class="controls"><label for="lgSel" class="muted">League</label><select id="lgSel"></select></div>
    <div class="tablewrap"><table id="ratingTable"></table></div>
  </section>

  <section id="upcoming">
    <h2>Upcoming fixtures</h2>
    <div id="upcomingBody"></div>
  </section>

  <section id="method" class="method">
    <h2>Method</h2>
    <dl>
      <dt>Data</dt><dd>football-data.co.uk results, shots and odds (Bet365, market best price; Pinnacle and closing prices when downloaded directly).</dd>
      <dt>Ratings</dt><dd>Elo with goal-difference weighting and a self-adjusting home advantage; pi-ratings (Constantinou &amp; Fenton 2013); online Dixon-Coles attack/defence on goals and on a pseudo-xG blend of goals and shots on target; shots-on-target ratings. Promoted and relegated teams are re-based using how earlier movers settled.</dd>
      <dt>Probabilities</dt><dd>Ordered logit, multinomial logistic regression and gradient boosting on the rating features, re-trained every season on prior data only (walk-forward). The blend adds the bookmaker's own probabilities as inputs.</dd>
      <dt>Backtest</dt><dd>One bet per match when model probability × price − 1 exceeds the edge threshold. Development window picks; holdout judges.</dd>
      <dt>Caveats</dt><dd>Best-price odds assume you can always get the top price; soft bookmakers limit winning accounts; no closing-line check in the mirror data. Many strategies were tested, so a few will look good by chance.</dd>
    </dl>
    <pre>pip install -e ".[dashboard]"
footy all                    # fetch, build, report
streamlit run app/dashboard.py</pre>
  </section>
  <footer id="foot"></footer>
</div>
<div id="tip" hidden></div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const S = D.summary || {};
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const pct = (x, d = 1) => x == null ? '–' : (x >= 0 ? '+' : '−') + Math.abs(x * 100).toFixed(d) + '%';
  const num = (x, d = 2) => x == null ? '–' : Number(x).toFixed(d);
  const int = (x) => x == null ? '–' : Math.round(x).toLocaleString('en-GB');
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const LABEL = S.model_labels || {};
  const NS = 'http://www.w3.org/2000/svg';
  const tip = $('tip');
  function showTip(html, ev) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
    tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
  }
  const hideTip = () => { tip.hidden = true; };
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function legend(id, items, dot) {
    $(id).innerHTML = items.map(([name, color]) => `<span><i class="${dot ? 'dot' : ''}" style="background:${color}"></i>${esc(name)}</span>`).join('');
  }
  function frame(container, h, padL) {
    const w = Math.max(300, container.clientWidth || 600);
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, class: 'chart', role: 'img' });
    container.innerHTML = ''; container.appendChild(svg);
    return { svg, w, h, L: padL || 46, R: 16, T: 10, B: 26 };
  }
  // ---------- line chart with crosshair tooltip
  function lineChart(container, xs, series, opt) {
    opt = opt || {};
    const f = frame(container, opt.height || 240, opt.padL);
    const all = series.flatMap(s => s.values.filter(v => v != null));
    let lo = opt.min != null ? opt.min : Math.min(...all), hi = opt.max != null ? opt.max : Math.max(...all);
    if (opt.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const pad = (hi - lo) * 0.08 || 1; lo -= opt.zero && lo === 0 ? 0 : pad; hi += pad;
    const xi = (i) => f.L + (f.w - f.L - f.R) * (xs.length > 1 ? i / (xs.length - 1) : 0.5);
    const y = (v) => f.T + (f.h - f.T - f.B) * (1 - (v - lo) / (hi - lo));
    const ticks = niceTicks(lo, hi, 4);
    for (const t of ticks) {
      el('line', { x1: f.L, x2: f.w - f.R, y1: y(t), y2: y(t), stroke: css(t === 0 && opt.zero ? '--muted' : '--rule'), 'stroke-width': 1 }, f.svg);
      const tx = el('text', { x: f.L - 6, y: y(t) + 4, 'text-anchor': 'end' }, f.svg); tx.textContent = opt.fmt ? opt.fmt(t) : t;
    }
    const nx = Math.min(xs.length, Math.max(2, Math.floor((f.w - f.L) / 90)));
    for (let k = 0; k < nx; k++) {
      const i = Math.round(k * (xs.length - 1) / (nx - 1));
      const tx = el('text', { x: xi(i), y: f.h - 6, 'text-anchor': k === 0 ? 'start' : k === nx - 1 ? 'end' : 'middle' }, f.svg);
      tx.textContent = opt.xfmt ? opt.xfmt(xs[i]) : xs[i];
    }
    series.forEach(s => {
      let d = '', pen = false;
      s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + xi(i).toFixed(1) + ' ' + y(v).toFixed(1); pen = true; });
      el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, f.svg);
      const last = s.values.map((v, i) => [v, i]).filter(p => p[0] != null).pop();
      if (last) el('circle', { cx: xi(last[1]), cy: y(last[0]), r: 4, fill: s.color, stroke: css('--surface'), 'stroke-width': 2 }, f.svg);
    });
    const cross = el('line', { y1: f.T, y2: f.h - f.B, stroke: css('--muted'), 'stroke-width': 1, visibility: 'hidden' }, f.svg);
    const hit = el('rect', { x: f.L, y: f.T, width: f.w - f.L - f.R, height: f.h - f.T - f.B, fill: 'transparent' }, f.svg);
    hit.addEventListener('pointermove', (ev) => {
      const r = f.svg.getBoundingClientRect(), px = (ev.clientX - r.left) * f.w / r.width;
      const i = Math.max(0, Math.min(xs.length - 1, Math.round((px - f.L) / (f.w - f.L - f.R) * (xs.length - 1))));
      cross.setAttribute('x1', xi(i)); cross.setAttribute('x2', xi(i)); cross.setAttribute('visibility', 'visible');
      const rows = series.map(s => `<div><i style="display:inline-block;width:10px;height:3px;background:${s.color};vertical-align:middle;margin-right:6px"></i>${esc(s.name)}: <b>${s.values[i] == null ? '–' : (opt.fmt ? opt.fmt(s.values[i]) : s.values[i])}</b></div>`).join('');
      showTip(`<b>${esc(opt.xfmt ? opt.xfmt(xs[i]) : xs[i])}</b>${opt.extra ? opt.extra(i) : ''}${rows}`, ev);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  }
  // ---------- horizontal dot plot around zero
  function dotPlot(container, rows, opt) {
    const rowH = 30, f = frame(container, rows.length * rowH + 34, opt.padL || 170);
    const vals = rows.map(r => r.value);
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals); const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
    const x = (v) => f.L + (f.w - f.L - f.R) * (v - lo) / (hi - lo);
    for (const t of niceTicks(lo, hi, 5)) {
      el('line', { x1: x(t), x2: x(t), y1: f.T, y2: f.h - f.B, stroke: css(t === 0 ? '--muted' : '--rule'), 'stroke-width': 1 }, f.svg);
      const tx = el('text', { x: x(t), y: f.h - 6, 'text-anchor': 'middle' }, f.svg); tx.textContent = t;
    }
    rows.forEach((r, i) => {
      const cy = f.T + i * rowH + rowH / 2;
      const lab = el('text', { x: f.L - 10, y: cy + 4, 'text-anchor': 'end' }, f.svg); lab.textContent = r.label;
      lab.style.fill = css('--ink-2');
      el('line', { x1: x(0), x2: x(r.value), y1: cy, y2: cy, stroke: r.color, 'stroke-width': 2 }, f.svg);
      const c = el('circle', { cx: x(r.value), cy, r: 5, fill: r.color, stroke: css('--surface'), 'stroke-width': 2 }, f.svg);
      const h = el('circle', { cx: x(r.value), cy, r: 14, fill: 'transparent' }, f.svg);
      h.addEventListener('pointermove', (ev) => showTip(r.tip, ev)); h.addEventListener('pointerleave', hideTip);
    });
  }
  // ---------- grouped columns around zero
  function columns(container, groups, seriesDefs, opt) {
    const f = frame(container, opt.height || 230, 48);
    const vals = groups.flatMap(g => seriesDefs.map(s => g[s.key]).filter(v => v != null));
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals); const pad = (hi - lo) * 0.1 || 0.01; lo -= pad; hi += pad;
    const y = (v) => f.T + (f.h - f.T - f.B) * (1 - (v - lo) / (hi - lo));
    for (const t of niceTicks(lo, hi, 4)) {
      el('line', { x1: f.L, x2: f.w - f.R, y1: y(t), y2: y(t), stroke: css(Math.abs(t) < 1e-12 ? '--muted' : '--rule'), 'stroke-width': 1 }, f.svg);
      const tx = el('text', { x: f.L - 6, y: y(t) + 4, 'text-anchor': 'end' }, f.svg); tx.textContent = opt.fmt(t);
    }
    const gw = (f.w - f.L - f.R) / groups.length, bw = Math.min(24, (gw - 24) / seriesDefs.length - 2);
    groups.forEach((g, gi) => {
      const cx = f.L + gw * gi + gw / 2;
      const lab = el('text', { x: cx, y: f.h - 6, 'text-anchor': 'middle' }, f.svg); lab.textContent = g.label;
      seriesDefs.forEach((s, si) => {
        const v = g[s.key]; if (v == null) return;
        const x0 = cx - (seriesDefs.length * (bw + 2)) / 2 + si * (bw + 2);
        const y0 = y(0), y1 = y(v), top = Math.min(y0, y1), hgt = Math.max(1, Math.abs(y1 - y0));
        const r = Math.min(4, hgt / 2), up = v >= 0;
        const d = up ? `M${x0} ${y0}V${top + r}Q${x0} ${top} ${x0 + r} ${top}H${x0 + bw - r}Q${x0 + bw} ${top} ${x0 + bw} ${top + r}V${y0}Z`
                     : `M${x0} ${y0}V${top + hgt - r}Q${x0} ${top + hgt} ${x0 + r} ${top + hgt}H${x0 + bw - r}Q${x0 + bw} ${top + hgt} ${x0 + bw} ${top + hgt - r}V${y0}Z`;
        const p = el('path', { d, fill: s.color }, f.svg);
        const h = el('rect', { x: x0 - 3, y: Math.min(top, y0) - 6, width: bw + 6, height: hgt + 12, fill: 'transparent' }, f.svg);
        h.addEventListener('pointermove', (ev) => showTip(opt.tip(g, s), ev)); h.addEventListener('pointerleave', hideTip);
      });
    });
  }
  // ---------- scatter
  function scatter(container, pts, opt) {
    const f = frame(container, 300, 52);
    const xsv = pts.map(p => p.x), ysv = pts.map(p => p.y);
    let x0 = Math.min(0, ...xsv), x1 = Math.max(0, ...xsv), y0 = Math.min(0, ...ysv), y1 = Math.max(0, ...ysv);
    const px = (x1 - x0) * .08, py = (y1 - y0) * .08; x0 -= px; x1 += px; y0 -= py; y1 += py;
    const X = (v) => f.L + (f.w - f.L - f.R) * (v - x0) / (x1 - x0), Y = (v) => f.T + (f.h - f.T - f.B - 14) * (1 - (v - y0) / (y1 - y0));
    for (const t of niceTicks(y0, y1, 4)) { el('line', { x1: f.L, x2: f.w - f.R, y1: Y(t), y2: Y(t), stroke: css(Math.abs(t) < 1e-12 ? '--muted' : '--rule') }, f.svg); const tx = el('text', { x: f.L - 6, y: Y(t) + 4, 'text-anchor': 'end' }, f.svg); tx.textContent = opt.fmt(t); }
    for (const t of niceTicks(x0, x1, 5)) { el('line', { x1: X(t), x2: X(t), y1: f.T, y2: f.h - f.B - 14, stroke: css(Math.abs(t) < 1e-12 ? '--muted' : '--rule') }, f.svg); const tx = el('text', { x: X(t), y: f.h - f.B + 2, 'text-anchor': 'middle' }, f.svg); tx.textContent = opt.fmt(t); }
    const xl = el('text', { x: (f.L + f.w - f.R) / 2, y: f.h - 4, 'text-anchor': 'middle' }, f.svg); xl.textContent = opt.xlabel;
    const yl = el('text', { x: 12, y: f.T + 4, 'text-anchor': 'start' }, f.svg); yl.textContent = opt.ylabel;
    pts.forEach(p => {
      el('circle', { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: p.color, stroke: css('--surface'), 'stroke-width': 2 }, f.svg);
      const h = el('circle', { cx: X(p.x), cy: Y(p.y), r: 10, fill: 'transparent' }, f.svg);
      h.addEventListener('pointermove', (ev) => showTip(p.tip, ev)); h.addEventListener('pointerleave', hideTip);
    });
  }

  // ======================================================= content
  const strat = Object.fromEntries((D.strategies || []).map(s => [s.key, s]));
  const main = strat.blend_max || {};
  $('meta-line').textContent = `${S.data_source === 'mirror' ? 'football-data.co.uk via GitHub mirror' : 'football-data.co.uk'} · ${int(S.matches)} matches · ${S.first_date} to ${S.last_result} · generated ${S.generated}`;
  $('lede-leagues').textContent = S.leagues;
  const holdStart = (S.holdout_window || [])[0] || '';
  const TT = D.byTier || [];
  const tierRow = (k, t, per) => TT.find(r => r.strategy === k && r.tier_grp === t && r.period === per) || {};
  const ouLow = tierRow('ou_blend_max', 0, 'holdout'), ouLowDev = tierRow('ou_blend_max', 0, 'dev'), ouTop = tierRow('ou_blend_max', 1, 'holdout');
  $('hero').innerHTML = `<div class="eyebrow">Best candidate · Over/Under 2.5 in divisions 2–5 · holdout</div>
    <div class="value">${pct(ouLow.roi)}</div>
    <div class="ci">95% range ${pct(ouLow.roi_lo)} to ${pct(ouLow.roi_hi)} · ${int(ouLow.bets)} bets · p = ${num(ouLow.p_value, 3)}</div>
    <p class="sub" style="font-size:13.5px">Model + market blend at the best available price, edge of 3% or more, from ${holdStart.slice(0, 4)} onwards.
      The same rule made ${pct(ouLowDev.roi)} in lower divisions during development and ${pct(ouTop.roi)} in top flights on the holdout.
      The lower-division split was examined after seeing results, so treat it as a lead to paper-trade, not a proven edge.</p>`;
  const ov = (D.overall || []).filter(r => r.period === 'holdout');
  const rps = Object.fromEntries(ov.map(r => [r.model, r.rps]));
  const bestModel = ['pxg', 'ens', 'gbm'].reduce((a, b) => (rps[b] < rps[a] ? b : a), 'pxg');
  const cons = strat.consensus_max || {};
  const tierRows = D.byTier || [];
  $('tiles').innerHTML = [
    ['Matches evaluated', int(S.eval_matches), 'out-of-sample, from 2005'],
    ['Market RPS (holdout)', num(rps.mkt, 4), 'Bet365, margin removed'],
    ['Best ratings-only RPS', num(rps[bestModel], 4), LABEL[bestModel] || bestModel],
    ['Model+market @ Bet365', pct((strat.blend_b365 || {}).hold_roi), `${int((strat.blend_b365 || {}).hold_bets)} holdout bets at a price you can actually get`],
  ].map(([l, v, n]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${esc(n)}</div></div>`).join('');

  // findings generated from the numbers
  const F = [];
  const gapBest = (rps[bestModel] - rps.mkt) * 1000, gapBlend = (rps.blend - rps.mkt) * 1000;
  const blendTxt = Math.abs(gapBlend) < 0.1 ? 'only matches the market: on average the model adds nothing the price does not already know' : `${gapBlend < 0 ? 'beats' : 'trails'} it by ${Math.abs(gapBlend).toFixed(1)}`;
  F.push([gapBest < 0 ? 'good' : 'bad', `Ratings alone ${gapBest < 0 ? 'beat' : 'trail'} the bookmaker by ${Math.abs(gapBest).toFixed(1)} RPS×10⁻³ on the holdout (best: ${LABEL[bestModel]}). Feeding the bookmaker's own prices into the model ${blendTxt}.`]);
  const lg = (D.byLeague || []).filter(r => r.gap_ens != null);
  if (lg.length) {
    const t1 = lg.filter(r => r.tier === 1), lo = lg.filter(r => r.tier >= 2);
    const avg = (a, k) => a.reduce((s, r) => s + r[k], 0) / Math.max(1, a.length);
    F.push(['info', `Bet365's margin averages ${pct(avg(t1, 'b365_margin'), 1).replace('+', '')} in top flights and ${pct(avg(lo, 'b365_margin'), 1).replace('+', '')} in lower divisions. The ratings-only model's gap to the market is ${(avg(t1, 'gap_ens') * 1000).toFixed(1)} in top flights vs ${(avg(lo, 'gap_ens') * 1000).toFixed(1)} below (RPS×10⁻³, lower is better).`]);
  }
  const cap5 = (D.capped || []).find(r => r.strategy === 'blend_max' && Math.abs(r.max_edge - 0.05) < 1e-9);
  if (cap5) F.push(['bad', `1X2 profits at the best price come mostly from edges above 8%, where the top quote is usually stale or an outlier. Ignoring edges above 5%, the model+market strategy returns ${pct(cap5.hold_roi)} on the holdout (${int(cap5.hold_bets)} bets).`]);
  const RH = (D.research || {}).holdout;
  if (RH) F.push(['info', `New, pre-registered: bookmakers over-react to finishing and goalkeeper luck (holdout t = ${RH.H1.t.d_fin_luck} and ${RH.H1.t.d_gk_luck}). The effect is real but smaller than Bet365's margin; the best rule made ${pct(RH.H3.roi)} on Asian handicaps (not significant).`]);
  const posHold = (D.strategies || []).filter(s => s.hold_roi > 0 && s.hold_bets >= 200);
  F.push([posHold.length ? 'good' : 'bad', posHold.length ? `${posHold.length} of ${D.strategies.length} fixed strategies made a profit on the holdout: ${posHold.map(s => `${s.name} (${pct(s.hold_roi)})`).join(', ')}.` : `None of the ${(D.strategies || []).length} fixed strategies made a profit on the holdout.`]);
  const g = (D.grid || []).filter(r => r.dev_bets >= 300 && r.hold_bets >= 100);
  if (g.length > 3) {
    const mx = g.reduce((a, r) => a + r.dev_roi, 0) / g.length, my = g.reduce((a, r) => a + r.hold_roi, 0) / g.length;
    let sxy = 0, sxx = 0, syy = 0; g.forEach(r => { sxy += (r.dev_roi - mx) * (r.hold_roi - my); sxx += (r.dev_roi - mx) ** 2; syy += (r.hold_roi - my) ** 2; });
    const corr = sxy / Math.sqrt(sxx * syy);
    F.push([corr > 0.3 ? 'good' : 'info', `Across ${g.length} strategy variants, development ROI and holdout ROI correlate at ${corr.toFixed(2)}. ${corr > 0.3 ? 'Development results carried over reasonably.' : 'Picking the best development variant would mostly have picked noise.'}`]);
  }
  $('findings').innerHTML = F.map(([k, t]) => `<li><span class="pill ${k}" aria-hidden="true">${k === 'good' ? '✓' : k === 'bad' ? '×' : 'i'}</span><span>${esc(t)}</span></li>`).join('');

  const COLORS = { mkt: '--s1', pxg: '--s2', ens: '--s3', blend: '--s4', gbm: '--s5', elo: '--s6', pi: '--s6', pois: '--s2' };
  function drawAll() {
    // gap chart
    const order = ['blend', 'gbm', 'ens', 'pxg', 'pois', 'pi', 'elo'].filter(m => rps[m] != null);
    dotPlot($('gapChart'), order.map(m => ({
      label: LABEL[m] || m, value: +(((rps[m] - rps.mkt) * 1000).toFixed(2)),
      color: css((rps[m] - rps.mkt) < 0 ? '--pos' : '--neg'),
      tip: `<b>${esc(LABEL[m] || m)}</b><div>RPS ${num(rps[m], 4)} vs market ${num(rps.mkt, 4)}</div>`,
    })), { padL: Math.min(190, $('gapChart').clientWidth * 0.42) });
    // by year
    const yr = D.byYear || [];
    const ys = [['mkt', 'Bet365'], ['pxg', 'Pseudo-xG Dixon-Coles'], ['ens', 'Ratings ensemble'], ['blend', 'Ensemble + market']];
    legend('yearLegend', ys.map(([k, n]) => [n, css(COLORS[k])]));
    lineChart($('yearChart'), yr.map(r => r.year), ys.map(([k, n]) => ({ name: n, color: css(COLORS[k]), values: yr.map(r => r[k]) })),
      { fmt: (v) => Number(v).toFixed(3), xfmt: (y) => `${y}-${String(y + 1).slice(2)}`, padL: 50 });
    lineChart($('marginChart'), yr.map(r => r.year), [{ name: 'Bet365 margin', color: css('--s1'), values: yr.map(r => r.b365_margin) }],
      { fmt: (v) => (v * 100).toFixed(1) + '%', xfmt: (y) => `${y}-${String(y + 1).slice(2)}`, height: 200, padL: 50 });
    // calibration
    const cal = (D.calibration || []).filter(r => r.outcome === 'h');
    const cm = [['mkt', 'Bet365'], ['ens', 'Ratings ensemble'], ['blend', 'Ensemble + market']];
    legend('calLegend', cm.map(([k, n]) => [n, css(COLORS[k])]).concat([['Perfect', css('--muted')]]));
    const bins = [...new Set(cal.map(r => r.bin))].sort((a, b) => a - b);
    const calSeries = cm.map(([k, n]) => ({ name: n, color: css(COLORS[k]), values: bins.map(b => { const r = cal.find(c => c.model === k && c.bin === b); return r && r.n >= 50 ? r.obs : null; }) }));
    calSeries.push({ name: 'Perfect', color: css('--muted'), values: bins.map(b => (b + 0.5) / 10) });
    lineChart($('calChart'), bins.map(b => b * 10 + 5), calSeries, { fmt: (v) => (v * 100).toFixed(0) + '%', min: 0, max: 1, height: 220, xfmt: (m) => `${m}% predicted` , padL: 50 });
    // tier chart
    const tr = D.byTier || [];
    const tierName = { 1: 'Top flights', 2: 'Second tiers', 3: 'Tiers 3–5', 0: 'All 2–5' };
    const groups = [];
    ['consensus_max', 'blend_max', 'blend_b365', 'ou_blend_max'].forEach(k => [1, 2, 3, 0].forEach(t => {
      const dv = tr.find(r => r.strategy === k && r.tier_grp === t && r.period === 'dev');
      const hv = tr.find(r => r.strategy === k && r.tier_grp === t && r.period === 'holdout');
      if (dv || hv) groups.push({ label: `${tierName[t]}`, strat: k, dev: dv ? dv.roi : null, hold: hv ? hv.roi : null, dn: dv ? dv.bets : 0, hn: hv ? hv.bets : 0 });
    }));
    const sel = $('edgeStrat').value;
    const tg = groups.filter(gr => gr.strat === sel);
    legend('tierLegend', [['Development', css('--s1')], ['Holdout', css('--s2')]]);
    columns($('tierChart'), tg, [{ key: 'dev', color: css('--s1'), name: 'Development' }, { key: 'hold', color: css('--s2'), name: 'Holdout' }],
      { fmt: (v) => (v * 100).toFixed(0) + '%', tip: (gr, s) => `<b>${gr.label}</b><div>${s.name}: <b>${pct(gr[s.key])}</b></div><div>${int(s.key === 'dev' ? gr.dn : gr.hn)} bets</div>` });
    // research: H1 t-stats dev vs holdout
    const RS = D.research || {};
    if (RS.dev && RS.holdout) {
      const nm = { d_surp_pts_10: 'Results', d_surp_sot_10: 'Shots', d_fin_luck: 'Finishing', d_gk_luck: 'Keeper' };
      const grp = Object.keys(nm).map(k => ({ label: nm[k], dev: RS.dev.H1.t[k], hold: RS.holdout.H1.t[k] }));
      legend('h1Legend', [['Development 2009–15', css('--s1')], ['Holdout 2016–26', css('--s2')]]);
      columns($('h1Chart'), grp, [{ key: 'dev', color: css('--s1'), name: 'Development' }, { key: 'hold', color: css('--s2'), name: 'Holdout' }],
        { fmt: (v) => Number(v).toFixed(0), tip: (gr, sd) => `<b>${gr.label}</b><div>${sd.name}: t = <b>${Number(gr[sd.key]).toFixed(2)}</b></div>` });
    }
    // edge bands
    const bs = $('bandStrat').value, B = (D.bands || []).filter(r => r.strategy === bs);
    const blabels = [...new Set(B.map(r => r.band_lo))].sort((a, b) => a - b);
    const bg = blabels.map(lo => { const dv = B.find(r => r.band_lo === lo && r.period === 'dev'), hv = B.find(r => r.band_lo === lo && r.period === 'holdout');
      const hi = (dv || hv).band_hi; return { label: `${Math.round(Math.max(lo, 0.02) * 100)}–${hi >= 1 ? '∞' : Math.round(hi * 100)}%`, dev: dv ? dv.roi : null, hold: hv ? hv.roi : null, dn: dv ? dv.bets : 0, hn: hv ? hv.bets : 0 }; });
    legend('bandLegend', [['Development', css('--s1')], ['Holdout', css('--s2')]]);
    columns($('bandChart'), bg, [{ key: 'dev', color: css('--s1'), name: 'Development' }, { key: 'hold', color: css('--s2'), name: 'Holdout' }],
      { fmt: (v) => (v * 100).toFixed(0) + '%', tip: (gr, sd) => `<b>Edge ${gr.label}</b><div>${sd.name}: <b>${pct(gr[sd.key])}</b></div><div>${int(sd.key === 'dev' ? gr.dn : gr.hn)} bets</div>` });
    // equity
    const eq = D.curves || {}, keys = Object.keys(eq);
    const cols = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6'];
    const allDates = [...new Set(keys.flatMap(k => eq[k].date))].sort();
    const ser = keys.map((k, i) => {
      const m = new Map(eq[k].date.map((d, j) => [d, eq[k].units[j]])); let last = 0;
      return { name: (strat[k] || {}).name || k, color: css(cols[i % cols.length]), values: allDates.map(d => { if (m.has(d)) last = m.get(d); return last; }) };
    });
    const step = Math.max(1, Math.floor(allDates.length / 360));
    const idx = allDates.map((_, i) => i).filter(i => i % step === 0 || i === allDates.length - 1);
    legend('eqLegend', ser.map(s => [s.name, s.color]));
    lineChart($('eqChart'), idx.map(i => allDates[i]), ser.map(s => ({ ...s, values: idx.map(i => s.values[i]) })),
      { fmt: (v) => (v >= 0 ? '+' : '') + Number(v).toFixed(0) + 'u', height: 280, zero: true, padL: 52, xfmt: (d) => d.slice(0, 7) });
    // grid scatter
    const gg = (D.grid || []).filter(r => r.dev_bets >= 100 && r.hold_bets >= 50);
    const mcol = { '1x2': css('--s1'), 'ou25': css('--s2'), 'ah': css('--s3') };
    legend('gridLegend', [['1X2', mcol['1x2']], ['Over/Under 2.5', mcol.ou25], ['Asian handicap', mcol.ah]], true);
    scatter($('gridChart'), gg.map(r => ({ x: r.dev_roi, y: r.hold_roi, color: mcol[r.market] || css('--s4'),
      tip: `<b>${esc(r.strategy)}</b><div>Development ${pct(r.dev_roi)} (${int(r.dev_bets)} bets)</div><div>Holdout ${pct(r.hold_roi)} (${int(r.hold_bets)} bets)</div>` })),
      { fmt: (v) => (v * 100).toFixed(0) + '%', xlabel: 'Development ROI', ylabel: 'Holdout ROI' });
  }

  // ---------- tables
  function heat(v, scale) {
    if (v == null) return '';
    const w = Math.min(40, Math.abs(v) / scale * 40);
    return `<span class="${v >= 0 ? 'bar-pos' : 'bar-neg'}" style="width:${w.toFixed(0)}px"></span>`;
  }
  function leagueTable() {
    const k = $('edgeStrat').value;
    const rows = (D.byLeague || []);
    $('leagueTable').innerHTML = `<thead><tr><th class="l">League</th><th>Tier</th><th>Matches</th><th>Bet365 margin</th><th>Model gap ×10⁻³</th><th>Blend gap ×10⁻³</th><th>Dev ROI</th><th>Dev bets</th><th>Holdout ROI</th><th>Holdout bets</th></tr></thead><tbody>` +
      rows.map(r => {
        const dv = r[`${k}_dev_roi`], hv = r[`${k}_holdout_roi`];
        return `<tr><td class="l">${esc(r.name)} <span class="muted">${esc(r.league)}</span></td><td>${r.tier}</td><td>${int(r.matches)}</td><td>${pct(r.b365_margin).replace('+', '')}</td>
        <td>${num(r.gap_ens * 1000, 1)}${heat(-r.gap_ens, 0.006)}</td><td>${num(r.gap_blend * 1000, 1)}${heat(-r.gap_blend, 0.002)}</td>
        <td>${pct(dv)}${heat(dv, 0.15)}</td><td>${int(r[`${k}_dev_bets`])}</td><td>${pct(hv)}${heat(hv, 0.15)}</td><td>${int(r[`${k}_holdout_bets`])}</td></tr>`;
      }).join('') + '</tbody>';
  }
  function stratTable() {
    $('stratTable').innerHTML = `<thead><tr><th class="l">Strategy</th><th>Dev bets</th><th>Dev ROI</th><th>Holdout bets</th><th>Holdout ROI</th><th>95% range</th><th>p-value</th><th>Avg odds</th><th>Max drawdown</th></tr></thead><tbody>` +
      (D.strategies || []).map(s => `<tr><td class="l">${esc(s.name)}<div class="muted" style="font-size:11.5px">${esc(s.label)}</div></td><td>${int(s.dev_bets)}</td><td>${pct(s.dev_roi)}</td><td>${int(s.hold_bets)}</td><td><b>${pct(s.hold_roi)}</b></td>
        <td>${pct(s.hold_roi_lo)} to ${pct(s.hold_roi_hi)}</td><td>${num(s.hold_p_value, 3)}</td><td>${num(s.hold_avg_odds)}</td><td>${s.hold_max_drawdown_units == null ? '–' : num(s.hold_max_drawdown_units, 0) + 'u'}</td></tr>`).join('') + '</tbody>';
  }
  function research() {
    const RS = D.research || {};
    if (!RS.holdout) { $('idea').hidden = true; return; }
    const dv = RS.dev || {}, hv = RS.holdout, adj = hv.holm_adjusted_p || {};
    const label = { H1: 'Market over-reacts to luck (effect test)', H2: 'Bet 1X2 at Bet365, edge ≥ 3%', H3: 'Bet Asian handicap at Bet365', H4: '1X2 at Bet365 using best-price data, edge ≥ 4%' };
    const pass = (k) => adj[k] < 0.05 && (k === 'H1' ? hv.H1.signs_as_predicted : hv[k].roi > 0);
    const cell = (k, w) => k === 'H1' ? `${int(w.H1.n)} matches · joint p ${w.H1.joint_p < 1e-6 ? '< 0.000001' : num(w.H1.joint_p, 4)}` : `${pct(w[k].roi)} · ${int(w[k].bets)} bets`;
    $('hypTable').innerHTML = `<thead><tr><th class="l">Pre-registered hypothesis</th><th class="l">Verdict</th><th>Holdout 2016–26</th><th>Holm p</th><th>Development 2009–15</th></tr></thead><tbody>` +
      ['H1', 'H2', 'H3', 'H4'].map(k => `<tr><td class="l wraptext">${esc(label[k])}</td>
        <td class="l"><span class="pill ${pass(k) ? 'good' : 'bad'}" aria-hidden="true">${pass(k) ? '✓' : '×'}</span> ${pass(k) ? 'Confirmed' : 'Failed'}</td>
        <td><b>${cell(k, hv)}</b></td><td>${adj[k] < 0.001 ? '< 0.001' : num(adj[k], 3)}</td><td>${dv[k] ? cell(k, dv) : '–'}</td></tr>`).join('') + '</tbody>';
    const P = hv.H3_margin_projection || [];
    $('projTable').innerHTML = `<thead><tr><th class="l">Bookmaker margin</th><th>ROI</th><th>Bets</th></tr></thead><tbody>` +
      P.map(r => `<tr><td class="l">${(r.margin * 100).toFixed(1)}%${Math.abs(r.margin - 0.036) < 1e-9 ? ' (Bet365, actual)' : Math.abs(r.margin - 0.02) < 1e-9 ? ' (Pinnacle-like)' : ''}</td><td>${pct(r.roi)}</td><td>${int(r.bets)}</td></tr>`).join('') + '</tbody>';
    const fw = RS.forward;
    $('fwdNote').textContent = fw && fw.played_after_freeze ? `Forward test since ${fw.since}: ${int(fw.played_after_freeze)} new matches scored.` :
      'Forward test: rules frozen on 3 Sep 2026. No newer matches in this build; run footy fetch, build and forward on a machine with internet access to score them.';
  }
  function capTable() {
    const C = D.capped || [];
    $('capTable').innerHTML = `<thead><tr><th class="l">Same strategies, ignoring edges above a cap</th><th>Cap</th><th>Dev bets</th><th>Dev ROI</th><th>Holdout bets</th><th>Holdout ROI</th><th>Holdout p</th></tr></thead><tbody>` +
      C.map(r => `<tr><td class="l">${esc(r.name)}</td><td>${Math.round(r.max_edge * 100)}%</td><td>${int(r.dev_bets)}</td><td>${pct(r.dev_roi)}</td><td>${int(r.hold_bets)}</td><td><b>${pct(r.hold_roi)}</b></td><td>${num(r.hold_p_value, 3)}</td></tr>`).join('') + '</tbody>';
  }
  function gridTable() {
    const g = (D.grid || []).filter(r => r.eligible).slice(0, 15);
    $('gridTable').innerHTML = `<thead><tr><th class="l">Top 15 by development ROI</th><th>Dev bets</th><th>Dev ROI</th><th>Holdout bets</th><th>Holdout ROI</th><th>Holdout p</th></tr></thead><tbody>` +
      g.map(r => `<tr><td class="l code">${esc(r.strategy)}</td><td>${int(r.dev_bets)}</td><td>${pct(r.dev_roi)}</td><td>${int(r.hold_bets)}</td><td><b>${pct(r.hold_roi)}</b></td><td>${num(r.hold_p_value, 3)}</td></tr>`).join('') + '</tbody>';
  }
  function ouTable() {
    const o = (D.ou || []).filter(r => r.period === 'holdout');
    const nm = { oumkt: 'Bet365 (margin removed)', pxg: 'Pseudo-xG Dixon-Coles', ouens: 'Ratings ensemble', oublend: 'Ensemble + market' };
    $('ouTable').innerHTML = `<thead><tr><th class="l">Over/Under 2.5, holdout</th><th>Matches</th><th>Log loss</th><th>Brier</th></tr></thead><tbody>` +
      o.map(r => `<tr><td class="l">${esc(nm[r.model] || r.model)}</td><td>${int(r.n)}</td><td>${num(r.logloss, 4)}</td><td>${num(r.brier, 4)}</td></tr>`).join('') + '</tbody>';
  }
  function ratings() {
    const R = D.ratings || [];
    const lgs = [...new Map(R.map(r => [r.league, `${r.league_name} (${r.league})`])).entries()];
    const s = $('lgSel');
    if (!s.options.length) { s.innerHTML = lgs.map(([c, n]) => `<option value="${esc(c)}">${esc(n)}</option>`).join(''); if (lgs.some(([c]) => c === 'E0')) s.value = 'E0'; }
    const rows = R.filter(r => r.league === s.value);
    $('ratingTable').innerHTML = `<thead><tr><th class="l">#</th><th class="l">Team</th><th>Elo</th><th>Attack</th><th>Defence</th><th>Shots-on-target att.</th><th>Matches</th><th>Last match</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr><td class="l muted">${i + 1}</td><td class="l">${esc(r.team)}</td><td>${num(r.elo, 0)}</td><td>${pct(Math.exp(r.pxg_att) - 1, 0)}</td><td>${pct(Math.exp(r.pxg_def) - 1, 0)}</td><td>${r.sot_att == null ? '–' : pct(Math.exp(r.sot_att) - 1, 0)}</td><td>${int(r.matches)}</td><td>${esc(String(r.last_match).slice(0, 10))}</td></tr>`).join('') + '</tbody>';
  }
  function upcoming() {
    const U = D.upcoming || [];
    if (!U.length) {
      $('upcomingBody').innerHTML = `<p class="sub">No fixtures in this build. Fixtures and current odds come from football-data.co.uk, which this build could not reach.
        Run <code>footy all</code> on a machine with internet access to fill this table with predictions and value flags.</p>`;
      return;
    }
    $('upcomingBody').innerHTML = `<div class="tablewrap"><table><thead><tr><th class="l">Date</th><th class="l">Match</th><th>Home</th><th>Draw</th><th>Away</th><th>Best H/D/A</th><th>Best value</th></tr></thead><tbody>` +
      U.map(r => { const evs = [['H', r.ev_h], ['D', r.ev_d], ['A', r.ev_a]].filter(x => x[1] != null).sort((a, b) => b[1] - a[1]); const top = evs[0];
        return `<tr><td class="l">${esc(String(r.date).slice(0, 10))}</td><td class="l">${esc(r.home)} – ${esc(r.away)} <span class="muted">${esc(r.league)}</span></td><td>${pct(r.blend_ph ?? r.ens_ph, 0).replace('+', '')}</td><td>${pct(r.blend_pd ?? r.ens_pd, 0).replace('+', '')}</td><td>${pct(r.blend_pa ?? r.ens_pa, 0).replace('+', '')}</td><td>${num(r.best_h)} / ${num(r.best_d)} / ${num(r.best_a)}</td><td>${top && top[1] > 0.03 ? `<span class="tag">${top[0]} ${pct(top[1])}</span>` : '<span class="muted">none</span>'}</td></tr>`; }).join('') + '</tbody></table></div>';
  }
  $('edgeStrat').addEventListener('change', () => { leagueTable(); drawAll(); });
  $('bandStrat').addEventListener('change', drawAll);
  $('lgSel').addEventListener('change', ratings);
  research(); leagueTable(); stratTable(); capTable(); gridTable(); ouTable(); ratings(); upcoming(); drawAll();
  $('foot').textContent = `Built by the footy engine. Development window ${S.dev_window ? S.dev_window.join(' to ') : ''}; holdout from ${holdStart}. Past results do not guarantee future returns.`;
  let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
  if (window.matchMedia) matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
  new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
</script>
"""
