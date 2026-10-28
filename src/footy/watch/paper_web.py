"""The paper-trading page (/paper), its JSON and CSV."""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime

from .core import sport_name, utcnow
from .notify import book_name, local_time, price_text
from .paper import VENUES, summarize, verdict
from .rules import selection_text

PAPER_CSS = """
:root{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--grid:#e1e0d9;--axis:#c3c2b7;--plot:#fcfcfb}
@media (prefers-color-scheme:dark){:root{--s1:#3987e5;--s2:#d95926;--s3:#199e70;--grid:#2c2c2a;--axis:#383835;--plot:#1a1a19}}
a{color:inherit}.back{font-size:13px;color:var(--ink-2);text-decoration:none}
.seg{display:flex;flex-wrap:wrap;gap:6px}.seg a{text-decoration:none;border:1px solid var(--rule);background:var(--surface);
border-radius:10px;padding:7px 12px;font-size:13.5px;color:var(--ink-2)}.seg a small{display:block;font-size:12px;color:var(--muted)}
.seg a.on{border-color:var(--ink);color:var(--ink);font-weight:600}.seg.sm a{padding:5px 10px;border-radius:999px;font-size:12.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:12px;padding:12px 14px}
.tile .lab{font-size:12.5px;color:var(--muted)}.tile .val{font-size:22px;font-weight:650;margin-top:2px}
.tile .sub{font-size:12.5px;color:var(--ink-2)}.tile.hero .val{font-size:34px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:12px;padding:14px}
.card h2{margin:0 0 2px}.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--ink-2);margin:8px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:6px}.key{width:18px;height:0;border-top:2px solid}
.key.k1{border-color:var(--s1)}.key.k2{border-color:var(--s2);border-top-style:dashed}
.key.k3{border-color:var(--s3);border-top-style:dotted}.key.band{height:10px;border:0;background:var(--s2);opacity:.14;border-radius:2px}
#pchart{position:relative;width:100%;min-height:220px}#pchart svg{display:block;width:100%;outline:none}
#pchart svg:focus-visible{outline:2px solid var(--ink);outline-offset:2px;border-radius:6px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--rule);border-radius:10px;
padding:8px 10px;font-size:12.5px;box-shadow:0 4px 18px rgba(0,0,0,.12);min-width:190px;max-width:260px;display:none;z-index:2}
.tip .h{color:var(--muted);margin-bottom:4px}.tip .r{display:flex;align-items:center;gap:7px;white-space:nowrap}
.tip .r b{font-variant-numeric:tabular-nums;min-width:72px}.tip .r .key{width:12px}
.verdict{font-size:14.5px}
details>summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px}details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸";color:var(--muted);font-size:14px;margin-top:8px}details[open]>summary::before{content:"▾"}
details>summary h2{display:inline}details .tablewrap{margin-top:10px}.note{font-size:12.5px;color:var(--muted);margin-top:6px}
"""

CHART_JS = r"""
(function(){
var el=document.getElementById('pchart');if(!el)return;
var raw=JSON.parse(document.getElementById('pchart-data').textContent);
var cur=el.getAttribute('data-cur');var NS='http://www.w3.org/2000/svg';
var pts=[{n:0,actual:0,ev:0,clv:0,band:0,when:'',match:'Start',status:'',profit:0}].concat(raw);
var S=JSON.parse(el.getAttribute('data-series')||'null')||[{k:'actual',name:'Actual profit',c:'var(--s1)',d:''},
{k:'ev',name:'Expected (EV when bet)',c:'var(--s2)',d:'6 4'},{k:'clv',name:'Expected (closing line)',c:'var(--s3)',d:'2 3'}];
function money(v){var a=Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
return (v<0?'−':'+')+cur+a;}
function mk(t,a,p){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function nice(lo,hi,n){var span=hi-lo||1,step=Math.pow(10,Math.floor(Math.log10(span/n))),err=span/n/step;
step*=err>=7.5?10:err>=3.5?5:err>=1.5?2:1;var out=[];for(var v=Math.floor(lo/step)*step;v<=hi+step*0.001;v+=step)out.push(+v.toFixed(10));return out;}
var tip=document.createElement('div');tip.className='tip';el.appendChild(tip);var svg,idx=-1,geo;
function draw(){
 if(svg)svg.remove();var W=el.clientWidth,H=Math.round(Math.max(220,Math.min(320,W*0.55)));
 var wide=W>=560,m={l:58,r:wide?150:14,t:12,b:30},pw=W-m.l-m.r,ph=H-m.t-m.b,n=pts.length-1;
 var lo=0,hi=0;pts.forEach(function(p){lo=Math.min(lo,p.actual,p.ev-p.band,p.clv);hi=Math.max(hi,p.actual,p.ev+p.band,p.clv);});
 var ticks=nice(lo,hi,5);lo=Math.min(lo,ticks[0]);hi=Math.max(hi,ticks[ticks.length-1]);
 var X=function(i){return m.l+(n?i/n*pw:pw/2);},Y=function(v){return m.t+(hi-v)/(hi-lo||1)*ph;};
 geo={X:X,Y:Y,m:m,pw:pw,ph:ph,n:n};
 svg=mk('svg',{viewBox:'0 0 '+W+' '+H,height:H,tabindex:'0',role:'img','aria-label':'Cumulative profit, actual versus expected'},null);
 el.insertBefore(svg,tip);
 ticks.forEach(function(t){mk('line',{x1:m.l,x2:m.l+pw,y1:Y(t),y2:Y(t),stroke:t===0?'var(--axis)':'var(--grid)','stroke-width':1},svg);
  var tx=mk('text',{x:m.l-8,y:Y(t)+4,'text-anchor':'end','font-size':11,fill:'var(--muted)'},svg);
  tx.textContent=(t<0?'−':'')+cur+Math.abs(t).toLocaleString();});
 var xt=nice(0,n,Math.max(2,Math.floor(pw/90)));xt.forEach(function(t){if(t>n||t!==Math.round(t))return;
  var tx=mk('text',{x:X(t),y:H-10,'text-anchor':'middle','font-size':11,fill:'var(--muted)'},svg);tx.textContent=t;});
 var xl=mk('text',{x:m.l+pw,y:H-10,'text-anchor':'end','font-size':11,fill:'var(--muted)'},svg);
 if(!wide)xl.textContent='';
 var up=pts.map(function(p,i){return X(i)+','+Y(p.ev+p.band);}),dn=pts.map(function(p,i){return X(i)+','+Y(p.ev-p.band);}).reverse();
 mk('polygon',{points:up.concat(dn).join(' '),fill:'var(--s2)','fill-opacity':0.12,stroke:'none'},svg);
 S.forEach(function(s){mk('path',{d:pts.map(function(p,i){return (i?'L':'M')+X(i).toFixed(1)+','+Y(p[s.k]).toFixed(1);}).join(''),
  fill:'none',stroke:s.c,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round','stroke-dasharray':s.d},svg);});
 var last=pts[n];S.forEach(function(s){mk('circle',{cx:X(n),cy:Y(last[s.k]),r:4,fill:s.c,stroke:'var(--plot)','stroke-width':2},svg);});
 if(wide){var labs=S.map(function(s){return {s:s,y:Y(last[s.k])};}).sort(function(a,b){return a.y-b.y;});
  var ok=true;for(var i=1;i<labs.length;i++)if(labs[i].y-labs[i-1].y<15)ok=false;
  if(ok)labs.forEach(function(l){var t=mk('text',{x:X(n)+10,y:l.y+4,'font-size':12,fill:'var(--ink-2)'},svg);
   t.textContent=money(last[l.s.k])+' '+(l.s.k==='actual'?'actual':l.s.k==='ev'?'EV':'CLV');});}
 geo.cross=mk('line',{y1:m.t,y2:m.t+ph,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'},svg);
 geo.dots=S.map(function(s){return mk('circle',{r:4,fill:s.c,stroke:'var(--plot)','stroke-width':2,visibility:'hidden'},svg);});
 mk('rect',{x:m.l,y:m.t,width:pw,height:ph,fill:'transparent'},svg);
 svg.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect();
  show(Math.round(Math.max(0,Math.min(1,(e.clientX-r.left-m.l)/pw))*n));});
 svg.addEventListener('pointerleave',hide);svg.addEventListener('blur',hide);
 svg.addEventListener('keydown',function(e){if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();
  show(Math.max(0,Math.min(n,(idx<0?n:idx)+(e.key==='ArrowRight'?1:-1))));}});
 svg.addEventListener('focus',function(){show(n);});
}
function hide(){tip.style.display='none';if(!geo)return;geo.cross.setAttribute('visibility','hidden');
 geo.dots.forEach(function(d){d.setAttribute('visibility','hidden');});idx=-1;}
function show(i){idx=i;var p=pts[i],x=geo.X(i);geo.cross.setAttribute('x1',x);geo.cross.setAttribute('x2',x);
 geo.cross.setAttribute('visibility','visible');
 S.forEach(function(s,j){var d=geo.dots[j];d.setAttribute('cx',x);d.setAttribute('cy',geo.Y(p[s.k]));d.setAttribute('visibility','visible');});
 tip.textContent='';var h=document.createElement('div');h.className='h';
 h.textContent=i?('Bet '+i+' · '+p.when):'Start';tip.appendChild(h);
 if(i){var mt=document.createElement('div');mt.textContent=p.match+' · '+p.status+' '+money(p.profit);mt.style.marginBottom='4px';tip.appendChild(mt);}
 S.forEach(function(s){var r=document.createElement('div');r.className='r';var k=document.createElement('span');
  k.className='key '+(s.k==='actual'?'k1':s.k==='ev'?'k2':'k3');var b=document.createElement('b');b.textContent=money(p[s.k]);
  var t=document.createElement('span');t.textContent=s.name;r.appendChild(k);r.appendChild(b);r.appendChild(t);tip.appendChild(r);});
 if(i){var lr=document.createElement('div');lr.className='r';lr.style.color='var(--muted)';
  lr.textContent='Luck range '+money(p.ev-p.band)+' to '+money(p.ev+p.band);tip.appendChild(lr);}
 tip.style.display='block';var tw=tip.offsetWidth,left=x+12;if(left+tw>el.clientWidth)left=x-tw-12;
 tip.style.left=Math.max(0,left)+'px';tip.style.top=(geo.m.t+4)+'px';}
draw();var t0;window.addEventListener('resize',function(){clearTimeout(t0);t0=setTimeout(function(){hide();draw();},150);});
})();
"""


def _money(x, cur: str, signed: bool = True) -> str:
    if x is None:
        return "–"
    sign = ("+" if x >= 0 else "−") if signed else ("−" if x < 0 else "")
    return f"{sign}{cur}{abs(x):,.2f}"


def _pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{'+' if x >= 0 else '−'}{abs(x) * 100:.{digits}f}%"


def _cls(x) -> str:
    return "" if x is None or abs(x) < 1e-12 else ("pos" if x > 0 else "neg")


def _place(b: dict) -> str:
    return f"{price_text(b['price'], b['book'])} {book_name(b['book'])}"


def _sel(b: dict) -> str:
    return selection_text(b["market"], b["line"], b["outcome"], b["home"], b["away"])


def render_paper(w, base_css: str, venue: str = "best", strategy: str = "all") -> str:
    e = html.escape
    pcfg = w.cfg["paper"]
    cur, start = pcfg.get("currency", "€"), float(pcfg["bankroll"])
    tz = w.cfg["server"].get("timezone", "UTC")
    venues = [v for v in pcfg.get("venues", list(VENUES)) if v in VENUES]
    venue = venue if venue in venues else (venues[0] if venues else "best")
    strategies = w.paper.strategies()
    if strategy != "all" and strategy not in {s for s, _ in strategies}:
        strategy = "all"
    bets = w.paper.bets(venue=venue, strategy=strategy)
    s = summarize(bets, start)
    if strategy != "all":  # the bankroll belongs to the place, whatever the filter
        s["bankroll"] = summarize(w.paper.bets(venue=venue), start)["bankroll"]

    def link(v=None, st=None):
        return f"/paper?venue={e(v or venue)}&amp;strategy={e(st or strategy)}"

    tabs = []
    for v in venues:
        vs = summarize(w.paper.bets(venue=v), start)
        tabs.append(f"<a class='{'on' if v == venue else ''}' href='{link(v)}'>{e(VENUES[v])}"
                    f"<small class='{_cls(vs['profit'])}'>{_money(vs['profit'], cur)} · {vs['bets']} bets</small></a>")
    chips = [f"<a class='{'on' if strategy == 'all' else ''}' href='{link(st='all')}'>All strategies</a>"]
    for code, name in strategies:
        chips.append(f"<a class='{'on' if strategy == code else ''}' href='{link(st=code)}'>{e(name)}</a>")
    staking = (f"{cur}{float(pcfg['flat_stake']):,.0f} flat stakes" if pcfg.get("staking", "flat") != "kelly" else
               f"{float(pcfg['kelly_fraction']):g} Kelly stakes (max {float(pcfg['max_stake_pct']):.0%})")
    tiles = f"""
<div class="tile hero"><div class="lab">Bankroll, {e(VENUES[venue])}</div><div class="val">{_money(s['bankroll'], cur, False)}</div>
<div class="sub">started at {_money(start, cur, False)}</div></div>
<div class="tile"><div class="lab">Profit</div><div class="val {_cls(s['profit'])}">{_money(s['profit'], cur)}</div>
<div class="sub">ROI {_pct(s['roi'])} · {s['settled']} settled · {_money(s['staked'], cur, False)} staked</div></div>
<div class="tile"><div class="lab">Average EV when bet</div><div class="val">{_pct(s['avg_ev'])}</div>
<div class="sub">expected {_money(s['expected_profit'], cur)} on settled bets</div></div>
<div class="tile"><div class="lab">Average closing-line value</div><div class="val {_cls(s['avg_clv'])}">{_pct(s['avg_clv'])}</div>
<div class="sub">beat the close on {_pct(s['beat_close'], 0).lstrip('+')} of {s['clv_n']} bets</div></div>
<div class="tile"><div class="lab">Open bets</div><div class="val">{s['open']}</div>
<div class="sub">{_money(s['open_stake'], cur, False)} at stake{f" · {s['void']} void" if s['void'] else ''}</div></div>"""

    series = [{**p, "when": local_time(datetime.fromisoformat(p["kickoff"]), tz)} for p in s["series"]]
    if series:
        data = json.dumps(series).replace("</", "<\\/")
        chart = (f"<div id='pchart' data-cur='{e(cur)}'></div>"
                 f"<script type='application/json' id='pchart-data'>{data}</script>"
                 f"<script>{CHART_JS}</script>")
    else:
        chart = "<p class='muted'>No settled bets yet. The chart starts once the first results are in.</p>"

    # comparison table: every place x strategy
    comp = []
    for v in venues:
        for code, name in [("all", "All strategies")] + strategies:
            cs = summarize(w.paper.bets(venue=v, strategy=code), start)
            if cs["bets"] == 0 and code != "all":
                continue
            comp.append(f"<tr><td class='l'>{e(VENUES[v]) if code == 'all' else ''}</td><td class='l'>{e(name)}</td>"
                        f"<td class='{_cls(cs['profit'])}'>{_money(cs['profit'], cur)}</td><td>{_pct(cs['roi'])}</td>"
                        f"<td class='{_cls(cs['avg_clv'])}'>{_pct(cs['avg_clv'])}</td><td>{cs['settled']}</td>"
                        f"<td>{cs['open']}</td><td>{_money(cs['staked'], cur, False)}</td><td>{_pct(cs['avg_ev'])}</td>"
                        f"<td>{_pct(cs['beat_close'], 0).lstrip('+')}</td></tr>")

    by_sport = []
    sports_seen = sorted({b.get("sport") or "soccer" for b in bets})
    if len(sports_seen) > 1:
        for sp in sports_seen:
            ss = summarize([b for b in bets if (b.get("sport") or "soccer") == sp], start)
            by_sport.append(f"<tr><td class='l'>{e(sport_name(sp))}</td>"
                            f"<td class='{_cls(ss['profit'])}'>{_money(ss['profit'], cur)}</td><td>{_pct(ss['roi'])}</td>"
                            f"<td class='{_cls(ss['avg_clv'])}'>{_pct(ss['avg_clv'])}</td><td>{ss['settled']}</td>"
                            f"<td>{ss['open']}</td><td>{_pct(ss['avg_ev'])}</td></tr>")
    sport_table = ("" if not by_sport else
                   f"<h2>By sport, {e(VENUES[venue])}</h2><div class='tablewrap'><table><thead><tr>"
                   "<th class='l'>Sport</th><th>Profit</th><th>ROI</th><th>Avg CLV</th><th>Settled</th><th>Open</th>"
                   f"<th>Avg EV</th></tr></thead><tbody>{''.join(by_sport)}</tbody></table></div>")
    open_rows, done_rows = [], []
    for b in sorted(bets, key=lambda x: x["kickoff"]):
        ko = local_time(datetime.fromisoformat(b["kickoff"]), tz)
        close = f"{b['close_fair']:.2f}" if b["close_fair"] else "–"
        how = " *" if b["close_how"] == "converted" else ""
        sp = b.get("sport") or "soccer"
        sp_txt = "" if sp == "soccer" else f"<div class='muted small'>{e(sport_name(sp))}</div>"
        row = (f"<td class='l'>{e(ko)}</td><td class='l'>{e(b['home'])} – {e(b['away'])}{sp_txt}</td>"
               f"<td class='l'>{e(_sel(b))}<div class='muted small'>{e(b['strategy_name'])}</div></td>"
               f"<td>{e(_place(b))}</td><td>{_pct(b['ev'])}</td><td>{close}{how}</td>"
               f"<td class='{_cls(b['clv'])}'>{_pct(b['clv'])}</td>")
        if b["status"] == "open":
            open_rows.append(f"<tr>{row}<td>{_money(b['stake'], cur, False)}</td></tr>")
        else:
            score = f"{b['home_goals']}–{b['away_goals']}" if b["home_goals"] is not None else "–"
            done_rows.append(f"<tr>{row}<td>{score}</td><td class='l'>{e(b['status'])}</td>"
                             f"<td class='{_cls(b['profit'])}'>{_money(b['profit'], cur)}</td></tr>")
    done_rows = done_rows[::-1][:150]
    head = ("<th class='l'>Kickoff</th><th class='l'>Match</th><th class='l'>Bet</th><th>Price</th><th>EV</th>"
            "<th>Close (fair)</th><th>CLV</th>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta http-equiv="refresh" content="300">
<title>Paper Trading</title><style>{base_css}{PAPER_CSS}</style></head><body><div class="wrap">
<a class="back" href="/">← Odds watch</a>
<h1>Paper trading</h1>
<div class="muted small">Imaginary bets, placed automatically when an alert fires. Each place has its own
{_money(start, cur, False)} bankroll and uses {e(staking)}. Updated {e(local_time(utcnow(), tz))}.</div>
<nav class="seg" aria-label="Where the bets are placed">{''.join(tabs)}</nav>
<nav class="seg sm" aria-label="Strategy">{''.join(chips)}</nav>
<div class="tiles">{tiles}</div>
<div class="card verdict">{e(verdict(s))}</div>
<div class="card"><h2>Profit: actual vs expected</h2>
<div class="muted small">{e(VENUES[venue])} · settled bets in kickoff order. If the edge is real, the blue line
should follow the expected lines; the shaded band is the normal luck range (95%) around the EV line.</div>
<div class="legend"><span><i class="key k1"></i>Actual profit</span><span><i class="key k2"></i>Expected from EV when bet</span>
<span><i class="key k3"></i>Expected from closing line</span><span><i class="key band"></i>Luck range</span></div>
{chart}</div>
<h2>All places and strategies</h2>
<div class="tablewrap"><table><thead><tr><th class="l">Place</th><th class="l">Strategy</th><th>Profit</th><th>ROI</th>
<th>Avg CLV</th><th>Settled</th><th>Open</th><th>Staked</th><th>Avg EV</th><th>Beat close</th></tr></thead>
<tbody>{''.join(comp) or "<tr><td class='l muted' colspan='10'>No bets yet.</td></tr>"}</tbody></table></div>
{sport_table}
<h2>Open bets</h2>
<div class="tablewrap"><table><thead><tr>{head}<th>Stake</th></tr></thead>
<tbody>{''.join(open_rows) or "<tr><td class='l muted' colspan='8'>None right now.</td></tr>"}</tbody></table></div>
<details{' open' if len(done_rows) <= 10 else ''}><summary><h2>Settled bets ({s['settled'] + s['void']})</h2></summary>
<div class="tablewrap"><table><thead><tr>{head}<th>Score</th><th class="l">Result</th><th>Profit</th></tr></thead>
<tbody>{''.join(done_rows) or "<tr><td class='l muted' colspan='10'>None yet.</td></tr>"}</tbody></table></div>
{'<div class="note">Latest 150 shown; the CSV has them all.</div>' if len(done_rows) >= 150 else ''}</details>
<div class="note">CLV = price taken (after commission) ÷ no-margin closing price of Pinnacle (else Betfair) − 1.
* = the reference closed on a different line, so its price was moved to ours with a goals model.
<a href="/api/paper.csv">Download all bets (CSV)</a></div>
</div></body></html>"""


CSV_COLS = ["id", "placed", "venue", "strategy", "strategy_name", "league", "kickoff", "home", "away", "market",
            "line", "outcome", "book", "price", "net_price", "fair_odds", "ev", "stake", "close_ref", "close_how",
            "close_fair", "close_price", "close_book", "close_at", "clv", "status", "home_goals", "away_goals",
            "result", "profit", "settled", "result_source"]


def paper_csv(w) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(CSV_COLS)
    for b in w.paper.bets():
        wr.writerow([b.get(c) for c in CSV_COLS])
    return buf.getvalue()


def paper_json(w) -> dict:
    start = float(w.cfg["paper"]["bankroll"])
    out = {"venues": {}, "bets": w.paper.bets()}
    for v in w.cfg["paper"].get("venues", list(VENUES)):
        s = summarize(w.paper.bets(venue=v), start)
        s.pop("series")
        out["venues"][v] = s
    return out
