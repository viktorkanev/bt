"""Prediction-market strategies: price-consistency arbitrage and weather."""

import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone

from footy.watch import config as C
from footy.watch.pm import Signal
from footy.watch.pm import gamma
from footy.watch.pm.consistency import ConsistencyScanner, date_of, number, template
from footy.watch.pm.ledger import PMBook, summarize
from footy.watch.pm.notify import format_signal
from footy.watch.pm.weather import WeatherScanner, bucket_probs, members, parse_bucket, parse_ensemble, \
    parse_title, station_of

NOW = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)   # 10:00 in New York
END = (NOW + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")


def mk(mid, q, ask, bid, title=None, **kw):
    return {"id": mid, "question": q, "groupItemTitle": title or q, "outcomes": '["Yes", "No"]',
            "clobTokenIds": json.dumps([f"{mid}y", f"{mid}n"]), "bestAsk": ask, "bestBid": bid,
            "active": True, "closed": False, "acceptingOrders": True, **kw}


def books_for(events, size=1000):
    """Order books matching each market's Gamma summary (NO ask = 1 - YES bid)."""
    out = {}
    for e in events:
        for m in e["markets"]:
            y, n = gamma.yes_no(m)
            out[y] = {"ask": m["bestAsk"], "ask_size": size, "bid": m["bestBid"], "bid_size": size}
            out[n] = {"ask": round(1 - m["bestBid"], 4), "ask_size": size, "bid": round(1 - m["bestAsk"], 4),
                      "bid_size": size}
    return out


def cfg(tmp_path, **over):
    c = C.load(None)
    c["server"]["db"] = str(tmp_path / "w.db")
    for k, v in over.items():
        c[k].update(v)
    return c


# ---------------------------------------------------------- consistency
def test_parsers():
    assert template("Will Bitcoin be above $110,000 on November 30?") == "will bitcoin be above <n> on <d> ?"
    assert number("$110k") == 110000 and number("2.5%") == 2.5 and number("between 1 and 2") is None
    assert date_of("by March 31, 2027", 2026).year == 2027 and date_of("by June 30", 2026).month == 6


def test_outcomes_that_dont_add_up(tmp_path):
    e = {"id": "1", "title": "Who wins?", "slug": "who-wins", "negRisk": True, "endDate": END,
         "markets": [mk("a", "A?", 0.30, 0.28), mk("b", "B?", 0.30, 0.28), mk("c", "C?", 0.31, 0.29)]}
    sc = ConsistencyScanner(cfg(tmp_path))
    sigs = sc.run(NOW, [e], books_for([e]))
    assert [s.kind for s in sigs] == ["all-yes"]
    s = sigs[0]
    cost = sum(p + gamma.fee(p, 0.05) for p in (0.30, 0.30, 0.31))
    assert abs(s.edge - (1 / cost - 1)) < 1e-9 and s.payout_per_set == 1.0
    assert abs(s.apy - s.edge * 365 / 10) < 1e-6
    # the NO side: YES bids adding to more than $1 -> n-1 of the NO shares pay
    e2 = {**e, "id": "2", "markets": [mk("d", "D?", 0.45, 0.43), mk("e", "E?", 0.40, 0.38), mk("f", "F?", 0.30, 0.28)]}
    sigs = sc.run(NOW, [e2], books_for([e2]))
    assert [s.kind for s in sigs] == ["all-no"] and sigs[0].payout_per_set == 2.0
    # not mutually exclusive, already decided, or too little money on offer -> nothing
    assert sc.run(NOW, [{**e, "negRisk": False}], books_for([e])) == []
    decided = {**e, "markets": e["markets"] + [mk("z", "Z?", 0.9, 0.9, closed=True, outcomePrices='["1", "0"]')]}
    assert sc.run(NOW, [decided], books_for([e])) == []
    assert sc.run(NOW, [e], books_for([e], size=10)) == []


def test_ladders_numbers_and_dates(tmp_path):
    sc = ConsistencyScanner(cfg(tmp_path))
    btc = {"id": "3", "title": "Bitcoin on November 30", "slug": "btc", "negRisk": False, "endDate": END, "markets": [
        mk("p", "Will Bitcoin be above $100,000 on November 30?", 0.60, 0.58, "100,000"),
        mk("q", "Will Bitcoin be above $110,000 on November 30?", 0.70, 0.68, "110,000"),   # narrower but dearer
        mk("r", "Will Bitcoin dip to $90,000 in November?", 0.20, 0.18, "90,000")]}
    sigs = sc.run(NOW, [btc], books_for([btc]))
    assert len(sigs) == 1 and sigs[0].kind == "ladder"
    legs = {(lg["market_id"], lg["side"]) for lg in sigs[0].legs}
    assert legs == {("p", "YES"), ("q", "NO")}        # YES on the wider, NO on the narrower: pays $1 or $2
    by = {"id": "4", "title": "X by ...?", "slug": "x", "negRisk": False, "endDate": END, "markets": [
        mk("s", "Will X happen by March 31?", 0.40, 0.38, "March 31"),
        mk("t", "Will X happen by June 30?", 0.30, 0.28, "June 30")]}
    sigs = sc.run(NOW, [by], books_for([by]))
    assert len(sigs) == 1 and {(lg["market_id"], lg["side"]) for lg in sigs[0].legs} == {("t", "YES"), ("s", "NO")}
    mixed = {**by, "markets": [mk("s", "Will X happen by March 31, 2027?", 0.40, 0.38, "March 31, 2027"),
                               mk("t", "Will X happen by June 30?", 0.30, 0.28, "June 30")]}
    assert sc.run(NOW, [mixed], books_for([mixed])) == []   # year unclear for one of them


def test_yearly_return_filter(tmp_path):
    sc = ConsistencyScanner(cfg(tmp_path, consistency={"min_apy": 0.15}))
    e = {"id": "5", "title": "Who wins?", "slug": "s", "negRisk": True, "endDate": END,
         "markets": [mk("a", "A?", 0.31, 0.29), mk("b", "B?", 0.31, 0.29), mk("c", "C?", 0.31, 0.29)]}
    sigs = sc.run(NOW, [e], books_for([e]))                  # ~3.9% in 10 days: well over 15%/yr
    assert len(sigs) == 1 and sigs[0].apy > 1.0
    slow = {**e, "endDate": (NOW + timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    assert sc.run(NOW, [slow], books_for([slow])) == []      # same 3.9% over 400 days: ~3.6%/yr, not worth it


# ---------------------------------------------------------------- ledger
def test_ledger_sizes_settles_and_summarizes(tmp_path):
    c = cfg(tmp_path)
    book = PMBook(c["server"]["db"], c)
    s = Signal("consistency", "k1", "1", "Who wins?", "who-wins", NOW + timedelta(days=2), "all-yes",
               [{"market_id": "a", "label": "A", "token": "ay", "side": "YES", "price": 0.3, "fee": 0.0105, "size": 50},
                {"market_id": "b", "label": "B", "token": "by", "side": "YES", "price": 0.3, "fee": 0.0105, "size": 500},
                {"market_id": "c", "label": "C", "token": "cy", "side": "YES", "price": 0.3, "fee": 0.0105, "size": 500}],
               1.0, None, 0.07, 10.0, "test")
    row = book.place(s, NOW)
    assert row and json.loads(row["legs"])[0]["shares"] == 50          # capped by the thinnest leg
    assert book.place(s, NOW) is None                                   # never twice
    title, body = format_signal(s, row)
    assert "arbitrage" in title.lower() and "polymarket.com/event/who-wins" in body
    res = {"a": {"closed": True, "outcomePrices": '["0", "1"]'}, "b": {"closed": True, "outcomePrices": '["1", "0"]'},
           "c": {"closed": True, "outcomePrices": '["0", "1"]'}}
    assert book.settle(NOW + timedelta(days=3), fetch=lambda ids: res) == 1
    r = book.positions()[0]
    assert r["status"] == "settled" and abs(r["payout"] - 50) < 1e-9 and abs(r["profit"] - (50 - r["cost"])) < 1e-6
    sm = summarize(book.positions(), 1000)
    assert sm["settled"] == 1 and sm["series"][0]["actual"] == round(r["profit"], 2)


# --------------------------------------------------------------- weather
def test_weather_parsers():
    assert parse_title("Highest temperature in NYC on November 8?")["kind"] == "max"
    assert parse_bucket("62-63°F") == (62, 63, "F") and parse_bucket("61°F or below") == (-math.inf, 61, "F")
    assert parse_bucket("13°C") == (13, 13, "C") and parse_bucket("20°C or higher") == (20, math.inf, "C")
    assert station_of({"title": "x", "description": "see https://www.weather.gov/wrh/timeseries?site=KLGA"}) == "KLGA"
    assert station_of({"title": "Highest temperature in Paris on May 1?", "description": ""},
                      {"Paris": "LFPG"}) == "LFPG"


def _ensemble(center: float, day="2026-11-08", spread=1.0, n=20):
    """Hourly members for two models; the day's high is center +- spread."""
    times = [f"2026-11-0{d}T{h:02d}:00" for d in (7, 8, 9) for h in range(24)]
    hourly = {"time": times}
    for model in ("gfs_seamless", "ecmwf_ifs025"):
        for k in range(n):
            peak = center + spread * ((k / (n - 1)) * 2 - 1)
            vals = [peak - 8 + 8 * math.sin(math.pi * (int(t[11:13]) - 6) / 12) ** 2 if 6 <= int(t[11:13]) <= 18
                    else peak - 10 for t in times]
            key = f"temperature_2m_member{k:02d}_{model}" if k else f"temperature_2m_{model}"
            hourly[key] = vals
    return parse_ensemble({"timezone": "America/New_York", "hourly": hourly})


def test_ensemble_probabilities_and_floor():
    ens = _ensemble(62.5)
    vals = members(ens, "2026-11-08", "max")
    assert len(vals) == 40 and abs(sum(w for w, _ in vals) - 1) < 1e-9
    assert abs(sum(w * v for w, v in vals) - 62.5) < 0.2
    buckets = [(-math.inf, 57), (58, 59), (60, 61), (62, 63), (64, 65), (66, math.inf)]
    p = bucket_probs(vals, buckets, 1.6)
    assert abs(sum(p) - 1) < 1e-9 and max(range(6), key=lambda i: p[i]) == 3
    floored = bucket_probs(vals, buckets, 1.6, floor=61.6)   # 62 already observed today
    assert floored[0] == floored[1] == floored[2] == 0 and abs(sum(floored) - 1) < 1e-9


def _weather_event(asks):
    labels = ["57°F or below", "58-59°F", "60-61°F", "62-63°F", "64-65°F", "66°F or higher"]
    return {"id": "w1", "title": "Highest temperature in NYC on November 8?", "slug": "nyc-nov-8", "negRisk": True,
            "endDate": "2026-11-09T05:00:00Z", "description": "Resolves per https://www.weather.gov/wrh/timeseries?site=KLGA",
            "markets": [mk(f"m{i}", lb, a, max(a - 0.02, 0.001), lb) for i, (lb, a) in enumerate(zip(labels, asks))]}


def test_weather_signal_end_to_end(tmp_path):
    sc = WeatherScanner(cfg(tmp_path))
    e = _weather_event([0.02, 0.08, 0.30, 0.20, 0.25, 0.10])      # 62-63°F too cheap at 20¢
    metars = {"KLGA": {"lat": 40.78, "lon": -73.87, "obs": []}}
    ens = {("KLGA", "F"): _ensemble(62.5)}
    sigs = sc.run(NOW, [e], books_for([e]), ens, metars)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.legs[0]["label"] == "62-63°F" and s.legs[0]["side"] == "YES" and s.prob > 0.3 and s.edge > 0.08
    assert "KLGA" == s.extra["station"] and s.link.endswith("nyc-nov-8")
    # model and market several degrees apart: probably the wrong station or unit, so no bet
    far = _weather_event([0.02, 0.03, 0.05, 0.10, 0.30, 0.50])
    assert sc.run(NOW, [far], books_for([far]), {("KLGA", "F"): _ensemble(56.0)}, metars) == []


def test_weather_bias_is_learned(tmp_path):
    sc = WeatherScanner(cfg(tmp_path))
    for d in range(6):
        sc._learn("KLGA", "max", "F", f"2026-10-0{d + 1}", 60.0, 61.5)
    assert sc.bias("KLGA", "max", "F") == 1.5 and sc.bias("KJFK", "max", "F") == 0.0


# ---------------------------------------------------------------- watcher
def test_watcher_runs_strategies_and_serves_page(tmp_path):
    from footy.watch.server import Watcher, serve
    c = cfg(tmp_path, consistency={"enabled": True})
    c["notify"]["console"] = False
    w = Watcher(c, provider_list=[])
    e = {"id": "1", "title": "Who wins?", "slug": "who-wins", "negRisk": True, "endDate": END,
         "markets": [mk("a", "A?", 0.30, 0.28), mk("b", "B?", 0.30, 0.28), mk("c", "C?", 0.31, 0.29)]}
    sc = w.scanners[0]
    sc.fetch = lambda now: [e]
    import footy.watch.pm.consistency as pc
    orig = pc.gamma.books
    pc.gamma.books = lambda tokens: books_for([e])
    try:
        placed = w.run_scanners(NOW, force=True)
        again = w.run_scanners(NOW, force=True)
    finally:
        pc.gamma.books = orig
    assert len(placed) == 1 and w.notifier.sent == 1 and w.pm.positions()[0]["kind"] == "all-yes"
    assert again == [] and w.notifier.sent == 1 and len(w.pm.positions()) == 1   # same opportunity: once
    assert w.scan_info["consistency"]["signals"] == 1 and not w.scan_info["consistency"]["error"]
    srv = serve(w)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/pm").read().decode()
        assert "Prediction markets" in page and "Who wins?" in page
        assert "Prediction markets" in urllib.request.urlopen(base + "/").read().decode()
        data = json.loads(urllib.request.urlopen(base + "/api/pm").read())
        assert data["summary"]["consistency"]["positions"] == 1
        assert urllib.request.urlopen(base + "/api/pm.csv").read().decode().startswith("id,placed")
    finally:
        srv.shutdown()
