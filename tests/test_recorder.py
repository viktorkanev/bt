"""Feed recorder: what each feed delivered, who moves first, and what a plan allows."""

import json
import urllib.request
from datetime import datetime, timedelta, timezone

from footy.watch import config as C
from footy.watch.core import EventIndex, Quote
from footy.watch.recorder import Recorder, minutes_between, monthly_requests

NOW = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=8)


def _cfg(tmp_path):
    c = C.load(None)
    c["server"].update({"db": str(tmp_path / "w.db"), "picks_file": str(tmp_path / "p.json"), "port": 0})
    return c


def test_records_changes_only_and_compares_feeds(tmp_path):
    c = _cfg(tmp_path)
    rec = Recorder(c)
    assert rec.path == str(tmp_path / "footy-feeds.db")             # next to the main database
    idx = EventIndex()
    a = idx.add("x", "1", "Arsenal", "Chelsea", KO)
    b = idx.add("x", "2", "Ajax", "PSV", KO)
    t = NOW
    # OddsPapi sees Pinnacle move from 2.10 to 2.00 five minutes before The Odds API does
    for minute, (op, oa) in enumerate([(2.10, 2.10), (2.00, 2.10), (2.00, 2.10), (2.00, 2.10), (2.00, 2.10),
                                       (2.00, 2.10), (2.00, 2.00)]):
        now = t + timedelta(minutes=minute)
        rec.record("oddspapi", now, [Quote(a.id, "1x2", None, "home", "pinnacle", op),
                                     Quote(a.id, "1x2", None, "home", "singbet", 2.2)], idx, 1.0, 3)
        rec.record("oddsapi", now, [Quote(a.id, "1x2", None, "home", "pinnacle", oa),
                                    Quote(b.id, "1x2", None, "home", "sport888", 1.9)], idx, 0.5, 3)
    with rec.lock:
        n_prices = rec.db.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert n_prices == 2 + 2 + 1 + 1                                     # first sight + one change per feed
    rep = rec.report(NOW + timedelta(hours=1), 7)["providers"]
    assert rep["oddspapi"]["matches_total"] == 1 and rep["oddsapi"]["matches_total"] == 2
    assert rep["oddsapi"]["only_here"] == 1 and rep["oddspapi"]["only_here"] == 0
    assert rep["oddspapi"]["exclusive_books"] == ["singbet"] and rep["oddsapi"]["exclusive_books"] == ["sport888"]
    assert rep["oddspapi"]["lead"]["oddsapi"]["median_minutes_ahead"] == 5.0    # OddsPapi first by 5 minutes
    assert rep["oddsapi"]["lead"]["oddspapi"]["median_minutes_ahead"] == -5.0
    assert rep["oddspapi"]["requests_per_poll"] == 3 and rep["oddspapi"]["polls"] == 7
    rec.record_scan("weather", NOW, 4.2, 30, 2, "")
    assert rec.report(NOW + timedelta(hours=1), 7)["providers"]["pm:weather"]["polls"] == 1
    rec.prune(NOW + timedelta(days=40))
    with rec.lock:
        assert rec.db.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 0


def test_paper_bets_that_needed_a_feed(tmp_path):
    from footy.watch.paper import PaperBook
    from footy.watch.rules import Alert
    c = _cfg(tmp_path)
    rec = Recorder(c)
    idx = EventIndex()
    ev = idx.add("x", "1", "Arsenal", "Chelsea", KO)
    rec.record("oddspapi", NOW, [Quote(ev.id, "1x2", None, "home", "singbet", 2.3)], idx)
    rec.record("oddsapi", NOW, [Quote(ev.id, "1x2", None, "home", "pinnacle", 2.0)], idx)
    book = PaperBook(c["server"]["db"], c)
    book.place([Alert("value", ev.id, "Arsenal v Chelsea", KO, None, "1x2", None, "home", "singbet", 2.3, 2.0, 0.08)],
               idx, NOW)
    rep = rec.report(NOW + timedelta(hours=1), 7, book)["providers"]
    assert rep["oddspapi"]["paper_bets_only_here"] >= 1 and rep["oddsapi"]["paper_bets_only_here"] == 0
    assert rep["oddspapi"]["paper_ev_only_here"] > 0


def test_paper_bets_on_matches_only_one_feed_listed(tmp_path):
    """A National League match only OddsPapi lists: Polymarket prices it only because OddsPapi
    listed it, and its Pinnacle price is OddsPapi's even though The Odds API has Pinnacle elsewhere."""
    from footy.watch.paper import PaperBook
    from footy.watch.rules import Alert
    c = _cfg(tmp_path)
    c["paper"]["venues"] = ["best"]
    rec = Recorder(c)
    idx = EventIndex()
    nl = idx.add("oddspapi", "op1", "Forest Green", "Eastleigh", KO)
    idx.add("polymarket", "pm1", "Forest Green", "Eastleigh", KO)
    prem = idx.add("oddsapi", "oa1", "Arsenal", "Chelsea", KO)
    rec.record("oddspapi", NOW, [Quote(nl.id, "ou", 2.5, "under", "pinnacle", 2.6)], idx)
    rec.record("polymarket", NOW, [Quote(nl.id, "ou", 2.5, "under", "polymarket", 2.78)], idx)
    rec.record("oddsapi", NOW, [Quote(prem.id, "1x2", None, "home", "pinnacle", 2.0)], idx)
    book = PaperBook(c["server"]["db"], c)
    pick = {"rule_code": "F1", "rule": "Lower-league goals"}
    book.place([Alert("pick", nl.id, "Forest Green v Eastleigh", KO, None, "ou", 2.5, "under", "polymarket",
                      2.78, 2.5, 0.08, extra=pick),
                Alert("pick", nl.id, "Forest Green v Eastleigh", KO, None, "ou", 2.5, "over", "pinnacle",
                      2.3, 2.2, 0.04, extra=pick)], idx, NOW)
    rep = rec.report(NOW + timedelta(hours=1), 7, book)["providers"]
    assert rep["oddspapi"]["paper_bets_only_here"] == 2       # the match came only from OddsPapi
    assert rep["polymarket"]["paper_bets_only_here"] == 1     # the Polymarket price came only from Polymarket
    assert rep["oddsapi"]["paper_bets_only_here"] == 0


def test_plan_maths():
    assert monthly_requests(3, 15) == 3 * 30 * 24 * 4                     # 8,640 a month
    assert abs(minutes_between(3, 20_000) - 30 * 24 * 60 * 3 / 20_000) < 1e-9   # ~6.5 min
    assert minutes_between(3, 250, 10) > 500


def test_watcher_records_polls_and_serves_page(tmp_path):
    from footy.watch.server import Watcher, serve
    odds = {"events": [{"id": "1", "home": "Arsenal", "away": "Chelsea", "kickoff": KO.isoformat()}],
            "quotes": [{"event": "1", "market": "1x2", "line": None, "outcome": o, "book": "pinnacle", "price": p,
                        "updated": NOW.isoformat()} for o, p in (("home", 2.0), ("draw", 3.5), ("away", 4.0))]}
    (tmp_path / "odds.json").write_text(json.dumps(odds))
    c = _cfg(tmp_path)
    c["file"].update({"enabled": True, "path": str(tmp_path / "odds.json")})
    c["notify"]["console"] = False
    w = Watcher(c)
    w.poll_once(NOW)
    w.next_poll = {k: 0 for k in w.next_poll}
    w.poll_once(NOW + timedelta(minutes=1))
    rep = w.recorder.report(NOW + timedelta(minutes=5), 7)["providers"]["file"]
    assert rep["polls"] == 2 and rep["matches_total"] == 1
    srv = serve(w)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/feeds?days=7").read().decode()
        assert "Feeds compared" in page and "File feed" in page
        assert "Feeds compared" in urllib.request.urlopen(base + "/").read().decode()
        assert "providers" in json.loads(urllib.request.urlopen(base + "/api/feeds").read())
    finally:
        srv.shutdown()
