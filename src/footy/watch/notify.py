"""Notifications: console, Telegram bot, ntfy.sh (or a self-hosted ntfy)."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import timezone

from .rules import Alert

log = logging.getLogger(__name__)
BOOKS = {"pinnacle": "Pinnacle", "betfair_ex_eu": "Betfair Exchange", "betfair_ex_uk": "Betfair Exchange",
         "betfair": "Betfair Exchange", "matchbook": "Matchbook", "smarkets": "Smarkets",
         "williamhill": "William Hill", "onexbet": "1xBet", "marathonbet": "Marathonbet",
         "unibet_eu": "Unibet", "sport888": "888sport", "betsson": "Betsson", "nordicbet": "NordicBet"}


def book_name(key: str) -> str:
    return BOOKS.get(key, key)


def local_time(dt, tzname: str) -> str:
    try:
        from zoneinfo import ZoneInfo
        dt = dt.astimezone(ZoneInfo(tzname))
    except Exception:  # noqa: BLE001
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%a %d %b %H:%M")


def format_alert(a: Alert, home: str, away: str, tzname: str = "UTC") -> tuple[str, str]:
    """(title, body) in plain words."""
    sel = a.selection(home, away)
    when = local_time(a.kickoff, tzname)
    if a.kind == "value":
        title = f"Wrong odds: {sel} @ {a.price:.2f} ({book_name(a.book)})"
        body = (f"{a.match} · {when}\n"
                f"Edge {a.edge:+.1%} · fair price {a.fair_odds:.2f}\n"
                f"Reference: {a.detail}")
    else:
        rule = a.extra.get("rule", "Research")
        moved = ""
        if a.market == "ah" and a.extra.get("pick_line") is not None and a.line != a.extra["pick_line"]:
            moved = " (line moved; edge recalculated)"
        title = f"Pick ({rule}): {sel} @ {a.price:.2f} ({book_name(a.book)})"
        body = (f"{a.match} · {when}\n"
                f"Edge {a.edge:+.1%}{moved} · bet at {a.fair_odds:.2f} or higher\n{a.detail}")
    return title, body


class Notifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sent = 0
        self.last_error = ""

    def send(self, title: str, body: str, priority: str = "default") -> None:
        n = self.cfg
        if n.get("console", True):
            log.info("ALERT %s | %s", title, body.replace("\n", " | "))
        try:
            if n.get("telegram_token") and n.get("telegram_chat_id"):
                data = urllib.parse.urlencode({"chat_id": n["telegram_chat_id"], "text": f"{title}\n{body}",
                                               "disable_web_page_preview": "true"}).encode()
                urllib.request.urlopen(urllib.request.Request(
                    f"https://api.telegram.org/bot{n['telegram_token']}/sendMessage", data=data), timeout=15).read()
            if n.get("ntfy_topic"):
                urllib.request.urlopen(urllib.request.Request(
                    f"{n.get('ntfy_server', 'https://ntfy.sh').rstrip('/')}/{n['ntfy_topic']}",
                    data=body.encode("utf-8"),
                    headers={"Title": title.encode("ascii", "replace").decode(), "Priority": priority,
                             "Tags": "soccer"}), timeout=15).read()
            self.sent += 1
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            log.warning("notification failed: %s", e)


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
