"""Notifications: console, Telegram bot, ntfy.sh (or a self-hosted ntfy)."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
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


TELEGRAM_GAP_S = 3.2       # Telegram takes about 20 messages a minute in one group
MAX_RETRY_AFTER_S = 60     # waiting longer would stall the watcher; give up on that message instead


class Notifier:
    """Sends each alert to every configured channel; one failing channel doesn't stop the others."""

    def __init__(self, cfg: dict, sleep=time.sleep, clock=time.monotonic):
        self.cfg = cfg
        self.sent = 0
        self.last_error = ""
        self._sleep, self._clock = sleep, clock
        self._telegram_at: float | None = None

    def send(self, title: str, body: str, priority: str = "default") -> bool:
        """True if at least one channel delivered the alert (or only the console is configured)."""
        n = self.cfg
        if n.get("console", True):
            log.info("ALERT %s | %s", title, body.replace("\n", " | "))
        channels = []
        if n.get("telegram_token") and n.get("telegram_chat_id"):
            channels.append(("telegram", lambda: self._telegram(f"{title}\n{body}")))
        if n.get("ntfy_topic"):
            channels.append(("ntfy", lambda: self._ntfy(title, body, priority)))
        errors = []
        for name, deliver in channels:
            try:
                deliver()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
                log.warning("%s notification failed: %s", name, e)
        self.last_error = "; ".join(errors)
        delivered = not channels or len(errors) < len(channels)
        if delivered:
            self.sent += 1
        return delivered

    def _telegram(self, text: str) -> None:
        n = self.cfg
        api = n.get("telegram_api", "https://api.telegram.org").rstrip("/")
        req = urllib.request.Request(f"{api}/bot{n['telegram_token']}/sendMessage", data=urllib.parse.urlencode(
            {"chat_id": n["telegram_chat_id"], "text": text, "disable_web_page_preview": "true"}).encode())
        for attempt in range(2):
            if self._telegram_at is not None:
                wait = TELEGRAM_GAP_S - (self._clock() - self._telegram_at)
                if wait > 0:
                    self._sleep(wait)
            try:
                self._post_telegram(req)
                return
            except urllib.error.HTTPError as e:
                reply = _json_reply(e)
                retry = (reply.get("parameters") or {}).get("retry_after") if e.code == 429 else None
                if attempt or retry is None or float(retry) > MAX_RETRY_AFTER_S:
                    raise RuntimeError(f"HTTP {e.code}: {reply.get('description') or e.reason}") from None
                self._sleep(float(retry))

    def _post_telegram(self, req: urllib.request.Request) -> None:
        try:
            urllib.request.urlopen(req, timeout=15).read()
        finally:
            self._telegram_at = self._clock()

    def _ntfy(self, title: str, body: str, priority: str) -> None:
        n = self.cfg
        urllib.request.urlopen(urllib.request.Request(
            f"{n.get('ntfy_server', 'https://ntfy.sh').rstrip('/')}/{n['ntfy_topic']}", data=body.encode("utf-8"),
            headers={"Title": title.encode("ascii", "replace").decode(), "Priority": priority, "Tags": "soccer"}),
            timeout=15).read()


def _json_reply(e: urllib.error.HTTPError) -> dict:
    """Telegram explains errors in a JSON body (description, parameters.retry_after)."""
    try:
        reply = json.loads(e.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return {}
    return reply if isinstance(reply, dict) else {}


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
