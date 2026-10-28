"""Alert delivery: Telegram's group rate limit, and channels that fail independently."""

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from footy.watch import config as C
from footy.watch.core import Quote
from footy.watch.notify import Notifier

TOKEN, CHAT, TOPIC = "123456:ABC-test", "-5388746305", "footy-test-topic"
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)

# Replies as the real services send them.
TG_OK = {"ok": True, "result": {"message_id": 7, "from": {"id": 123456, "is_bot": True, "first_name": "bt",
                                                          "username": "titobtbot"},
                                "chat": {"id": -5388746305, "title": "bt", "type": "group",
                                         "all_members_are_administrators": True},
                                "date": 1793178000, "text": "title\nbody"}}
TG_429 = {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 7",
          "parameters": {"retry_after": 7}}
TG_429_LONG = {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 600",
               "parameters": {"retry_after": 600}}
TG_400 = {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
NTFY_OK = {"id": "27Sk2QLNq6c1", "time": 1793177754, "expires": 1793220954, "event": "message", "topic": TOPIC,
           "title": "title", "message": "body", "priority": 3, "tags": ["soccer"]}


class FakeServices:
    """A local stand-in for api.telegram.org and ntfy.sh that records every request."""

    def __init__(self, telegram=()):
        self.telegram = list(telegram)   # scripted (status, reply) for Telegram; afterwards 200 OK
        self.requests = []               # (service, path, body)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
                if self.path.startswith("/bot"):
                    outer.requests.append(("telegram", self.path, body))
                    status, reply = outer.telegram.pop(0) if outer.telegram else (200, TG_OK)
                else:
                    outer.requests.append(("ntfy", self.path, body))
                    status, reply = 200, NTFY_OK
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def count(self, service: str) -> int:
        return sum(1 for r in self.requests if r[0] == service)


class Clock:
    """Fake monotonic clock: time only moves when the code under test sleeps (or the test says so)."""

    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def services():
    made = []

    def make(**kw):
        made.append(FakeServices(**kw))
        return made[-1]
    yield make
    for f in made:
        f.srv.shutdown()


def _notify_cfg(fake, telegram=True, ntfy=True) -> dict:
    cfg = {"console": False}
    if telegram:
        cfg.update({"telegram_token": TOKEN, "telegram_chat_id": CHAT, "telegram_api": fake.url})
    if ntfy:
        cfg.update({"ntfy_topic": TOPIC, "ntfy_server": fake.url})
    return cfg


def test_rate_limited_telegram_message_is_retried_after_the_wait_telegram_asks_for(services):
    fake, clock = services(telegram=[(429, TG_429)]), Clock()
    n = Notifier(_notify_cfg(fake), sleep=clock.sleep, clock=clock.now)
    assert n.send("title", "body") is True
    assert fake.count("telegram") == 2 and fake.count("ntfy") == 1
    assert clock.sleeps == [7]
    assert n.sent == 1 and n.last_error == ""
    tg = [r for r in fake.requests if r[0] == "telegram"]
    assert all(r[1] == f"/bot{TOKEN}/sendMessage" and f"chat_id={CHAT}" in r[2] for r in tg)


def test_telegram_failure_does_not_stop_ntfy(services):
    fake, clock = services(telegram=[(400, TG_400)]), Clock()
    n = Notifier(_notify_cfg(fake), sleep=clock.sleep, clock=clock.now)
    assert n.send("title", "body") is True                 # ntfy still got it
    assert fake.count("ntfy") == 1 and "chat not found" in n.last_error
    assert n.send("title 2", "body 2") is True             # Telegram works again: the error clears
    assert n.last_error == "" and n.sent == 2 and fake.count("telegram") == 2


def test_telegram_messages_are_spaced_to_stay_under_the_group_limit(services):
    fake, clock = services(), Clock()
    n = Notifier(_notify_cfg(fake, ntfy=False), sleep=clock.sleep, clock=clock.now)
    for i in range(3):
        n.send(f"alert {i}", "body")
    assert fake.count("telegram") == 3
    # Telegram allows about 20 messages a minute in a group: at least 3 s between messages
    assert len(clock.sleeps) == 2 and all(s >= 3.0 for s in clock.sleeps)
    clock.t += 60                                          # after a quiet minute there is nothing to wait for
    n.send("alert 3", "body")
    assert len(clock.sleeps) == 2


def test_long_rate_limit_is_not_waited_out(services):
    fake, clock = services(telegram=[(429, TG_429_LONG)]), Clock()
    n = Notifier(_notify_cfg(fake), sleep=clock.sleep, clock=clock.now)
    assert n.send("title", "body") is True                 # delivered by ntfy
    assert clock.sleeps == [] and fake.count("telegram") == 1 and "429" in n.last_error


def test_nothing_delivered_is_reported(services):
    fake, clock = services(telegram=[(400, TG_400)]), Clock()
    n = Notifier(_notify_cfg(fake, ntfy=False), sleep=clock.sleep, clock=clock.now)
    assert n.send("title", "body") is False and n.sent == 0


def test_undelivered_alert_is_sent_again_on_the_next_sweep(services, tmp_path):
    from footy.watch.server import Watcher
    fake, clock = services(telegram=[(400, TG_400)]), Clock()
    cfg = C.load(None)
    cfg["server"].update({"db": str(tmp_path / "w.db"), "picks_file": str(tmp_path / "picks.json")})
    cfg["paper"]["enabled"] = False
    cfg["alerts"]["arb_alerts"] = False      # these prices are also an arbitrage; this test is about one alert
    cfg["notify"] = _notify_cfg(fake, ntfy=False)
    w = Watcher(cfg, provider_list=[])
    w.notifier = Notifier(cfg["notify"], sleep=clock.sleep, clock=clock.now)
    ev = w.index.add("x", "1", "Bolton", "Wigan", NOW + timedelta(hours=3))
    # Pinnacle 2.00 / 3.60 / 4.00 against 2.30 elsewhere: a wrong-odds alert on Bolton
    w.quotes = {(ev.id, "1x2", None, bk, o): Quote(ev.id, "1x2", None, o, bk, p)
                for bk, o, p in (("pinnacle", "home", 2.00), ("pinnacle", "draw", 3.60), ("pinnacle", "away", 4.00),
                                 ("sport888", "home", 2.30))}
    assert w.evaluate(NOW) == []                           # Telegram refused it: not marked as sent
    sent = w.evaluate(NOW + timedelta(minutes=5))
    assert [(a.book, a.outcome) for a in sent] == [("sport888", "home")]
