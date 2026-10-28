# Live odds watcher on a Raspberry Pi Zero 2 W

The watcher polls live odds, sends a phone notification when a price is wrong or a
research pick becomes playable, and shows a status page on your home network.
It uses only Python's standard library (about 30 MB of RAM), so nothing heavy
needs installing on the Pi. The models stay on your PC: `footy upcoming` there
produces the picks and `footy push-picks` sends them to the Pi.

```
 PC (Windows)                               Raspberry Pi Zero 2 W
 footy upcoming  ── picks.json ──push──▶    footy.watch  ◀── odds feeds (The Odds API / Betfair)
                                                 │
                                                 ├──▶ Telegram / ntfy notification on your phone
                                                 └──▶ http://raspberrypi.local:8080 status page
```

## 1. Odds feeds

| Feed | What you get | Cost |
|---|---|---|
| [The Odds API](https://the-odds-api.com) | Pinnacle, Betfair Exchange and many other books in one call (region `eu`) | Free: 500 credits/month (enough to try it). Paid plans for real tracking |
| [Betfair Exchange API](https://developer.betfair.com) | Exchange back/lay prices and liquidity | Free delayed key (prices 1-180 s late); live key is a one-off £299 |
| Your own script | Anything that writes `odds.json` (see the `file` provider) | - |

Pinnacle closed its public API in July 2025; The Odds API is the simplest way to
see its prices. BetInAsia's BLACK platform runs on Mollybet, which has an API for
some customers; ask BetInAsia support. A provider for it can be added the same way.

Credits on The Odds API: each odds request costs `markets x regions` (3 markets x 1
region = 3 credits per league). The watcher only polls leagues with matches in the
next 36 hours and spaces polls so `monthly_credits` last until the reset. With
the free 500 credits and 5 leagues (15 credits a sweep) that is about one sweep a
day: fine for testing, too slow to catch wrong odds. Limiting `bookmakers` to the
two or three you need, fewer markets, or a paid plan shortens the gap.

## 2. Notifications

* **ntfy** (simplest): install the ntfy app, subscribe to a long random topic name,
  put the same name in `ntfy_topic`.
* **Telegram**: create a bot with @BotFather, send it any message, open
  `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat id.

## 3. Install on the Pi (Raspberry Pi OS Lite, Bookworm)

```bash
sudo apt update && sudo apt install -y git python3
git clone https://github.com/viktorkanev/bt.git ~/bt
cd ~/bt
PYTHONPATH=src python3 -m footy.watch init-config watch.toml
nano watch.toml                      # API keys, notifications, timezone
PYTHONPATH=src python3 -m footy.watch notify-test --config watch.toml
PYTHONPATH=src python3 -m footy.watch sports --config watch.toml   # check league keys (The Odds API)
PYTHONPATH=src python3 -m footy.watch check --config watch.toml    # one poll, prints what it sees

sudo cp deploy/raspberry-pi/footy-watch.service /etc/systemd/system/
sudo systemctl enable --now footy-watch
journalctl -u footy-watch -f          # logs
```

Open `http://raspberrypi.local:8080` (or the Pi's IP) on your phone.

## 4. Send research picks from your PC

```bash
footy upcoming
footy push-picks http://raspberrypi.local:8080
```

The Pi then alerts you when a pick's live price reaches its minimum odds. Handicap
picks carry the full goal-difference distribution, so if the bookmaker moves the
line (say from -0.25 to -0.5) the edge is recalculated for the new line.

## 5. Reading an alert

```
Wrong odds: Over 2.5 goals @ 2.10 (Betfair Exchange)
Bolton v Wigan · Sat 01 Nov 17:00
Edge +5.8% · fair price 1.98
Reference: pinnacle 1.95 / 1.92 (no-margin 1.98)
```

*Fair price* is the reference book's price with its margin removed. *Edge* is the
expected profit per unit stake at the alerted price after commission. Prices move
fast; check the price before betting and keep stakes small until the alerts have
a track record. Big "edges" are often stale prices or feed errors, which is why
`max_edge` filters them out.
