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
                                                 ├──▶ http://raspberrypi.local:8080 status page
                                                 └──▶ http://raspberrypi.local:8080/paper paper trading
```

## 1. Odds feeds

| Feed | What you get | Cost |
|---|---|---|
| [The Odds API](https://the-odds-api.com) | Pinnacle, Betfair Exchange and many other books in one call (region `eu`) | Free: 500 credits/month (enough to try it). Paid plans for real tracking |
| [Betfair Exchange API](https://developer.betfair.com) | Exchange back/lay prices and liquidity | Free delayed key (prices 1-180 s late); live key is a one-off £299 |
| [OddsPapi](https://oddspapi.io) | 350+ bookmakers in any sport, including Pinnacle and the Asian books BetInAsia offers (SBOBet, Singbet) | Free: 250 requests/month (a few sweeps a day). Paid plans for more |
| [Polymarket](https://polymarket.com) | Football match winner, totals and spreads from the order book (best ask, best bid, money available) | Free, no key. Taker fee about 0.05 x p x (1-p) per share; limit (maker) orders pay none |
| Your own script | Anything that writes `odds.json` (see the `file` provider) | - |

Kalshi is not included: its member agreement lists Bulgaria among the countries
whose residents may not trade. Polymarket's own restricted list did not include
Bulgaria when this was written; check it before depositing.

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
  For a group, add the bot to it; for a channel, make it an admin that can post.
  Post a message there and `getUpdates` shows the chat id (negative for groups and
  channels). Telegram takes about 20 messages a minute in a group, so the watcher
  spaces them about 3 seconds apart; a burst of alerts takes a minute or two to arrive.

Both can be on at once. If one fails, the other still gets the alert, and an alert
nothing delivered is tried again on the next sweep. A research pick sends one alert at
its best price and line, and repeats only when that price improves by `reprice_step`.

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

## 6. Paper trading (forward test with an imaginary bankroll)

Every alert also becomes an imaginary bet, so you can see whether the alerts make
money before risking any. Each bet is recorded at three places, each with its own
bankroll (default €1,000 and €10 flat stakes; Kelly staking is an option):

| Place | Price used |
|---|---|
| Best price | the best price among `best_books` (all books by default; limit it to the ones you can use) |
| Pinnacle | Pinnacle's own price |
| Betfair Exchange | the back price, after 5% commission |

A place only gets a bet when its own price clears the alert's minimum edge, so
the three bankrolls answer "would this work if I could only bet there?". For
"wrong odds" alerts Pinnacle is usually the reference, so the Pinnacle bankroll
mostly holds research picks.

For each bet the watcher keeps:

* **EV when bet**: the expected profit at the moment it was placed.
* **CLV (closing-line value)**: our price against the margin-free closing price of
  Pinnacle (else Betfair), captured just before kickoff. If the reference closed on
  a different handicap or total, its price is moved to our line with a goals model
  (marked `*`). On a small Odds API plan the watcher makes one extra request per
  match about 10 minutes before kickoff (3 credits) to get the closing price.
* **Result and profit**: from the-odds-api scores (2 credits per league, only
  while bets are waiting), else football-data.co.uk (free, updated a few times a
  week). A bet with no result after 7 days is voided.

Open `http://raspberrypi.local:8080/paper`, or run `python3 -m footy.watch paper`
over SSH. The chart shows actual profit against the profit the EV and the closing
line predicted, with a shaded luck range. How to read it:

* Profit alone needs many hundreds of bets before it means anything; the luck
  range shows how far it can swing by chance.
* CLV is the early signal. Beating the closing line on average over 100+ bets is
  the best evidence of a real edge; consistently losing to it means the prices
  taken were not really value, whatever the profit says so far.

`/api/paper.csv` downloads every bet. `python3 -m footy.watch paper-reset --yes`
starts the bankrolls again.

## 7. Polymarket and arbitrage

Turn on `[polymarket]` and its prices join everything else: "wrong odds" alerts
when a Polymarket share is cheaper than Pinnacle's no-margin price (after the taker
fee), research picks priced at Polymarket, and a separate Polymarket bankroll on
the paper-trading page. `python3 -m footy.watch polymarket` lists its leagues and
the next matches it can see.

A Polymarket price of 40c is decimal odds 1/0.40 = 2.50 before the fee. The fee
for takers is `rate x p x (1-p)` per share (about 1.2c at 50c with the 0.05
rate), so the alerts use 1/(0.40 + 0.012) = 2.43. Check the fee shown on the
market and set `fees.polymarket` to match.

**Arbitrage alerts** look at every book together. When backing all outcomes of
a market at the best prices (after commission and fees) costs less than the
payout, you get an alert with the stake split. Example: Arsenal at Polymarket 40c,
draw at Polymarket 24c, Chelsea 3.50 at Pinnacle is a 5.6% locked-in return.
Things to know:

* Set `arb_books` to the books you can use, or most arbs will be at books you can't reach.
* Place the least liquid leg first (usually Polymarket or a soft book), then the rest.
* Prices move. An arb that is only half placed is a normal bet.
* Soft bookmakers limit accounts that arb. Pinnacle, Betfair and Polymarket don't
  limit winners, so arbs between them are the ones that last.

Paper trading records each arbitrage in its own "Arbitrage" bankroll (default
€100 split across the legs), so you can see how often they appear and what they
would have made.

## 8. More sports and OddsPapi

Besides football the watcher handles baseball, basketball, ice hockey, tennis,
volleyball, rugby union, rugby league and American football. "Wrong odds",
arbitrage and paper trading work the same way: Pinnacle's (or Betfair's) price
without its margin is the fair price, and anything clearly better is an alert.
No historical data is needed for that.

Only markets with the same rules everywhere are compared:

| Sport | Markets compared |
|---|---|
| Football | match result (90 min), totals, Asian handicap |
| Baseball, basketball, American football | winner (incl. extra innings / overtime), totals, handicap (run line / spread) |
| Ice hockey | winner incl. overtime, regular-time result, totals, puck line |
| Rugby union and league | match result, totals, handicap |
| Tennis, volleyball | winner only (books mix games, sets and points in handicaps and totals) |

Turn sports on per feed:

* **OddsPapi**: add the sport under `[oddspapi.sports]`. An empty list follows the
  busiest tournaments in the next 36 hours; put tournament ids there to choose.
* **Betfair**: `sports = ["soccer", "tennis", "baseball", ...]` (winner markets).
* **The Odds API**: `groups = ["Baseball", "Tennis", ...]` (costs credits per league).
* **Polymarket**: `sports = ["mlb", "nba", "nhl", "atp", ...]` (codes from `python3 -m footy.watch polymarket`).

**OddsPapi on the free plan (250 requests a month).** One request returns every
match in up to 10 tournaments for up to 3 bookmakers, and the sport list, market
list and fixture list are kept on disk (`oddspapi-cache.json`). The watcher counts
requests and spaces sweeps so they last the month. Each sport costs one fixture
request a day plus one odds request per sweep, so on the free plan one sport gets a
sweep about every 3-4 hours and two sports about every 8 hours. Fine for paper
trading the idea; live alerts need a paid plan (set `monthly_requests` to it).

Check it once installed (uses about 4 requests the first time, 1 after):

```bash
python3 -m footy.watch oddspapi --sport soccer --bookmakers
python3 -m footy.watch oddspapi --sport baseball --save oddspapi-samples
```

If prices don't show up, send the files in `oddspapi-samples/` (they contain no key).

Results for paper bets: The Odds API scores and football-data settle by final
score; the Betfair feed settles winner markets in any sport from Betfair's own
result. Tennis and volleyball bets need the Betfair feed for that.
