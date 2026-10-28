# Closing-line value of the research rules (football-data.co.uk data)

Data: a direct football-data.co.uk download (256,578 matches to 2026-10-27),
which has closing odds that the GitHub mirror lacks: Pinnacle closing 1X2 from
2012, closing over/under and Asian handicap odds (Pinnacle, Bet365, market
average and max) from 2019/20, and from 2026-27 **Betfair Exchange** prices and
**expected goals (xG)**. Pinnacle prices stop part-way through 2025-26, after
Pinnacle closed its public API.

Reproduce: `footy clv` (writes `reports/research/clv/results.json`). Rules are
the pre-registered ones; models are walk-forward (each season fitted on earlier
seasons only). This is secondary analysis of frozen rules on seasons already used
for the holdout test, plus the forward period after 2026-09-03.

Closing-line value (CLV) = price taken x closing probability (margin removed) - 1.
A positive average means the bets were struck at better prices than the market
settled on.

## 1. The luck bias survives to the closing price

Goal-difference residual against **Pinnacle's closing price**, 2016-2026,
64,145 matches:

| Feature | t-statistic |
|---|---|
| Finishing luck | **-4.27** |
| Goalkeeper luck | **-3.75** |
| Results vs market | -0.72 |
| Shot dominance | -2.45 (not pre-registered in this direction) |

Even the sharpest closing line over-rates teams whose goals outran their shots.

## 2. Luck-fade Asian handicap (F2)

| Price | Seasons | Bets | ROI | p | CLV |
|---|---|---|---|---|---|
| Bet365, pre-match | 2019-2026 | 4,549 | +1.2% | 0.17 | -1.9% |
| Pinnacle, pre-match | 2019-2025 | 4,243 | +1.3% | 0.17 | -1.4% |
| Pinnacle, closing | 2019-2025 | 4,224 | +1.1% | 0.21 | |
| Market average, closing | 2019-2026 | 4,416 | -0.1% | 0.54 | |
| Betfair Exchange, pre-match, after 5% commission | 2026-27 | 102 | +10.5% | 0.12 | +0.7% |
| Betfair Exchange, closing, after 5% commission | 2026-27 | 149 | +10.4% | 0.08 | |

Baseline: betting **every** handicap side at Pinnacle's pre-match price has CLV
-2.9% (the cost of the margin and of line movement). F2's picks at -1.4% are
1.4 points better, so the signal does point the way the market later moves, but
not far enough to overcome the margin.

At sharp prices F2 is worth about +1%. With handicap returns this noisy, proving
a 1% edge at p < 0.05 takes roughly 24,000 bets. It is not proven, and at Bet365
or market-average prices it is break-even.

## 3. Lower-division over/under (F1)

| Price | Bets | ROI | p | CLV | Beat close |
|---|---|---|---|---|---|
| Best price (market max), 2016-2026 | 3,660 | **+4.8%** | **0.002** | **+1.2%** (t = 9.3) | 56% |
| Pinnacle | 604 | +1.7% | 0.35 | -1.8% | 35% |
| Bet365 | 67 | +3.4% | 0.39 | -0.7% | 42% |
| No model: best price beats Bet365's fair price by 3-10% | 638 | +4.9% | 0.10 | +1.5% (t = 5.5) | 61% |
| Baseline: every side at best price | | | | -2.5% | |

F1 makes money and beats the close **only at the best price across bookmakers**.
A rule with no model at all gets similar CLV on fewer bets. F1 is mainly a
line-shopping edge in lower leagues; the model adds volume. It needs accounts
with many bookmakers, and those bookmakers limit winning accounts.

The "best price" column in football-data.co.uk now covers fewer bookmakers: its
lower-league margin rose from about 1.8% (2019) to 4.5% (2026-27). That is why F1
found only a handful of bets this season.

## 4. Skill/luck 1X2 model at Bet365 (H2)

2,182 bets, CLV -3.6% (t = -16). No edge at Bet365's 1X2 prices.

## Forward test since 2026-09-03 (frozen rules, unseen matches)

| Rule and venue | Bets | ROI | CLV vs Betfair close |
|---|---|---|---|
| F2 at Betfair Exchange (after 5% commission) | 68 | +11.4% | +1.3% (38 bets) |
| F2 at Bet365 | 91 | +4.9% | -2.1% (51 bets) |
| F1 at best price | 8 | +54% | -1.3% |

Far too few bets to judge.

## What to do with this

1. **Proven:** bookmakers, including Pinnacle at the close, over-react to
   finishing and goalkeeper luck.
2. **Not proven:** that it pays. At sharp prices it is worth about 1%.
3. **Best venue to keep testing F2: Betfair Exchange.** No account limits, its
   closing price is now the sharpest benchmark in the data, and the first 2026-27
   numbers are positive. `footy upcoming` now prefers exchange prices for F2.
4. **F1 is a line-shopping strategy.** Profitable historically at the best price,
   but only if you can get the best price.
5. **Next upgrade:** football-data.co.uk now publishes xG for most leagues.
   Once a season of xG has built up, rebuild the luck features on real xG instead
   of the shots-on-target proxy and pre-register that version.
