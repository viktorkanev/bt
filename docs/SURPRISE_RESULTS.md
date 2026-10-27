# Results: market-surprise decomposition (pre-registered test)

Rules were registered in `docs/PREREGISTRATION.md` (git commit `6e38115`,
2026-10-27 23:13 UTC) before this holdout run. Reproduce with
`footy confirm --window holdout`.

## Short version

* **The inefficiency is real and survived out of sample.** Bookmakers
  over-react to finishing luck and goalkeeper luck: a team that has scored more
  than its shots on target justify (or conceded fewer) is priced too short in
  its next match. On 68,270 unseen holdout matches the effect is as strong as
  in development (t = -5.2 and -4.7; joint p = 1.7 x 10^-16).
* **It does not beat Bet365's margin.** None of the three pre-registered
  betting rules passed. The handicap version came closest (+1.2% over 6,034
  bets, not significant).
* Two parts of the original idea did *not* survive: over-reaction to results
  versus the market's expectation and under-reaction to shot dominance were
  significant before 2016 and close to zero after it. Either the market learned
  or the development effect was partly noise.

## Hypotheses

| | Development 2009-2015 | Holdout 2016-2026 | Holm p (holdout) | Verdict |
|---|---|---|---|---|
| H1 effect: GD residual vs market on the four features | t: results -3.6, shots +4.6, finishing luck -1.7, keeper luck -1.7 | t: results -0.6, shots +0.5, **finishing luck -5.2, keeper luck -4.7** | < 0.001 | **Confirmed** (all signs as predicted) |
| H2 1X2 at Bet365, EV >= 3% | +5.2% (2,135 bets) | -0.6% (2,411 bets, 95% -7.1% to +5.7%) | 1.0 | Failed |
| H3 Asian handicap at Bet365, signal >= 0.05 | +0.3% (5,572 bets) | +1.2% (6,034 bets, 95% -1.1% to +3.4%) | 0.46 | Failed |
| H4 1X2 at Bet365 with best-price features, EV >= 4% | +10.2% (3,011 bets) | -1.7% (2,643 bets) | 1.0 | Failed |

H4's large development return fell apart on the holdout. That fits the caveat
registered in advance (the best-price snapshot may have been taken later than
Bet365's price in older seasons) and is a warning about "stale bookmaker"
signals built from aggregated odds.

## How big is the confirmed effect in money terms?

Post-hoc projection, not a test: re-pricing the H3 holdout bets at Bet365's
margin-free handicap odds plus a smaller margin.

| Margin on the handicap price | Holdout, 6,034 bets | Development, 5,572 bets |
|---|---|---|
| 3.6% (about Bet365's) | +1.1% | +1.0% |
| 3.0% | +1.6% | +1.5% |
| 2.0% (Pinnacle / exchange territory) | +2.5% | +2.4% |
| 0% | +4.2% | +4.1% |

The effect is the same size in both periods. Bet365's own margins were
higher before 2016, which is why the actual development ROI was only +0.3%.

This assumes a sharper book's fair line equals Bet365's de-vigged line. A
sharp book may already price part of the luck effect, so treat +2.5% as an
upper bound. The forward test below measures it properly.

## Ideas explored on development data and dropped

| Idea | What we found (2005-2016 only) |
|---|---|
| 1X2 odds imply a total; bet over/under where it disagrees with the totals market | The totals market was right; goals followed its price, not the 1X2-implied one |
| The reverse: totals price implies a draw probability | No signal |
| Red card in the previous match as a suspension proxy | +1 point draw residual, too small |
| Rest-day difference | Nothing |
| Promoted/relegated teams early in the season | Under 1 point, inconsistent |
| Winter totals in lower leagues | Fewer goals than priced in Feb-Mar (about 2 points) but under bets still lost to the margin |
| Bet365 is the best price on an outcome | Positive but noisy; profit did not rise with the edge |

## What would prove profitability now: forward test

The holdout is spent. The only clean evidence left is matches the research
never saw: everything after **2026-09-03**, the last result in this data.
Two rules are frozen in `config/forward_rules.json`:

* **F1** lower-division over/under 2.5 (tiers 2-5, model+market blend, edge
  3-10% at the best available price, odds <= 4.0); the lead from the first
  report.
* **F2** luck-fade Asian handicap (frozen coefficients, signal >= 0.05), scored
  at Pinnacle, Bet365, market max and market average prices.

Run on a machine that can reach football-data.co.uk:

```bash
footy fetch && footy build && footy forward
```

Matches from 4 September 2026 onwards are already available there (about
eight weeks of games at the time of writing). The summary reports ROI, p-value
and **closing-line value against Pinnacle** for each rule and price.

Decision rule, fixed now:

* Stake real money on a rule only if, after at least 500 settled bets, its
  average CLV against Pinnacle's closing price is positive with p < 0.05.
* Drop it if CLV is negative after 500 bets, whatever the ROI.

CLV needs far fewer bets than ROI to separate skill from luck, because it does
not depend on match outcomes.
