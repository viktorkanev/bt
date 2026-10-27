# Pre-registration: market-surprise decomposition strategies

**Registered:** 2026-10-27, committed to git before the holdout evaluation below
was run. The holdout result is in `docs/SURPRISE_RESULTS.md` (written after).

## Idea

Bookmaker prices are a forecast. After every match a team has a "surprise"
relative to that forecast. We split each team's recent surprise into parts
and test whether the market over- or under-reacts to each part:

| Feature (EWMA over the team's previous ~10 league matches) | Meaning |
|---|---|
| `surp_pts` | points won minus market-expected points (3*P(win) + P(draw), margin removed) |
| `surp_sot` | shots-on-target difference minus the pre-match expectation of our shots model |
| `fin_luck` | goals scored minus league conversion rate x shots on target |
| `gk_luck` | conversion rate x shots on target conceded minus goals conceded |

Features enter as home minus away. Every value uses previous matches only.

**Prediction:** the market over-reacts to results and finishing/goalkeeping luck
(negative coefficients) and under-reacts to shot dominance (positive coefficient).

## What was looked at before registration

* Development window: matches before 2016-07-01. All exploration of these
  signals, and the choice of every threshold below, used only that window.
* Explored on development data and **dropped** (not tested on the holdout):
  1X2-implied totals vs the over/under market (and the reverse, totals-implied
  draw vs the draw price); red card in the previous match as a suspension
  proxy; rest-day differences; promoted/relegated team mispricing; winter
  totals in lower leagues; "Bet365 is the best price" as a stand-alone rule.
* Disclosure: the original report (`reports/latest`) already showed holdout
  results for other strategies (model+market blend, consensus, lower-league
  over/under). Those strategies are not part of this test. The blend model there
  included shot-rating features but not these surprise/luck features.

## Data and protocol

* Matches: GitHub mirror of football-data.co.uk, eligible matches (both teams
  >= 10 previous matches, league >= 200 matches) from 2005-07-01.
* **Holdout: seasons starting 2016-07-01 to the last result (2026-09-03).**
* Walk-forward: for each holdout season the model is re-fitted on every
  eligible match from 2005-07-01 up to the start of that season, then predicts
  the season. Same code path as the development check (`footy.research.confirm`).
* Odds: Bet365 1X2 or Asian-handicap prices as recorded by football-data.co.uk.
  All bets are settled at Bet365's price. Flat 1-unit stakes.

## Hypotheses and success criteria

Significance: one-sided, Holm-corrected across H1-H4, alpha = 0.05.

* **H1 (the effect exists out of sample).** OLS of the goal-difference residual
  versus the 1X2-market-implied goal difference on the four features (no
  interactions), HC1 errors. Success: all four coefficient signs as predicted
  (surp_pts -, surp_sot +, fin_luck -, gk_luck -) **and** joint Wald p < 0.05.
* **H2 (profitable at a price you can get).** Multinomial logistic regression
  (standardised, C = 1) on Bet365 log-probabilities (power method), the four
  features and their interactions with "tier >= 2". Bet the home or away win
  (no draws) with the highest expected value at Bet365 odds, if EV >= 3% and
  odds are between 1.7 and 7.0. Success: ROI > 0 with p < 0.05 (t-test on
  per-bet returns).
* **H3 (Asian handicap).** OLS of the goal-difference residual on the four
  features plus tier interactions gives a supremacy shift; apply it to the
  1X2-implied expected goals, and bet the AH side at Bet365 where the shift
  adds >= 0.05 to expected value. Success: ROI > 0 with p < 0.05.
* **H4 (adds the best-price consensus).** As H2 plus best-price (market max)
  log-probabilities as features; EV >= 4%. Success: ROI > 0 with p < 0.05.
  *Caveat declared in advance:* if football-data.co.uk collected the market
  max later than Bet365's price, H4 could profit from information that was not
  available when the Bet365 price was quoted. H4 passing is therefore weaker
  evidence than H2 passing.

## Development results that informed the rules (rolling origin, 2009-2015)

| | Bets | ROI | p |
|---|---|---|---|
| H1 joint effect | 34,095 matches | all signs as predicted | 1.3e-11 |
| H2 | 2,135 | +5.2% | 0.063 |
| H3 | 5,572 | +0.3% | 0.415 |
| H4 | 3,011 | +10.2% | 0.0003 |

## What counts as failure

Any hypothesis with Holm-adjusted p >= 0.05 has failed, whatever its ROI. A
holdout ROI below zero for H2 means the strategy should not be bet. Secondary
analyses (other EV thresholds, per season, per tier) are descriptive only.
