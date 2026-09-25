# Scalping methodology — cost first, signal second

Written 2026-09-25, after nine experiments and no wins.

---

## 1. The diagnosis everything so far has missed

    LIVE   n=388   gross +0.1175%   fee 0.0998%   NET +0.0177%   fee = 85% of gross
    DEMO   n=762   gross +0.0897%   fee 0.0998%   NET -0.0101%   fee = 111% of gross

**The fee is 85% of the edge on live and more than 100% on demo.** This is not
a strategy with a signal problem. It is a strategy whose economics leave
almost nothing on the table regardless of how good the signal gets.

Nine experiments, all of them signal:

    CRT · jev verdict · jev confidence · jev components · jev picks ·
    room ahead · shape · callback multiplier · exit rules

Not one tested cost. Cost is the larger term.

### What each lever is worth, per trade, in % of price

    today               gross +0.1175   fee 0.0998   NET +0.0177
    maker ENTRY         gross +0.1175   fee 0.0700   NET +0.0475    2.7x
    maker BOTH sides    gross +0.1175   fee 0.0400   NET +0.0775    4.4x

To reach a gross/fee ratio of 2.0 by signal alone, the gross move must rise
**+70%**. Nothing measured this week moved it by more than ~10%. To reach it
by cost alone, the fee must fall to 0.0587% — maker on both sides gets to
0.0400% and clears it outright.

**This is the whole argument for what follows.**

---

## 2. The method

### 2.1 One metric governs

`gross / fee` per trade, leverage-free. Currently **1.18 live, 0.90 demo**.
Target **≥ 2.0 sustained over 200 trades**.

Leverage cancels in this ratio — it scales gross and fee together — so it is
the only figure comparable across instances and across leverage changes. Use
it in preference to P&L, ROI, win rate and median price move, all of which
have misled at some point in this investigation.

### 2.2 Pre-register before deploying

Written down BEFORE the config changes, or the experiment does not count:

    HYPOTHESIS   one sentence, falsifiable
    METRIC       which number moves, and in which direction
    SIZE         n per arm, decided in advance
    KILL         what result reverts the change
    CONFOUNDS    what else changes at the same time (ideally: nothing)

Every failure this week is traceable to skipping one of these. The callback
experiment changed the floor and silently handed 29% of trades to a different
sizing rule. The shape verdict was drawn against a cohort from a different
market period. The exit replay's headline moved 33 points on 4 of 200 trades.

### 2.3 Read the paired number, never the total

Totals are seductive and unstable. `trail only` totalled +17.5 at n=196 and
−16.0 at n=200 — four trades. The paired 95% interval included zero
throughout and was right throughout.

When the same trades appear under both arms, difference them per trade. Most
of the variance is "which trade was it" and it cancels.

### 2.4 One variable

Two changes at once and neither is readable. If a second change is
unavoidable, the experiment is abandoned, not reinterpreted.

### 2.5 Check the cohorts are comparable before reading the result

- **median ATR** within 25% — else the arms traded different markets
- **time ranges** overlapping — a field added mid-flight splits into
  "populated" and "everything before the deploy"
- **exit mix** similar — else the comparison is partly about exit paths
- **n ≥ 30 per arm** as a floor, not a target

The tooling prints all four. They exist because each one silently corrupted a
conclusion here first.

---

## 3. The experiment queue, in order of expected value

### A. Maker entry — post-only limit instead of trailing-stop market

**Expected: 2.7x net edge.** Larger than every signal effect measured this
week combined.

A `TRAILING_STOP_MARKET` entry fires as a market order and pays taker
(0.05%). A post-only `LIMIT` resting at the same retracement level pays maker
(0.02%).

    HYPOTHESIS  fee per trade falls from ~0.100% to ~0.070% of price with no
                material fall in gross move
    METRIC      fee/notional per trade; gross/fee ratio
    SIZE        100 trades
    KILL        gross move falls more than 15%, or fill rate below 50%

The risk is unfilled entries: price retraces to the level and keeps going, so
the limit never fills. That is not purely a loss — an unfilled entry costs
nothing, and 79% of current entries are underwater at first sight, so being
structurally pickier may help twice. Measure the fill rate; do not assume it.

**Note this is a deterministic saving.** It requires no forecast. Every other
item in this queue requires predicting the market.

### B. Maker take-profit alongside the existing trail

**Expected: a further ~0.03% on whatever fraction exits at the target.**

Keep the trail as protection; add a post-only limit at a target. When the
limit fills, the round trip is 0.04% rather than 0.07%.

Distinct from the replay's TP test, which REPLACED the trail and hurt. Here
the trail still owns the downside; the limit only changes how winners exit.

    HYPOTHESIS  a maker TP improves gross/fee without cutting the right tail
    METRIC      gross/fee, and p90 of gross move (must not fall)
    SIZE        100 trades
    KILL        p90 falls more than 10%

### C. Fee tier

VIP1 needs 30-day volume of 15M USDT. At ~$30 notional × 40 trades/day you
are nowhere near it, so this is not available and should not be planned
around. **Worth stating so nobody proposes it again.**

BNB fee discount is already settled as declined — constant fee rate preferred
over accounting complexity.

### D. RSI 80+ ceiling

The only signal finding with support from three independent datasets: refused
candidates (n=37 in band), live trades (n=8, net −0.97), demo trades (n=3,
net −29.21). All negative.

    HYPOTHESIS  removing RSI>80 shorts raises gross/fee
    METRIC      gross/fee on shorts
    SIZE        50 shorts
    KILL        no improvement, or short volume falls more than 20%

It REMOVES trades rather than adding them, so it cannot make the fee problem
worse. Cheapest item in the queue.

### E. Trade less, hold longer — only after A and B

With the fee at 85% of gross, the arithmetic says fewer, larger trades beat
more, smaller ones. But every selection attempt here has failed (nine of
them), so do not attempt this by picking better entries.

The version that does not require prediction: raise the minimum expected
move. `AUTO_MIN_ATR_PCT` currently admits coins whose typical candle is
smaller than the round-trip fee. At ATR 0.5% and a 1.69-minute hold, the
achievable move is barely above cost before the trade starts.

    HYPOTHESIS  AUTO_MIN_ATR_PCT=1.0 raises gross/fee
    METRIC      gross/fee; trades per day
    SIZE        100 trades
    KILL        trades/day falls below 15 without gross/fee reaching 1.6

---

## 4. What to stop doing

**Stop testing entry signals until the cost work is done.** Nine attempts, no
wins, and the largest observed effect was ~10% on gross — against a 30-60%
effect available from fees with no forecasting.

**Stop reading never_green.** It halved (30% → 15%) while outcomes got worse.
It measures the cost of a wide callback, not a defect.

**Stop comparing across instances on anything but entry metrics.** Demo is
live data with a ~0.751 factor on ATR, different leverage, and ROI-denominated
thresholds that act at half the price distance. Same settings, different
strategy.

**Stop reading ROI.** It multiplies by leverage. Four wrong conclusions here
came from reasoning in ROI about a price-denominated quantity.

---

## 5. Honest position on whether this can work

At 0.1175% gross against 0.0998% fee, live currently nets +0.0177% per trade
before the tail. With ~3.8% of wallet deployed per trade at 10x, that is
roughly +0.27%/day — which matches the observed account return.

**That is a real edge, and it is thin enough that a single bad day erases a
week.** Both the −6.90 and −267.51 days happened with medians only slightly
negative; the damage came from the tail, not the centre.

Cost reduction is the only lever measured here that improves the centre
without requiring a better forecast. If maker entry lands as predicted, the
same signal produces 2.7x the net edge, and the tail matters proportionally
less.

If maker entry does NOT land — fills are too rare, or the gross move collapses
because the good entries are exactly the ones that run away — then the honest
conclusion is that this strategy is fee-bound at retail tier and the timeframe
needs to lengthen until the move per trade is several times the cost.
