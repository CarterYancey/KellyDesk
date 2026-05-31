## Table of Contents

1. [The Problem](#the-problem)
2. [Core Concepts](#core-concepts)
   - [The Binary Contract](#the-binary-contract)
   - [Edge and Expected Value](#edge-and-expected-value)
   - [Kelly Criterion — Single Bet](#kelly-criterion--single-bet)
   - [Why Single-Bet Kelly Breaks for Simultaneous Positions](#why-single-bet-kelly-breaks-for-simultaneous-positions)
3. [The Joint Kelly Solver](#the-joint-kelly-solver)
   - [Objective Function](#objective-function)
   - [Why Over-Leverage Is Impossible by Design](#why-over-leverage-is-impossible-by-design)
   - [The Three-Contract Example](#the-three-contract-example)
   - [Solver Algorithm](#solver-algorithm)
4. [Risk of Ruin](#risk-of-ruin)
   - [Long-Run Drawdown Probability](#long-run-drawdown-probability)
   - [Single-Round Catastrophe Risk](#single-round-catastrophe-risk)
   - [Why Simultaneity Raises Risk](#why-simultaneity-raises-risk)
5. [Sizing Policy](#sizing-policy)
   - [Fractional Kelly](#fractional-kelly)
   - [Target Risk of Ruin Mode](#target-risk-of-ruin-mode)
   - [Hard Exposure Cap](#hard-exposure-cap)

---

## The Problem

Suppose you have a bankroll of $10,000 and three independent contracts on your desk, each priced at $0.25 with your estimated win probability of 50%. The naive approach — sizing each bet independently using single-contract Kelly — recommends allocating **33.3%** of bankroll to each. Applied simultaneously, that is 100% exposure. If all three contracts lose — a 12.5% probability event — you are completely wiped out.

This is not a quirk or edge case. It is the fundamental error of applying a single-bet formula to a multi-bet portfolio: the formula assumes you own the whole bankroll for each bet, so three bets each get 33%, for a total claim of 100%. Kelly Desk solves this by treating the entire portfolio as a single joint optimization problem, producing allocations that account for all contracts simultaneously. The same three contracts correctly sized land at roughly **24.9% each** — total exposure ~75%, leaving a 25% cash cushion that survives any single-round outcome.

---

## Core Concepts

### The Binary Contract

A binary contract has exactly two outcomes at resolution:

| Outcome | Probability | Net change |
|---------|-------------|------------|
| Win | *p* | Receive $1 per contract, net gain = $(1 − c) per contract |
| Lose | *1 − p* | Receive $0, net loss = $c per contract |

where *c* is the purchase price (cost), constrained to (0, 1).

If you allocate fraction *f* of your bankroll *B* to a contract costing *c*, you purchase *fB/c* contracts. Defining **net odds** *b = (1 − c)/c*, your bankroll is multiplied by:

```
Win:   1 + f·b
Lose:  1 − f
```

This is a lever: a contract at *c* = 0.25 has *b* = 3, meaning each dollar risked can return three dollars of profit.

### Edge and Expected Value

**Edge** is the raw advantage of a contract:

```
e = p − c
```

This is the expected payout minus the cost, in dollars per contract. A contract with *p* = 0.55 and *c* = 0.40 has edge *e* = +$0.15 per contract — you expect to earn 15 cents on every contract you buy.

A contract is only worth holding when *e* > 0, i.e. when your probability estimate exceeds the price. The Kelly solver automatically assigns zero allocation to any contract with non-positive edge.

**Expected value in dollar terms** (for *N* contracts):

```
EV = N · (p − c)
```

Expected value tells you the average profit but says nothing about risk or appropriate size. Kelly criterion bridges the gap.

### Kelly Criterion — Single Bet

For a single binary contract held in isolation, the fraction of bankroll that maximizes the long-run expected growth rate is:

```
f* = (p − c) / (1 − c)
```

**Derivation.** Maximizing expected log-wealth is equivalent to maximizing the growth rate. If you bet fraction *f*:

```
G(f) = p·ln(1 + f·b) + (1−p)·ln(1 − f)
```

Setting G'(f) = 0 and solving:

```
p·b/(1 + f·b) = (1−p)/(1−f)
p·b·(1−f) = (1−p)·(1+f·b)
f* = (p·b − (1−p)) / b = p − (1−p)/b
   = [p(1−c) − (1−p)c] / (1−c)
   = (p − c) / (1 − c)
```

The number of contracts to purchase:

```
N = f* · B / c
```

where *B* is your bankroll. With a $10,000 bankroll, *c* = 0.40, and *f\** = 25%, you buy $2,500 / $0.40 = **6,250 contracts**.

### Why Single-Bet Kelly Breaks for Simultaneous Positions

The derivation above assumes your entire bankroll is available for this one bet and that no other bets are placed at the same time. When you hold *n* contracts simultaneously, each bet implicitly stakes a fraction of the same pool of capital. If three bets each claim 33%, the combined claim is 100%. A single round in which all three contracts lose — probability (1−p)³ — results in total ruin.

More precisely: single-bet Kelly ignores the **joint outcome distribution**. The worst single-round event is not "one contract loses" — it is "all contracts lose simultaneously." Correct sizing must account for this tail.

---

## The Joint Kelly Solver

### Objective Function

Instead of optimizing each contract independently, Kelly Desk maximizes expected log-wealth over the **full joint outcome space** of all *n* simultaneous contracts. With *n* contracts, there are 2ⁿ possible outcomes (each subset *S* of contracts can win). The probability of outcome *S* for independent contracts is:

```
P(S) = ∏_{i∈S} pᵢ · ∏_{i∉S} (1 − pᵢ)
```

The bankroll multiplier in outcome *S* is:

```
M(S, f) = 1 + Σ_{i∈S} fᵢ·bᵢ − Σ_{i∉S} fᵢ
```

The contracts in *S* return their net odds *bᵢ* on the staked fraction; the contracts not in *S* forfeit the staked fraction. The joint objective is:

```
G(f) = Σ_S P(S) · ln( M(S, f) )
```

The optimal allocation vector **f\*** maximizes *G* subject to *fᵢ ≥ 0*.

### Why Over-Leverage Is Impossible by Design

The all-lose outcome (*S* = ∅) contributes the term:

```
P(∅) · ln( 1 − Σfᵢ )
```

As total exposure Σ*fᵢ* → 1, this term diverges to −∞, dragging the objective to −∞ regardless of how profitable the other outcomes are. The optimizer therefore never reaches 100% total exposure. No explicit constraint is required — the **log-utility objective itself enforces a strict exposure ceiling** determined by the edge and probability structure of your portfolio.

Adding more positive-edge contracts does not push you past 100%; it redistributes the existing allocation. Each new contract competes for the same bankroll, and the optimizer accounts for the marginal value of adding capital to the new position against the increased catastrophe risk in the all-lose tail.

### The Three-Contract Example

Three contracts: *c* = 0.25, *p* = 0.50, so *b* = 3.

By symmetry, the optimal *f* is the same for all three. With *j* contracts winning out of 3, the multiplier is:

```
M_j = 1 + f(4j − 3)
```

The outcome probabilities are Binomial(3, 0.5):

| Wins *j* | Multiplier | Probability |
|----------|------------|-------------|
| 0 | 1 − 3f | 1/8 |
| 1 | 1 + f | 3/8 |
| 2 | 1 + 5f | 3/8 |
| 3 | 1 + 9f | 1/8 |

Setting G'(f) = 0:

```
(1/8)·(−3)/(1−3f) + (3/8)·(1)/(1+f) + (3/8)·(5)/(1+5f) + (1/8)·(9)/(1+9f) = 0
```

Solving numerically: **f\* ≈ 0.249**. Total exposure ≈ 74.7%. The worst-case single-round loss is 74.7% of bankroll (when all three lose, probability 12.5%) — painful, but survivable and never a wipeout.

A fourth identical contract would cause the solver to redistribute to roughly 20% each, total ~80%, rather than exceeding any natural bound.

### Solver Algorithm

Kelly Desk uses **coordinate-wise Newton ascent** with incremental multiplier maintenance.

The algorithm maintains the array *M*[*S*] for all 2ⁿ outcomes. For each coordinate *i*, it computes the exact gradient and second derivative of *G* with respect to *fᵢ* (holding all others fixed):

```
G'(fᵢ)  = Σ_S P(S) · aᵢ(S) / M(S)
G''(fᵢ) = −Σ_S P(S) · aᵢ(S)² / M(S)²
```

where *aᵢ*(*S*) = *bᵢ* if *i* ∈ *S* (contract wins), −1 otherwise (contract loses). The Newton step *Δ = −G'/G''* is then clamped to keep all *M*(*S*) > 0 and *fᵢ* ≥ 0. When *fᵢ* changes by *Δ*, every *M*(*S*) is updated in O(2ⁿ) time by adding *aᵢ*(*S*)·*Δ*.

The algorithm sweeps all coordinates to convergence. The full objective is strictly concave — since log is concave and *M*(*S*) is linear in *f* — so coordinate ascent converges to the global optimum.

**Computational note.** Exact enumeration of 2ⁿ outcomes is tractable up to *n* ≈ 16 contracts (65,536 outcomes) in the browser. The tool caps the joint solver at 16 positive-edge contracts. In practice, portfolios of that size are common; beyond it you would want a Monte Carlo gradient estimator.

---

## Risk of Ruin

### Long-Run Drawdown Probability

Over many rounds of simultaneous portfolio betting, the log-wealth process `ln W_t` behaves (by the central limit theorem applied to i.i.d. log returns) approximately as a Brownian motion with drift *m* (mean log-return per round) and diffusion coefficient *v* (variance of log-return per round):

```
m = G(f) = Σ_S P(S) · ln M(S)
v = Σ_S P(S) · [ln M(S)]² − m²
```

For a Brownian motion with positive drift, the probability of ever reaching a level *d* below the starting point is:

```
P(ruin to level) = e^(−2m·d/v)
```

Setting the ruin level as "bankroll drops to fraction *α* of its initial value" means *d* = −ln *α* = ln(1/*α*), and:

```
P(ever draw down to α · B₀) ≈ α^(2m/v)
```

This formula has a clean interpretation. The exponent *k* = 2*m*/*v* is the ratio of the return's drift to its variance — a measure of reward per unit of risk. Higher *k* means the long-run growth force dominates the random fluctuations, so the probability of hitting the ruin barrier is lower. The formula is **general**: it applies to any repeated round where the per-round log-return has mean *m* and variance *v*, whether those rounds involve one contract or twenty simultaneous ones.

The key distinction from the previous single-asset approximation is that *m* and *v* are now computed over the **joint** outcome distribution — all 2ⁿ outcomes weighted by their joint probabilities — rather than approximated from a single-bet diffusion model.

### Single-Round Catastrophe Risk

Distinct from long-run drawdown risk, the **single-round catastrophe** is the probability and magnitude of the worst outcome in any given round:

```
P(all contracts lose) = ∏ᵢ (1 − pᵢ)
Loss if all lose       = Σᵢ fᵢ  (as fraction of bankroll)
```

Because the joint solver keeps Σ*fᵢ* < 1, a single all-lose round can never fully wipe you out. The tool reports this number explicitly because it is the most intuitive single-round tail risk and the one most likely to cause behavioral errors (panic-selling or abandoning a sound strategy after one bad round).

### Why Simultaneity Raises Risk

Contrast two strategies with identical aggregate exposure:

- **Sequential**: bet 30% of bankroll on one contract per round, one at a time.
- **Simultaneous**: bet 10% each on three identical contracts per round.

Both have the same expected log-return *m* (since edges are identical), but the simultaneous strategy has higher per-round variance *v* — the three contracts can all win or all lose together, generating larger tail outcomes in both directions. Higher *v* at the same *m* means lower *k* = 2*m*/*v*, which means higher ruin probability *α^k*. This is the core reason a joint optimizer is necessary: it picks allocations that account for the cross-contract variance, not just each contract's individual contribution.

---

## Sizing Policy

### Fractional Kelly

Full Kelly (*λ* = 1) maximizes the asymptotic growth rate but subjects you to large drawdowns along the way. For the three-contract example, full Kelly gives roughly 40% chance of ever halving your bankroll — a psychological and practical challenge even if the long-run growth is optimal.

The standard mitigation is **fractional Kelly**: scale the entire optimal allocation vector by *λ* ∈ (0, 1]:

```
f_applied = λ · f*
```

Scaling the whole vector (rather than adjusting each contract independently) preserves the relative allocation across contracts — you stay on the efficient frontier — while shifting the aggressiveness of the overall position. The resulting *m* and *v* at the scaled allocation determine a new ruin probability *α^(2m(λ)/v(λ))*, which decreases as *λ* decreases.

### Target Risk of Ruin Mode

Rather than guessing an appropriate *λ*, Kelly Desk lets you specify the risk constraint directly: set *α* (what "ruin" means — e.g. 50% drawdown) and *R* (the maximum probability of ever hitting *α* you are willing to accept — e.g. 10%). The tool binary-searches over *λ* ∈ (0, 1] to find the largest fraction of full Kelly consistent with your risk tolerance, then sizes every contract accordingly.

This mode directly answers the practitioner's question: *"Given my edge estimates and my risk appetite, what is the most aggressive I can responsibly be?"*

For single-asset Kelly, an explicit closed form exists:

```
λ_max = 2·ln(α) / ln(α·R)
```

For the joint multi-asset case the relationship between *λ* and ruin probability is not cleanly separable, so the tool solves it numerically. The binary search is fast (50 iterations covers 15 significant figures of precision) and runs after every parameter change.

### Hard Exposure Cap

As an additional safety layer, the tool offers a **hard exposure cap**: a maximum fraction of bankroll (e.g. 80%) that may be staked simultaneously, regardless of what the Kelly solver and risk policy suggest. If the scaled allocation's total exposure exceeds the cap, the entire vector is scaled down proportionally until total exposure equals the cap.

The cap is off by default (set to 100%) since the joint solver already guarantees exposure below 100%. It is most useful for investors who want an explicit comfort limit — for example, always keeping 30% in cash — regardless of the mathematical optimum.
