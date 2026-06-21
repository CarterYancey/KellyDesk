# Kelly Desk — Binary Contract Portfolio Manager

A bankroll management tool for investors who buy binary financial contracts — instruments that pay exactly **$1** or **$0** at resolution — and need principled, mathematically sound sizing for an entire simultaneous portfolio. Kelly Desk implements *joint* Kelly criterion optimization and derives risk-of-ruin directly from the portfolio's full outcome distribution.

---

### Install dependencies and start the dev server:
 
```bash
npm install
npm run dev
```
 
Vite will print a local URL — typically `http://localhost:5173`. Open it in your browser. The dashboard hot-reloads on any file save.
 
**Optional — production build:**
 
```bash
npm run build    # outputs to dist/
npm run preview  # serves the production build locally
```
 
The `dist/` folder is a self-contained static site you can host anywhere (GitHub Pages, Netlify, S3, etc.).
 
### Persistence Locally
 
The `main.jsx` polyfill implementation backed by `localStorage`, giving the local build equivalent persistence — your state survives page refreshes and browser restarts, stored in the browser's local storage under the key `kelly-desk-state-v1`.
 
If you want to reset all saved state, run this in the browser console:
 
```js
localStorage.removeItem('kelly-desk-state-v1')
```
 
then refresh the page.

---

## Using the Tool

### Entering Contracts

Each row in the contracts table represents one open or candidate position:

| Field | Description |
|-------|-------------|
| **Name** | A label for the contract (free text). |
| **Cost** | The purchase price *c* per contract, as a decimal between 0 and 1 exclusive. A contract trading at 62 cents is entered as `0.62`. |
| **P(win)** | Your probability estimate *p* that the contract resolves at $1. This is your signal, not the market price. Enter as a decimal: 70% = `0.70`. |

Contracts with *p* ≤ *c* (zero or negative edge) are automatically excluded from the joint optimizer and assigned zero allocation. They remain visible in the table so you can track them and update your estimate if conditions change.

**Whole contracts mode.** Because you can only buy integer quantities, enabling "Whole contracts only" floors each contract's allocation to the nearest whole number of contracts. The displayed stake and quantity reflect the integer-rounded position; the risk and growth metrics are computed on the continuous (pre-rounding) allocation as your sizing *intent*. For small bankrolls or high-cost contracts, rounding can materially reduce actual exposure below the policy target.

### Reading the Output

**Sizing Policy card** shows the active λ (fraction of full joint Kelly) and, in risk-target mode, the implied Kelly fraction derived from your risk tolerance inputs.

**Risk of Ruin card** shows:
- The headline probability of ever drawing down to your defined ruin level, computed from joint log-return moments.
- The drawdown curve: *P*(ever reach fraction *x*) = *x^k* for *k* = 2*m*/*v*, plotted across all drawdown levels. The red marker shows your chosen *α*.

**Portfolio card** shows:
- **Exposure bar**: the total fraction of bankroll committed (sum of applied fractions). Color-coded: green below 60%, amber 60–85%, red above 85%.
- **All-lose catastrophe**: the single-round worst case — its probability and the fraction of bankroll lost.
- **Expected profit / round**: arithmetic expectation of dollar profit across all outcomes.
- **Log-growth / round**: *m* = *G*(**f**), the geometric mean growth rate. This is the quantity Kelly maximizes. Positive log-growth compounds to long-run wealth; a contract mix with negative log-growth slowly ruins you even if the arithmetic EV is positive.

**Contracts table** shows per-contract:
- **Edge**: *p* − *c* in cents per contract.
- **Full Kelly**: the contract's share of the *jointly* optimal portfolio. This number already accounts for all other contracts — it is not the single-contract Kelly fraction.
- **Sized**: the applied fraction after your policy and cap adjustments.
- **Stake**: total dollars to commit to this contract (*= Qty × c*).
- **Qty**: number of contracts to purchase.

### The Monte Carlo Check

The Monte Carlo simulator replays your current portfolio over thousands of independent paths. Each step in a path resolves all *n* contracts simultaneously and independently, multiplying wealth by *M*(*S*) for the realized outcome *S*. The simulation tracks the minimum wealth reached over the path and compares it to *α* · starting wealth to assess whether "ruin" (as you have defined it) was ever hit.

The empirical ruin probability should sit just below the theoretical *α^(2m/v)* because the simulation uses a finite number of rounds (400 by default) while the formula assumes an infinite horizon. Larger divergence suggests the asymptotic formula is less reliable — typically because individual bets are large relative to the bankroll, violating the continuous-limit assumption underlying the Brownian motion approximation.

The median terminal wealth after 400 rounds indicates the compounding trajectory: a value of 2.5× means a typical path grows to 2.5 times its starting value over 400 simultaneous rounds. This gives intuition for how long the growth takes to manifest versus how often bad runs occur.

---

## Assumptions and Limitations

**Independence (by default).** Unless you say otherwise, all contracts resolve independently and *P*(*S*) is the product of marginals. If two contracts concern related events — the same political outcome, correlated markets, or any shared underlying uncertainty — that product is wrong. Positively correlated contracts make the all-lose tail more likely than the independent model predicts (the tool would understate ruin risk); natural hedges make it safer (overstating ruin risk, under-allocating).

**Declaring correlations.** The **Linked Contracts** panel removes the independence assumption for any group of contracts you link. Three relationship types are supported:

- **Nested / implication chain** — when one event logically implies another (e.g. "price > $7" ⟹ "price > $5"). The marginals alone pin down the joint: only the *k*+1 "staircase" outcomes are possible.
- **Mutually exclusive** — at most one member can win; outcomes with two or more winners have probability zero.
- **Pairwise correlation** — for a pair, supply any one of *P*(*A*\|*B*), *P*(*B*\|*A*), *P*(*A*\|¬*B*), *P*(*B*\|¬*A*), correlation *ρ*, or odds ratio *θ*; the joint *P*(*A*∧*B*) is solved from it and shown alongside all six derived quantities.

Each link's joint replaces the independence product over its members when building *P*(*S*); the solver, risk metrics, and the exact 2ⁿ enumeration all consume the corrected distribution with no change to the optimization math. Inputs that imply a logically impossible joint are rejected against the Fréchet–Hoeffding bounds with a message naming the valid range, and the offending link falls back to independence until corrected. A contract may belong to at most one link. (Implementation: `src/joint.js`.)

**Probability estimation.** Kelly criterion is exquisitely sensitive to the accuracy of your *p* estimates. A systematic overestimate of 5 percentage points in all your *p* values will cause the solver to recommend significantly larger positions than warranted, with correspondingly higher actual (as opposed to estimated) ruin probability. The fractional-Kelly reduction exists largely to hedge against overconfident probability estimates.

**Stationary edge.** The long-run drawdown formula assumes the portfolio is re-run repeatedly with the same contracts and the same edge. In practice, contracts resolve, new opportunities arise, and edge estimates change. The formula gives the right order-of-magnitude intuition for a dynamically managed portfolio with typical turnover, but it is not literally applicable to a fixed one-off set of bets.

**Finite round approximation.** The Brownian motion derivation of the drawdown formula is a diffusion limit — it becomes exact as bets become small and numerous. For large fractional positions (e.g., 30%+ of bankroll per contract), the discrete nature of the bet outcomes means the true ruin probability may differ noticeably from the formula. The Monte Carlo check provides a more reliable estimate in this regime.

**Integer rounding.** Flooring to whole contract quantities reduces your actual exposure below the policy target, which is conservative and safe. The risk metrics reflect your policy intent, not the exact integer position.

**Browser performance.** The exact joint solver enumerates 2ⁿ outcomes and is capped at 16 positive-edge contracts (65,536 outcomes) for performance. Portfolios larger than this require approximation — Monte Carlo gradient estimation being the natural extension.

---

## Quick Reference — Key Formulas

| Quantity | Formula |
|----------|---------|
| Net odds | *b* = (1 − *c*) / *c* |
| Edge | *e* = *p* − *c* |
| Single-contract full Kelly | *f\** = (*p* − *c*) / (1 − *c*) |
| Contracts to buy | *N* = *f\** · *B* / *c* |
| Joint objective | *G*(**f**) = Σ_S *P*(*S*) · ln *M*(*S*, **f**) |
| All-lose multiplier | *M*(∅) = 1 − Σ *fᵢ* |
| Per-round log-growth | *m* = *G*(**f\***) |
| Per-round log-variance | *v* = Σ_S *P*(*S*) [ln *M*(*S*)]² − *m*² |
| Long-run ruin to *α* | *P* ≈ *α^(2m/v)* |
| All-lose catastrophe prob | ∏ᵢ (1 − *pᵢ*) |
| Applied allocation | **f**_applied = *λ* · **f\*** (then cap if needed) |
| Single-asset λ from risk | *λ* = 2·ln(*α*) / ln(*α*·*R*) |

---

*Probabilities are estimates. All models are wrong; some are useful. Never bet more than you can afford to lose.*
