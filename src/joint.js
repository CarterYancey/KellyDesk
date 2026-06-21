// ============================================================
//  Correlated outcomes — construct the joint distribution from
//  user-declared links instead of assuming independence.
//
//  Nothing here touches the optimization math; these helpers only
//  change how the prob[s] array fed into the solver is built. With
//  no links every cluster is a singleton and jointProbArray()
//  reproduces the independent product ∏ᵢ (sᵢ ? pᵢ : 1−pᵢ) exactly.
// ============================================================

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// Fréchet–Hoeffding bounds on the pairwise joint P(A∧B) given marginals a, b.
export const frechet = (a, b) => ({ lo: Math.max(0, a + b - 1), hi: Math.min(a, b) });

// odds ratio θ = (p11·p00)/(p10·p01) as a function of the joint j.
export const oddsAt = (j, a, b) => {
  const p10 = a - j, p01 = b - j;
  if (p10 <= 1e-15 || p01 <= 1e-15) return Infinity;
  return (j * (1 - a - b + j)) / (p10 * p01);
};

// Pick the root of the odds-ratio quadratic that lands inside the Fréchet range.
export function solveOdds(theta, a, b, lo, hi) {
  if (!(theta >= 0)) return NaN;
  const A2 = theta - 1;
  const B2 = -(theta * (a + b) + (1 - a - b));
  const C2 = theta * a * b;
  let roots;
  if (Math.abs(A2) < 1e-12) {
    roots = Math.abs(B2) > 1e-15 ? [-C2 / B2] : [];
  } else {
    const disc = B2 * B2 - 4 * A2 * C2;
    if (disc < 0) return NaN;
    const sq = Math.sqrt(disc);
    roots = [(-B2 + sq) / (2 * A2), (-B2 - sq) / (2 * A2)];
  }
  for (const r of roots) if (r >= lo - 1e-9 && r <= hi + 1e-9) return clamp(r, lo, hi);
  return NaN;
}

// Feasible range of the chosen input quantity, derived from the Fréchet j-range.
export function pairwiseInputRange(input, a, b, lo, hi) {
  switch (input) {
    case "P(A|B)": return b > 0 ? [lo / b, hi / b] : [0, 0];
    case "P(B|A)": return a > 0 ? [lo / a, hi / a] : [0, 0];
    case "P(A|~B)": return (1 - b) > 0 ? [(a - hi) / (1 - b), (a - lo) / (1 - b)] : [0, 0];
    case "P(B|~A)": return (1 - a) > 0 ? [(b - hi) / (1 - a), (b - lo) / (1 - a)] : [0, 0];
    case "corr": { const d = Math.sqrt(a * (1 - a) * b * (1 - b)); return d > 0 ? [(lo - a * b) / d, (hi - a * b) / d] : [0, 0]; }
    case "odds": return [oddsAt(lo, a, b), oddsAt(hi, a, b)];
    default: return [0, 0];
  }
}

// The six (well, seven) pairwise quantities derived from (a, b, j), for display.
export function pairwiseQuantities(a, b, j) {
  const d = Math.sqrt(a * (1 - a) * b * (1 - b));
  return {
    "P(A)": a, "P(B)": b, "P(A∧B)": j,
    "P(A|B)": b > 0 ? j / b : NaN,
    "P(B|A)": a > 0 ? j / a : NaN,
    "P(A|~B)": (1 - b) > 0 ? (a - j) / (1 - b) : NaN,
    "P(B|~A)": (1 - a) > 0 ? (b - j) / (1 - a) : NaN,
    corr: d > 0 ? (j - a * b) / d : NaN,
    odds: oddsAt(j, a, b),
  };
}

export const PW_LABEL = {
  "P(A|B)": "P(A|B)", "P(B|A)": "P(B|A)", "P(A|~B)": "P(A|¬B)", "P(B|~A)": "P(B|¬A)",
  corr: "correlation ρ", odds: "odds ratio θ",
};

// Solve for the joint j implied by a pairwise association input, with feasibility
// check. Returns { ok, j, lo, hi, range, error? }.
export function derivePairwise(a, b, assoc) {
  const { lo, hi } = frechet(a, b);
  const input = assoc ? assoc.input : null;
  const x = assoc ? Number(assoc.value) : NaN;
  const range = input ? pairwiseInputRange(input, a, b, lo, hi) : [0, 0];
  if (assoc == null || !isFinite(x)) return { ok: false, j: NaN, lo, hi, range, error: "Enter an association value." };
  let j;
  switch (input) {
    case "P(A|B)": j = x * b; break;
    case "P(B|A)": j = x * a; break;
    case "P(A|~B)": j = a - x * (1 - b); break;
    case "P(B|~A)": j = b - x * (1 - a); break;
    case "corr": j = a * b + x * Math.sqrt(a * (1 - a) * b * (1 - b)); break;
    case "odds": j = solveOdds(x, a, b, lo, hi); break;
    default: return { ok: false, j: NaN, lo, hi, range, error: "Unknown association input." };
  }
  const label = PW_LABEL[input] || input;
  if (!isFinite(j) || j < lo - 1e-9 || j > hi + 1e-9) {
    const [rlo, rhi] = range;
    const rangeStr = `[${isFinite(rlo) ? rlo.toFixed(3) : "−∞"}, ${isFinite(rhi) ? rhi.toFixed(3) : "∞"}]`;
    const jStr = isFinite(j) ? j.toFixed(3) : "no real solution";
    return {
      ok: false, j, lo, hi, range,
      error: `${label}=${x} with P(A)=${a.toFixed(3)}, P(B)=${b.toFixed(3)} requires P(A∧B)=${jStr}, but given these marginals the joint must lie in [${lo.toFixed(3)}, ${hi.toFixed(3)}]. Valid ${label} range: ${rangeStr} (raise/lower the marginals or pick a value in range).`,
    };
  }
  return { ok: true, j: clamp(j, lo, hi), lo, hi, range };
}

// Iterative Proportional Fitting: max-entropy joint over 2^k cells from
// (mask, target) constraints. Future-proofs k≥3 correlated clusters; with only
// marginal constraints it converges to the independent product.
export function ipfJoint(k, constraints) {
  const N = 1 << k;
  const t = new Float64Array(N).fill(1 / N);
  for (let iter = 0; iter < 200; iter++) {
    let maxViol = 0;
    for (const { mask, target } of constraints) {
      let cur = 0;
      for (let s = 0; s < N; s++) if ((s & mask) === mask) cur += t[s];
      maxViol = Math.max(maxViol, Math.abs(cur - target));
      const up = target / (cur || 1e-15), dn = (1 - target) / ((1 - cur) || 1e-15);
      for (let s = 0; s < N; s++) t[s] *= (s & mask) === mask ? up : dn;
    }
    if (maxViol < 1e-10) break;
  }
  return t;
}

// Staircase table for a nested/implication chain (members in descending-P order).
export function nestedTable(p) {
  const k = p.length, table = new Float64Array(1 << k);
  table[0] = 1 - p[0];
  for (let m = 1; m < k; m++) table[(1 << m) - 1] = p[m - 1] - p[m];
  table[(1 << k) - 1] = p[k - 1];
  return table;
}

// Build one cluster's 2^k cell table from a link. Returns { memberIdx, table }
// or null (pushing a descriptive error) if the link is infeasible.
export function buildClusterTable(link, present, items, errors) {
  const k = present.length;
  const p = present.map((i) => items[i].p);
  const nm = present.map((i) => items[i].name || items[i].rowId);

  if (link.type === "nested") {
    for (let i = 0; i + 1 < k; i++) {
      if (p[i] < p[i + 1] - 1e-9) {
        errors.push({ id: link.id, msg: `Nested link invalid: "${nm[i]}" (P=${p[i].toFixed(3)}) is ordered ahead of "${nm[i + 1]}" (P=${p[i + 1].toFixed(3)}), but nesting requires the outer event to be at least as likely as the inner one. Reorder so probabilities are non-increasing.` });
        return null;
      }
    }
    return { memberIdx: present.slice(), table: nestedTable(p) };
  }

  if (link.type === "exclusive") {
    const sum = p.reduce((s, x) => s + x, 0);
    if (sum > 1 + 1e-9) {
      errors.push({ id: link.id, msg: `Mutually-exclusive link invalid: member win-probabilities sum to ${sum.toFixed(3)} > 1 (${nm.map((n, i) => `${n}=${p[i].toFixed(2)}`).join(", ")}). Lower one or more so they sum to ≤ 1.` });
      return null;
    }
    const table = new Float64Array(1 << k);
    table[0] = 1 - sum;
    for (let i = 0; i < k; i++) table[1 << i] = p[i];
    return { memberIdx: present.slice(), table };
  }

  if (link.type === "pairwise") {
    if (k === 2) {
      const a = p[0], b = p[1];
      const r = derivePairwise(a, b, link.assoc);
      if (!r.ok) { errors.push({ id: link.id, msg: r.error }); return null; }
      const j = r.j;
      const table = Float64Array.of(1 - a - b + j, a - j, b - j, j); // [p00, p10(A), p01(B), p11]
      return { memberIdx: present.slice(), table };
    }
    // k ≥ 3 pairwise cluster → max-entropy IPF (UI keeps pairwise at 2 members).
    const constraints = present.map((_, i) => ({ mask: 1 << i, target: p[i] }));
    return { memberIdx: present.slice(), table: ipfJoint(k, constraints) };
  }

  return null;
}

// Partition items (positive-edge contracts) into independent clusters — one per
// active link, singletons for everything else — and build each cluster's joint
// cell table. Returns { clusters, errors }.
export function buildClusters(items, links) {
  const errors = [];
  const idxByRow = new Map(items.map((it, i) => [it.rowId, i]));
  const used = new Set();
  const clusters = [];
  for (const link of links) {
    const present = [];
    for (const rid of link.members) {
      const i = idxByRow.get(rid);
      if (i !== undefined && !used.has(i)) present.push(i);
    }
    if (present.length < 2) continue; // too few active members → treat as independent
    const cl = buildClusterTable(link, present, items, errors);
    if (cl) { cl.memberIdx.forEach((i) => used.add(i)); clusters.push(cl); }
    // invalid link: members fall through to singletons (independent fallback)
  }
  for (let i = 0; i < items.length; i++) {
    if (!used.has(i)) {
      const pp = items[i].p;
      clusters.push({ memberIdx: [i], table: Float64Array.of(1 - pp, pp) });
    }
  }
  return { clusters, errors };
}

// Compose the full 2^n joint as the product across mutually-independent clusters.
export function jointProbArray(items, clusters) {
  const n = items.length, N = 1 << n;
  const prob = new Float64Array(N);
  for (let s = 0; s < N; s++) {
    let pr = 1;
    for (const cl of clusters) {
      let local = 0;
      for (let bit = 0; bit < cl.memberIdx.length; bit++) {
        if (s & (1 << cl.memberIdx[bit])) local |= 1 << bit;
      }
      pr *= cl.table[local];
    }
    prob[s] = pr;
  }
  return prob;
}
