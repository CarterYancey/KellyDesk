// Ad-hoc verification of the joint-distribution construction.
// Run: node src/joint.test.mjs
import {
  frechet, oddsAt, derivePairwise, nestedTable,
  buildClusters, jointProbArray, ipfJoint,
} from "./joint.js";

let pass = 0, fail = 0;
const approx = (x, y, t = 1e-9) => Math.abs(x - y) <= t;
const ok = (name, cond) => { if (cond) { pass++; } else { fail++; console.error("FAIL:", name); } };
const sum = (a) => a.reduce((s, x) => s + x, 0);

// ---- 1. Independence: all singletons reproduce ∏ pᵢ exactly ----
{
  const items = [
    { rowId: "a", p: 0.5, b: 3, name: "A" },
    { rowId: "b", p: 0.3, b: 2, name: "B" },
    { rowId: "c", p: 0.7, b: 1, name: "C" },
  ];
  const { clusters, errors } = buildClusters(items, []);
  const prob = jointProbArray(items, clusters);
  const n = items.length, N = 1 << n;
  let maxErr = 0;
  for (let s = 0; s < N; s++) {
    let pr = 1;
    for (let i = 0; i < n; i++) pr *= s & (1 << i) ? items[i].p : 1 - items[i].p;
    maxErr = Math.max(maxErr, Math.abs(pr - prob[s]));
  }
  ok("singletons reproduce independent product", maxErr < 1e-12);
  ok("independence: no errors", errors.length === 0);
  ok("independence: sums to 1", approx(sum([...prob]), 1));
}

// ---- 2. Nested staircase ----
{
  // w>5 (p=0.7) ⟹ ... actually outer must be MORE likely. Outer=0.7, mid=0.5, inner=0.2
  const items = [
    { rowId: "x", p: 0.2, b: 1, name: "w>10" }, // inner (least likely)
    { rowId: "y", p: 0.7, b: 1, name: "w>5" },  // outer (most likely)
    { rowId: "z", p: 0.5, b: 1, name: "w>7" },  // mid
  ];
  const link = { id: "L1", type: "nested", members: ["y", "z", "x"] }; // declared high→low
  const { clusters, errors } = buildClusters(items, [link]);
  ok("nested: no errors", errors.length === 0);
  const prob = jointProbArray(items, clusters);
  ok("nested: sums to 1", approx(sum([...prob]), 1));
  // Only k+1 = 4 nonzero cells. Marginals must be preserved.
  const nz = [...prob].filter((x) => x > 1e-12).length;
  ok("nested: exactly k+1 nonzero cells", nz === 4);
  // check marginal of each member recovered
  const n = items.length, N = 1 << n;
  const marg = items.map((_, i) => { let m = 0; for (let s = 0; s < N; s++) if (s & (1 << i)) m += prob[s]; return m; });
  ok("nested marginal y=0.7", approx(marg[1], 0.7, 1e-9));
  ok("nested marginal z=0.5", approx(marg[2], 0.5, 1e-9));
  ok("nested marginal x=0.2", approx(marg[0], 0.2, 1e-9));
  // nesting implies P(inner ∧ ¬outer)=0: P(x win, y lose) must be 0
  let pXnotY = 0;
  for (let s = 0; s < N; s++) if ((s & 1) && !(s & 2)) pXnotY += prob[s];
  ok("nested: inner⟹outer (P(x∧¬y)=0)", approx(pXnotY, 0, 1e-12));

  // invalid order: declare inner before outer
  const bad = { id: "L2", type: "nested", members: ["x", "y", "z"] };
  const r2 = buildClusters(items, [bad]);
  ok("nested invalid order flagged", r2.errors.length === 1 && r2.errors[0].id === "L2");
  // fallback to independence: prob sums to 1 and equals product
  const p2 = jointProbArray(items, r2.clusters);
  ok("nested invalid falls back to independence sum=1", approx(sum([...p2]), 1));
}

// ---- 3. Mutually exclusive ----
{
  const items = [
    { rowId: "a", p: 0.3, b: 1, name: "A" },
    { rowId: "b", p: 0.4, b: 1, name: "B" },
  ];
  const link = { id: "E1", type: "exclusive", members: ["a", "b"] };
  const { clusters, errors } = buildClusters(items, [link]);
  ok("exclusive: no errors", errors.length === 0);
  const prob = jointProbArray(items, clusters);
  ok("exclusive sums to 1", approx(sum([...prob]), 1));
  // both win cell (s=3) must be 0
  ok("exclusive: P(both)=0", approx(prob[3], 0, 1e-12));
  ok("exclusive: P(none)=0.3", approx(prob[0], 0.3, 1e-9));
  ok("exclusive: marginals preserved", approx(prob[1], 0.3) && approx(prob[2], 0.4));

  // infeasible: sum > 1
  const items2 = [{ rowId: "a", p: 0.7, b: 1, name: "A" }, { rowId: "b", p: 0.6, b: 1, name: "B" }];
  const r = buildClusters(items2, [{ id: "E2", type: "exclusive", members: ["a", "b"] }]);
  ok("exclusive sum>1 flagged", r.errors.length === 1);
}

// ---- 4. Pairwise: all six inputs map to the same j, feasibility, bounds ----
{
  const a = 0.5, b = 0.75;
  const items = [{ rowId: "a", p: a, b: 1, name: "A" }, { rowId: "b", p: b, b: 1, name: "B" }];
  const mkProb = (assoc) => {
    const { clusters, errors } = buildClusters(items, [{ id: "P1", type: "pairwise", members: ["a", "b"], assoc }]);
    return { prob: jointProbArray(items, clusters), errors };
  };
  // independence check: P(A|B) = a = 0.5 should give j = ab = 0.375
  const ind = derivePairwise(a, b, { input: "P(A|B)", value: a });
  ok("pairwise indep j = ab", ind.ok && approx(ind.j, a * b));
  // and corr=0 same j
  const c0 = derivePairwise(a, b, { input: "corr", value: 0 });
  ok("pairwise corr0 j = ab", c0.ok && approx(c0.j, a * b));
  // and odds=1 same j
  const o1 = derivePairwise(a, b, { input: "odds", value: 1 });
  ok("pairwise odds1 j = ab", o1.ok && approx(o1.j, a * b, 1e-7));
  // independence prob equals product
  const { prob } = mkProb({ input: "corr", value: 0 });
  ok("pairwise corr0 prob = product",
    approx(prob[0], (1 - a) * (1 - b)) && approx(prob[1], a * (1 - b)) &&
    approx(prob[2], (1 - a) * b) && approx(prob[3], a * b));

  // marginal preservation for a non-trivial association
  const { prob: pc, errors: ec } = mkProb({ input: "P(B|A)", value: 1 }); // B certain given A → j = a = 0.5 (feasible, j∈[0.25,0.5])
  ok("pairwise P(B|A)=1 feasible", ec.length === 0);
  ok("pairwise marginal A preserved", approx(pc[1] + pc[3], a));
  ok("pairwise marginal B preserved", approx(pc[2] + pc[3], b));
  ok("pairwise sums to 1", approx(sum([...pc]), 1));

  // infeasible: P(A|B)=1 → j=0.75 > min(a,b)=0.5  → rejected
  const bad = derivePairwise(a, b, { input: "P(A|B)", value: 1 });
  ok("pairwise P(A|B)=1 infeasible", !bad.ok && /range/.test(bad.error));
  const r = mkProb({ input: "P(A|B)", value: 1 });
  ok("pairwise infeasible flagged + fallback sum=1", r.errors.length === 1 && approx(sum([...r.prob]), 1));

  // Fréchet bounds sanity
  const fb = frechet(a, b);
  ok("frechet bounds", approx(fb.lo, 0.25) && approx(fb.hi, 0.5));
  // odds monotonic: larger odds → larger j
  const lowJ = derivePairwise(a, b, { input: "odds", value: 0.5 });
  const hiJ = derivePairwise(a, b, { input: "odds", value: 4 });
  ok("odds monotonic in j", lowJ.ok && hiJ.ok && lowJ.j < a * b && hiJ.j > a * b);
  // round-trip: oddsAt(j) recovers θ
  ok("oddsAt round trips", approx(oddsAt(hiJ.j, a, b), 4, 1e-6));
}

// ---- 5. Two independent clusters multiply correctly ----
{
  const items = [
    { rowId: "a", p: 0.5, b: 1, name: "A" },
    { rowId: "b", p: 0.75, b: 1, name: "B" },
    { rowId: "c", p: 0.4, b: 1, name: "C" }, // singleton
  ];
  const link = { id: "P1", type: "pairwise", members: ["a", "b"], assoc: { input: "corr", value: 0.3 } };
  const { clusters } = buildClusters(items, [link]);
  const prob = jointProbArray(items, clusters);
  ok("mixed cluster + singleton sums to 1", approx(sum([...prob]), 1));
  // marginal of singleton c must be 0.4 regardless of the a-b link
  const N = 1 << 3; let mc = 0;
  for (let s = 0; s < N; s++) if (s & (1 << 2)) mc += prob[s];
  ok("singleton marginal independent of cluster", approx(mc, 0.4, 1e-9));
}

// ---- 6. IPF with only marginals → independence ----
{
  const t = ipfJoint(3, [{ mask: 1, target: 0.5 }, { mask: 2, target: 0.3 }, { mask: 4, target: 0.8 }]);
  // independent product for bit pattern
  const p = [0.5, 0.3, 0.8];
  let maxErr = 0;
  for (let s = 0; s < 8; s++) {
    let pr = 1;
    for (let i = 0; i < 3; i++) pr *= s & (1 << i) ? p[i] : 1 - p[i];
    maxErr = Math.max(maxErr, Math.abs(pr - t[s]));
  }
  ok("IPF marginals-only → independence", maxErr < 1e-9);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
