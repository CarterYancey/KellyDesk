import React, { useState, useEffect, useMemo, useCallback } from "react";

/* ============================================================
   KELLY DESK — simultaneous binary-contract bankroll manager

   Sizes ALL current contracts as ONE simultaneous portfolio by
   maximizing expected log-growth over the joint outcome space:

     G(f) = Σ_S P(S) · ln( 1 + Σ_{i∈S} fᵢbᵢ − Σ_{i∉S} fᵢ )

   where S ranges over the 2ⁿ win/lose combinations and
   bᵢ = (1−cᵢ)/cᵢ. The all-lose term ln(1−Σfᵢ) drives the
   objective to −∞ as exposure → 100%, so the optimum is NEVER
   over-leveraged. Risk of ruin uses the joint per-round
   log-return moments (m, v):   P(reach α) ≈ α^(2m/v).
   ============================================================ */

const fmtMoney = (x) =>
  (x < 0 ? "-$" : "$") +
  Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = (x, d = 1) => (x * 100).toFixed(d) + "%";
// compact money for axis ticks: $1.2k, $3.4M, $850
const fmtMoneyShort = (x) => {
  const a = Math.abs(x), s = x < 0 ? "-$" : "$";
  if (a >= 1e6) return s + (a / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e3) return s + (a / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k";
  return s + a.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : 2);
};
// "nice" round tick values spanning [min,max]
const niceTicks = (min, max, count = 5) => {
  const range = (max - min) || 1;
  const mag = Math.pow(10, Math.floor(Math.log10(range / count)));
  const norm = range / count / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(v);
  return out;
};
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const MAX_N = 16; // exact 2ⁿ enumeration cap

// ---- portfolio solver: maximizes expected CRRA power utility ----
// a = relative risk aversion. a = 1 is log utility = full Kelly.
// a > 1 tilts toward safer (higher-p) contracts and punishes ruin tails harder.
// The objective is globally concave in f for every a > 0, so Newton converges.
function solvePortfolio(items, a = 1, init = null, jointProb = null) {
  const n = items.length;
  if (n === 0) return [];
  const N = 1 << n;
  const b = items.map((it) => it.b);
  let prob = jointProb;
  if (!prob) {
    prob = new Float64Array(N);
    for (let s = 0; s < N; s++) {
      let pr = 1;
      for (let i = 0; i < n; i++) pr *= s & (1 << i) ? items[i].p : 1 - items[i].p;
      prob[s] = pr;
    }
  }
  const sign = (s, i) => (s & (1 << i) ? b[i] : -1);
  const multipliers = (f) => {
    const M = new Float64Array(N);
    for (let s = 0; s < N; s++) {
      let m = 1;
      for (let i = 0; i < n; i++) m += sign(s, i) * f[i];
      M[s] = m;
    }
    return M;
  };
  const isLog = Math.abs(a - 1) < 1e-9;
  const Uval = (M) => (isLog ? Math.log(M) : Math.pow(M, 1 - a) / (1 - a)); // utility
  const Up = (M) => (isLog ? 1 / M : Math.pow(M, -a)); // u'(M) > 0
  const Upp = (M) => (isLog ? -1 / (M * M) : -a * Math.pow(M, -a - 1)); // u''(M) < 0
  const G = (f) => {
    const M = multipliers(f);
    let g = 0;
    for (let s = 0; s < N; s++) {
      if (M[s] <= 0) return -Infinity;
      g += prob[s] * Uval(M[s]); }
    return g;
  };

  let f = init ? init.slice() : new Array(n).fill(0.5 / n);
  for (let iter = 0; iter < 60; iter++) {
    const M = multipliers(f);
    const grad = new Float64Array(n);
    const H = Array.from({ length: n }, () => new Float64Array(n));
    for (let s = 0; s < N; s++) {
      const up = prob[s] * Up(M[s]);
      const upp = prob[s] * Upp(M[s]);
      for (let i = 0; i < n; i++) {
        const ai = sign(s, i);
        grad[i] += up * ai;
        for (let j = i; j < n; j++) H[i][j] += upp * ai * sign(s, j); // negative definite
      }
    }
    for (let i = 0; i < n; i++) {
      H[i][i] -= 1e-9; // damping
      for (let j = 0; j < i; j++) H[i][j] = H[j][i];
    }

    const delta = solveLinear(H, Array.from(grad, (x) => -x)); // H δ = −grad
    const gCur = G(f);
    const dirDeriv = grad.reduce((s, gi, i) => s + gi * delta[i], 0);
    let t = 1, fNew = f;
    for (let ls = 0; ls < 50; ls++) {
      const cand = f.map((x, i) => x + t * delta[i]);
      if (cand.every((x) => x >= 0) && G(cand) >= gCur + 1e-4 * t * dirDeriv) { fNew = cand; break; }
      t *= 0.5;
      if (t < 1e-14) { fNew = f; break; }
    }
    const change = Math.max(...fNew.map((x, i) => Math.abs(x - f[i])));
    f = fNew;
    if (change < 1e-11) break;
  }
  return f;
}

const solveJointKelly = (items, prob = null) => solvePortfolio(items, 1, null, prob); // full Kelly = log utility

// Continuation (homotopy) solve: walk risk aversion up from 1 (full Kelly) in small
// steps, warm-starting each step from the previous solution. This keeps every Newton
// solve interior and well-conditioned, avoiding the boundary stiffness that makes a
// single high-a solve started from the full-Kelly point fail to converge.
function solveAtAversion(items, fKelly, a, prob = null) {
  if (items.length === 0) return [];
  if (a <= 1.0001) return fKelly;
  const steps = Math.max(2, Math.ceil((a - 1) / 0.5));
  let warm = fKelly;
  for (let k = 1; k <= steps; k++) warm = solvePortfolio(items, 1 + ((a - 1) * k) / steps, warm, prob);
  return warm;
}

// Gauss–Jordan solve of A x = rhs for small dense systems.
function solveLinear(A, rhs) {
  const n = rhs.length;
  const M = A.map((row, i) => Array.from(row).concat(rhs[i]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    if (Math.abs(M[col][col]) < 1e-15) M[col][col] = M[col][col] < 0 ? -1e-15 : 1e-15;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// ---- joint per-round log-return moments at a given allocation ----
function jointMoments(items, alloc, jointProb = null) {
  const n = items.length;
  if (n === 0) return { m: 0, v: 0, allLoseP: 0, exposure: 0 };
  const N = 1 << n;
  let m = 0, m2 = 0;
  for (let s = 0; s < N; s++) {
    let pr = jointProb ? jointProb[s] : 1, M = 1;
    for (let i = 0; i < n; i++) {
      const w = (s & (1 << i)) !== 0;
      if (!jointProb) pr *= w ? items[i].p : 1 - items[i].p;
      M += w ? alloc[i] * items[i].b : -alloc[i];
    }
    const l = Math.log(Math.max(M, 1e-12));
    m += pr * l;
    m2 += pr * l * l;
  }
  const v = Math.max(m2 - m * m, 1e-12);
  const allLoseP = jointProb ? jointProb[0] : items.reduce((a, it) => a * (1 - it.p), 1);
  const exposure = alloc.reduce((a, x) => a + x, 0);
  return { m, v, allLoseP, exposure };
}

const ruinProb = (alpha, m, v) => (m <= 0 ? 1 : Math.pow(alpha, (2 * m) / v));

// ---- pairwise dependence solver ----------------------------------------
// Given marginals pA = P(A), pB = P(B) (from the two contracts) and any of the
// four conditionals, recover the full 2×2 joint of a linked pair:
//   p11 = P(A∧B), p10 = P(A∧¬B), p01 = P(¬A∧B), p00 = P(¬A∧¬B).
// With pA, pB fixed there is a single free parameter p11; every conditional
// pins it down. If none is given the pair is independent (p11 = pA·pB).
// `fields` carries the raw string inputs { pAgB, pAgNb, pBgA, pBgNa }.
function solvePair(pA, pB, fields) {
  const valid = Number.isFinite(pA) && Number.isFinite(pB) &&
    pA >= 0 && pA <= 1 && pB >= 0 && pB <= 1;
  // each provided conditional gives a candidate p11
  const cands = [];
  const num = (s) => (s === "" || s == null ? NaN : parseFloat(s));
  const within01 = (x) => Number.isFinite(x) && x >= 0 && x <= 1;
  const cAgB = num(fields.pAgB), cAgNb = num(fields.pAgNb),
        cBgA = num(fields.pBgA), cBgNa = num(fields.pBgNa);
  if (within01(cAgB)) cands.push(cAgB * pB);                 // P(A|B)·P(B)
  if (within01(cBgA)) cands.push(cBgA * pA);                 // P(B|A)·P(A)
  if (within01(cAgNb)) cands.push(pA - cAgNb * (1 - pB));    // pA − P(A|¬B)(1−pB)
  if (within01(cBgNa)) cands.push(pB - cBgNa * (1 - pA));    // pB − P(B|¬A)(1−pA)

  let error = null;
  if (!valid) error = "invalid";
  // consistency: all candidates must agree
  if (!error && cands.length > 1) {
    const spread = Math.max(...cands) - Math.min(...cands);
    if (spread > 1e-6) error = "inconsistent";
  }

  let p11 = cands.length ? cands.reduce((a, b) => a + b, 0) / cands.length
                         : pA * pB; // no conditional → independent
  // Fréchet bounds: p11 ∈ [max(0, pA+pB−1), min(pA,pB)]
  const lo = Math.max(0, pA + pB - 1), hi = Math.min(pA, pB);
  if (!error && (p11 < lo - 1e-9 || p11 > hi + 1e-9)) error = "infeasible";
  p11 = clamp(p11, lo, hi);

  const p10 = pA - p11, p01 = pB - p11, p00 = 1 - pA - pB + p11;
  const safeDiv = (x, d) => (d > 1e-12 ? x / d : NaN);
  const derived = {
    pA, pB,
    pAgB: safeDiv(p11, pB),
    pAgNb: safeDiv(p10, 1 - pB),
    pBgA: safeDiv(p11, pA),
    pBgNa: safeDiv(p01, 1 - pA),
    pAandB: p11,
  };
  const provided = cands.length > 0;
  return { p11, p10, p01, p00, derived, error, provided };
}

// ---- joint distribution over the full 2ⁿ outcome space ------------------
// Builds P(s) for every win/lose combo of `items` using the link forest. The
// links define pairwise joints; over a forest the exact joint factorizes as
//   P(x) = ∏_v P(x_v) · ∏_edges P(x_u,x_v)/(P(x_u)P(x_v))
// which reduces to the independent product when there are no links. Returns
// null (use the fast independent path) when no link is active. `warnings`, if
// passed, is mutated with any { type, ... } notices for the UI.
function buildJointProb(items, links, warnings) {
  const n = items.length;
  if (n === 0 || !links || links.length === 0) return null;
  const idxById = {};
  items.forEach((it, i) => (idxById[it.rowId] = i));

  // union-find for cycle detection over item indices
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };

  const edges = [];
  for (const lk of links) {
    const u = idxById[lk.aId], v = idxById[lk.bId];
    if (u == null || v == null || u === v) continue; // endpoint not in solver
    const pair = solvePair(items[u].p, items[v].p, lk);
    if (!pair.provided) continue; // no conditional → independent, nothing to do
    if (pair.error) { warnings && warnings.push({ type: pair.error, id: lk.id }); }
    const ru = find(u), rv = find(v);
    if (ru === rv) { warnings && warnings.push({ type: "cycle", id: lk.id }); continue; }
    parent[ru] = rv;
    edges.push({ u, v, cells: [pair.p00, pair.p01, pair.p10, pair.p11] });
    // cells indexed by (bitA<<1 | bitB): 00,01,10,11 where bitA = win A, bitB = win B
  }
  if (edges.length === 0) return null;

  const N = 1 << n;
  const prob = new Float64Array(N);
  for (let s = 0; s < N; s++) {
    // node marginals
    let pr = 1;
    for (let i = 0; i < n; i++) pr *= s & (1 << i) ? items[i].p : 1 - items[i].p;
    // edge correction factors
    for (const e of edges) {
      const wa = (s & (1 << e.u)) !== 0 ? 1 : 0;
      const wb = (s & (1 << e.v)) !== 0 ? 1 : 0;
      const joint = e.cells[(wa << 1) | wb];
      const marg = (wa ? items[e.u].p : 1 - items[e.u].p) * (wb ? items[e.v].p : 1 - items[e.v].p);
      pr *= marg > 1e-12 ? joint / marg : 0;
    }
    prob[s] = pr;
  }
  return prob;
}

const STORE_KEY = "kelly-desk-state-v2";
const blank = (n) => ({ id: Math.random().toString(36).slice(2), name: n, cost: "", prob: "" });

export default function KellyDesk() {
  const [bankroll, setBankroll] = useState(10000);
  const [aversion, setAversion] = useState(2); // relative risk aversion (re-weights toward safety)
  const [alpha, setAlpha] = useState(0.5);
  const [rows, setRows] = useState([
    { id: "a", name: "Contract A", cost: "0.25", prob: "0.50" },
    { id: "b", name: "Contract B", cost: "0.25", prob: "0.50" },
    { id: "c", name: "Contract C", cost: "0.25", prob: "0.50" },
  ]);
  const [links, setLinks] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [sim, setSim] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        let r = await window.storage.get(STORE_KEY);
        if (!(r && r.value)) r = await window.storage.get("kelly-desk-state-v1"); // migrate old saves
        if (r && r.value) {
          const s = JSON.parse(r.value);
          if (s.bankroll != null) setBankroll(s.bankroll);
          if (s.aversion != null) setAversion(s.aversion);
          if (s.alpha != null) setAlpha(s.alpha);
          if (Array.isArray(s.rows)) setRows(s.rows);
          if (Array.isArray(s.links)) setLinks(s.links);
        }
      } catch (e) {/* defaults */} finally { setLoaded(true); }
    })();
  }, []);

  useEffect(() => {
    if (!loaded) return;
    (async () => {
      try {
        await window.storage.set(STORE_KEY, JSON.stringify(
          { bankroll, aversion, alpha, rows, links }));
      } catch (e) {/* in-memory only */}
    })();
  }, [bankroll, aversion, alpha, rows, links, loaded]);

  // valid positive-edge contracts feed the solver
  const items = useMemo(() => {
    const list = rows.map((r) => {
      const c = parseFloat(r.cost), p = parseFloat(r.prob);
      const valid = c > 0 && c < 1 && p >= 0 && p <= 1;
      return { rowId: r.id, p, c, b: (1 - c) / c, valid, edge: valid ? p - c : -1 };
    }).filter((x) => x.valid && x.edge > 0);
    return list.slice(0, MAX_N);
  }, [rows]);
  const overCap = rows.filter((r) => {
    const c = parseFloat(r.cost), p = parseFloat(r.prob);
    return c > 0 && c < 1 && p > c;
  }).length > MAX_N;

  const itemsKey = items.map((i) => i.rowId + ":" + i.p + ":" + i.c).join("|");
  const linksKey = links.map((l) => [l.aId, l.bId, l.pAgB, l.pAgNb, l.pBgA, l.pBgNa].join(":")).join("|");

  // joint outcome distribution over all 2ⁿ combos, accounting for linked
  // (correlated) contracts. null ⇒ contracts independent ⇒ fast product path.
  const joint = useMemo(() => {
    const warnings = [];
    const prob = buildJointProb(items, links, warnings);
    return { prob, warnings };
  }, [itemsKey, linksKey]); // eslint-disable-line
  const jointProb = joint.prob;

  const fStar = useMemo(() => solveJointKelly(items, jointProb), [itemsKey, jointProb]); // eslint-disable-line

  // Resolve the applied allocation: re-optimize at relative risk aversion a (CRRA),
  // re-weighting toward safer contracts.
  const alloc = useMemo(() => {
    if (items.length === 0) return [];
    return solveAtAversion(items, fStar, aversion, jointProb);
  }, [items, itemsKey, fStar, aversion, jointProb]); // eslint-disable-line

  const mom = useMemo(() => jointMoments(items, alloc, jointProb), [items, alloc, jointProb]);
  const kEff = mom.v > 0 ? (2 * mom.m) / mom.v : 0;
  const risk = ruinProb(alpha, mom.m, mom.v);

  // per-row execution figures
  const calc = useMemo(() => {
    const byId = {};
    items.forEach((it, idx) => (byId[it.rowId] = { fk: fStar[idx], ap: alloc[idx], item: it }));
    return rows.map((r) => {
      const c = parseFloat(r.cost), p = parseFloat(r.prob);
      const valid = c > 0 && c < 1 && p >= 0 && p <= 1;
      const e = byId[r.id];
      const fk = e ? e.fk : 0, ap = e ? e.ap : 0;
      const dollars = ap * bankroll;
      const n = e && c > 0 ? Math.floor(dollars / c) : 0;
      const stake = n * c;
      return { ...r, c, p, valid, edge: valid ? p - c : 0, posEdge: !!e, fk, ap, n, stake };
    });
  }, [rows, items, fStar, alloc, bankroll]);

  const totals = useMemo(() => {
    const staked = calc.reduce((s, x) => s + (x.stake > 0 ? x.stake : 0), 0);
    const ev = calc.reduce((s, x) => s + (x.n > 0 ? x.n * x.edge : 0), 0);
    return { staked, ev, cash: bankroll - staked };
  }, [calc, bankroll]);

  // exact single-round outcome enumeration over all 2ⁿ joint win/lose combos
  const TOP_ROWS = 30;
  const outcomes = useMemo(() => {
    const n = items.length;
    if (n === 0) return { list: [], belowProb: 0, n: 0, minMult: 0, maxMult: 0 };
    const N = 1 << n;
    const list = new Array(N);
    let belowProb = 0, minMult = Infinity, maxMult = -Infinity;
    for (let s = 0; s < N; s++) {
      let pr = jointProb ? jointProb[s] : 1, M = 1;
      for (let i = 0; i < n; i++) {
        const w = (s & (1 << i)) !== 0;
        if (!jointProb) pr *= w ? items[i].p : 1 - items[i].p;
        M += w ? alloc[i] * items[i].b : -alloc[i];

      }
      list[s] = { mask: s, prob: pr, mult: M };
      if (M < alpha) belowProb += pr;
      if (M < minMult) minMult = M;
      if (M > maxMult) maxMult = M;
    }
    return { list, belowProb, n, minMult, maxMult };
  }, [items, alloc, alpha, jointProb]);

  // top outcomes by probability for the table (full 2ⁿ list is too large to render)
  const topOutcomes = useMemo(
    () => [...outcomes.list].sort((a, b) => b.prob - a.prob).slice(0, TOP_ROWS),
    [outcomes]
  );
  const topProb = topOutcomes.reduce((s, o) => s + o.prob, 0);

  // probability-weighted end-value distribution, binned by multiplier
  const histo = useMemo(() => {
    const BINS = 40;
    if (outcomes.n === 0) return { bins: [], maxP: 0, lo: 0, hi: 1, mean: 1 };
    let lo = outcomes.minMult, hi = outcomes.maxMult;
    if (hi - lo < 1e-9) { lo -= 0.01; hi += 0.01; }
    const width = (hi - lo) / BINS;
    const bins = Array.from({ length: BINS }, (_, k) => ({ x0: lo + k * width, x1: lo + (k + 1) * width, p: 0 }));
    let mean = 0;
    for (const o of outcomes.list) {
      mean += o.prob * o.mult;
      let k = Math.floor((o.mult - lo) / width);
      if (k < 0) k = 0; else if (k >= BINS) k = BINS - 1;
      bins[k].p += o.prob;
    }
    const maxP = bins.reduce((m, b) => Math.max(m, b.p), 0);
    return { bins, maxP, lo, hi, mean };
  }, [outcomes]);

  const updateRow = (id, field, val) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, [field]: val } : r)));
  const addRow = () => setRows((rs) => [...rs, blank("Contract " + String.fromCharCode(65 + rs.length))]);
  const delRow = (id) => setRows((rs) => {
    setLinks((ls) => ls.filter((l) => l.aId !== id && l.bId !== id)); // drop links referencing this contract
    return rs.filter((r) => r.id !== id);
  });

  const updateLink = (id, field, val) => setLinks((ls) => ls.map((l) => (l.id === id ? { ...l, [field]: val } : l)));
  const addLink = () => setLinks((ls) => {
    const a = rows[0]?.id || "", b = rows[1]?.id || rows[0]?.id || "";
    return [...ls, { id: Math.random().toString(36).slice(2), aId: a, bId: b, pAgB: "", pAgNb: "", pBgA: "", pBgNa: "" }];
  });
  const delLink = (id) => setLinks((ls) => ls.filter((l) => l.id !== id));

  const curve = useMemo(() => {
    const W = 280, H = 90, pts = [];
    for (let i = 0; i <= 60; i++) {
      const x = 0.02 + (i / 60) * 0.96;
      const y = Math.pow(x, kEff);
      pts.push([W - x * W, H - y * H]);
    }
    return { W, H, d: "M" + pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L"),
      mark: [W - alpha * W, H - Math.pow(alpha, kEff) * H] };
  }, [kEff, alpha]);

  const expColor = mom.exposure > 0.85 ? "#f85149" : mom.exposure > 0.6 ? "#d29922" : "#3fb950";

  // per-link view: marginals from the contracts, inferred conditionals, warnings
  const linkViews = useMemo(() => {
    const inSolver = new Set(items.map((it) => it.rowId));
    const probById = {};
    rows.forEach((r) => (probById[r.id] = parseFloat(r.prob)));
    return links.map((lk) => {
      const a = rows.find((r) => r.id === lk.aId), b = rows.find((r) => r.id === lk.bId);
      const pA = probById[lk.aId], pB = probById[lk.bId];
      const solved = solvePair(pA, pB, lk);
      const active = inSolver.has(lk.aId) && inSolver.has(lk.bId) && lk.aId !== lk.bId;
      return { lk, aName: a?.name || "—", bName: b?.name || "—", pA, pB, solved, active };
    });
  }, [links, rows, items]);
  // cycle warnings come from the joint builder (which enforces the forest)
  const cycleLinkIds = useMemo(() => new Set(
    joint.warnings.filter((w) => w.type === "cycle").map((w) => w.id)
  ), [joint]);

  return (
    <div className="kd-root">
      <style>{CSS}</style>
      <header className="kd-head">
        <div>
          <h1>KELLY&nbsp;DESK</h1>
          <span className="kd-sub">simultaneous portfolio · joint Kelly · $1 / $0 contracts</span>
        </div>
        <div className="kd-bankroll">
          <label>BANKROLL</label>
          <div className="kd-bankin">
            <span>$</span>
            <input type="number" value={bankroll}
              onChange={(e) => setBankroll(Math.max(0, parseFloat(e.target.value) || 0))} />
          </div>
        </div>
      </header>

      <div className="kd-grid">
        <section className="kd-card kd-policy">
          <div className="kd-card-h">SIZING POLICY</div>
          <div className="kd-controls">
            <Slider label="Ruin = drawdown to" value={alpha} min={0.05} max={0.95} step={0.05}
              onChange={setAlpha} fmt={(v) => fmtPct(v, 0) + " of bankroll"} />
            <Slider label="Relative risk aversion (a)" value={aversion} min={1} max={10} step={0.25}
              onChange={setAversion} fmt={(v) => (v <= 1 ? "1.00 (full Kelly)" : v.toFixed(2))} />
            <div className="kd-derived">→ {fmtPct(risk, 1)} chance of ever reaching {fmtPct(alpha, 0)} of bankroll</div>
          </div>
        </section>

        <section className="kd-card kd-risk">
          <div className="kd-card-h">RISK OF RUIN (JOINT)</div>
          <div className="kd-bignum" style={{ color: risk > 0.25 ? "#f85149" : risk > 0.1 ? "#d29922" : "#3fb950" }}>
            {fmtPct(risk, 1)}
          </div>
          <div className="kd-bignum-cap">long-run chance of ever reaching {fmtPct(alpha, 0)} of bankroll</div>
          <svg viewBox={`0 0 ${curve.W} ${curve.H}`} className="kd-curve" preserveAspectRatio="none">
            <path d={curve.d} fill="none" stroke="#58a6ff" strokeWidth="1.6" />
            <line x1={curve.mark[0]} y1={curve.mark[1]} x2={curve.mark[0]} y2={curve.H} stroke="#f85149" strokeWidth="1" strokeDasharray="3 2" />
            <circle cx={curve.mark[0]} cy={curve.mark[1]} r="3" fill="#f85149" />
          </svg>
          <div className="kd-curve-cap"><span>← deeper drawdown</span><span>P(reach level)</span></div>
        </section>

        <section className="kd-card kd-summary">
          <div className="kd-card-h">PORTFOLIO</div>
          <div className="kd-expbar">
            <div className="kd-expbar-top"><span>Exposure</span><b style={{ color: expColor }}>{fmtPct(mom.exposure, 0)}</b></div>
            <div className="kd-expbar-track"><div className="kd-expbar-fill" style={{ width: fmtPct(Math.min(mom.exposure, 1), 0), background: expColor }} /></div>
          </div>
          <Stat label="Total to stake" value={fmtMoney(totals.staked)} />
          <Stat label="Cash remaining" value={fmtMoney(totals.cash)} />
          <Stat label="Expected profit / round" value={fmtMoney(totals.ev)} accent={totals.ev >= 0 ? "pos" : "neg"} />
          <Stat label="Log-growth / round" value={(mom.m >= 0 ? "+" : "") + (mom.m * 100).toFixed(2) + "%"} accent={mom.m >= 0 ? "pos" : "neg"} />
          <div className="kd-cat">
            ☠ All-lose round: <b>{fmtPct(mom.allLoseP, 1)}</b> chance, costs <b>{fmtPct(mom.exposure, 0)}</b> of bankroll.
            Exposure is solver-capped below 100%, so a single round can never fully wipe you out.
          </div>
        </section>
      </div>

      <section className="kd-card">
        <div className="kd-card-h kd-th-row"><span>CONTRACTS &nbsp;·&nbsp; SIZED AS ONE SIMULTANEOUS PORTFOLIO</span><button className="kd-add" onClick={addRow}>+ add</button></div>
        <div className="kd-table">
          <div className="kd-tr kd-thead">
            <span>Name</span><span>Cost</span><span>P(win)</span><span>Edge</span>
            <span>Full Kelly</span><span>Sized</span><span>Stake</span><span>Qty</span><span></span>
          </div>
          {calc.map((r) => (
            <div className="kd-tr" key={r.id}>
              <input className="kd-name" value={r.name} onChange={(e) => updateRow(r.id, "name", e.target.value)} />
              <input className="kd-num" type="number" step="0.01" value={r.cost} placeholder="0.00"
                onChange={(e) => updateRow(r.id, "cost", e.target.value)} />
              <input className="kd-num" type="number" step="0.01" value={r.prob} placeholder="0.00"
                onChange={(e) => updateRow(r.id, "prob", e.target.value)} />
              <span className={"kd-cell " + (r.edge > 0 ? "pos" : r.valid ? "neg" : "mut")}>
                {r.valid ? (r.edge >= 0 ? "+" : "") + (r.edge * 100).toFixed(1) + "¢" : "—"}
              </span>
              <span className="kd-cell mut">{r.posEdge ? fmtPct(r.fk, 1) : r.valid ? "0%" : "—"}</span>
              <span className="kd-cell">{r.posEdge ? fmtPct(r.ap, 1) : "—"}</span>
              <span className="kd-cell">{r.n > 0 ? fmtMoney(r.stake) : "—"}</span>
              <span className="kd-cell hl">{r.n > 0 ? r.n.toLocaleString() : "—"}</span>
              <button className="kd-del" onClick={() => delRow(r.id)}>×</button>
            </div>
          ))}
        </div>
        <div className="kd-note">
          “Full Kelly” is each contract’s share of the <i>jointly</i> optimal portfolio (already accounts for the others).
          “Sized” applies your risk policy and exposure cap. {overCap && <b style={{ color: "#d29922" }}>Only the first {MAX_N} positive-edge contracts are solved exactly.</b>}
        </div>
      </section>

      <section className="kd-card">
        <div className="kd-card-h kd-th-row"><span>LINKED / CORRELATED CONTRACTS</span><button className="kd-add" onClick={addLink}>+ add link</button></div>
        {links.length === 0 ? (
          <div className="kd-note">
            Link two related contracts whose outcomes are <i>not independent</i> (e.g. “price &gt; $5” and “price &gt; $7”).
            Each contract’s P(win) supplies the marginals P(A) and P(B); enter any one conditional below and the rest of the
            joint distribution is inferred and fed into the solver and outcome table.
          </div>
        ) : (
          <div className="kd-links">
            {linkViews.map(({ lk, pA, pB, solved, active }) => {
              const d = solved.derived;
              const cell = (v) => (Number.isFinite(v) ? fmtPct(v, 1) : "—");
              const isCycle = cycleLinkIds.has(lk.id);
              return (
                <div className="kd-link" key={lk.id}>
                  <div className="kd-link-sel">
                    <select className="kd-select" value={lk.aId} onChange={(e) => updateLink(lk.id, "aId", e.target.value)}>
                      {rows.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
                    </select>
                    <span className="kd-link-amp">A &nbsp;·&nbsp; B</span>
                    <select className="kd-select" value={lk.bId} onChange={(e) => updateLink(lk.id, "bId", e.target.value)}>
                      {rows.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
                    </select>
                    <button className="kd-del" onClick={() => delLink(lk.id)}>×</button>
                  </div>
                  <div className="kd-link-marg">
                    P(A) = <b>{Number.isFinite(pA) ? fmtPct(pA, 1) : "—"}</b> &nbsp;·&nbsp;
                    P(B) = <b>{Number.isFinite(pB) ? fmtPct(pB, 1) : "—"}</b>
                    &nbsp; (from contract P(win))
                  </div>
                  <div className="kd-link-grid">
                    {[
                      ["P(A|B)", "pAgB", d.pAgB],
                      ["P(A|¬B)", "pAgNb", d.pAgNb],
                      ["P(B|A)", "pBgA", d.pBgA],
                      ["P(B|¬A)", "pBgNa", d.pBgNa],
                    ].map(([label, field, inferred]) => (
                      <label className="kd-link-field" key={field}>
                        <span>{label}</span>
                        <input className="kd-num" type="number" step="0.01" min="0" max="1"
                          value={lk[field]} placeholder={Number.isFinite(inferred) ? inferred.toFixed(3) : "—"}
                          onChange={(e) => updateLink(lk.id, field, e.target.value)} />
                        <i className="kd-link-inf">{lk[field] === "" ? "inferred " + cell(inferred) : "← entered"}</i>
                      </label>
                    ))}
                  </div>
                  <div className="kd-link-foot">
                    P(A∧B) = <b>{cell(d.pAandB)}</b>
                    {!solved.provided && <span className="kd-link-warn"> · no conditional set → treated as independent</span>}
                    {solved.error === "infeasible" && <span className="kd-link-warn"> · ⚠ conditional violates P(A)/P(B) bounds (clamped)</span>}
                    {solved.error === "inconsistent" && <span className="kd-link-warn"> · ⚠ entered conditionals disagree</span>}
                    {solved.error === "invalid" && <span className="kd-link-warn"> · ⚠ both contracts need a valid P(win)</span>}
                    {isCycle && <span className="kd-link-warn"> · ⚠ creates a loop with other links — this link is ignored</span>}
                    {!active && !isCycle && solved.provided && !solved.error &&
                      <span className="kd-link-warn"> · ⚠ inactive: both contracts must be positive-edge to affect sizing</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="kd-card">
        <div className="kd-card-h">EXACT SINGLE-ROUND OUTCOMES (2ⁿ ENUMERATION)</div>
        {outcomes.n === 0 ? (
          <div className="kd-note">Add positive-edge contracts to enumerate every possible outcome of one simultaneous round exactly.</div>
        ) : (
          <>
            <div className="kd-bignum" style={{ color: outcomes.belowProb > 0.25 ? "#f85149" : outcomes.belowProb > 0.1 ? "#d29922" : "#3fb950" }}>
              {fmtPct(outcomes.belowProb, 2)}
            </div>
            <div className="kd-bignum-cap">
              chance this single round ends below {fmtPct(alpha, 0)} of bankroll ({fmtMoney(alpha * bankroll)})
            </div>
            <EndValuePdf histo={histo} bankroll={bankroll} alpha={alpha} />

            <div className="kd-otable">
              <div className="kd-otr kd-othead">
                <span>Outcome (W/L per contract)</span><span>Probability</span><span>×bankroll</span><span>End value</span>
              </div>
              {topOutcomes.map((o) => {
                const below = o.mult < alpha;
                return (
                  <div className={"kd-otr" + (below ? " kd-below" : "")} key={o.mask}>
                    <span className="kd-wl">
                      {items.map((it, i) => (
                        <b key={i} className={(o.mask & (1 << i)) ? "w" : "l"}>{(o.mask & (1 << i)) ? "W" : "L"}</b>
                      ))}
                    </span>
                    <span className="kd-cell">{fmtPct(o.prob, 2)}</span>
                    <span className="kd-cell">{o.mult.toFixed(3)}×</span>
                    <span className={"kd-cell " + (below ? "neg" : "")}>{fmtMoney(o.mult * bankroll)}</span>
                  </div>
                );
              })}
            </div>

            {outcomes.list.length > TOP_ROWS && (
              <div className="kd-note">
                Showing top {TOP_ROWS} of {outcomes.list.length.toLocaleString()} outcomes by probability — the remaining {(outcomes.list.length - TOP_ROWS).toLocaleString()} account for {fmtPct(1 - topProb, 2)} of total probability.
              </div>
            )}
            <div className="kd-note">
              Every row resolves all {outcomes.n} contracts simultaneously; W/L columns follow the contracts table order. {jointProb ? "Linked contracts use their joint distribution, so impossible combinations show 0%; unlinked contracts are independent." : "Contracts are treated as independent (add a link to model correlated outcomes)."} Probabilities are exact and sum to 100% over all 2^{outcomes.n} = {outcomes.list.length.toLocaleString()} outcomes. Red marks outcomes finishing below the drawdown threshold.
            </div>
          </>
        )}
      </section>

      <details className="kd-card kd-math">
        <summary>the math</summary>
        <div className="kd-mathbody">
          <p><b>Joint objective:</b> <code>G(f) = Σ_S P(S)·ln(1 + Σ_{"{i∈S}"} fᵢbᵢ − Σ_{"{i∉S}"} fᵢ)</code>, maximized over fᵢ ≥ 0, where S = each win/lose combo, bᵢ = (1−cᵢ)/cᵢ.</p>
          <p><b>No over-leverage:</b> the all-lose term <code>ln(1 − Σfᵢ)</code> → −∞ as Σfᵢ → 1, so the optimum keeps total exposure strictly below 100% on its own.</p>
          <p><b>Risk of ruin:</b> from the joint per-round log-return moments m (mean) and v (variance), <code>P(ever reach α) ≈ α^(2m/v)</code>. Simultaneous bets raise v, which raises ruin risk versus betting them one at a time.</p>
          <p><b>Risk aversion:</b> raising a re-solves the portfolio under CRRA utility <code>U(M) = M^(1−a)/(1−a)</code> (log utility at a = 1, full Kelly), tilting weight toward safer contracts.</p>
          <p><b>Linked contracts:</b> for correlated events the joint P(S) is no longer the product of marginals. Each link fixes a pair’s 2×2 joint from the two contracts’ P(win) plus one conditional; over a forest of links the full joint factorizes exactly as <code>∏ᵥ P(xᵥ) · ∏_edges P(xᵤ,xᵥ)/(P(xᵤ)P(xᵥ))</code>. This P(S) then drives the objective, ruin moments and outcome table.</p>
          <p className="kd-caveat">Contracts you don’t link are still assumed <i>independent</i>. Links must form a forest (no loops); a link that closes a loop is ignored. And Kelly is savagely sensitive to your P(win) estimates, which is the real reason to bet a fraction of it.</p>
        </div>
      </details>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }) {
  return (
    <div className="kd-slider">
      <div className="kd-slider-top"><span>{label}</span><b>{fmt(value)}</b></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
    </div>
  );
}
function Stat({ label, value, accent }) {
  return (
    <div className="kd-stat"><span className="kd-stat-l">{label}</span><span className={"kd-stat-v " + (accent || "")}>{value}</span></div>
  );
}

// Probability density of end-of-round portfolio size, from the exact 2ⁿ
// outcome distribution. X-axis = dollar value of the bankroll after one
// round; Y-axis = probability mass in each bin. The ruin threshold α,
// the starting bankroll and the expected end value are marked.
function EndValuePdf({ histo, bankroll, alpha }) {
  if (!histo.bins.length || histo.maxP <= 0) return null;
  const W = 720, H = 260, m = { l: 60, r: 18, t: 20, b: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const { lo, hi, maxP, mean } = histo;
  const x$ = (mult) => m.l + ((mult - lo) / (hi - lo)) * iw;   // multiplier → px
  const yP = (p) => m.t + (1 - p / maxP) * ih;                 // probability → px
  const base = m.t + ih;

  const xticks = niceTicks(lo * bankroll, hi * bankroll, 6)
    .filter((d) => d >= lo * bankroll - 1e-9 && d <= hi * bankroll + 1e-9);
  const yticks = niceTicks(0, maxP, 4);

  const ruinX = x$(alpha), startX = x$(1), meanX = x$(mean);
  const inRange = (mult) => mult >= lo - 1e-9 && mult <= hi + 1e-9;

  return (
    <div className="kd-pdf-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="kd-pdf" preserveAspectRatio="xMidYMid meet">
        {/* y gridlines + probability labels */}
        {yticks.map((p, i) => (
          <g key={"y" + i}>
            <line x1={m.l} y1={yP(p)} x2={m.l + iw} y2={yP(p)} stroke="#1b2230" strokeWidth="1" />
            <text x={m.l - 8} y={yP(p) + 3} textAnchor="end" className="kd-pdf-tick">{fmtPct(p, p < 0.1 ? 1 : 0)}</text>
          </g>
        ))}
        {/* shaded ruin region (below α) */}
        {ruinX > m.l && (
          <rect x={m.l} y={m.t} width={Math.min(ruinX, m.l + iw) - m.l} height={ih} fill="#f8514910" />
        )}
        {/* density bars */}
        {histo.bins.map((b, k) => {
          const xa = x$(b.x0), xb = x$(b.x1);
          const h = (b.p / maxP) * ih;
          if (h <= 0) return null;
          return <rect key={k} x={xa} y={base - h} width={Math.max(xb - xa - 0.6, 0.4)} height={h}
            fill={b.x1 <= alpha ? "#f85149" : "#58a6ff"} opacity="0.9" />;
        })}
        {/* axes */}
        <line x1={m.l} y1={base} x2={m.l + iw} y2={base} stroke="#30363d" strokeWidth="1" />
        <line x1={m.l} y1={m.t} x2={m.l} y2={base} stroke="#30363d" strokeWidth="1" />
        {/* x ticks: portfolio size in $ */}
        {xticks.map((d, i) => (
          <g key={"x" + i}>
            <line x1={x$(d / bankroll)} y1={base} x2={x$(d / bankroll)} y2={base + 4} stroke="#30363d" strokeWidth="1" />
            <text x={x$(d / bankroll)} y={base + 16} textAnchor="middle" className="kd-pdf-tick">{fmtMoneyShort(d)}</text>
          </g>
        ))}
        {/* markers: ruin α, starting bankroll, expected end value */}
        {inRange(alpha) && (
          <g>
            <line x1={ruinX} y1={m.t - 4} x2={ruinX} y2={base} stroke="#d29922" strokeWidth="1.3" strokeDasharray="4 3" />
            <text x={ruinX} y={m.t - 8} textAnchor="middle" className="kd-pdf-mark" fill="#d29922">ruin α · {fmtMoneyShort(alpha * bankroll)}</text>
          </g>
        )}
        {inRange(1) && (
          <line x1={startX} y1={m.t} x2={startX} y2={base} stroke="#6b7785" strokeWidth="1" strokeDasharray="2 3" />
        )}
        {inRange(mean) && (
          <g>
            <line x1={meanX} y1={m.t - 4} x2={meanX} y2={base} stroke="#3fb950" strokeWidth="1.3" strokeDasharray="4 3" />
            <text x={meanX} y={m.t - 8} textAnchor="middle" className="kd-pdf-mark" fill="#3fb950">E[end] · {fmtMoneyShort(mean * bankroll)}</text>
          </g>
        )}
        {/* axis titles */}
        <text x={m.l + iw / 2} y={H - 6} textAnchor="middle" className="kd-pdf-axis">portfolio size after one round ($)</text>
        <text x={14} y={m.t + ih / 2} textAnchor="middle" className="kd-pdf-axis" transform={`rotate(-90 14 ${m.t + ih / 2})`}>probability</text>
      </svg>
      <div className="kd-pdf-legend">
        <span><i style={{ background: "#f85149" }} />below α (ruin)</span>
        <span><i style={{ background: "#58a6ff" }} />above α</span>
        <span><i className="dash" style={{ background: "#6b7785" }} />start ({fmtMoneyShort(bankroll)})</span>
      </div>
    </div>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.kd-root{font-family:'IBM Plex Sans',system-ui,sans-serif;background:#0a0e14;color:#c9d1d9;padding:22px;border-radius:10px;max-width:1000px;margin:0 auto;}
.kd-root *{box-sizing:border-box;}
.kd-head{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #1f2730;padding-bottom:16px;margin-bottom:18px;}
.kd-head h1{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:22px;letter-spacing:4px;margin:0;color:#e6edf3;}
.kd-sub{font-size:11px;color:#6b7785;letter-spacing:.5px;}
.kd-bankroll{text-align:right;}
.kd-bankroll label{display:block;font-size:10px;letter-spacing:2px;color:#6b7785;margin-bottom:4px;}
.kd-bankin{display:flex;align-items:center;background:#0d1117;border:1px solid #2a3441;border-radius:6px;padding:4px 10px;font-family:'IBM Plex Mono',monospace;}
.kd-bankin span{color:#6b7785;}
.kd-bankin input{background:none;border:none;color:#3fb950;font-family:inherit;font-size:18px;font-weight:600;width:120px;text-align:right;outline:none;}
.kd-grid{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:14px;margin-bottom:14px;}
.kd-card{background:#0d1117;border:1px solid #1f2730;border-radius:8px;padding:14px;}
.kd-card-h{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:2px;color:#6b7785;margin-bottom:12px;}
.kd-th-row{display:flex;justify-content:space-between;align-items:center;}
.kd-controls{display:flex;flex-direction:column;gap:14px;margin-bottom:14px;}
.kd-slider-top{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;}
.kd-slider-top b{font-family:'IBM Plex Mono',monospace;color:#e6edf3;}
.kd-slider input[type=range]{width:100%;accent-color:#58a6ff;}
.kd-derived{font-size:12px;color:#8b949e;background:#0a0e14;border:1px solid #1f2730;border-radius:6px;padding:8px 10px;font-family:'IBM Plex Mono',monospace;}
.kd-derived b{color:#58a6ff;}
.kd-bignum{font-family:'IBM Plex Mono',monospace;font-size:40px;font-weight:600;line-height:1;}
.kd-bignum-cap{font-size:11px;color:#6b7785;margin:6px 0 10px;}
.kd-curve{width:100%;height:70px;}
.kd-curve-cap{display:flex;justify-content:space-between;font-size:9px;color:#4a5568;margin-top:2px;}
.kd-expbar{margin-bottom:10px;}
.kd-expbar-top{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;}
.kd-expbar-top b{font-family:'IBM Plex Mono',monospace;}
.kd-expbar-track{height:7px;background:#0a0e14;border:1px solid #21262d;border-radius:4px;overflow:hidden;}
.kd-expbar-fill{height:100%;transition:width .2s;}
.kd-summary .kd-stat{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #161b22;}
.kd-stat-l{font-size:12px;color:#8b949e;}
.kd-stat-v{font-family:'IBM Plex Mono',monospace;font-size:14px;color:#e6edf3;font-weight:500;}
.kd-stat-v.pos{color:#3fb950;} .kd-stat-v.neg{color:#f85149;}
.kd-cat{font-size:11px;color:#8b949e;background:#0a0e1480;border:1px solid #21262d;border-radius:6px;padding:8px;margin-top:10px;line-height:1.5;}
.kd-cat b{color:#d29922;font-family:'IBM Plex Mono',monospace;}
.kd-add{background:#1f6feb22;border:1px solid #1f6feb55;color:#58a6ff;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;font-family:inherit;}
.kd-table{font-family:'IBM Plex Mono',monospace;}
.kd-tr{display:grid;grid-template-columns:1.6fr .8fr .8fr .8fr 1fr .9fr 1fr .7fr .4fr;gap:6px;align-items:center;padding:5px 0;border-bottom:1px solid #161b22;}
.kd-thead{font-size:9.5px;color:#6b7785;letter-spacing:.5px;border-bottom:1px solid #2a3441;padding-bottom:8px;}
.kd-thead span{text-align:right;} .kd-thead span:first-child{text-align:left;}
.kd-name{background:#0a0e14;border:1px solid #21262d;color:#e6edf3;border-radius:4px;padding:5px 7px;font-family:'IBM Plex Sans',sans-serif;font-size:12px;outline:none;}
.kd-num{background:#0a0e14;border:1px solid #21262d;color:#c9d1d9;border-radius:4px;padding:5px;font-family:inherit;font-size:12px;text-align:right;outline:none;width:100%;}
.kd-num:focus,.kd-name:focus{border-color:#1f6feb;}
.kd-links{display:flex;flex-direction:column;gap:12px;}
.kd-link{border:1px solid #1f2730;border-radius:8px;padding:12px;background:#0a0e1480;}
.kd-link-sel{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.kd-select{background:#0a0e14;border:1px solid #21262d;color:#e6edf3;border-radius:4px;padding:5px 7px;font-family:'IBM Plex Sans',sans-serif;font-size:12px;outline:none;}
.kd-select:focus{border-color:#1f6feb;}
.kd-link-amp{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7785;letter-spacing:1px;}
.kd-link-sel .kd-del{margin-left:auto;}
.kd-link-marg{font-size:11px;color:#8b949e;margin:8px 0 10px;font-family:'IBM Plex Mono',monospace;}
.kd-link-marg b{color:#58a6ff;}
.kd-link-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}
.kd-link-field{display:flex;flex-direction:column;gap:4px;font-size:10.5px;color:#8b949e;}
.kd-link-field span{font-family:'IBM Plex Mono',monospace;letter-spacing:.5px;}
.kd-link-inf{font-style:normal;font-size:9.5px;color:#6b7785;text-align:right;}
.kd-link-foot{font-size:11px;color:#8b949e;margin-top:10px;font-family:'IBM Plex Mono',monospace;}
.kd-link-foot b{color:#e6edf3;}
.kd-link-warn{color:#d29922;}
.kd-cell{text-align:right;font-size:12.5px;color:#c9d1d9;}
.kd-cell.pos{color:#3fb950;} .kd-cell.neg{color:#f85149;} .kd-cell.mut{color:#6b7785;} .kd-cell.hl{color:#58a6ff;font-weight:600;}
.kd-del{background:none;border:none;color:#4a5568;font-size:16px;cursor:pointer;line-height:1;}
.kd-del:hover{color:#f85149;}
.kd-note{font-size:11px;color:#6b7785;line-height:1.5;margin-top:12px;}
.kd-pdf-wrap{margin:14px 0 4px;}
.kd-pdf{width:100%;height:auto;display:block;}
.kd-pdf-tick{font-family:'IBM Plex Mono',monospace;font-size:10px;fill:#6b7785;}
.kd-pdf-mark{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;}
.kd-pdf-axis{font-size:10px;letter-spacing:.5px;fill:#8b949e;}
.kd-pdf-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:#8b949e;margin-top:6px;padding-left:6px;}
.kd-pdf-legend span{display:inline-flex;align-items:center;gap:5px;}
.kd-pdf-legend i{width:11px;height:11px;border-radius:2px;display:inline-block;}
.kd-pdf-legend i.dash{width:14px;height:0;border-top:2px dashed #6b7785;background:none!important;}
.kd-otable{font-family:'IBM Plex Mono',monospace;margin-top:14px;max-height:420px;overflow-y:auto;}
.kd-otr{display:grid;grid-template-columns:2fr 1fr 1fr 1.1fr;gap:8px;align-items:center;padding:4px 0;border-bottom:1px solid #161b22;}
.kd-othead{font-size:9.5px;color:#6b7785;letter-spacing:.5px;border-bottom:1px solid #2a3441;padding-bottom:8px;position:sticky;top:0;background:#0d1117;z-index:1;}
.kd-othead span{text-align:right;} .kd-othead span:first-child{text-align:left;}
.kd-otr.kd-below{background:#f8514910;}
.kd-wl{display:flex;gap:3px;flex-wrap:wrap;}
.kd-wl b{display:inline-flex;width:16px;height:16px;align-items:center;justify-content:center;border-radius:3px;font-size:9px;font-weight:600;}
.kd-wl b.w{background:#3fb95022;color:#3fb950;}
.kd-wl b.l{background:#f8514922;color:#f85149;}
.kd-math{margin-top:14px;}
.kd-math summary{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:1px;color:#8b949e;cursor:pointer;}
.kd-mathbody{font-size:13px;line-height:1.6;color:#c9d1d9;margin-top:12px;}
.kd-mathbody code{background:#0a0e14;border:1px solid #21262d;border-radius:4px;padding:1px 6px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#58a6ff;}
.kd-caveat{color:#8b949e;font-size:12px;border-left:2px solid #d29922;padding-left:10px;margin-top:14px;}
@media(max-width:760px){.kd-grid{grid-template-columns:1fr;}
.kd-tr{grid-template-columns:1.5fr .7fr .7fr .7fr .9fr .6fr .4fr;}
.kd-tr span:nth-child(6),.kd-tr span:nth-child(7),.kd-thead span:nth-child(6),.kd-thead span:nth-child(7){display:none;}
.kd-link-grid{grid-template-columns:repeat(2,1fr);}}
`;
