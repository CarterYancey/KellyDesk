import React, { useState, useEffect, useMemo, useCallback } from "react";
import { buildClusters, jointProbArray, derivePairwise, pairwiseQuantities, nestedTable, PW_LABEL } from "./joint.js";

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
const MAX_N = 16; // exact 2ⁿ enumeration cap
const LINK_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#39c5cf", "#f778ba", "#db6d28", "#a5d6ff"];

// ---- portfolio solver: maximizes expected CRRA power utility ----
// a = relative risk aversion. a = 1 is log utility = full Kelly.
// a > 1 tilts toward safer (higher-p) contracts and punishes ruin tails harder.
// The objective is globally concave in f for every a > 0, so Newton converges.
function solvePortfolio(items, prob, a = 1, init = null) {
  const n = items.length;
  if (n === 0) return [];
  const N = 1 << n;
  const b = items.map((it) => it.b);
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

const solveJointKelly = (items, prob) => solvePortfolio(items, prob, 1); // full Kelly = log utility

// Continuation (homotopy) solve: walk risk aversion up from 1 (full Kelly) in small
// steps, warm-starting each step from the previous solution. This keeps every Newton
// solve interior and well-conditioned, avoiding the boundary stiffness that makes a
// single high-a solve started from the full-Kelly point fail to converge.
function solveAtAversion(items, prob, fKelly, a) {
  if (items.length === 0) return [];
  if (a <= 1.0001) return fKelly;
  const steps = Math.max(2, Math.ceil((a - 1) / 0.5));
  let warm = fKelly;
  for (let k = 1; k <= steps; k++) warm = solvePortfolio(items, prob, 1 + ((a - 1) * k) / steps, warm);
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
function jointMoments(items, prob, alloc) {
  const n = items.length;
  if (n === 0) return { m: 0, v: 0, allLoseP: 0, exposure: 0 };
  const N = 1 << n;
  let m = 0, m2 = 0;
  for (let s = 0; s < N; s++) {
    let M = 1;
    for (let i = 0; i < n; i++) {
      const w = (s & (1 << i)) !== 0;
      M += w ? alloc[i] * items[i].b : -alloc[i];
    }
    const pr = prob[s];
    const l = Math.log(Math.max(M, 1e-12));
    m += pr * l;
    m2 += pr * l * l;
  }
  const v = Math.max(m2 - m * m, 1e-12);
  const allLoseP = prob[0]; // joint P(all contracts lose)
  const exposure = alloc.reduce((a, x) => a + x, 0);
  return { m, v, allLoseP, exposure };
}

const ruinProb = (alpha, m, v) => (m <= 0 ? 1 : Math.pow(alpha, (2 * m) / v));

const STORE_KEY = "kelly-desk-state-v1";
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
  const [links, setLinks] = useState([]); // declared correlations between contracts
  const [loaded, setLoaded] = useState(false);
  const [sim, setSim] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await window.storage.get(STORE_KEY);
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
      return { rowId: r.id, name: r.name, p, c, b: (1 - c) / c, valid, edge: valid ? p - c : -1 };
    }).filter((x) => x.valid && x.edge > 0);
    return list.slice(0, MAX_N);
  }, [rows]);
  const overCap = rows.filter((r) => {
    const c = parseFloat(r.cost), p = parseFloat(r.prob);
    return c > 0 && c < 1 && p > c;
  }).length > MAX_N;

  const itemsKey = items.map((i) => i.rowId + ":" + i.p + ":" + i.c).join("|");
  const linksKey = links.map((l) => `${l.id}:${l.type}:${l.members.join(",")}:${JSON.stringify(l.assoc || null)}`).join("|");

  // Build correlated clusters and compose the true joint prob[s]. Both recompute
  // together on the same deps; the optimizer math downstream is unchanged.
  const { clusters, linkErrors } = useMemo(
    () => buildClusters(items, links),
    [itemsKey, linksKey] // eslint-disable-line
  );
  const prob = useMemo(
    () => jointProbArray(items, clusters),
    [itemsKey, linksKey] // eslint-disable-line
  );

  const fStar = useMemo(() => solveJointKelly(items, prob), [itemsKey, linksKey]); // eslint-disable-line

  // Resolve the applied allocation: re-optimize at relative risk aversion a (CRRA),
  // re-weighting toward safer contracts.
  const alloc = useMemo(() => {
    if (items.length === 0) return [];
    return solveAtAversion(items, prob, fStar, aversion);
  }, [itemsKey, linksKey, fStar, aversion]); // eslint-disable-line

  const mom = useMemo(() => jointMoments(items, prob, alloc), [items, prob, alloc]);
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
      let M = 1;
      for (let i = 0; i < n; i++) {
        const w = (s & (1 << i)) !== 0;
        M += w ? alloc[i] * items[i].b : -alloc[i];
      }
      const pr = prob[s];
      list[s] = { mask: s, prob: pr, mult: M };
      if (M < alpha) belowProb += pr;
      if (M < minMult) minMult = M;
      if (M > maxMult) maxMult = M;
    }
    return { list, belowProb, n, minMult, maxMult };
  }, [items, prob, alloc, alpha]);

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
  const delRow = (id) => {
    setRows((rs) => rs.filter((r) => r.id !== id));
    // drop the deleted contract from any link; remove links that fall below 2 members
    setLinks((ls) => ls
      .map((l) => ({ ...l, members: l.members.filter((m) => m !== id) }))
      .filter((l) => l.members.length >= 2));
  };

  // link CRUD + per-link color for the membership dots
  const addLink = useCallback((link) => setLinks((ls) => [...ls, link]), []);
  const removeLink = useCallback((id) => setLinks((ls) => ls.filter((l) => l.id !== id)), []);
  const linkErrorMap = useMemo(() => {
    const m = {};
    for (const e of linkErrors) m[e.id] = e.msg;
    return m;
  }, [linkErrors]);
  const colorByRow = useMemo(() => {
    const map = {};
    links.forEach((l, idx) => {
      const c = LINK_COLORS[idx % LINK_COLORS.length];
      l.members.forEach((m) => (map[m] = c));
    });
    return map;
  }, [links]);

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
            {links.length > 0
              ? " This probability reflects the correlations you've declared between contracts (not an independence assumption)."
              : " Exposure is solver-capped below 100%, so a single round can never fully wipe you out."}
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
              <div className="kd-namewrap">
                {colorByRow[r.id] && <span className="kd-link-dot" style={{ background: colorByRow[r.id] }} title="linked contract" />}
                <input className="kd-name" value={r.name} onChange={(e) => updateRow(r.id, "name", e.target.value)} />
              </div>
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

      <LinkPanel rows={rows} links={links} linkErrorMap={linkErrorMap} colorByRow={colorByRow} onAdd={addLink} onRemove={removeLink} />

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
              Every row resolves all {outcomes.n} contracts simultaneously and independently; W/L columns follow the contracts table order. Probabilities are exact and sum to 100% over all 2^{outcomes.n} = {outcomes.list.length.toLocaleString()} outcomes. Red marks outcomes finishing below the drawdown threshold.
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
          <p><b>Correlated contracts:</b> by default outcomes are independent (<code>prob[s] = ∏ᵢ pᵢ</code>). Declaring a <i>link</i> (nested implication, mutually exclusive, or a pairwise association) replaces the independent product over those contracts with the correct joint — staircase for nesting, single-winner for exclusivity, or the Fréchet-feasible two-cell solve for a pairwise input — and the solver, risk, and outcome enumeration all consume that joint unchanged.</p>
          <p className="kd-caveat">Kelly is savagely sensitive to your P(win) estimates, which is the real reason to bet a fraction of it. Linked correlations are equally sensitive to the association value you enter — an infeasible value is rejected against the Fréchet–Hoeffding bounds.</p>
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

// ---------- Linked Contracts panel (declare correlations) ----------
const QUANT_ORDER = ["P(A)", "P(B)", "P(A∧B)", "P(A|B)", "P(A|~B)", "P(B|A)", "P(B|~A)", "corr", "odds"];
const fmtQ = (k, x) => {
  if (!isFinite(x)) return k === "odds" ? "∞" : "—";
  if (k === "odds") return x.toFixed(2);
  if (k === "corr") return (x >= 0 ? "+" : "") + x.toFixed(2);
  return x.toFixed(3);
};
const fmtRange = (v) => (isFinite(v) ? v.toFixed(3) : v > 0 ? "∞" : "−∞");

function LinkPanel({ rows, links, linkErrorMap, colorByRow, onAdd, onRemove }) {
  const [draft, setDraft] = useState(null);

  const isValidContract = (r) => {
    const c = parseFloat(r.cost), p = parseFloat(r.prob);
    return c > 0 && c < 1 && p >= 0 && p <= 1;
  };
  const linkedSet = new Set(links.flatMap((l) => l.members));
  const rowById = (id) => rows.find((r) => r.id === id);
  const nameOf = (id) => rowById(id)?.name || id;
  const pOf = (id) => parseFloat(rowById(id)?.prob);
  const linkable = rows.filter((r) => isValidContract(r) && !linkedSet.has(r.id));

  const startDraft = () => setDraft({ members: [], type: "pairwise", assoc: { input: "P(A|B)", value: "" } });

  const toggleMember = (id) =>
    setDraft((d) => {
      const has = d.members.includes(id);
      let members = has ? d.members.filter((m) => m !== id) : [...d.members, id];
      if (d.type === "nested") members = [...members].sort((x, y) => pOf(y) - pOf(x));
      return { ...d, members };
    });
  const setType = (type) =>
    setDraft((d) => {
      let members = d.members;
      if (type === "nested") members = [...members].sort((x, y) => pOf(y) - pOf(x));
      return { ...d, type, members };
    });

  const canSave =
    !!draft &&
    (draft.type === "pairwise"
      ? draft.members.length === 2 && draft.assoc.value !== "" && isFinite(Number(draft.assoc.value))
      : draft.members.length >= 2);

  const save = () => {
    const id = Math.random().toString(36).slice(2);
    const link =
      draft.type === "pairwise"
        ? { id, type: "pairwise", members: draft.members.slice(0, 2), assoc: { input: draft.assoc.input, value: Number(draft.assoc.value) } }
        : { id, type: draft.type, members: draft.members.slice() };
    onAdd(link);
    setDraft(null);
  };

  return (
    <section className="kd-card">
      <div className="kd-card-h kd-th-row">
        <span>LINKED CONTRACTS &nbsp;·&nbsp; CORRELATED OUTCOMES</span>
        {!draft && <button className="kd-add" onClick={startDraft}>+ add link</button>}
      </div>

      {links.length === 0 && !draft && (
        <div className="kd-note">
          No links — every contract is treated as independent. Link contracts that concern related events
          (nested price thresholds, mutually exclusive outcomes, or a correlated pair) so the joint distribution
          is built correctly instead of as a product of marginals.
        </div>
      )}

      {links.map((l) => (
        <LinkRow key={l.id} link={l} error={linkErrorMap[l.id]} color={colorByRow[l.members[0]]} nameOf={nameOf} pOf={pOf} onRemove={() => onRemove(l.id)} />
      ))}

      {draft && (
        <DraftForm
          draft={draft}
          linkable={linkable}
          nameOf={nameOf}
          pOf={pOf}
          onToggle={toggleMember}
          onType={setType}
          onAssoc={(assoc) => setDraft((d) => ({ ...d, assoc }))}
          canSave={canSave}
          onSave={save}
          onCancel={() => setDraft(null)}
        />
      )}
    </section>
  );
}

function LinkRow({ link, error, color, nameOf, pOf, onRemove }) {
  const valid = !error;
  let body = null;
  if (link.type === "pairwise") {
    const A = link.members[0], B = link.members[1];
    const a = pOf(A), b = pOf(B);
    const r = isFinite(a) && isFinite(b) ? derivePairwise(a, b, link.assoc) : { ok: false };
    const q = r.ok ? pairwiseQuantities(a, b, r.j) : null;
    body = (
      <>
        <div className="kd-link-sub">A = <b>{nameOf(A)}</b> · B = <b>{nameOf(B)}</b> · you set <b>{PW_LABEL[link.assoc.input]} = {Number(link.assoc.value)}</b></div>
        {q && (
          <div className="kd-qgrid">
            {QUANT_ORDER.map((k) => (
              <span key={k} className={"kd-quant" + (k === link.assoc.input ? " set" : "")}><i>{PW_LABEL[k] || k}</i>{fmtQ(k, q[k])}</span>
            ))}
          </div>
        )}
      </>
    );
  } else if (link.type === "nested") {
    body = (
      <div className="kd-link-sub">implication chain: {link.members.map((m, i) => (<span key={m}>{i > 0 ? " ⟹ " : ""}<b>{nameOf(m)}</b> ({fmtPct(pOf(m), 0)})</span>))}</div>
    );
  } else {
    const sum = link.members.reduce((s, m) => s + (pOf(m) || 0), 0);
    body = (
      <div className="kd-link-sub">mutually exclusive: {link.members.map((m, i) => (<span key={m}>{i > 0 ? ", " : ""}<b>{nameOf(m)}</b> ({fmtPct(pOf(m), 0)})</span>))} · Σp = {fmtPct(sum, 0)}</div>
    );
  }
  return (
    <div className="kd-link">
      <div className="kd-link-head">
        <span className="kd-link-dot" style={{ background: color }} />
        <span className="kd-link-type">{link.type}</span>
        <span className={"kd-badge " + (valid ? "ok" : "bad")}>{valid ? "✓ valid" : "✕ invalid · using independence"}</span>
        <button className="kd-del" onClick={onRemove}>×</button>
      </div>
      {body}
      {error && <div className="kd-link-err">{error}</div>}
    </div>
  );
}

function DraftForm({ draft, linkable, nameOf, pOf, onToggle, onType, onAssoc, canSave, onSave, onCancel }) {
  const { type, members } = draft;
  let preview = null;

  if (type === "pairwise" && members.length === 2) {
    const a = pOf(members[0]), b = pOf(members[1]);
    const r = derivePairwise(a, b, draft.assoc);
    const [rlo, rhi] = r.range || [0, 0];
    const q = r.ok ? pairwiseQuantities(a, b, r.j) : null;
    preview = (
      <div className="kd-draft-prev">
        <div className="kd-link-sub">A = <b>{nameOf(members[0])}</b> (P={fmtPct(a, 0)}) · B = <b>{nameOf(members[1])}</b> (P={fmtPct(b, 0)})</div>
        <div className="kd-row">
          <select className="kd-sel" value={draft.assoc.input} onChange={(e) => onAssoc({ ...draft.assoc, input: e.target.value })}>
            {Object.keys(PW_LABEL).map((k) => <option key={k} value={k}>{PW_LABEL[k]}</option>)}
          </select>
          <input className="kd-num" type="number" step="0.01" value={draft.assoc.value} placeholder="value"
            onChange={(e) => onAssoc({ ...draft.assoc, value: e.target.value })} />
        </div>
        <div className="kd-hint">valid range for {PW_LABEL[draft.assoc.input]}: [{fmtRange(rlo)}, {fmtRange(rhi)}]</div>
        {q ? (
          <div className="kd-qgrid">
            {QUANT_ORDER.map((k) => (
              <span key={k} className={"kd-quant" + (k === draft.assoc.input ? " set" : "")}><i>{PW_LABEL[k] || k}</i>{fmtQ(k, q[k])}</span>
            ))}
          </div>
        ) : <div className="kd-link-err">{r.error}</div>}
      </div>
    );
  } else if (type === "nested" && members.length >= 2) {
    const p = members.map(pOf);
    const k = members.length, table = nestedTable(p);
    const stairs = [{ label: "none win", p: table[0] }];
    for (let m = 1; m <= k; m++) stairs.push({ label: m === k ? "all win" : `top ${m} win`, p: table[(1 << m) - 1] });
    const monotone = p.every((x, i) => i === 0 || p[i - 1] >= p[i] - 1e-9);
    preview = (
      <div className="kd-draft-prev">
        <div className="kd-link-sub">chain (auto-ordered high→low P): {members.map((m, i) => (<span key={m}>{i > 0 ? " ⟹ " : ""}<b>{nameOf(m)}</b> ({fmtPct(pOf(m), 0)})</span>))}</div>
        <div className="kd-qgrid">
          {stairs.map((s, i) => <span key={i} className="kd-quant"><i>{s.label}</i>{fmtPct(s.p, 1)}</span>)}
        </div>
        {!monotone && <div className="kd-link-err">marginals are not monotonically ordered — this will be rejected.</div>}
      </div>
    );
  } else if (type === "exclusive" && members.length >= 2) {
    const sum = members.reduce((s, m) => s + (pOf(m) || 0), 0);
    preview = (
      <div className="kd-draft-prev">
        <div className="kd-link-sub">at most one wins: {members.map((m, i) => (<span key={m}>{i > 0 ? ", " : ""}<b>{nameOf(m)}</b> ({fmtPct(pOf(m), 0)})</span>))}</div>
        <div className={"kd-hint" + (sum > 1 ? " bad" : "")}>Σp = {fmtPct(sum, 1)}{sum > 1 ? " — exceeds 100%, infeasible" : ""} · P(none) = {fmtPct(Math.max(0, 1 - sum), 1)}</div>
      </div>
    );
  }

  return (
    <div className="kd-draft">
      <div className="kd-draft-h">New link</div>
      <div className="kd-chooser">
        {linkable.length === 0 ? (
          <span className="kd-note">All valid contracts are already linked. Remove a link or add contracts.</span>
        ) : linkable.map((r) => (
          <label key={r.id} className={"kd-chip" + (members.includes(r.id) ? " on" : "")}>
            <input type="checkbox" checked={members.includes(r.id)} onChange={() => onToggle(r.id)} />
            {r.name} <span className="kd-chip-p">{fmtPct(parseFloat(r.prob), 0)}</span>
          </label>
        ))}
      </div>
      <div className="kd-typesel">
        {[["pairwise", "Pairwise correlation"], ["nested", "Nested (implication)"], ["exclusive", "Mutually exclusive"]].map(([t, lbl]) => (
          <button key={t} className={type === t ? "on" : ""} onClick={() => onType(t)} disabled={t === "pairwise" && members.length > 2}>{lbl}</button>
        ))}
      </div>
      {type === "pairwise" && members.length !== 2 && <div className="kd-hint">select exactly 2 contracts for a pairwise link</div>}
      {type !== "pairwise" && members.length < 2 && <div className="kd-hint">select 2 or more contracts</div>}
      {preview}
      <div className="kd-draft-actions">
        <button className="kd-add" disabled={!canSave} onClick={onSave}>save link</button>
        <button className="kd-cancel" onClick={onCancel}>cancel</button>
      </div>
    </div>
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
.kd-namewrap{display:flex;align-items:center;gap:6px;min-width:0;}
.kd-namewrap .kd-name{flex:1;min-width:0;}
.kd-link-dot{width:8px;height:8px;border-radius:50%;flex:none;display:inline-block;}
.kd-link{background:#0a0e14;border:1px solid #21262d;border-radius:6px;padding:10px 12px;margin-bottom:10px;}
.kd-link-head{display:flex;align-items:center;gap:8px;margin-bottom:6px;}
.kd-link-type{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#8b949e;}
.kd-badge{font-size:10px;font-family:'IBM Plex Mono',monospace;padding:2px 7px;border-radius:10px;}
.kd-badge.ok{background:#3fb95022;color:#3fb950;}
.kd-badge.bad{background:#f8514922;color:#f85149;}
.kd-link-head .kd-del{margin-left:auto;}
.kd-link-sub{font-size:12px;color:#c9d1d9;line-height:1.6;}
.kd-link-sub b{color:#e6edf3;}
.kd-link-err{font-size:11px;color:#f85149;background:#f8514910;border:1px solid #f8514933;border-radius:5px;padding:7px 9px;margin-top:7px;line-height:1.5;}
.kd-qgrid{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.kd-quant{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;background:#0d1117;border:1px solid #21262d;border-radius:5px;padding:3px 7px;display:inline-flex;gap:5px;align-items:baseline;}
.kd-quant i{font-style:normal;color:#6b7785;}
.kd-quant.set{border-color:#1f6feb;background:#1f6feb18;color:#58a6ff;}
.kd-quant.set i{color:#58a6ff;}
.kd-draft{background:#0a0e14;border:1px dashed #2a3441;border-radius:6px;padding:12px;margin-top:6px;}
.kd-draft-h{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:1px;color:#8b949e;margin-bottom:10px;}
.kd-chooser{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}
.kd-chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:#c9d1d9;background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:5px 9px;cursor:pointer;}
.kd-chip.on{border-color:#1f6feb;background:#1f6feb18;color:#58a6ff;}
.kd-chip input{accent-color:#58a6ff;}
.kd-chip-p{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6b7785;}
.kd-typesel{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap;}
.kd-typesel button{background:#0d1117;border:1px solid #2a3441;color:#8b949e;border-radius:6px;padding:6px 10px;font-size:11px;cursor:pointer;font-family:inherit;}
.kd-typesel button.on{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff;}
.kd-typesel button:disabled{opacity:.4;cursor:not-allowed;}
.kd-hint{font-size:11px;color:#6b7785;font-family:'IBM Plex Mono',monospace;margin-top:6px;}
.kd-hint.bad{color:#f85149;}
.kd-draft-prev{margin-top:8px;border-top:1px solid #161b22;padding-top:10px;}
.kd-draft-prev .kd-row{display:flex;gap:8px;margin-top:8px;align-items:center;}
.kd-sel{background:#0d1117;border:1px solid #21262d;color:#c9d1d9;border-radius:5px;padding:5px 7px;font-family:inherit;font-size:12px;outline:none;}
.kd-sel:focus{border-color:#1f6feb;}
.kd-draft-prev .kd-num{width:110px;}
.kd-draft-actions{display:flex;gap:8px;margin-top:12px;}
.kd-add:disabled{opacity:.4;cursor:not-allowed;}
.kd-cancel{background:none;border:1px solid #2a3441;color:#8b949e;border-radius:6px;padding:4px 12px;font-size:11px;cursor:pointer;font-family:inherit;}
@media(max-width:760px){.kd-grid{grid-template-columns:1fr;}
.kd-tr{grid-template-columns:1.5fr .7fr .7fr .7fr .9fr .6fr .4fr;}
.kd-tr span:nth-child(6),.kd-tr span:nth-child(7),.kd-thead span:nth-child(6),.kd-thead span:nth-child(7){display:none;}}
`;
