import React, { useState, useEffect, useMemo } from "react";

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
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const MAX_N = 16; // exact 2ⁿ enumeration cap

// ---- joint full-Kelly solver (full-vector Newton ascent) ----
function solveJointKelly(items) {
  const n = items.length;
  if (n === 0) return [];
  const N = 1 << n;
  const b = items.map((it) => it.b);
  const prob = new Float64Array(N);
  for (let s = 0; s < N; s++) {
    let pr = 1;
    for (let i = 0; i < n; i++) pr *= s & (1 << i) ? items[i].p : 1 - items[i].p;
    prob[s] = pr;
  }
  const sign = (s, i) => (s & (1 << i) ? b[i] : -1); // payoff multiple of f_i in outcome s
  const multipliers = (f) => {
    const M = new Float64Array(N);
    for (let s = 0; s < N; s++) {
      let m = 1;
      for (let i = 0; i < n; i++) m += sign(s, i) * f[i];
      M[s] = m;
    }
    return M;
  };
  const G = (f) => {
    const M = multipliers(f);
    let g = 0;
    for (let s = 0; s < N; s++) {
      if (M[s] <= 0) return -Infinity;
      g += prob[s] * Math.log(M[s]);
    }
    return g;
  };

  // Full-vector Newton ascent. The optimum is interior w.r.t. f_i >= 0 for every
  // positive-edge contract, so we only enforce the M(S) > 0 barrier via line search.
  let f = new Array(n).fill(0.5 / n); // feasible, symmetric start (sum = 0.5)
  for (let iter = 0; iter < 60; iter++) {
    const M = multipliers(f);
    const grad = new Float64Array(n);
    const H = Array.from({ length: n }, () => new Float64Array(n));
    for (let s = 0; s < N; s++) {
      const invM = 1 / M[s];
      const w = prob[s] * invM * invM;
      const wg = prob[s] * invM;
      for (let i = 0; i < n; i++) {
        const ai = sign(s, i);
        grad[i] += wg * ai;
        for (let j = i; j < n; j++) H[i][j] -= w * ai * sign(s, j); // Hessian (neg. def.)
      }
    }
    for (let i = 0; i < n; i++) {
      H[i][i] -= 1e-9; // tiny damping against ill-conditioning near the boundary
      for (let j = 0; j < i; j++) H[i][j] = H[j][i];
    }
    // Newton step solves  H * delta = -grad  (ascent direction since H is neg. def.)
    const delta = solveLinear(H, Array.from(grad, (x) => -x));
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
function jointMoments(items, alloc) {
  const n = items.length;
  if (n === 0) return { m: 0, v: 0, allLoseP: 0, exposure: 0 };
  const N = 1 << n;
  let m = 0, m2 = 0;
  for (let s = 0; s < N; s++) {
    let pr = 1, M = 1;
    for (let i = 0; i < n; i++) {
      const w = (s & (1 << i)) !== 0;
      pr *= w ? items[i].p : 1 - items[i].p;
      M += w ? alloc[i] * items[i].b : -alloc[i];
    }
    const l = Math.log(Math.max(M, 1e-12));
    m += pr * l;
    m2 += pr * l * l;
  }
  const v = Math.max(m2 - m * m, 1e-12);
  const allLoseP = items.reduce((a, it) => a * (1 - it.p), 1);
  const exposure = alloc.reduce((a, x) => a + x, 0);
  return { m, v, allLoseP, exposure };
}

const ruinProb = (alpha, m, v) => (m <= 0 ? 1 : Math.pow(alpha, (2 * m) / v));

// ---- solve fraction-of-full-Kelly λ that hits a target ruin risk ----
function lambdaForRisk(items, fStar, alpha, R) {
  const risk = (lam) => {
    const { m, v } = jointMoments(items, fStar.map((x) => x * lam));
    return ruinProb(alpha, m, v);
  };
  if (risk(1) <= R) return 1;
  let lo = 1e-4, hi = 1;
  for (let it = 0; it < 50; it++) {
    const mid = (lo + hi) / 2;
    if (risk(mid) > R) hi = mid;
    else lo = mid;
  }
  return (lo + hi) / 2;
}

const STORE_KEY = "kelly-desk-state-v1";
const blank = (n) => ({ id: Math.random().toString(36).slice(2), name: n, cost: "", prob: "" });

export default function KellyDesk() {
  const [bankroll, setBankroll] = useState(10000);
  const [mode, setMode] = useState("risk");
  const [lambda, setLambda] = useState(0.5);
  const [alpha, setAlpha] = useState(0.5);
  const [maxRisk, setMaxRisk] = useState(0.1);
  const [maxExposure, setMaxExposure] = useState(1);
  const [integer, setInteger] = useState(true);
  const [rows, setRows] = useState([
    { id: "a", name: "Contract A", cost: "0.25", prob: "0.50" },
    { id: "b", name: "Contract B", cost: "0.25", prob: "0.50" },
    { id: "c", name: "Contract C", cost: "0.25", prob: "0.50" },
  ]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await window.storage.get(STORE_KEY);
        if (r && r.value) {
          const s = JSON.parse(r.value);
          if (s.bankroll != null) setBankroll(s.bankroll);
          if (s.mode) setMode(s.mode);
          if (s.lambda != null) setLambda(s.lambda);
          if (s.alpha != null) setAlpha(s.alpha);
          if (s.maxRisk != null) setMaxRisk(s.maxRisk);
          if (s.maxExposure != null) setMaxExposure(s.maxExposure);
          if (s.integer != null) setInteger(s.integer);
          if (Array.isArray(s.rows)) setRows(s.rows);
        }
      } catch (e) {/* defaults */} finally { setLoaded(true); }
    })();
  }, []);

  useEffect(() => {
    if (!loaded) return;
    (async () => {
      try {
        await window.storage.set(STORE_KEY, JSON.stringify(
          { bankroll, mode, lambda, alpha, maxRisk, maxExposure, integer, rows }));
      } catch (e) {/* in-memory only */}
    })();
  }, [bankroll, mode, lambda, alpha, maxRisk, maxExposure, integer, rows, loaded]);

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
  const fStar = useMemo(() => solveJointKelly(items), [itemsKey]); // eslint-disable-line

  const effLambda = useMemo(
    () => (mode === "risk" ? lambdaForRisk(items, fStar, alpha, maxRisk) : lambda),
    [mode, items, fStar, alpha, maxRisk, lambda]
  );

  // applied allocation = λ·f*  then clamp total exposure to user cap
  const alloc = useMemo(() => {
    let a = fStar.map((x) => x * effLambda);
    const tot = a.reduce((s, x) => s + x, 0);
    if (tot > maxExposure && tot > 0) a = a.map((x) => (x * maxExposure) / tot);
    return a;
  }, [fStar, effLambda, maxExposure]);

  const mom = useMemo(() => jointMoments(items, alloc), [items, alloc]);
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
      const n = e && c > 0 ? (integer ? Math.floor(dollars / c) : dollars / c) : 0;
      const stake = n * c;
      return { ...r, c, p, valid, edge: valid ? p - c : 0, posEdge: !!e, fk, ap, n, stake };
    });
  }, [rows, items, fStar, alloc, bankroll, integer]);

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
      let pr = 1, M = 1;
      for (let i = 0; i < n; i++) {
        const w = (s & (1 << i)) !== 0;
        pr *= w ? items[i].p : 1 - items[i].p;
        M += w ? alloc[i] * items[i].b : -alloc[i];

      }
      list[s] = { mask: s, prob: pr, mult: M };
      if (M < alpha) belowProb += pr;
      if (M < minMult) minMult = M;
      if (M > maxMult) maxMult = M;
    }
    return { list, belowProb, n, minMult, maxMult };
  }, [items, alloc, alpha]);

  // top outcomes by probability for the table (full 2ⁿ list is too large to render)
  const topOutcomes = useMemo(
    () => [...outcomes.list].sort((a, b) => b.prob - a.prob).slice(0, TOP_ROWS),
    [outcomes]
  );
  const topProb = topOutcomes.reduce((s, o) => s + o.prob, 0);

  // probability-weighted end-value distribution, binned by multiplier
  const histo = useMemo(() => {
    const BINS = 28;
    if (outcomes.n === 0) return { bins: [], maxP: 0, lo: 0, hi: 1, alphaPos: 0 };
    let lo = outcomes.minMult, hi = outcomes.maxMult;
    if (hi - lo < 1e-9) { lo -= 0.01; hi += 0.01; }
    const width = (hi - lo) / BINS;
    const bins = Array.from({ length: BINS }, (_, k) => ({ x0: lo + k * width, x1: lo + (k + 1) * width, p: 0 }));
    for (const o of outcomes.list) {
      let k = Math.floor((o.mult - lo) / width);
      if (k < 0) k = 0; else if (k >= BINS) k = BINS - 1;
      bins[k].p += o.prob;
    }
    const maxP = bins.reduce((m, b) => Math.max(m, b.p), 0);
    return { bins, maxP, lo, hi, alphaPos: clamp((alpha - lo) / (hi - lo), 0, 1) };
  }, [outcomes, alpha]);

  const updateRow = (id, field, val) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, [field]: val } : r)));
  const addRow = () => setRows((rs) => [...rs, blank("Contract " + String.fromCharCode(65 + rs.length))]);
  const delRow = (id) => setRows((rs) => rs.filter((r) => r.id !== id));

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
          <div className="kd-toggle">
            <button className={mode === "risk" ? "on" : ""} onClick={() => setMode("risk")}>Target risk of ruin</button>
            <button className={mode === "manual" ? "on" : ""} onClick={() => setMode("manual")}>Set Kelly fraction</button>
          </div>
          {mode === "risk" ? (
            <div className="kd-controls">
              <Slider label="Ruin = drawdown to" value={alpha} min={0.05} max={0.95} step={0.05}
                onChange={setAlpha} fmt={(v) => fmtPct(v, 0) + " of bankroll"} />
              <Slider label="Max acceptable risk" value={maxRisk} min={0.01} max={0.5} step={0.01}
                onChange={setMaxRisk} fmt={(v) => fmtPct(v, 0) + " chance"} />
              <div className="kd-derived">→ solver uses <b>{(effLambda * 100).toFixed(0)}%</b> of full joint Kelly</div>
            </div>
          ) : (
            <div className="kd-controls">
              <Slider label="Fraction of full joint Kelly (λ)" value={lambda} min={0.05} max={1} step={0.05}
                onChange={setLambda} fmt={(v) => v.toFixed(2) + "×"} />
              <div className="kd-derived">→ {fmtPct(risk, 1)} chance of ever reaching {fmtPct(alpha, 0)} of bankroll</div>
            </div>
          )}
          <Slider label="Hard exposure cap" value={maxExposure} min={0.25} max={1} step={0.05}
            onChange={setMaxExposure} fmt={(v) => (v >= 1 ? "off (100%)" : fmtPct(v, 0))} />
          <label className="kd-check">
            <input type="checkbox" checked={integer} onChange={(e) => setInteger(e.target.checked)} />
            Whole contracts only
          </label>
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
            <svg viewBox="0 0 280 90" className="kd-histo" preserveAspectRatio="none">
              {histo.bins.map((b, k) => {
                const bw = 280 / histo.bins.length;
                const h = histo.maxP > 0 ? (b.p / histo.maxP) * 88 : 0;
                return <rect key={k} x={k * bw + 0.5} y={90 - h} width={Math.max(bw - 1, 0.5)} height={h}
                  fill={b.x1 <= alpha ? "#f85149" : "#58a6ff"} opacity="0.85" />;
              })}
              <line x1={histo.alphaPos * 280} y1="0" x2={histo.alphaPos * 280} y2="90" stroke="#d29922" strokeWidth="1" strokeDasharray="3 2" />
            </svg>
            <div className="kd-curve-cap"><span>← lower end value</span><span>P-weighted end value · ↑ drawdown α</span></div>

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
          <p><b>Fractional Kelly:</b> the whole optimal vector is scaled by λ ∈ (0,1]; the risk-target mode binary-searches λ to hit your tolerance.</p>
          <p className="kd-caveat">Assumes contracts are <i>independent</i>. Correlated outcomes (e.g. two contracts on related events) need a joint distribution, not the product of marginals — coming next if useful. And Kelly is savagely sensitive to your P(win) estimates, which is the real reason to bet a fraction of it.</p>
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
.kd-toggle{display:flex;gap:6px;margin-bottom:14px;}
.kd-toggle button{flex:1;background:#0a0e14;border:1px solid #2a3441;color:#8b949e;border-radius:6px;padding:7px 4px;font-size:11px;cursor:pointer;font-family:inherit;transition:.15s;}
.kd-toggle button.on{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff;}
.kd-controls{display:flex;flex-direction:column;gap:14px;margin-bottom:14px;}
.kd-slider-top{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;}
.kd-slider-top b{font-family:'IBM Plex Mono',monospace;color:#e6edf3;}
.kd-slider input[type=range]{width:100%;accent-color:#58a6ff;}
.kd-derived{font-size:12px;color:#8b949e;background:#0a0e14;border:1px solid #1f2730;border-radius:6px;padding:8px 10px;font-family:'IBM Plex Mono',monospace;}
.kd-derived b{color:#58a6ff;}
.kd-check{display:flex;align-items:center;gap:8px;font-size:12px;color:#8b949e;margin-top:4px;cursor:pointer;}
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
.kd-histo{width:100%;height:80px;margin:10px 0 0;}
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
.kd-tr span:nth-child(6),.kd-tr span:nth-child(7),.kd-thead span:nth-child(6),.kd-thead span:nth-child(7){display:none;}}
`;
