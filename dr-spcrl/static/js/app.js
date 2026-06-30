/* DR-SPCRL interactive site logic. */

const COLORS = {
  Nominal:   '#888888',
  Fixed:     '#7e57c2',
  Space:     '#26a69a',
  Naive:     '#42a5f5',
  Accel:     '#ffa726',
  DomRand:   '#8d6e63',
  SelfPaced: '#d23a3a',
};

let DATA = null;
const state = { alg: 'PPO', env: 'HalfCheetah', perturb: 'Action' };

/* ---------- KL ball slider with hover-video dots ---------- */

function preloadVideos(urls) {
  // Use <link rel=prefetch> instead of preload as=video — Safari/older Chrome
  // reject as=video as unsupported. Prefetch warms the HTTP cache without
  // the spec-strict "as" validation.
  urls.forEach((u) => {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = u;
    document.head.appendChild(link);
  });
}

async function initKLExplainer() {
  const meta = await fetch('./static/js/hover_envs.json').then((r) => r.json());
  const dots = meta.dots;

  const urls = [meta.nominal_video, ...dots.map((d) => d.video)].filter(Boolean);
  preloadVideos(urls);

  const svg = document.getElementById('kl-canvas');
  const worldsG = document.getElementById('kl-worlds');
  const slider = document.getElementById('kl-slider');
  const ball = document.getElementById('kl-ball');
  const epsVal = document.getElementById('kl-eps-val');
  const coverageEl = document.getElementById('kl-coverage');
  const video = document.getElementById('kl-hover-video');

  // Place a dot per real perturbed env. Distance from P₀ = ball radius at
  // that ε. Ball radius scales LINEARLY: 40 (ε=0) → 180 (ε=0.5).
  // Same formula must be used for the ball (see update()) so they line up.
  const cx = 200, cy = 200;
  const radiusForEps = (eps) => 40 + (eps / 0.5) * 140;
  dots.forEach((d) => {
    const r = radiusForEps(d.epsilon);
    const theta = (2 * Math.PI * d.id) / dots.length + 0.3;
    d._x = cx + r * Math.cos(theta);
    d._y = cy + r * Math.sin(theta);
    d._dist = r;
  });

  const xmlns = 'http://www.w3.org/2000/svg';
  const stats = document.getElementById('kl-hover-stats');
  const fmt = (n) => (n == null ? '—' : Math.round(n).toLocaleString());

  const playClip = (src, label, ret, steps) => {
    if (video.dataset.src !== src) {
      video.dataset.src = src;
      video.src = src;
    } else {
      // Same clip — restart from the top for visible feedback.
      try { video.currentTime = 0; } catch (e) {}
    }
    video.classList.add('is-active');
    video.play().catch(() => {});
    stats.classList.add('is-active');
    stats.innerHTML =
      `<span class="kl-stat-label">${label}</span>` +
      `<span class="kl-stat-num">return ${fmt(ret)}</span>` +
      `<span class="kl-stat-num">${fmt(steps)} steps</span>`;
  };

  // RdYlGn diverging color map (low return = red, high = green).
  // Stops sampled from matplotlib's RdYlGn at 0, .25, .5, .75, 1.
  const RDYLGN_STOPS = [
    [0.00, [165,  0, 38]],   // dark red
    [0.25, [244,109, 67]],   // orange-red
    [0.50, [255,255,191]],   // pale yellow
    [0.75, [166,217,106]],   // light green
    [1.00, [ 26,152, 80]],   // dark green
  ];
  const interpRGB = (t) => {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < RDYLGN_STOPS.length; i++) {
      const [t1, c1] = RDYLGN_STOPS[i - 1];
      const [t2, c2] = RDYLGN_STOPS[i];
      if (t <= t2) {
        const u = (t - t1) / (t2 - t1);
        const r = Math.round(c1[0] + (c2[0] - c1[0]) * u);
        const g = Math.round(c1[1] + (c2[1] - c1[1]) * u);
        const b = Math.round(c1[2] + (c2[2] - c1[2]) * u);
        return `rgb(${r},${g},${b})`;
      }
    }
    return 'rgb(26,152,80)';
  };
  // Normalize each dot's return into [0, 1] using nominal as the ceiling.
  const nominalR = meta.nominal_return || Math.max(...dots.map((d) => d.episodic_return));
  const retColor = (r) => interpRGB(r / nominalR);

  const tooltipG = document.getElementById('kl-tooltip');
  const tooltipBg = document.getElementById('kl-tooltip-bg');
  const tooltipText = document.getElementById('kl-tooltip-text');
  const showTooltip = (d) => {
    const txt = `${d.perturbation} ε=${d.epsilon.toFixed(2)} · ret ${Math.round(d.episodic_return)}`;
    tooltipText.textContent = txt;
    // Measure & center.
    const bbox = tooltipText.getBBox();
    const w = Math.max(110, bbox.width + 16);
    const h = 26;
    tooltipBg.setAttribute('width', w);
    tooltipBg.setAttribute('height', h);
    // Position above the dot, nudging back inside the SVG if it would clip.
    let x = d._x - w / 2;
    let y = d._y - h - 10;
    x = Math.max(2, Math.min(400 - w - 2, x));
    if (y < 2) y = d._y + 12;
    tooltipBg.setAttribute('x', x);
    tooltipBg.setAttribute('y', y);
    tooltipText.setAttribute('x', x + w / 2);
    tooltipText.setAttribute('y', y + 17);
    tooltipG.style.display = 'block';
  };
  const hideTooltip = () => { tooltipG.style.display = 'none'; };

  dots.forEach((d) => {
    const c = document.createElementNS(xmlns, 'circle');
    c.setAttribute('cx', d._x.toFixed(1));
    c.setAttribute('cy', d._y.toFixed(1));
    c.setAttribute('r', 4);
    d._color = retColor(d.episodic_return);
    c.setAttribute('fill', '#cccccc');     // grey until ball reaches it
    c.dataset.id = d.id;
    c.style.cursor = 'pointer';
    worldsG.appendChild(c);

    const label = `${d.perturbation} · ε=${d.epsilon.toFixed(2)}`;
    const onEnter = () => { showTooltip(d); playClip(d.video, label, d.episodic_return, d.episode_steps); };
    c.addEventListener('mouseenter', onEnter);
    c.addEventListener('focus', onEnter);
    c.addEventListener('click', onEnter);
    c.addEventListener('mouseleave', hideTooltip);
    c.addEventListener('blur', hideTooltip);
  });

  // Center P₀ marker plays the nominal (unperturbed) rollout.
  if (meta.nominal_video) {
    const nominalDot = document.getElementById('kl-nominal-dot');
    const nominalPlay = () =>
      playClip(meta.nominal_video, 'nominal', meta.nominal_return, meta.nominal_steps);
    nominalDot.addEventListener('mouseenter', nominalPlay);
    nominalDot.addEventListener('focus', nominalPlay);
    nominalDot.addEventListener('click', nominalPlay);
    // Start the player with the nominal rollout so the area isn't blank.
    playClip(meta.nominal_video, 'nominal', meta.nominal_return, meta.nominal_steps);
  }

  function update() {
    const eps = parseFloat(slider.value);
    const radius = radiusForEps(eps);
    ball.setAttribute('r', radius.toFixed(1));
    epsVal.textContent = eps.toFixed(2);
    let covered = 0;
    worldsG.querySelectorAll('circle').forEach((c) => {
      const d = dots[parseInt(c.dataset.id)];
      if (d._dist <= radius) {
        c.setAttribute('fill', d._color);
        c.setAttribute('r', 5);
        covered++;
      } else {
        c.setAttribute('fill', '#cccccc');
        c.setAttribute('r', 4);
      }
    });
    coverageEl.textContent = covered;
  }
  slider.addEventListener('input', update);
  update();
}

/* ---------- Mechanism sparklines ---------- */

async function initMechanismSparks() {
  const explainer = document.querySelector('.mech-explainer');
  if (!explainer) return;

  const sparkLayout = (extras = {}) => ({
    margin: { l: 4, r: 4, t: 4, b: 4 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    xaxis: { visible: false, fixedrange: true },
    yaxis: { visible: false, fixedrange: true },
    ...extras,
  });
  const cfg = { displayModeBar: false, responsive: true };

  // β(s,a) along a synthetic rollout: tall bars are fragile states.
  const Nb = 18;
  const betaFull = Array.from({ length: Nb }, (_, i) => {
    const x = i / (Nb - 1);
    return 0.05 + 0.95 * (0.5 + 0.5 * Math.sin(2.4 * x * Math.PI + 1.1)) ** 2;
  });
  Plotly.newPlot('spark-beta', [{
    type: 'bar',
    x: betaFull.map((_, i) => i),
    y: betaFull.map(() => 0),
    marker: { color: betaFull.map((v) => `rgba(210,58,58,${0.35 + 0.5 * v})`) },
    hoverinfo: 'skip',
  }], sparkLayout({ yaxis: { range: [0, 1.1], visible: false, fixedrange: true } }), cfg);

  // Synthetic E[β(t)]: large early (fragile policy), decays to a small floor as
  // the policy hardens. Used by the SPCL update simulator.
  const Ne = 200;
  const epsBudget = 1.0;
  const betaFloor = 0.02;
  const betaSignal = Array.from({ length: Ne }, (_, t) => {
    const x = t / (Ne - 1);
    return betaFloor + 0.4 * Math.exp(-3.2 * x);
  });
  // Integrate ε(t+1) = ε(t) + η · (E[β] − α · (ε(t) − ε_budget)), clipped to [0, ε_budget].
  function simulateEpsilon(alpha, eta) {
    const out = new Array(Ne);
    let eps = 0.0;
    for (let t = 0; t < Ne; t++) {
      out[t] = eps;
      const grad = betaSignal[t] - alpha * (eps - epsBudget);
      eps = Math.min(epsBudget, Math.max(0, eps + eta * grad));
    }
    return out;
  }
  const defaultAlpha = 0.05;
  const defaultEta = 0.01;
  const initialCurve = simulateEpsilon(defaultAlpha, defaultEta);
  const epsCurve = initialCurve;
  const epsMax = epsBudget;
  Plotly.newPlot('spark-eps', [{
    x: epsCurve.map((_, i) => i),
    y: epsCurve.map(() => null),
    mode: 'lines',
    line: { color: '#d23a3a', width: 2.6 },
    fill: 'tozeroy', fillcolor: 'rgba(210,58,58,0.13)',
    hoverinfo: 'skip',
  }], sparkLayout({ yaxis: { range: [0, epsMax * 1.05], visible: false, fixedrange: true } }), cfg);

  const animateSpark = (gd, full, stepMs) => new Promise((resolve) => {
    let i = 0;
    const tick = () => {
      const y = full.slice(0, i + 1).concat(Array(full.length - i - 1).fill(null));
      Plotly.restyle(gd, { y: [y] }, [0]);
      i++;
      if (i >= full.length) return resolve();
      setTimeout(tick, stepMs);
    };
    tick();
  });
  const resetSparks = () => {
    Plotly.restyle('spark-beta', { y: [Array(Nb).fill(0)] }, [0]);
    Plotly.restyle('spark-eps', { y: [Array(Ne).fill(null)] }, [0]);
  };

  let running = false;
  let sliderUsed = false;       // suppress scroll-replay only while user is dragging
  const animateAll = async () => {
    if (running || sliderUsed) return;
    running = true;
    resetSparks();
    await Promise.all([
      animateSpark('spark-beta', betaFull, 50),
      animateSpark('spark-eps', epsCurve, 20),
    ]);
    running = false;
  };

  // Play once immediately so the visual is never blank, even if the section
  // is below the fold at load (IntersectionObserver-only kicks can miss when
  // MathJax reflows the page after layout).
  animateAll();

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) animateAll();
      else if (e.boundingClientRect.top > 0) resetSparks();
    });
  }, { threshold: 0.25 });
  io.observe(explainer);

  // Live ε(t) update from the α/η sliders.
  const alphaIn = document.getElementById('mech-alpha');
  const etaIn = document.getElementById('mech-eta');
  const alphaVal = document.getElementById('mech-alpha-val');
  const etaVal = document.getElementById('mech-eta-val');
  const onSliderInput = () => {
    sliderUsed = true;
    const a = parseFloat(alphaIn.value);
    const e = parseFloat(etaIn.value);
    alphaVal.textContent = a.toFixed(3);
    etaVal.textContent = e.toFixed(4);
    const curve = simulateEpsilon(a, e);
    Plotly.restyle('spark-eps', { y: [curve] }, [0]);
  };
  if (alphaIn && etaIn) {
    alphaIn.addEventListener('input', onSliderInput);
    etaIn.addEventListener('input', onSliderInput);
    // After the user lets go, allow the scroll-triggered animation to replay
    // on the next time they re-enter the section.
    const clearSliderLock = () => { sliderUsed = false; };
    alphaIn.addEventListener('change', clearSliderLock);
    etaIn.addEventListener('change', clearSliderLock);
  }
}

/* ---------- Motivation chart (scroll-triggered) ---------- */

const PERTURB_TITLES = {
  action: 'Action noise',
  observation: 'Observation noise',
  environment: 'Environment noise',
};
const PERTURB_SYMBOL = {
  action: 'p<sub>act</sub>',
  observation: 'σ<sub>obs</sub>',
  environment: 'δ<sub>env</sub>',
};

function dashFromLatex(ls) {
  return ls === '--' ? 'dash' : 'solid';
}

async function initMotivation() {
  const data = await fetch('./static/js/motivation.json').then((r) => r.json());
  const cleanLabel = (lbl) => lbl
    .replace(/\$/g, '')
    .replace(/\\varepsilon/g, 'ε');

  const traces = [];
  // Three subplots, one per perturbation type, laid out horizontally.
  const xaxisKey = ['xaxis', 'xaxis2', 'xaxis3'];
  const yaxisKey = ['yaxis', 'yaxis2', 'yaxis3'];
  const domains = [[0, 0.30], [0.36, 0.66], [0.72, 1.0]];
  data.noise_types.forEach((nt, col) => {
    Object.entries(data.series).forEach(([label, s]) => {
      const d = s.noise[nt];
      traces.push({
        x: d.levels,
        y: d.mean,
        type: 'scatter',
        mode: 'lines+markers',
        name: cleanLabel(label),
        legendgroup: label,
        showlegend: col === 0,
        xaxis: `x${col + 1}`,
        yaxis: `y${col + 1}`,
        line: { color: s.color, width: 2.4, dash: dashFromLatex(s.linestyle) },
        marker: { size: 7 },
      });
    });
  });

  const layout = {
    margin: { l: 65, r: 25, t: 50, b: 60 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: '"Noto Sans", sans-serif', size: 13, color: '#1a1a1a' },
    legend: { orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center', font: { size: 12 } },
    annotations: data.noise_types.map((nt, i) => ({
      text: `<b>${PERTURB_TITLES[nt]}</b>`,
      x: (domains[i][0] + domains[i][1]) / 2,
      y: 1.06,
      xref: 'paper', yref: 'paper',
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: { size: 14, color: 'rgba(26,26,26,0)' }, // start transparent
    })),
  };
  // Lock x and y ranges up-front so axes don't reflow during animation.
  const xMin = Math.min(...data.levels);
  const xMax = Math.max(...data.levels);
  const xPad = (xMax - xMin) * 0.08;
  const allY = [];
  traces.forEach((t) => t.y.forEach((v) => { if (v != null) allY.push(v); }));
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yPad = (yMax - yMin) * 0.08;

  data.noise_types.forEach((nt, i) => {
    layout[xaxisKey[i]] = {
      domain: domains[i],
      title: { text: `Noise level (${PERTURB_SYMBOL[nt]})`, font: { size: 12 } },
      tickvals: data.levels,
      ticktext: data.levels.map(String),
      range: [xMin - xPad, xMax + xPad],
      gridcolor: '#eef0f5',
      fixedrange: true,
    };
    layout[yaxisKey[i]] = {
      anchor: `x${i + 1}`,
      range: [yMin - yPad, yMax + yPad],
      gridcolor: '#eef0f5',
      title: i === 0 ? { text: 'Episodic return', font: { size: 13 } } : undefined,
      fixedrange: true,
    };
  });

  // Initialize traces blank (nulls render as gaps) so legend/axes lock in
  // but no points show yet.
  const blankTraces = traces.map((t) => ({
    ...t,
    x: data.levels.map(() => null),
    y: data.levels.map(() => null),
  }));
  await Plotly.newPlot('motivation-chart', blankTraces, layout, { displayModeBar: false, responsive: true });

  // Reveal order: trace-by-trace, point-by-point. Vanilla first, then ε=0.1,
  // then ε=1.0. Each series fills all 3 subplots in lock-step.
  const seriesKeys = Object.keys(data.series);
  const numPerturbs = data.noise_types.length;
  const numLevels = data.levels.length;
  const POINT_DELAY_MS = 370;        // ~1.5x faster than 550
  const SERIES_GAP_MS = 470;         // ~1.5x faster than 700

  let running = false;
  const hideTitles = () => {
    const ann = layout.annotations.map((a) => ({ ...a, font: { ...a.font, color: 'rgba(26,26,26,0)' } }));
    Plotly.relayout('motivation-chart', { annotations: ann });
  };
  const showTitlesFade = async (durationMs = 600, steps = 12) => {
    for (let k = 1; k <= steps; k++) {
      const alpha = k / steps;
      const ann = layout.annotations.map((a) => ({ ...a, font: { ...a.font, color: `rgba(26,26,26,${alpha})` } }));
      Plotly.relayout('motivation-chart', { annotations: ann });
      await new Promise((r) => setTimeout(r, durationMs / steps));
    }
  };
  const resetTraces = () => {
    const blanks = traces.map((t) => ({ ...t, x: data.levels.map(() => null), y: data.levels.map(() => null) }));
    Plotly.react('motivation-chart', blanks, layout, { displayModeBar: false, responsive: true });
  };
  const animate = async () => {
    if (running) return;
    running = true;
    resetTraces();
    hideTitles();
    // Fade all 3 subplot titles in together as the first series starts drawing.
    showTitlesFade(600, 14);
    for (let s = 0; s < seriesKeys.length; s++) {
      for (let p = 0; p < numLevels; p++) {
        const update = { x: [], y: [] };
        const idxs = [];
        for (let col = 0; col < numPerturbs; col++) {
          const traceIdx = col * seriesKeys.length + s;
          idxs.push(traceIdx);
          const full = traces[traceIdx];
          update.x.push(full.x.slice(0, p + 1));
          update.y.push(full.y.slice(0, p + 1));
        }
        Plotly.restyle('motivation-chart', update, idxs);
        await new Promise((r) => setTimeout(r, POINT_DELAY_MS));
      }
      await new Promise((r) => setTimeout(r, SERIES_GAP_MS));
    }
    running = false;
  };

  // Replay every time the chart re-enters the viewport.
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) animate(); });
  }, { threshold: 0.45 });
  observer.observe(document.getElementById('motivation-chart'));
}

/* ---------- Epsilon-trajectory interactive grid ---------- */

async function initEpsilonTrajectories() {
  const target = document.getElementById('eps-traj-grid');
  if (!target) return;
  const data = await fetch('./static/js/epsilon_trajectories.json').then((r) => r.json());
  const cells = data.cells;
  if (!cells.length) return;

  // Layout: 2 rows × 6 cols. Reserve generous room below the bottom row for
  // its x-axis tick labels, the legend, and breathing space.
  const COLS = 6, ROWS = 2;
  const padX = 0.025;
  const topMargin = 0.06;       // above top-row titles
  const bottomMargin = 0.20;    // below bottom-row plots — holds bottom x-ticks
  const rowGap = 0.10;          // between rows — holds top-row x-ticks
  const totalH = 1 - topMargin - bottomMargin;
  const cellH = (totalH - rowGap) / ROWS;
  const cellW = (1 - padX * (COLS + 1)) / COLS;
  const domainFor = (i) => {
    const r = Math.floor(i / COLS), c = i % COLS;
    const x0 = padX + c * (cellW + padX);
    const y1 = 1 - topMargin - r * (cellH + rowGap);
    const y0 = y1 - cellH;
    return { x: [x0, x0 + cellW], y: [y0, y1] };
  };

  const EPS_COLOR = '#2E86AB';
  const BETA_COLOR = '#E07B39';

  const traces = [];
  const annotations = [];
  cells.forEach((c, i) => {
    const xa = `x${i + 1}`;
    const ya = `y${i + 1}`;       // epsilon axis (left, linear)
    const ya2 = `y${i + 1 + cells.length}`;   // beta axis (right, log)
    // Epsilon trace (drawn first; starts blank).
    traces.push({
      x: c.steps, y: c.steps.map(() => null),
      mode: 'lines', name: 'ε(t)', legendgroup: 'eps',
      showlegend: i === 0,
      line: { color: EPS_COLOR, width: 2 },
      xaxis: xa, yaxis: ya, hoverinfo: 'skip',
    });
    // Beta trace.
    if (c.beta) {
      traces.push({
        x: c.steps, y: c.steps.map(() => null),
        mode: 'lines', name: 'β(t)', legendgroup: 'beta',
        showlegend: i === 0,
        line: { color: BETA_COLOR, width: 1.8, dash: 'dash' },
        xaxis: xa, yaxis: ya2, hoverinfo: 'skip',
      });
    } else {
      // Placeholder so trace indices stay 2 per cell.
      traces.push({
        x: [], y: [], mode: 'lines', xaxis: xa, yaxis: ya2,
        showlegend: false, hoverinfo: 'skip',
      });
    }
    annotations.push({
      text: `<b>${c.env}</b> · ${c.backbone}`,
      x: (domainFor(i).x[0] + domainFor(i).x[1]) / 2,
      y: domainFor(i).y[1] + 0.025,
      xref: 'paper', yref: 'paper',
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false, font: { size: 11, color: '#222' },
    });
  });

  const layout = {
    margin: { l: 50, r: 50, t: 30, b: 80 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: '"Noto Sans", sans-serif', size: 11, color: '#222' },
    showlegend: true,
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 0.06,
              yanchor: 'top', font: { size: 12 } },
    annotations,
  };
  cells.forEach((c, i) => {
    const d = domainFor(i);
    const isLeftCol = (i % COLS) === 0;
    const isRightCol = (i % COLS) === COLS - 1;
    const maxStep = c.steps[c.steps.length - 1];
    const tickVals = [0, 0.25, 0.5, 0.75, 1.0].map((f) => f * maxStep);
    const fmtStep = (v) => v === 0 ? '0' :
      v >= 1e6 ? (v / 1e6).toFixed(v % 1e6 === 0 ? 0 : 1) + 'M' :
      Math.round(v / 1e3) + 'k';
    layout[`xaxis${i + 1}`] = {
      domain: d.x,
      anchor: `y${i + 1}`,
      gridcolor: '#eef0f5',
      tickfont: { size: 9 },
      showticklabels: true,
      tickvals: tickVals,
      ticktext: tickVals.map(fmtStep),
      fixedrange: true,
    };
    layout[`yaxis${i + 1}`] = {
      domain: d.y,
      anchor: `x${i + 1}`,
      gridcolor: '#eef0f5',
      tickfont: { size: 9, color: EPS_COLOR },
      showticklabels: isLeftCol,
      range: [0, 1.05],
      fixedrange: true,
    };
    layout[`yaxis${i + 1 + cells.length}`] = {
      domain: d.y,
      anchor: `x${i + 1}`,
      overlaying: `y${i + 1}`,
      side: 'right',
      type: 'log',
      tickfont: { size: 9, color: BETA_COLOR },
      showticklabels: isRightCol,
      showgrid: false,
      fixedrange: true,
    };
  });

  await Plotly.newPlot(target, traces, layout, { displayModeBar: false, responsive: true });

  const ANIM_MS_EPS = 1400;       // ε draws across whole timeline in this long
  const ANIM_MS_BETA = 1400;
  const STEPS = 60;

  let running = false;
  const reset = () => {
    const blanks = traces.map((t, i) => ({ ...t, y: (t.x || []).map(() => null) }));
    Plotly.react(target, blanks, layout, { displayModeBar: false, responsive: true });
  };
  const drawLayer = async (offset, totalMs) => {
    // offset = 0 -> epsilon traces (even indices); 1 -> beta (odd).
    const idxs = traces.map((_, j) => j).filter((j) => j % 2 === offset);
    const tickMs = totalMs / STEPS;
    for (let k = 1; k <= STEPS; k++) {
      const frac = k / STEPS;
      const update = { y: idxs.map((j) => {
        const full = cells[Math.floor(j / 2)];
        const series = offset === 0 ? full.epsilon.mean : (full.beta ? full.beta.mean : null);
        if (!series) return [];
        const n = series.length;
        const cut = Math.max(1, Math.round(n * frac));
        return series.slice(0, cut).concat(Array(n - cut).fill(null));
      }) };
      Plotly.restyle(target, update, idxs);
      await new Promise((r) => setTimeout(r, tickMs));
    }
  };
  const animate = async () => {
    if (running) return;
    running = true;
    reset();
    await drawLayer(0, ANIM_MS_EPS);
    await new Promise((r) => setTimeout(r, 200));
    await drawLayer(1, ANIM_MS_BETA);
    running = false;
  };

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) animate(); });
  }, { threshold: 0.25 });
  io.observe(target);
}

/* ---------- Training-returns interactive grid ---------- */

async function initTrainingGrid() {
  const target = document.getElementById('training-grid');
  if (!target) return;
  const data = await fetch('./static/js/training_returns.json').then((r) => r.json());
  const cells = data.cells;
  if (!cells.length) return;

  // 3 rows (families = PPO/DDPG/SAC) × 4 cols (envs).
  const COLS = 4, ROWS = data.families.length;
  const padX = 0.05;     // wider so per-cell y-tick labels don't crowd
  const topMargin = 0.06;          // env titles at the top
  const bottomMargin = 0.16;       // bottom x-ticks + legend
  const rowGap = 0.06;             // tightened (each row keeps its own ticks below it)
  const totalH = 1 - topMargin - bottomMargin;
  const cellH = (totalH - rowGap * (ROWS - 1)) / ROWS;
  const cellW = (1 - padX * (COLS + 1)) / COLS;

  // Cells in JSON come row-major: for each family, all envs in order.
  // Verify by recomputing index from (family, env).
  const indexOf = (family, env) =>
    data.families.indexOf(family) * COLS + data.envs.indexOf(env);

  const domainFor = (i) => {
    const r = Math.floor(i / COLS), c = i % COLS;
    const x0 = padX + c * (cellW + padX);
    const y1 = 1 - topMargin - r * (cellH + rowGap);
    const y0 = y1 - cellH;
    return { x: [x0, x0 + cellW], y: [y0, y1] };
  };

  const traces = [];
  const annotations = [];
  // Dedup the legend by method *label* (Vanilla/Fixed/SPACE/...) so each
  // method appears exactly once even though it exists across 3 families.
  const legendShown = new Set();

  cells.forEach((cell, cellIdx) => {
    const xa = `x${cellIdx + 1}`;
    const ya = `y${cellIdx + 1}`;
    cell.traces.forEach((t) => {
      const legendKey = t.label;
      const showLegend = !legendShown.has(legendKey);
      if (showLegend) legendShown.add(legendKey);
      traces.push({
        x: t.x, y: t.x.map(() => null),
        type: 'scatter', mode: 'lines',
        name: t.label, legendgroup: legendKey,
        showlegend: showLegend,
        line: { color: t.color, width: t.is_ours ? 2.2 : 1.4,
                dash: t.is_ours ? 'dash' : 'solid' },
        xaxis: xa, yaxis: ya, hoverinfo: 'skip',
        _trace: t,
        _cellIdx: cellIdx,
      });
    });
    // Top-row title (env name) above each column.
    if (Math.floor(cellIdx / COLS) === 0) {
      const d = domainFor(cellIdx);
      annotations.push({
        text: `<b>${cell.env}</b>`,
        x: (d.x[0] + d.x[1]) / 2, y: d.y[1] + 0.02,
        xref: 'paper', yref: 'paper',
        xanchor: 'center', yanchor: 'bottom',
        showarrow: false, font: { size: 12, color: '#222' },
      });
    }
    // Left-column label (family name) outside the leftmost cells.
    if ((cellIdx % COLS) === 0) {
      const d = domainFor(cellIdx);
      annotations.push({
        text: `<b>${cell.family}</b>`,
        x: -0.005, y: (d.y[0] + d.y[1]) / 2,
        xref: 'paper', yref: 'paper',
        xanchor: 'right', yanchor: 'middle',
        textangle: -90,
        showarrow: false, font: { size: 12, color: '#222' },
      });
    }
  });

  const layout = {
    margin: { l: 70, r: 30, t: 30, b: 80 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: '"Noto Sans", sans-serif', size: 11, color: '#222' },
    showlegend: true,
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 0.04,
              yanchor: 'top', font: { size: 11 } },
    annotations,
  };

  cells.forEach((cell, i) => {
    const d = domainFor(i);
    const isLeftCol = (i % COLS) === 0;
    const xs = (cell.traces[0]?.x) || [];
    const maxStep = xs[xs.length - 1] || 1;
    const tickVals = [0, 0.25, 0.5, 0.75, 1.0].map((f) => f * maxStep);
    const fmtStep = (v) => v === 0 ? '0' :
      v >= 1e6 ? (v / 1e6).toFixed(v % 1e6 === 0 ? 0 : 1) + 'M' :
      Math.round(v / 1e3) + 'k';
    layout[`xaxis${i + 1}`] = {
      domain: d.x,
      anchor: `y${i + 1}`,
      gridcolor: '#eef0f5',
      tickfont: { size: 9 },
      showticklabels: true,
      tickvals: tickVals,
      ticktext: tickVals.map(fmtStep),
      fixedrange: true,
    };
    // Per-cell auto y-range: exactly the data extrema, with 10% headroom on top.
    const allMean = cell.traces.flatMap((t) => t.y).filter((v) => v != null);
    const rawMin = allMean.length ? Math.min(...allMean) : 0;
    const rawMax = allMean.length ? Math.max(...allMean) : 1;
    const span = Math.max(rawMax - rawMin, 1);
    const yMin = rawMin;
    const yMax = rawMax + span * 0.10;
    layout[`yaxis${i + 1}`] = {
      domain: d.y,
      anchor: `x${i + 1}`,
      side: 'left',
      gridcolor: '#eef0f5',
      tickfont: { size: 9 },
      showticklabels: true,
      range: [yMin, yMax],
      fixedrange: true,
      nticks: 3,
    };
  });

  await Plotly.newPlot(target, traces, layout, { displayModeBar: false, responsive: true });

  const ANIM_MS = 1800;
  const STEPS = 70;
  let running = false;
  const reset = () => {
    const blanks = traces.map((t) => ({ ...t, y: t.x.map(() => null) }));
    Plotly.react(target, blanks, layout, { displayModeBar: false, responsive: true });
  };
  const animate = async () => {
    if (running) return;
    running = true;
    reset();
    const tickMs = ANIM_MS / STEPS;
    for (let k = 1; k <= STEPS; k++) {
      const frac = k / STEPS;
      const idxs = traces.map((_, j) => j);
      const update = { y: idxs.map((j) => {
        const full = traces[j]._trace.y;
        const n = full.length;
        const cut = Math.max(1, Math.round(n * frac));
        return full.slice(0, cut).concat(Array(n - cut).fill(null));
      }) };
      Plotly.restyle(target, update, idxs);
      await new Promise((r) => setTimeout(r, tickMs));
    }
    running = false;
  };

  // Play once immediately so the visual isn't blank, then re-trigger on
  // each scroll-into-view.
  animate();
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) animate(); });
  }, { threshold: 0.2 });
  io.observe(target);
}

/* ---------- Results section ---------- */

function makeButtonGroup(containerId, items, activeValue, onSelect) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  items.forEach((it) => {
    const b = document.createElement('button');
    b.className = 'button is-small' + (it.value === activeValue ? ' is-active' : '');
    b.textContent = it.label;
    b.dataset.value = it.value;
    b.addEventListener('click', () => {
      [...el.children].forEach((c) => c.classList.remove('is-active'));
      b.classList.add('is-active');
      onSelect(it.value);
    });
    el.appendChild(b);
  });
}

function renderChart() {
  const methods = DATA.results[state.alg][state.env][state.perturb];
  const xs = DATA.epsilons;
  const traces = methods.map((m) => ({
    x: xs,
    y: m.series.map((p) => p.mean),
    error_y: {
      type: 'data',
      array: m.series.map((p) => p.std),
      visible: true,
      thickness: 1,
      width: 4,
      color: COLORS[m.key],
    },
    mode: 'lines+markers',
    name: m.label,
    line: { color: COLORS[m.key], width: m.is_ours ? 3 : 1.5 },
    marker: { size: m.is_ours ? 9 : 6, symbol: m.is_ours ? 'diamond' : 'circle' },
  }));
  const layout = {
    margin: { l: 55, r: 20, t: 20, b: 50 },
    xaxis: { title: 'Perturbation radius ε', dtick: 0.1, gridcolor: '#eee', zerolinecolor: '#ddd' },
    yaxis: { title: 'Episodic return', gridcolor: '#eee', zerolinecolor: '#ddd' },
    legend: { orientation: 'h', y: -0.22, font: { size: 10 } },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: '"Noto Sans", sans-serif', size: 12 },
  };
  Plotly.react('results-chart', traces, layout, { displayModeBar: false, responsive: true });
}

function renderTable() {
  const methods = DATA.results[state.alg][state.env][state.perturb];
  const xs = DATA.epsilons;
  const thead = document.querySelector('#results-table thead');
  const tbody = document.querySelector('#results-table tbody');
  thead.innerHTML = '<tr><th>Method</th>' +
    xs.map((e) => `<th>ε=${e}</th>`).join('') + '</tr>';
  tbody.innerHTML = methods.map((m) => {
    const rowClass = m.is_ours ? 'row-ours' : '';
    const cells = m.series.map((p) => {
      const cls = p.best ? 'best' : '';
      return `<td class="${cls}">${p.mean.toFixed(1)} <span style="color:#999">±${p.std.toFixed(1)}</span></td>`;
    }).join('');
    return `<tr class="${rowClass}"><td>${m.label}</td>${cells}</tr>`;
  }).join('');
}

function renderHighlight() {
  const methods = DATA.results[state.alg][state.env][state.perturb];
  const ours = methods.find((m) => m.is_ours);
  let totalGain = 0, n = 0;
  ours.series.forEach((p, i) => {
    const others = methods.filter((m) => !m.is_ours).map((m) => m.series[i].mean);
    const bestOther = Math.max(...others);
    if (bestOther > 1e-6) { totalGain += (p.mean - bestOther) / Math.abs(bestOther); n++; }
  });
  const pct = n ? (100 * totalGain / n) : 0;
  const sign = pct >= 0 ? '+' : '';
  document.getElementById('result-highlight').innerHTML =
    `DR-${state.alg}-SPCRL improves over the best baseline on ` +
    `<b>${state.env}</b> / <b>${state.perturb.toLowerCase()} perturbation</b> ` +
    `by <strong>${sign}${pct.toFixed(1)}%</strong> (averaged over ε).`;
}

function renderAll() {
  renderChart();
  renderTable();
  renderHighlight();
}

/* ---------- Ablation controls ---------- */

function initAblation() {
  let hp = 'alpha', env = 'HalfCheetah-v4';
  const update = () => {
    ['ppo', 'sac', 'ddpg'].forEach((alg) => {
      document.getElementById(`ablation-${alg}`).src =
        `./static/figures/ablation/${env}_${alg}_ablation_${hp}.png`;
    });
  };
  document.querySelectorAll('#ablation-hp-buttons button').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#ablation-hp-buttons button').forEach((x) => x.classList.remove('is-active'));
      b.classList.add('is-active');
      hp = b.dataset.hp;
      update();
    });
  });
  document.querySelectorAll('#ablation-env-buttons button').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#ablation-env-buttons button').forEach((x) => x.classList.remove('is-active'));
      b.classList.add('is-active');
      env = b.dataset.env;
      update();
    });
  });
}

/* ---------- Boot ---------- */

function initSectionReveal() {
  // Fades + slides each .reveal section in when it enters the viewport,
  // and re-arms it (so re-entry replays the animation) only after it has
  // been seen at least once AND has scrolled back below the viewport.
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;
  const seenOnce = new WeakSet();
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add('is-visible');
        seenOnce.add(e.target);
      } else if (seenOnce.has(e.target) && e.boundingClientRect.top > 0) {
        e.target.classList.remove('is-visible');
      }
    });
  }, { threshold: 0.12 });
  els.forEach((el) => io.observe(el));
}

async function boot() {
  initSectionReveal();
  initKLExplainer().catch((e) => console.error('kl:', e));
  initMechanismSparks();
  initMotivation().catch((e) => console.error('motivation:', e));
  initEpsilonTrajectories().catch((e) => console.error('eps-traj:', e));
  initTrainingGrid().catch((e) => console.error('training:', e));

  DATA = await fetch('./static/js/results.json').then((r) => r.json());

  makeButtonGroup('alg-buttons',
    DATA.algorithms.map((a) => ({ value: a, label: a })),
    state.alg,
    (v) => { state.alg = v; renderAll(); });
  makeButtonGroup('env-buttons',
    DATA.environments.map((e) => ({ value: e, label: e })),
    state.env,
    (v) => { state.env = v; renderAll(); });
  makeButtonGroup('perturb-buttons',
    DATA.perturbations.map((p) => ({ value: p, label: p })),
    state.perturb,
    (v) => { state.perturb = v; renderAll(); });

  renderAll();
}

document.addEventListener('DOMContentLoaded', boot);
