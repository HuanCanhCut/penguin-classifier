const SVG_NS = "http://www.w3.org/2000/svg";
const NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"];

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function statTile(label, value, note = "") {
  return el("div", { class: "card stat" },
    el("div", { class: "label" }, label),
    el("div", { class: "value" }, value),
    note ? el("div", { class: "small muted" }, note) : null,
  );
}

function renderLegends() {
  for (const box of document.querySelectorAll("[data-legend]")) {
    box.replaceChildren(...SPECIES.map((name) => el("span", {}, speciesLabel(name))));
  }
}

function renderSpecies(stats) {
  const counts = stats.species_distribution;
  const max = Math.max(...Object.values(counts));
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name, n]) => {
    const share = n / stats.n_samples;
    return el("div", { class: `bar-row sp-${name}`, "data-tip": `${name}: ${n} con (${percent(share, 1)})` },
      el("span", { class: "name" }, name),
      el("div", { class: "bar-track" }, el("div", { class: "bar-fill", style: `width:${(n / max) * 100}%` })),
      el("span", { class: "num" }, String(n)),
    );
  });
  document.querySelector("#species-chart").replaceChildren(el("div", { class: "bars" }, ...rows));
}

function renderIslands(stats) {
  const islands = Object.entries(stats.island_distribution);
  const max = Math.max(...islands.map(([, row]) => Object.values(row).reduce((a, b) => a + b, 0)));
  const rows = islands.map(([island, row]) => {
    const total = Object.values(row).reduce((a, b) => a + b, 0);
    const segments = SPECIES.filter((s) => row[s] > 0).map((s) =>
      el("span", { class: `sp-${s}`, style: `flex:${row[s]}`, "data-tip": `${island} · ${s}: ${row[s]} con` }));
    return el("div", { class: "bar-row" },
      el("span", { class: "name" }, island),
      el("div", {}, el("div", { class: "stack", style: `width:${(total / max) * 100}%` }, ...segments)),
      el("span", { class: "num" }, String(total)),
    );
  });
  document.querySelector("#island-chart").replaceChildren(el("div", { class: "bars" }, ...rows));
}

function niceTicks(min, max, count = 5) {
  const step = Math.pow(10, Math.floor(Math.log10((max - min) / count)));
  const err = (max - min) / count / step;
  const nice = step * (err >= 5 ? 10 : err >= 2 ? 5 : err >= 1.5 ? 2 : 1);
  const ticks = [];
  for (let v = Math.ceil(min / nice) * nice; v <= max + 1e-9; v += nice) ticks.push(+v.toFixed(6));
  return ticks;
}

function renderScatter(stats) {
  const W = 720, H = 360, m = { top: 12, right: 16, bottom: 44, left: 48 };
  const xs = stats.points.map((p) => p.bill_length_mm);
  const ys = stats.points.map((p) => p.bill_depth_mm);
  const [x0, x1] = [Math.floor(Math.min(...xs)) - 1, Math.ceil(Math.max(...xs)) + 1];
  const [y0, y1] = [Math.floor(Math.min(...ys)) - 0.5, Math.ceil(Math.max(...ys)) + 0.5];
  const sx = (v) => m.left + ((v - x0) / (x1 - x0)) * (W - m.left - m.right);
  const sy = (v) => H - m.bottom - ((v - y0) / (y1 - y0)) * (H - m.top - m.bottom);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img",
    "aria-label": "Biểu đồ phân tán chiều dài mỏ và độ dày mỏ theo loài" });

  for (const t of niceTicks(x0, x1)) {
    root.append(svg("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: m.top, y2: H - m.bottom }));
    const label = svg("text", { class: "axis-label", x: sx(t), y: H - m.bottom + 16, "text-anchor": "middle" });
    label.textContent = t;
    root.append(label);
  }
  for (const t of niceTicks(y0, y1)) {
    root.append(svg("line", { class: "gridline", x1: m.left, x2: W - m.right, y1: sy(t), y2: sy(t) }));
    const label = svg("text", { class: "axis-label", x: m.left - 8, y: sy(t) + 4, "text-anchor": "end" });
    label.textContent = t;
    root.append(label);
  }

  const xTitle = svg("text", { class: "axis-title", x: (m.left + W - m.right) / 2, y: H - 6, "text-anchor": "middle" });
  xTitle.textContent = "Chiều dài mỏ (mm)";
  const yTitle = svg("text", { class: "axis-title", x: -(m.top + H - m.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" });
  yTitle.textContent = "Độ dày mỏ (mm)";
  root.append(xTitle, yTitle);

  for (const p of stats.points) {
    const dot = svg("circle", { cx: sx(p.bill_length_mm), cy: sy(p.bill_depth_mm), r: 4.5, class: `sp-${p.species}`,
      "data-tip": `${p.species} · mỏ ${p.bill_length_mm} × ${p.bill_depth_mm} mm` });
    root.append(dot);
  }
  document.querySelector("#scatter").replaceChildren(root);
}

function renderMeans(stats) {
  const head = el("tr", {}, el("th", {}, "Loài"), ...NUMERIC.map((f) => el("th", { class: "num" }, FIELD_LABELS[f])));
  const rows = SPECIES.map((s) => el("tr", {},
    el("td", {}, speciesLabel(s)),
    ...NUMERIC.map((f) => el("td", { class: "num" }, String(stats.feature_means[s][f]))),
  ));
  const range = el("tr", {}, el("td", { class: "muted" }, "Min – max"),
    ...NUMERIC.map((f) => el("td", { class: "num muted" }, `${stats.feature_ranges[f].min} – ${stats.feature_ranges[f].max}`)));
  document.querySelector("#means").replaceChildren(
    el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows, range))));
}

function renderModel(info) {
  const metrics = info.metrics;
  const items = [
    ["Thuật toán", "K-Nearest Neighbors"],
    ["k (n_neighbors)", info.knn.n_neighbors],
    ["Metric", info.knn.metric],
    ["Weights", info.knn.weights],
    ["Features", info.input_features.join(", ")],
    ["Số mẫu train", info.n_train],
    ["File model", info.model_file],
    ["scikit-learn", info.sklearn_version],
  ];
  const children = [el("div", { class: "table-wrap" }, el("table", {}, el("tbody", {},
    ...items.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, String(v))))))),
  ];

  if (metrics?.confusion_matrix) {
    const labels = metrics.labels;
    const head = el("tr", {}, el("th", {}, "Thực tế \\ Dự đoán"), ...labels.map((l) => el("th", { class: "num" }, l)));
    const rows = metrics.confusion_matrix.map((row, i) => el("tr", {},
      el("td", {}, speciesLabel(labels[i])),
      ...row.map((n, j) => el("td", { class: "num", style: i === j ? "font-weight:700" : n ? "color:var(--error);font-weight:700" : "" }, String(n))),
    ));
    children.push(
      el("h2", { style: "margin-top:20px" }, `Confusion matrix (tập test, ${metrics.n_test} mẫu)`),
      el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows))),
    );
  }
  document.querySelector("#model").replaceChildren(...children);
}

async function main() {
  initTooltips();
  renderLegends();
  try {
    const [stats, info] = await Promise.all([apiFetch("/api/stats"), apiFetch("/api/model-info")]);
    document.querySelector("#stats").replaceChildren(
      statTile("Số mẫu (sau làm sạch)", String(stats.n_samples)),
      statTile("Test accuracy", percent(info.test_accuracy, 1), `${info.metrics?.n_test ?? "?"} mẫu test`),
      statTile("Test macro-F1", percent(info.test_macro_f1, 1)),
      statTile("Model", `KNN k=${info.knn.n_neighbors}`, `${info.knn.metric}, ${info.knn.weights}`),
    );
    renderSpecies(stats);
    renderIslands(stats);
    renderScatter(stats);
    renderMeans(stats);
    renderModel(info);
  } catch (error) {
    document.querySelector("#dash-alert").textContent = error.message;
  }
}

main();
