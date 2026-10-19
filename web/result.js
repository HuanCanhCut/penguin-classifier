const content = document.querySelector("#content");

function loadResult() {
  try {
    return JSON.parse(sessionStorage.getItem(RESULT_KEY));
  } catch {
    return null;
  }
}

function renderEmpty() {
  content.replaceChildren(
    el("h1", {}, "Kết quả dự đoán"),
    el("p", { class: "lead" }, "Chưa có kết quả nào. Hãy nhập thông tin một con chim cánh cụt trước."),
    el("a", { class: "button", href: "index.html" }, "Đi tới trang dự đoán"),
  );
}

function confidenceBars(confidence, votes) {
  return el("div", { class: "bars" }, ...SPECIES.map((name) => {
    const p = confidence[name] ?? 0;
    return el("div", { class: `bar-row sp-${name}`, "data-tip": `${name}: ${percent(p, 1)} · ${votes[name]}/5 hàng xóm` },
      el("span", { class: "name" }, name),
      el("div", { class: "bar-track" }, el("div", { class: "bar-fill", style: `width:${p * 100}%` })),
      el("span", { class: "num" }, percent(p)),
    );
  }));
}

function neighborsTable(neighbors) {
  const numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"];
  const hasFeatures = neighbors.some((n) => n.features);
  const head = el("tr", {},
    el("th", {}, "#"), el("th", {}, "Loài"), el("th", { class: "num" }, "Khoảng cách"),
    ...(hasFeatures ? [...numeric.map((f) => el("th", { class: "num" }, FIELD_LABELS[f])), el("th", {}, "Giới tính"), el("th", {}, "Đảo")] : []),
  );
  const rows = neighbors.map((n) => el("tr", {},
    el("td", {}, String(n.rank)),
    el("td", {}, speciesLabel(n.species)),
    el("td", { class: "num" }, n.distance.toFixed(3)),
    ...(hasFeatures ? [
      ...numeric.map((f) => el("td", { class: "num" }, String(n.features[f]))),
      el("td", {}, n.features.sex), el("td", {}, n.features.island),
    ] : []),
  ));
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows)));
}

function inputSummary(input) {
  return el("div", { class: "table-wrap" }, el("table", {}, el("tbody", {},
    ...Object.entries(input).map(([key, value]) => el("tr", {},
      el("th", {}, FIELD_LABELS[key] || key), el("td", { class: "num" }, String(value)),
    )),
  )));
}

function render(result) {
  const top = result.confidence[result.prediction];
  const warnings = result.warnings || [];

  content.replaceChildren(
    el("h1", {}, "Kết quả dự đoán"),
    el("p", { class: "lead" }, "Dựa trên 5 con chim gần nhất trong tập train (KNN, khoảng cách Manhattan sau chuẩn hoá)."),
    ...(warnings.length
      ? [el("div", { class: "alert warn", role: "status" },
          el("strong", {}, "Lưu ý: dữ liệu nhập có giá trị bất thường"),
          el("ul", {}, ...warnings.map((w) => el("li", {}, `${FIELD_LABELS[w.field] || w.field}: ${w.message}`))))]
      : []),
    el("div", { class: "grid grid-2" },
      el("section", { class: "card" },
        el("div", { class: "hero" },
          el("div", {},
            el("div", { class: "label" }, "Loài dự đoán"),
            el("div", { class: `value sp-${result.prediction}` }, el("span", { class: "swatch", style: "width:18px;height:18px;border-radius:5px" }), result.prediction),
            el("div", { class: "muted" }, `Độ tin cậy ${percent(top)} · ${result.neighbor_votes[result.prediction]}/5 hàng xóm cùng loài`),
          ),
        ),
      ),
      el("section", { class: "card" },
        el("h2", {}, "Độ tin cậy theo hàng xóm"),
        confidenceBars(result.confidence, result.neighbor_votes),
        el("p", { class: "small muted" }, "Mỗi hàng xóm bỏ phiếu với trọng số 1/khoảng cách, nên hàng xóm gần hơn có tiếng nói lớn hơn."),
      ),
    ),
    el("section", { class: "card", style: "margin-top:16px" },
      el("h2", {}, "5 hàng xóm gần nhất"),
      neighborsTable(result.neighbors),
    ),
    el("section", { class: "card", style: "margin-top:16px" },
      el("h2", {}, "Dữ liệu đã nhập"),
      inputSummary(result.input),
    ),
    el("div", { class: "actions" },
      el("a", { class: "button", href: "index.html" }, "Dự đoán con khác"),
      el("a", { class: "button secondary", href: "dashboard.html" }, "Xem dashboard"),
    ),
  );
}

initTooltips();
const result = loadResult();
if (result && result.prediction) render(result);
else renderEmpty();
