// Web được FastAPI phục vụ cùng origin; nếu mở từ server tĩnh khác thì gọi API ở cổng 8000
const API_BASE = location.port === "8000" ? "" : "http://localhost:8000";
const SPECIES = ["Adelie", "Chinstrap", "Gentoo"];
const RESULT_KEY = "penguin:lastResult";

const FIELD_LABELS = {
  bill_length_mm: "Chiều dài mỏ",
  bill_depth_mm: "Độ dày mỏ",
  flipper_length_mm: "Chiều dài vây",
  body_mass_g: "Cân nặng",
  sex: "Giới tính",
  island: "Đảo",
};

async function apiFetch(path, options = {}) {
  let response;
  try {
    response = await fetch(API_BASE + path, options);
  } catch {
    throw new Error("Không kết nối được API. Server FastAPI đã chạy chưa?");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(data?.error?.message || `API trả về HTTP ${response.status}`);
    error.details = data?.error?.details || [];
    throw error;
  }
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

function speciesLabel(name) {
  return el("span", { class: `sp-${name}` }, el("span", { class: "swatch" }), name);
}

function percent(value, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`;
}

// Một tooltip dùng chung cho mọi phần tử có data-tip
function initTooltips() {
  const tip = el("div", { class: "tooltip", role: "tooltip" });
  document.body.append(tip);
  document.addEventListener("pointerover", (event) => {
    const target = event.target.closest("[data-tip]");
    if (!target) return;
    tip.textContent = target.dataset.tip;
    tip.style.display = "block";
  });
  document.addEventListener("pointermove", (event) => {
    if (tip.style.display !== "block") return;
    const x = Math.min(event.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${event.clientY + 14}px`;
  });
  document.addEventListener("pointerout", (event) => {
    if (event.target.closest("[data-tip]")) tip.style.display = "none";
  });
}
