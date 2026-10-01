const NUMERIC_FIELDS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"];
const CATEGORICAL_FIELDS = { sex: ["female", "male"], island: ["Biscoe", "Dream", "Torgersen"] };
const EXAMPLE = {
  bill_length_mm: 45.2, bill_depth_mm: 17.1, flipper_length_mm: 190, body_mass_g: 3650,
  sex: "female", island: "Dream",
};

// Giá trị dự phòng, sẽ được ghi đè bằng /api/model-info để web và API luôn dùng cùng giới hạn
let hardLimits = {
  bill_length_mm: { min: 20, max: 80 },
  bill_depth_mm: { min: 8, max: 30 },
  flipper_length_mm: { min: 140, max: 260 },
  body_mass_g: { min: 2000, max: 7500 },
};
let typicalRanges = {
  bill_length_mm: { min: 32, max: 60 },
  bill_depth_mm: { min: 13, max: 21.5 },
  flipper_length_mm: { min: 170, max: 232 },
  body_mass_g: { min: 2700, max: 6300 },
};

const form = document.querySelector("#predict-form");
const submitButton = document.querySelector("#submit");
const alertBox = document.querySelector("#form-alert");
const warningBox = document.querySelector("#form-warning");
let confirmedWarnings = null;

function fieldBox(name) {
  return form.querySelector(`[data-field="${name}"]`);
}

function setFieldState(name, state, message = "") {
  const box = fieldBox(name);
  box.classList.toggle("has-error", state === "error");
  box.classList.toggle("has-warning", state === "warning");
  box.querySelector(".message").textContent = message;
  box.querySelector("input, select").setAttribute("aria-invalid", state === "error");
}

// Trả về { value, error, warning } cho một trường
function checkField(name) {
  const input = form.elements[name];
  const raw = input.value.trim();
  const label = FIELD_LABELS[name];

  if (name in CATEGORICAL_FIELDS) {
    if (!raw) return { error: `Thiếu dữ liệu: chọn ${label.toLowerCase()}` };
    if (!CATEGORICAL_FIELDS[name].includes(raw)) return { error: `${label} không hợp lệ` };
    return { value: raw };
  }

  // input[type=number] trả value rỗng khi nội dung không phải số → dùng validity để phân biệt
  if (input.validity.badInput) return { error: `${label} phải là một số` };
  if (!raw) return { error: `Thiếu dữ liệu: nhập ${label.toLowerCase()}` };
  const value = Number(raw);
  if (!Number.isFinite(value)) return { error: `${label} phải là một số` };

  const hard = hardLimits[name];
  if (value < hard.min || value > hard.max) {
    return { error: `Ngoài khoảng cho phép [${hard.min}, ${hard.max}] — kiểm tra lại đơn vị` };
  }
  const typical = typicalRanges[name];
  if (value < typical.min || value > typical.max) {
    return {
      value,
      warning: `Bất thường: dữ liệu train nằm trong [${typical.min}, ${typical.max}]`,
    };
  }
  return { value };
}

function validateAll() {
  const payload = {};
  const errors = [];
  const warnings = [];
  for (const name of [...NUMERIC_FIELDS, ...Object.keys(CATEGORICAL_FIELDS)]) {
    const result = checkField(name);
    if (result.error) {
      errors.push(`${FIELD_LABELS[name]}: ${result.error}`);
      setFieldState(name, "error", result.error);
    } else if (result.warning) {
      warnings.push(`${FIELD_LABELS[name]} = ${result.value}: ${result.warning}`);
      setFieldState(name, "warning", result.warning);
    } else {
      setFieldState(name, "ok");
    }
    payload[name] = result.value;
  }
  return { payload, errors, warnings };
}

function showList(box, title, items) {
  box.replaceChildren();
  if (!items.length) return;
  if (title) box.append(el("strong", {}, title));
  box.append(el("ul", {}, ...items.map((item) => el("li", {}, item))));
}

for (const name of [...NUMERIC_FIELDS, ...Object.keys(CATEGORICAL_FIELDS)]) {
  const input = form.elements[name];
  const recheck = () => {
    const result = checkField(name);
    if (result.error) setFieldState(name, "error", result.error);
    else if (result.warning) setFieldState(name, "warning", result.warning);
    else setFieldState(name, "ok");
    confirmedWarnings = null;
  };
  input.addEventListener("blur", recheck);
  input.addEventListener("change", recheck);
  input.addEventListener("input", () => {
    if (fieldBox(name).classList.contains("has-error")) recheck();
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const { payload, errors, warnings } = validateAll();

  showList(alertBox, "Vui lòng sửa các lỗi sau:", errors);
  if (errors.length) {
    showList(warningBox, "", []);
    form.querySelector(".has-error input, .has-error select")?.focus();
    return;
  }

  // Có giá trị bất thường: lần bấm đầu chỉ cảnh báo, bấm lần nữa mới gửi
  const warningKey = warnings.join("|");
  if (warnings.length && confirmedWarnings !== warningKey) {
    showList(warningBox, "Có giá trị nằm ngoài khoảng hợp lý. Nhấn “Dự đoán” lần nữa nếu chắc chắn:", warnings);
    confirmedWarnings = warningKey;
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Đang dự đoán...";
  try {
    const result = await apiFetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    sessionStorage.setItem(RESULT_KEY, JSON.stringify(result));
    location.href = "result.html";
  } catch (error) {
    const details = (error.details || []).map((d) => (d.field ? `${FIELD_LABELS[d.field] || d.field}: ${d.message}` : d.message));
    if (details.length) showList(alertBox, error.message, details);
    else showList(alertBox, "", [error.message]);
    for (const detail of error.details || []) {
      if (detail.field && fieldBox(detail.field)) setFieldState(detail.field, "error", detail.message);
    }
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Dự đoán";
  }
});

form.addEventListener("reset", () => {
  setTimeout(() => {
    for (const name of [...NUMERIC_FIELDS, ...Object.keys(CATEGORICAL_FIELDS)]) setFieldState(name, "ok");
    showList(alertBox, "", []);
    showList(warningBox, "", []);
    confirmedWarnings = null;
  });
});

document.querySelector("#fill-example").addEventListener("click", () => {
  for (const [name, value] of Object.entries(EXAMPLE)) form.elements[name].value = value;
  validateAll();
  showList(alertBox, "", []);
  showList(warningBox, "", []);
});

function renderHints() {
  for (const name of NUMERIC_FIELDS) {
    const { min, max } = typicalRanges[name];
    fieldBox(name).querySelector("[data-hint]").textContent = `Thường gặp: ${min} – ${max}`;
    const input = form.elements[name];
    input.min = hardLimits[name].min;
    input.max = hardLimits[name].max;
  }
}

apiFetch("/api/model-info")
  .then((info) => {
    hardLimits = info.hard_limits;
    typicalRanges = info.typical_ranges;
  })
  .catch(() => {})
  .finally(renderHints);
