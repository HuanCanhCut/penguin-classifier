const API_URL = "http://localhost:5000/api/health";

const button = document.querySelector("#check-health");
const result = document.querySelector("#result");

button.addEventListener("click", async () => {
  button.disabled = true;
  button.textContent = "Đang kiểm tra...";
  result.className = "result";
  result.textContent = "Đang kết nối tới API...";

  try {
    const response = await fetch(API_URL);
    if (!response.ok) {
      throw new Error(`API trả về HTTP ${response.status}`);
    }

    const data = await response.json();
    result.classList.add("success");
    result.textContent = `API hoạt động — ${data.service} (${data.status})`;
  } catch (error) {
    result.classList.add("error");
    result.textContent = `Không kết nối được API: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Check API health";
  }
});

