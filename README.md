# Penguin Classifier

Phân loại loài chim cánh cụt (Adelie / Chinstrap / Gentoo) bằng KNN.
Training chạy offline trong `ml/`; web và API chỉ load Pipeline đã lưu để dự đoán.

## Cấu trúc

```text
ml/                  Offline: làm sạch dữ liệu, thí nghiệm, GridSearchCV, train model cuối
  src/train_final.py   → ml/models/penguin_pipeline.joblib (+ .json metadata)
api/                 Online: FastAPI, load pipeline khi khởi động
  app.py             Endpoint, error handler, phục vụ web tĩnh
  schemas.py         JSON schema request/response (Pydantic)
  model_service.py   Load pipeline, predict, tìm 5 hàng xóm, thống kê dashboard
  tests/             API tests
  API.md             Tài liệu API + ví dụ request/response
web/                 3 màn hình HTML/CSS/JS thuần
  index.html         Dự đoán: form nhập + validation
  result.html        Kết quả: loài, confidence, 5 hàng xóm gần nhất
  dashboard.html     Phân bố dữ liệu + thông tin model
```

## Cài đặt

Yêu cầu Python 3.10+ (khi cài trên Windows, hãy tích chọn **"Add python.exe to PATH"**). Dùng chung một virtualenv `.venv` ở gốc repo:

### Trên Windows

- **PowerShell**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -r ml/requirements.txt -r api/requirements-dev.txt
  ```
  > **Lưu ý (PowerShell):** Nếu gặp lỗi `running scripts is disabled on this system`, hãy mở PowerShell và chạy:
  > ```powershell
  > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  > ```
  > Sau đó chạy lại lệnh kích hoạt `.venv\Scripts\Activate.ps1`.

- **Command Prompt (CMD)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  pip install -r ml/requirements.txt -r api/requirements-dev.txt
  ```

### Trên Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt -r api/requirements-dev.txt
```

## Train model (offline, chỉ cần khi muốn train lại)

Đảm bảo đã kích hoạt môi trường ảo `.venv`, sau đó chạy:

```bash
python ml/src/train_final.py
```

## Chạy web + API

Khởi động server (sau khi đã kích hoạt `.venv`):

```bash
python -m api.app
```

Mở trình duyệt:
- Web: `http://localhost:8000`
- Swagger / JSON schema: `http://localhost:8000/docs`

Các biến môi trường: `HOST`, `PORT`, `RELOAD=1`, `MODEL_PATH`, `CORS_ORIGIN`.

**Cách chạy kèm biến môi trường (ví dụ bật auto-reload `RELOAD=1`):**

- **Windows (PowerShell)**:
  ```powershell
  $env:RELOAD="1"; python -m api.app
  ```
- **Windows (CMD)**:
  ```cmd
  set RELOAD=1 && python -m api.app
  ```
- **Linux / macOS**:
  ```bash
  RELOAD=1 python -m api.app
  ```

## Chạy test

Đảm bảo đã kích hoạt `.venv`:

```bash
pytest
```
