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

Yêu cầu Python 3.10+. Dùng chung `.venv` ở gốc repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt -r api/requirements-dev.txt
```

## Train model (offline, chỉ cần khi muốn train lại)

```bash
python ml/src/train_final.py
```

## Chạy web + API

```bash
python -m api.app
```

Mở `http://localhost:8000` (web), `http://localhost:8000/docs` (Swagger / JSON schema).
Biến môi trường: `HOST`, `PORT`, `RELOAD=1`, `MODEL_PATH`, `CORS_ORIGIN`.

## Chạy test

```bash
pytest
```
