# Penguin Classifier

Dự án khởi tạo gồm Flask API và giao diện web thuần HTML/CSS/JavaScript.

## Cấu trúc

```text
api/                 Flask backend
  app.py             App factory và health endpoint
  requirements.txt   Dependency chạy ứng dụng
  requirements-dev.txt
  tests/             API tests
web/                 Static frontend
```

## Chạy backend

Yêu cầu Python 3.10 trở lên.

```bash
python3 -m venv api/.venv
source api/.venv/bin/activate
pip install -r api/requirements-dev.txt
python3 -m api.app
```

API mặc định chạy tại `http://localhost:5000`. Kiểm tra trực tiếp:

```bash
curl http://localhost:5000/api/health
```

Có thể cấu hình bằng các biến môi trường `HOST`, `PORT`, `FLASK_DEBUG` và
`CORS_ORIGIN`.

## Chạy frontend

Mở terminal khác tại thư mục gốc dự án:

```bash
python3 -m http.server 8000 --directory web
```

Truy cập `http://localhost:8000`, sau đó nhấn **Check API health**.

## Chạy test

```bash
pytest
```
