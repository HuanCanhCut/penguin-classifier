# Penguin Classifier API

FastAPI phục vụ Pipeline KNN đã train **offline** (`ml/src/train_final.py` →
`ml/models/penguin_pipeline.joblib`). API chỉ `joblib.load` pipeline khi khởi động và gọi
`predict_proba` / `kneighbors`, không bao giờ gọi `fit()` (có test kiểm tra điều này).

```text
                OFFLINE (ml/)                                  ONLINE (api/ + web/)
train.csv → tune (GridSearchCV) → train_final.py      User → web form → POST /api/predict
                                        │                                   │
                                        ▼                                   ▼
                          penguin_pipeline.joblib  ───── load khi khởi động ─→ Pipeline.predict
```

JSON schema đầy đủ và thử trực tiếp: `http://localhost:8000/docs` (Swagger) hoặc
`/openapi.json`.

## Endpoint

| Method | Path | Mô tả |
|---|---|---|
| POST | `/api/predict` | Dự đoán loài + confidence + 5 hàng xóm gần nhất |
| GET | `/api/model-info` | Thông tin model, giới hạn input, metrics trên tập test |
| GET | `/api/stats` | Phân bố dữ liệu cho dashboard |
| GET | `/api/health` | Trạng thái server và model |

## POST /api/predict

### Request

Model cuối dùng 4 số đo + `sex` + `island`, nên cả 6 trường đều bắt buộc.

| Trường | Kiểu | Cho phép (ngoài khoảng → 422) | Thường gặp (ngoài khoảng → cảnh báo) |
|---|---|---|---|
| `bill_length_mm` | number | 20 – 80 | 32 – 60 |
| `bill_depth_mm` | number | 8 – 30 | 13 – 21.5 |
| `flipper_length_mm` | number | 140 – 260 | 170 – 232 |
| `body_mass_g` | number | 2000 – 7500 | 2700 – 6300 |
| `sex` | `"female"` \| `"male"` | | |
| `island` | `"Biscoe"` \| `"Dream"` \| `"Torgersen"` | | |

Số phải là JSON number (chuỗi `"45.2"` hoặc `true` bị từ chối); trường thừa bị từ chối.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181,
       "body_mass_g": 3750, "sex": "male", "island": "Torgersen"}'
```

### Response 200

```json
{
  "prediction": "Adelie",
  "confidence": { "Adelie": 1.0, "Chinstrap": 0.0, "Gentoo": 0.0 },
  "neighbor_votes": { "Adelie": 5, "Chinstrap": 0, "Gentoo": 0 },
  "neighbors": [
    {
      "rank": 1,
      "species": "Adelie",
      "distance": 1.0895,
      "features": {
        "bill_length_mm": 39.7, "bill_depth_mm": 18.4, "flipper_length_mm": 190.0,
        "body_mass_g": 3900.0, "sex": "male", "island": "Torgersen"
      }
    }
  ],
  "warnings": [],
  "input": {
    "bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0, "sex": "male", "island": "Torgersen"
  }
}
```

(`neighbors` luôn có 5 phần tử, sắp theo khoảng cách tăng dần; ví dụ trên rút gọn còn 1.)

- `confidence`: xác suất theo hàng xóm của KNN (`weights="distance"`: mỗi hàng xóm bỏ phiếu
  với trọng số 1/khoảng cách). Tổng = 1.
- `neighbor_votes`: đếm số hàng xóm thuộc từng loài (không trọng số).
- `distance`: khoảng cách Manhattan trong không gian đã chuẩn hoá (StandardScaler + one-hot).
- `warnings`: các giá trị hợp lệ nhưng ngoài khoảng thường gặp, ví dụ
  `{"field": "body_mass_g", "message": "6800 nằm ngoài khoảng thường gặp [2700, 6300], ..."}`.

### Error response

Mọi lỗi dùng chung một dạng:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Dữ liệu đầu vào không hợp lệ",
    "details": [
      { "field": "bill_length_mm", "message": "Input should be a valid number" },
      { "field": "body_mass_g", "message": "Input should be less than or equal to 7500" },
      { "field": "island", "message": "Field required" }
    ]
  }
}
```

| HTTP | `code` | Khi nào |
|---|---|---|
| 422 | `validation_error` | Thiếu trường, sai kiểu, ngoài khoảng, category lạ, JSON hỏng |
| 404 | `not_found` | Endpoint không tồn tại |
| 503 | `model_unavailable` | Không tìm thấy file pipeline khi khởi động |
| 500 | `internal_error` | Lỗi không mong muốn |

## Test

```bash
.venv/bin/python -m pytest
```

`api/tests/test_api.py` kiểm tra: load đúng pipeline đã lưu, API không gọi `fit()`, dạng response,
dự đoán đúng 3 mẫu điển hình, cảnh báo giá trị bất thường, các lỗi validation (thiếu trường, sai
kiểu, ngoài khoảng, category lạ, trường thừa, JSON hỏng), 404, 503, `/api/model-info`,
`/api/stats`, OpenAPI có example, và web được phục vụ.
