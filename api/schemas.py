"""JSON schema cho request/response của API (Pydantic sinh schema, hiển thị ở /docs)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Species = Literal["Adelie", "Chinstrap", "Gentoo"]

# Giới hạn cứng: ngoài khoảng này chắc chắn là nhập sai (sai đơn vị, gõ nhầm) → trả 422.
# Khoảng "hợp lý" hẹp hơn (theo dữ liệu train) nằm ở model_service.TYPICAL_RANGES → chỉ cảnh báo.
HARD_LIMITS = {
    "bill_length_mm": (20.0, 80.0),
    "bill_depth_mm": (8.0, 30.0),
    "flipper_length_mm": (140.0, 260.0),
    "body_mass_g": (2000.0, 7500.0),
}

EXAMPLE_REQUEST = {
    "bill_length_mm": 45.2,
    "bill_depth_mm": 17.1,
    "flipper_length_mm": 190,
    "body_mass_g": 3650,
    "sex": "female",
    "island": "Dream",
}


def _measure(name: str, description: str):
    low, high = HARD_LIMITS[name]
    # strict=True: không tự ép chuỗi "45.2" hay true thành số
    return Field(..., ge=low, le=high, strict=True, description=description)


class PenguinInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    bill_length_mm: float = _measure("bill_length_mm", "Chiều dài mỏ (mm)")
    bill_depth_mm: float = _measure("bill_depth_mm", "Độ dày mỏ (mm)")
    flipper_length_mm: float = _measure("flipper_length_mm", "Chiều dài vây (mm)")
    body_mass_g: float = _measure("body_mass_g", "Cân nặng (g)")
    sex: Literal["female", "male"] = Field(..., description="Giới tính")
    island: Literal["Biscoe", "Dream", "Torgersen"] = Field(..., description="Đảo nơi quan sát")


class Neighbor(BaseModel):
    rank: int = Field(..., description="Thứ tự, 1 = gần nhất")
    species: Species
    distance: float = Field(..., description="Khoảng cách Manhattan trong không gian đã chuẩn hoá")
    features: dict[str, float | str] | None = Field(
        None, description="Số đo gốc của con chim trong tập train (nếu có)"
    )


class InputWarning(BaseModel):
    field: str
    message: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prediction": "Adelie",
                    "confidence": {"Adelie": 0.8, "Chinstrap": 0.0, "Gentoo": 0.2},
                    "neighbor_votes": {"Adelie": 4, "Chinstrap": 0, "Gentoo": 1},
                    "neighbors": [
                        {"rank": 1, "species": "Adelie", "distance": 0.42, "features": None}
                    ],
                    "warnings": [],
                    "input": EXAMPLE_REQUEST,
                }
            ]
        }
    )

    prediction: Species
    confidence: dict[str, float] = Field(
        ..., description="Xác suất theo hàng xóm (có trọng số 1/khoảng cách), tổng = 1"
    )
    neighbor_votes: dict[str, int] = Field(..., description="Số hàng xóm thuộc mỗi loài")
    neighbors: list[Neighbor]
    warnings: list[InputWarning] = Field(
        default_factory=list, description="Giá trị hợp lệ nhưng nằm ngoài khoảng thường gặp"
    )
    input: PenguinInput


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "validation_error",
                        "message": "Dữ liệu đầu vào không hợp lệ",
                        "details": [
                            {"field": "body_mass_g", "message": "Input should be less than or equal to 7500"}
                        ],
                    }
                }
            ]
        }
    )

    error: ErrorBody
