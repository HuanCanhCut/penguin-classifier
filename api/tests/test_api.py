import math

import pytest
from fastapi.testclient import TestClient
from sklearn.neighbors import KNeighborsClassifier

from api.app import create_app
from api.model_service import MODEL_PATH, ModelService

VALID = {
    "bill_length_mm": 45.2,
    "bill_depth_mm": 17.1,
    "flipper_length_mm": 190,
    "body_mass_g": 3650,
    "sex": "female",
    "island": "Dream",
}
SPECIES = {"Adelie", "Chinstrap", "Gentoo"}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def predict(client, **overrides):
    payload = {**VALID, **overrides}
    payload = {k: v for k, v in payload.items() if v is not ...}
    return client.post("/api/predict", json=payload)


def assert_validation_error(response, field):
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert field in [d["field"] for d in body["error"]["details"]]


# ---------- Model được load từ Pipeline đã lưu ----------

def test_model_is_loaded_from_saved_pipeline(client):
    model = client.app.state.model
    assert isinstance(model, ModelService)
    assert MODEL_PATH.exists()
    assert isinstance(model.pipeline.named_steps["knn"], KNeighborsClassifier)


def test_api_never_trains(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("API không được gọi fit()")

    monkeypatch.setattr(KNeighborsClassifier, "fit", forbidden)
    with TestClient(create_app()) as c:
        assert predict(c).status_code == 200


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------- Predict thành công ----------

def test_predict_response_shape(client):
    response = predict(client)
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] in SPECIES
    assert set(body["confidence"]) == SPECIES
    assert math.isclose(sum(body["confidence"].values()), 1.0, abs_tol=1e-3)
    assert body["prediction"] == max(body["confidence"], key=body["confidence"].get)

    neighbors = body["neighbors"]
    assert len(neighbors) == 5
    assert [n["rank"] for n in neighbors] == [1, 2, 3, 4, 5]
    distances = [n["distance"] for n in neighbors]
    assert distances == sorted(distances)
    assert all(n["species"] in SPECIES for n in neighbors)
    assert sum(body["neighbor_votes"].values()) == 5
    assert body["warnings"] == []
    assert body["input"] == VALID


@pytest.mark.parametrize(
    "sample, expected",
    [
        ({"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750,
          "sex": "male", "island": "Torgersen"}, "Adelie"),
        ({"bill_length_mm": 46.5, "bill_depth_mm": 17.9, "flipper_length_mm": 192, "body_mass_g": 3500,
          "sex": "female", "island": "Dream"}, "Chinstrap"),
        ({"bill_length_mm": 46.1, "bill_depth_mm": 13.2, "flipper_length_mm": 211, "body_mass_g": 4500,
          "sex": "female", "island": "Biscoe"}, "Gentoo"),
    ],
)
def test_predict_typical_penguins(client, sample, expected):
    assert client.post("/api/predict", json=sample).json()["prediction"] == expected


def test_integer_values_are_accepted(client):
    assert predict(client, bill_length_mm=45, body_mass_g=3650).status_code == 200


def test_unusual_value_returns_warning(client):
    response = predict(client, body_mass_g=7000)
    assert response.status_code == 200
    assert [w["field"] for w in response.json()["warnings"]] == ["body_mass_g"]


# ---------- Validation / error response ----------

@pytest.mark.parametrize("field", list(VALID))
def test_missing_field(client, field):
    assert_validation_error(predict(client, **{field: ...}), field)


@pytest.mark.parametrize("value", ["45.2", "abc", True, None, [45.2]])
def test_wrong_type(client, value):
    assert_validation_error(predict(client, bill_length_mm=value), "bill_length_mm")


@pytest.mark.parametrize(
    "field, value",
    [("bill_length_mm", 5), ("bill_depth_mm", 100), ("flipper_length_mm", -1), ("body_mass_g", 36500)],
)
def test_out_of_range(client, field, value):
    assert_validation_error(predict(client, **{field: value}), field)


@pytest.mark.parametrize("field, value", [("sex", "unknown"), ("island", "Hawaii"), ("sex", "Male")])
def test_invalid_category(client, field, value):
    assert_validation_error(predict(client, **{field: value}), field)


def test_extra_field_rejected(client):
    assert_validation_error(predict(client, species="Adelie"), "species")


def test_malformed_json(client):
    response = client.post("/api/predict", content="{bad json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unknown_endpoint_uses_error_format(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_model_unavailable_returns_503(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("penguin_pipeline.joblib")

    monkeypatch.setattr(ModelService, "load", classmethod(missing))
    with TestClient(create_app()) as c:
        response = predict(c)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


# ---------- Endpoint cho dashboard ----------

def test_model_info(client):
    body = client.get("/api/model-info").json()
    assert body["knn"]["n_neighbors"] == 5
    assert set(body["classes"]) == SPECIES
    assert set(body["hard_limits"]) == set(body["typical_ranges"])


def test_stats(client):
    body = client.get("/api/stats").json()
    assert set(body["species_distribution"]) == SPECIES
    assert sum(body["species_distribution"].values()) == body["n_samples"]
    assert len(body["points"]) == body["n_samples"]


def test_openapi_schema_has_examples(client):
    schema = client.get("/openapi.json").json()
    assert "/api/predict" in schema["paths"]
    assert schema["components"]["schemas"]["PenguinInput"]["examples"]


def test_web_pages_are_served(client):
    for page in ["/", "/result.html", "/dashboard.html"]:
        assert client.get(page).status_code == 200
