import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import model_service
from api.model_service import ModelService
from api.schemas import HARD_LIMITS, ErrorResponse, PenguinInput, PredictionResponse

WEB_DIR = model_service.ROOT_DIR / "web"

ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Dữ liệu đầu vào không hợp lệ"},
    503: {"model": ErrorResponse, "description": "Model chưa được load"},
}


def error_response(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details or []}}
    return JSONResponse(status_code=status_code, content=body)


def create_app(service: ModelService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load Pipeline một lần khi khởi động; nếu thiếu file thì API vẫn chạy và trả 503
        if app.state.model is None:
            try:
                app.state.model = ModelService.load()
            except FileNotFoundError as exc:
                app.state.model_error = str(exc)
        yield

    app = FastAPI(
        title="Penguin Classifier API",
        version="1.0.0",
        description="Dự đoán loài chim cánh cụt bằng KNN Pipeline đã train offline.",
        lifespan=lifespan,
    )
    app.state.model = service
    app.state.model_error = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGIN", "http://localhost:8000").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            # loc có dạng ("body", "bill_length_mm"); lỗi JSON hỏng thì chỉ có ("body", vị trí)
            field = ".".join(str(part) for part in err["loc"][1:]) or None
            details.append({"field": field, "message": err["msg"]})
        return error_response(422, "validation_error", "Dữ liệu đầu vào không hợp lệ", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        return error_response(500, "internal_error", "Lỗi không mong muốn ở server")

    def get_model() -> ModelService | None:
        return app.state.model

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "service": "penguin-classifier-api",
            "model_loaded": get_model() is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/api/predict", response_model=PredictionResponse, responses=ERROR_RESPONSES)
    def predict(payload: PenguinInput):
        model = get_model()
        if model is None:
            return error_response(503, "model_unavailable", "Model chưa được load", [
                {"field": None, "message": app.state.model_error or "unknown"}
            ])
        result = model.predict(payload.model_dump())
        return {**result, "input": payload}

    @app.get("/api/model-info", responses={503: ERROR_RESPONSES[503]})
    def model_info():
        model = get_model()
        if model is None:
            return error_response(503, "model_unavailable", "Model chưa được load")
        return {
            **model.info,
            "hard_limits": {k: {"min": lo, "max": hi} for k, (lo, hi) in HARD_LIMITS.items()},
            "typical_ranges": {
                k: {"min": lo, "max": hi} for k, (lo, hi) in model_service.TYPICAL_RANGES.items()
            },
            "metrics": model_service.model_metrics(),
        }

    @app.get("/api/stats")
    def stats():
        return model_service.dataset_stats()

    # Phục vụ luôn web tĩnh để chỉ cần chạy một server; mount sau cùng để không che /api/*
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
