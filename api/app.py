import os
from datetime import datetime, timezone

from flask import Flask, jsonify


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        """Allow the local static web app to call the API during development."""
        response.headers["Access-Control-Allow-Origin"] = os.getenv(
            "CORS_ORIGIN", "http://localhost:5500"
        )
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return response

    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok",
            service="penguin-classifier-api",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )

