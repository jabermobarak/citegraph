"""Local, read-only graph workspace with bounded requests and optional token access."""

import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .service import Workspace

STATIC = Path(__file__).parent / "static"


class StressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str = Field(min_length=1, max_length=20, pattern=r"^\d+$")
    drop_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=2026, ge=0, le=1000000)


def create_app(data_dir: Path | None = None, artifacts: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.workspace = Workspace(
            data_dir or Path(os.getenv("CG_DATA", "data")),
            artifacts or Path(os.getenv("CG_ARTIFACTS", "artifacts")),
        )
        yield

    app = FastAPI(title="CiteGraph", version="0.1.0", lifespan=lifespan)
    token = os.getenv("CG_API_TOKEN", "")
    counters = {"requests": 0, "stress": 0, "elapsed_ms": 0.0}
    counter_lock = Lock()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if token and (request.url.path.startswith("/api/") or request.url.path == "/metrics"):
            value = request.headers.get("Authorization", "")
            if not secrets.compare_digest(value, "Bearer " + token):
                return JSONResponse({"detail": "Bearer token required"}, status_code=401)
        start = time.perf_counter()
        result = await call_next(request)
        if request.url.path.startswith("/api/"):
            with counter_lock:
                counters["requests"] += 1
                counters["elapsed_ms"] += (time.perf_counter() - start) * 1000
                counters["stress"] += int(request.url.path == "/api/stress")
        return result

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ready", "nodes": len(app.state.workspace.graph.ids)}

    @app.get("/api/overview")
    def overview():
        return app.state.workspace.overview()

    @app.get("/api/node/{paper_id}")
    def node(paper_id: str):
        try:
            return app.state.workspace.node(paper_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown paper ID") from exc

    @app.get("/api/neighborhood/{paper_id}")
    def neighborhood(paper_id: str):
        try:
            return app.state.workspace.neighborhood(paper_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown paper ID") from exc

    @app.post("/api/stress")
    def stress(request: StressRequest):
        try:
            return app.state.workspace.stress(request.paper_id, request.drop_rate, request.seed)
        except KeyError as exc:
            raise HTTPException(404, "Unknown paper ID") from exc

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        with counter_lock:
            values = counters.copy()
        return Response(
            f"# TYPE citegraph_requests_total counter\ncitegraph_requests_total {values['requests']}\n"
            f"# TYPE citegraph_stress_total counter\ncitegraph_stress_total {values['stress']}\n"
            f"# TYPE citegraph_api_latency_ms_sum counter\ncitegraph_api_latency_ms_sum {values['elapsed_ms']:.3f}\n",
            media_type="text/plain; version=0.0.4",
        )

    return app


app = create_app()
