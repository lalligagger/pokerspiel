from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI

from .router import router
from .service import service


app = FastAPI(
    title="Freerunning Solver Live Probe API",
    version="0.1.0",
    description=(
        "Schema-first API for a long-running solver that answers selected-node range "
        "queries on demand without changing the existing solver code path."
    ),
)


@app.get("/openapi.json", include_in_schema=False)
def custom_openapi() -> Dict[str, Any]:
    if app.openapi_schema is None:
        app.openapi_schema = app.openapi()
    return app.openapi_schema


@app.get("/summary")
def summary() -> Dict[str, Any]:
    status = service.status()
    return {
        "solver": status.solver,
        "iteration": status.iteration,
        "stable": status.stable,
        "ready_for_queries": status.ready_for_queries,
        "status": "ok",
        "summary_url": "http://0.0.0.0:8080/summary",
        "api_docs": "http://0.0.0.0:8080/docs",
        "dashboard": "http://0.0.0.0:8765",
    }


@app.on_event("startup")
def startup_event() -> None:
    service.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.stop()


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
