from __future__ import annotations

from fastapi import FastAPI

from .router import router

app = FastAPI(
    title="Freerunning Solver Live Probe API",
    version="0.1.0",
    description=(
        "Schema-first API for a long-running solver that answers selected-node range "
        "queries on demand without changing the existing solver code path."
    ),
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
