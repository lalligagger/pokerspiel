from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .contracts import (
    BulkProbeRequest,
    BulkProbeResponse,
    HealthStatus,
    ProbeRequest,
    ProbeResponse,
    SolverStatusResponse,
)
from .service import SolverService

router = APIRouter()
service = SolverService()


@router.get("/health", response_model=HealthStatus)
def get_health() -> HealthStatus:
    return service.health()


@router.get("/status", response_model=SolverStatusResponse)
def get_status() -> SolverStatusResponse:
    return service.status()


@router.post("/probe", response_model=ProbeResponse)
def request_probe(request: ProbeRequest) -> ProbeResponse:
    response = service.request_probe(request)
    if not response.ready:
        raise HTTPException(status_code=409, detail={"error": response.message or "probe not ready"})
    return response


@router.post("/bulk-probe", response_model=BulkProbeResponse)
def request_bulk_probe(request: BulkProbeRequest) -> BulkProbeResponse:
    return service.request_bulk_probe(request)
