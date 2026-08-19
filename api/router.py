from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .contracts import (
    BulkProbeRequest,
    BulkProbeResponse,
    HealthStatus,
    PostflopExactRequest,
    PostflopExactResponse,
    PostflopRangeRequest,
    PostflopRangeResponse,
    PreflopRangeResponse,
    ProbeRequest,
    ProbeResponse,
    SolverStatusResponse,
    SpotFrequencyResponse,
)
from .service import service

router = APIRouter()


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


@router.get("/preflop/{spot}/{hand}", response_model=SpotFrequencyResponse)
def get_preflop_spot(spot: str, hand: str) -> SpotFrequencyResponse:
    return service.get_preflop_spot(spot=spot, hand=hand)


@router.get("/preflop/open", response_model=SpotFrequencyResponse)
def get_preflop_open_spot(hand: str) -> SpotFrequencyResponse:
    return service.get_preflop_spot(spot="open", hand=hand)


@router.get("/preflop/{spot}/range", response_model=PreflopRangeResponse)
def get_preflop_range(spot: str) -> PreflopRangeResponse:
    return service.get_preflop_range(spot=spot)


@router.post("/postflop/exact", response_model=PostflopExactResponse)
def request_postflop_exact(request: PostflopExactRequest) -> PostflopExactResponse:
    response = service.request_postflop_exact(request)
    if not response.ready:
        raise HTTPException(status_code=409, detail={"error": response.message or "postflop exact lookup not ready"})
    return response


@router.post("/postflop/range", response_model=PostflopRangeResponse)
def request_postflop_range(request: PostflopRangeRequest) -> PostflopRangeResponse:
    response = service.request_postflop_range(request)
    if not response.ready:
        raise HTTPException(status_code=409, detail={"error": response.message or "postflop range estimate not ready"})
    return response
