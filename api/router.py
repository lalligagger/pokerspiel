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


@router.get(
    "/status",
    response_model=SolverStatusResponse,
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {
                            "solver": "external",
                            "iteration": 125000,
                            "stable": True,
                            "ready_for_queries": True,
                            "min_iteration": 100000,
                            "probe_budget_remaining": 1326,
                            "selected_node_summary": [
                                {"node_name": "first_to_act", "label": "first_to_act", "history": []}
                            ],
                        }
                    }
                }
            }
        }
    },
)
def get_status() -> SolverStatusResponse:
    return service.status()


@router.post(
    "/probe",
    response_model=ProbeResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "first_to_act": {
                            "summary": "first-to-act probe",
                            "value": {
                                "node": "first_to_act",
                                "history": [],
                                "samples": 1326,
                                "min_iteration": 0,
                                "include_stability": True,
                                "include_hands": True,
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {
                            "iteration": 125000,
                            "node": "first_to_act",
                            "display_name": "first_to_act",
                            "history": [],
                            "sample_count": 1326,
                            "action_frequencies": {"fold": 0.12, "check_call": 0.45, "bet_raise": 0.43},
                            "hands": [{"hand": "TT", "policy": {"fold": 0.08, "check_call": 0.22, "bet_raise": 0.7}}],
                            "ready": True,
                        }
                    }
                }
            }
        },
    },
)
def request_probe(request: ProbeRequest) -> ProbeResponse:
    response = service.request_probe(request)
    if not response.ready:
        raise HTTPException(status_code=409, detail={"error": response.message or "probe not ready"})
    return response


@router.post("/bulk-probe", response_model=BulkProbeResponse)
def request_bulk_probe(request: BulkProbeRequest) -> BulkProbeResponse:
    return service.request_bulk_probe(request)


@router.get("/preflop/{spot}/range", response_model=PreflopRangeResponse)
def get_preflop_range(spot: str) -> PreflopRangeResponse:
    return service.get_preflop_range(spot=spot)


@router.get(
    "/preflop/{spot}/{hand}",
    response_model=SpotFrequencyResponse,
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {
                            "spot": "response_to_open",
                            "hand": "TT",
                            "iteration": 125000,
                            "frequencies": {"fold": 0.1, "check_call": 0.2, "bet_raise": 0.7},
                            "ready": True,
                        }
                    }
                }
            }
        }
    },
)
def get_preflop_spot(spot: str, hand: str) -> SpotFrequencyResponse:
    return service.get_preflop_spot(spot=spot, hand=hand)


@router.get("/preflop/open", response_model=SpotFrequencyResponse)
def get_preflop_open_spot(hand: str) -> SpotFrequencyResponse:
    return service.get_preflop_spot(spot="open", hand=hand)


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
