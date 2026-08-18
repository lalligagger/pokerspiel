from __future__ import annotations

from typing import List

from .contracts import (
    BulkProbeRequest,
    BulkProbeResponse,
    HealthStatus,
    ProbeRequest,
    ProbeResponse,
    SolverStatusResponse,
    StabilitySummary,
)


class SolverService:
    """Thin adapter layer for a long-running solver.

    This is intentionally a placeholder. The real implementation would connect to the
    active solver state later without changing the existing training code.
    """

    def __init__(self, *, iteration: int = 0, stable: bool = False):
        self.iteration = iteration
        self.stable = stable

    def health(self) -> HealthStatus:
        return HealthStatus(
            status="running",
            iteration=self.iteration,
            stable=self.stable,
            ready_for_queries=True,
            message="solver is running; live probing is enabled",
        )

    def status(self) -> SolverStatusResponse:
        return SolverStatusResponse(
            solver="hulh",
            iteration=self.iteration,
            stable=self.stable,
            stability=StabilitySummary(
                passed=self.stable,
                threshold=0.01,
                avg_abs_delta=0.0,
                max_abs_delta=0.0,
                consecutive_passes=0,
            ),
            ready_for_queries=True,
            last_probe_at=None,
            min_iteration=200000,
            probe_budget_remaining=None,
        )

    def request_probe(self, request: ProbeRequest) -> ProbeResponse:
        if request.min_iteration is not None and self.iteration < request.min_iteration:
            return ProbeResponse(
                iteration=self.iteration,
                node=request.node,
                display_name=request.node,
                history=request.history,
                sample_count=0,
                action_frequencies={},
                hands=[],
                ready=False,
                message=f"solver iteration {self.iteration} is below min_iteration {request.min_iteration}",
            )

        return ProbeResponse(
            iteration=self.iteration,
            node=request.node,
            display_name=request.node,
            history=request.history,
            sample_count=request.samples,
            action_frequencies={
                "fold": 0.15,
                "check_call": 0.35,
                "bet_raise": 0.50,
            },
            hands=[],
            stability=StabilitySummary(
                passed=self.stable,
                threshold=0.01,
                avg_abs_delta=0.0,
                max_abs_delta=0.0,
                consecutive_passes=0,
            ),
            ready=True,
            message="placeholder probe response; replace with live solver adapter later",
        )

    def request_bulk_probe(self, request: BulkProbeRequest) -> BulkProbeResponse:
        results = [self.request_probe(item) for item in request.requests]
        failed: List[str] = []
        for item in results:
            if not item.ready:
                failed.append(item.node)
        return BulkProbeResponse(results=results, failed=failed)
