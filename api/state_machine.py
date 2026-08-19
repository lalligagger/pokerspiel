from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SolverState(str, Enum):
    TRAINING = "training"
    SCORING = "scoring"
    AVAILABLE = "available"
    RUNNING = "running"
    STABLE = "stable"
    QUERYABLE = "queryable"
    PAUSED = "paused"
    STOPPED = "stopped"
    RECOVERING = "recovering"
    ERROR = "error"


@dataclass
class SolverSnapshot:
    """The latest durable snapshot that can be written to disk."""

    iteration: int
    state: SolverState
    timestamp: float
    summary: Dict[str, Any] = field(default_factory=dict)
    policy_snapshot: Optional[Dict[str, Any]] = None


@dataclass
class SolverRuntimeState:
    """In-memory state for a long-lived solver service."""

    state: SolverState = SolverState.RUNNING
    iteration: int = 0
    stable: bool = False
    ready_for_queries: bool = False
    current_average_policy: Optional[Dict[str, Any]] = None
    latest_stable_snapshot: Optional[SolverSnapshot] = None
    latest_checkpoint_on_disk: Optional[str] = None
    last_probe_at: Optional[int] = None
    last_stability_check: Optional[Dict[str, Any]] = None

    def mark_stable(self, iteration: int, summary: Dict[str, Any]) -> None:
        self.state = SolverState.STABLE
        self.stable = True
        self.ready_for_queries = True
        self.iteration = max(self.iteration, iteration)
        self.last_stability_check = summary
        self.latest_stable_snapshot = SolverSnapshot(
            iteration=iteration,
            state=SolverState.STABLE,
            timestamp=summary.get("timestamp", 0.0),
            summary=summary,
            policy_snapshot=self.current_average_policy,
        )

    def mark_queryable(self) -> None:
        self.state = SolverState.QUERYABLE
        self.ready_for_queries = True

    def mark_paused(self) -> None:
        self.state = SolverState.PAUSED
        self.ready_for_queries = False

    def mark_stopped(self) -> None:
        self.state = SolverState.STOPPED
        self.ready_for_queries = False

    def mark_recovering(self) -> None:
        self.state = SolverState.RECOVERING
        self.ready_for_queries = False

    def mark_error(self, message: Optional[str] = None) -> None:
        self.state = SolverState.ERROR
        self.ready_for_queries = False
        self.last_stability_check = {"error": message or "solver error"}

    def should_persist_snapshot(self) -> bool:
        return (
            self.latest_stable_snapshot is not None
            or self.state in {SolverState.PAUSED, SolverState.STOPPED, SolverState.RECOVERING}
        )
