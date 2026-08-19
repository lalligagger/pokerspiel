from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class StabilitySummary:
    """Compact convergence metadata for a node or a solver snapshot."""

    passed: bool
    max_abs_delta: Optional[float] = None
    avg_abs_delta: Optional[float] = None
    threshold: Optional[float] = None
    consecutive_passes: Optional[int] = None
    matched_nodes: Optional[int] = None
    top_moving: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HealthStatus:
    """Minimal liveness signal for a long-running solver process."""

    status: Literal["running", "idle", "stopped", "error"]
    iteration: Optional[int] = None
    stable: Optional[bool] = None
    last_probe_at: Optional[int] = None
    ready_for_queries: bool = False
    message: Optional[str] = None


@dataclass
class ProbeRequest:
    """Request to materialize a selected-node range snapshot on demand."""

    node: str
    history: Optional[List[str]] = None
    samples: int = 1326
    min_iteration: Optional[int] = None
    include_stability: bool = True
    include_hands: bool = True
    action_filter: Optional[List[str]] = None


@dataclass
class HandPolicy:
    """Per-hand action policy for a selected node."""

    hand: str
    policy: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpotFrequencyResponse:
    """Single-hand lookup for a preflop spot using compact combo keys like TT or AKs."""

    spot: str
    hand: str
    iteration: int
    frequencies: Dict[str, float] = field(default_factory=dict)
    ready: bool = True
    message: Optional[str] = None


@dataclass
class ProbeResponse:
    """Response payload for a single selected-node probe."""

    iteration: int
    node: str
    display_name: Optional[str] = None
    history: Optional[List[str]] = None
    sample_count: int = 0
    action_frequencies: Dict[str, float] = field(default_factory=dict)
    hands: List[HandPolicy] = field(default_factory=list)
    stability: Optional[StabilitySummary] = None
    ready: bool = True
    message: Optional[str] = None


@dataclass
class BulkProbeRequest:
    """Batch request for multiple node probes in one payload."""

    requests: List[ProbeRequest] = field(default_factory=list)


@dataclass
class BulkProbeResponse:
    """Batch response for multiple node probes."""

    results: List[ProbeResponse] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


@dataclass
class SolverStatusResponse:
    """Detailed, read-only convergence and readiness snapshot."""

    solver: str
    iteration: int
    stable: bool
    stability: Optional[StabilitySummary] = None
    ready_for_queries: bool = False
    last_probe_at: Optional[int] = None
    min_iteration: Optional[int] = None
    probe_budget_remaining: Optional[int] = None
