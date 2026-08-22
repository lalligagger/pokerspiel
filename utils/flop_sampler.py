"""Reduced-flop sampler helpers for benchmark PoCs.

This keeps the sampler logic isolated from the solver and service layers so the
main runtime stays stable while reduced-flop experiments are being evaluated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

from utils.flop_isomorphism import build_strategic_1755_subset, canonical_flop_key

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]


@dataclass
class FlopSubset:
    name: str
    members: List[Tuple[str, ...]]
    weights: Dict[Tuple, float]


def all_physical_flops() -> List[Tuple[str, ...]]:
    deck = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]
    return list(combinations(deck, 3))


def reduced_flop_subset_1755() -> FlopSubset:
    """Build the strategic-1755 reduced flop set based on suit-isomorphic collapse."""
    canonical, counts = build_strategic_1755_subset()
    members = list(canonical)
    weights = {canonical_flop_key(member): float(counts.get(canonical_flop_key(member), 1)) for member in members}
    return FlopSubset(name="strategic_1755", members=members, weights=weights)


def landmark_sampler_from_members(members: Sequence[Sequence[str]], seed: int | None = None) -> object:
    """Simple sampler object for a reduced-flop subset benchmark.

    This intentionally stays minimal and is meant to support a benchmark harness,
    not production runtime logic.
    """
    rng = Random(seed)
    members = [tuple(member) for member in members]

    class _Sampler:
        def sample_board(self):
            board = rng.choice(members)
            return tuple(board), 1.0

    return _Sampler()
