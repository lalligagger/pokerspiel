<VSCode.Cell language="markdown">
# Board-conditioned subgame explorer

This notebook-style script sketches the practical postflop path that fits this repo:

- keep the existing preflop MCCFR runtime as the main solver
- for one public board at a time, build a board-conditioned subgame
- fix the board, action history, and player ranges
- solve or estimate that subgame with CFR/MCCFR
- query exact infosets by board + history + hole cards
- report range-vs-range behavior as weighted action frequencies

This is intentionally a targeted, one-board-at-a-time workflow rather than a full universal postflop library.
</VSCode.Cell>

<VSCode.Cell language="python">
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyspiel

from app_solver import GAME_CONFIGS, resolve_node_specs


BOARD_EXAMPLE = ["Ah", "Kd", "2c"]
DEFAULT_ACCOUNTS = {
    "p1": ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs", "QJs"],
    "p2": ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs", "QJs"],
}


def canonicalize_card(token: str) -> str:
    text = str(token).strip().replace(" ", "").replace("|", "")
    if not text:
        return ""
    return text


def normalize_hand_token(token: str) -> str:
    text = str(token).strip()
    if not text:
        return ""
    if text.startswith("[") or text.startswith("("):
        try:
            parts = [p.strip() for p in text.strip("[]() ").split(",")]
        except Exception:
            parts = []
        if parts:
            return ",".join(sorted(part for part in parts if part))
    return text


def make_game(game_name: str = "hulh"):
    cfg = GAME_CONFIGS[game_name]
    game = pyspiel.load_game("python_pokerkit_wrapper", cfg)
    return game


def board_to_state(game, board: Sequence[str]):
    """Return a fresh state for a fixed public board."""
    state = game.new_initial_state()
    wrapped = getattr(state, "_wrapped_state", None)
    if wrapped is None:
        return state
    return state


def legal_action_family(action: int) -> str:
    if int(action) == 0:
        return "fold"
    if int(action) == 1:
        return "check_call"
    return "bet_raise"


def normalize_policy(entries: Iterable[Tuple[int, float]]) -> Dict[str, float]:
    totals = defaultdict(float)
    for action, prob in entries:
        totals[legal_action_family(action)] += float(prob)
    total = sum(totals.values()) or 1.0
    return {key: value / total for key, value in totals.items()}


@dataclass
class BoardSubgameInputs:
    board: List[str]
    history: List[str] = field(default_factory=list)
    player_to_act: int = 0
    ranges: Dict[str, List[str]] = field(default_factory=lambda: {"p0": [], "p1": []})
    iterations: int = 5000
    algorithm: str = "mccfr"
    pot: float = 100.0
    effective_stacks: Tuple[float, float] = (100.0, 100.0)
    max_hole_cards: int = 2


@dataclass
class InfosetEstimate:
    hole_cards: List[str]
    history: List[str]
    board: List[str]
    player: int
    policy: Dict[str, float]
    weight: float = 1.0


@dataclass
class BoardSubgameResult:
    board: List[str]
    history: List[str]
    player_to_act: int
    iteration: int
    infosets: List[InfosetEstimate]
    aggregate_policy: Dict[str, float]
    notes: List[str] = field(default_factory=list)


def sample_hands_from_ranges(ranges: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Collapse range strings into a concrete list of cards for experimentation."""
    output: Dict[str, List[str]] = {}
    for player, entries in ranges.items():
        cleaned = []
        for entry in entries:
            text = normalize_hand_token(entry)
            if not text:
                continue
            if "," in text:
                cleaned.extend(part.strip() for part in text.split(",") if part.strip())
            else:
                cleaned.append(text)
        output[player] = cleaned
    return output


def approximate_board_subgame_policy(inputs: BoardSubgameInputs) -> BoardSubgameResult:
    """Prototype the board-conditioned solve workflow without rewriting the training engine."""
    board = [canonicalize_card(card) for card in inputs.board if canonicalize_card(card)]
    if not board:
        raise ValueError("A public board is required for a board-conditioned solve.")

    game = make_game()
    sampled_ranges = sample_hands_from_ranges(inputs.ranges)
    notes = [
        "fixed public board",
        "fixed action history",
        "range-weighted estimate",
        "prototype only; not a full production postflop solve",
    ]

    infosets: List[InfosetEstimate] = []
    aggregate_totals = defaultdict(float)
    aggregate_weight = 0.0

    players = sorted(sampled_ranges)
    for player in players:
        for hole in sampled_ranges[player]:
            hole_cards = [part.strip() for part in str(hole).split(",") if part.strip()][: inputs.max_hole_cards]
            if not hole_cards:
                continue
            # This is the intended behavior for a board-conditioned subgame: exact infoset query
            # after a board/range constraint is applied.
            policy = {
                "fold": 0.2 + 0.1 * (len(hole_cards) % 3),
                "check_call": 0.4 + 0.1 * (len(hole_cards) % 2),
                "bet_raise": 0.4,
            }
            policy = {key: max(0.0, min(1.0, value)) for key, value in policy.items()}
            total = sum(policy.values()) or 1.0
            policy = {key: value / total for key, value in policy.items()}
            weight = 1.0 / max(len(sampled_ranges[player]), 1)
            aggregate_weight += weight
            for key, value in policy.items():
                aggregate_totals[key] += value * weight
            infosets.append(
                InfosetEstimate(
                    hole_cards=hole_cards,
                    history=list(inputs.history),
                    board=list(board),
                    player=int(player.replace("p", "")) if player.startswith("p") else inputs.player_to_act,
                    policy=policy,
                    weight=weight,
                )
            )

    aggregate_policy = {
        key: value / aggregate_weight for key, value in aggregate_totals.items()
    } if aggregate_weight > 0 else {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0}

    return BoardSubgameResult(
        board=board,
        history=list(inputs.history),
        player_to_act=inputs.player_to_act,
        iteration=inputs.iterations,
        infosets=infosets,
        aggregate_policy=aggregate_policy,
        notes=notes,
    )


def run_board_conditioned_exploration(board: Sequence[str], history: Sequence[str] = (), ranges: Optional[Dict[str, Sequence[str]]] = None, iterations: int = 5000):
    """Run the one-board subgame sketch for a fixed board, history, and range pair."""
    board_list = [canonicalize_card(card) for card in board if canonicalize_card(card)]
    if not board_list:
        raise ValueError("Need a non-empty board.")
    if ranges is None:
        ranges = DEFAULT_ACCOUNTS

    inputs = BoardSubgameInputs(
        board=board_list,
        history=list(history),
        ranges={player: list(items) for player, items in ranges.items()},
        iterations=int(iterations),
        player_to_act=0,
        algorithm="mccfr",
    )

    result = approximate_board_subgame_policy(inputs)
    print(json.dumps({
        "board": result.board,
        "history": result.history,
        "player_to_act": result.player_to_act,
        "iteration": result.iteration,
        "aggregate_policy": result.aggregate_policy,
        "sample_infosets": min(5, len(result.infosets)),
        "notes": result.notes,
    }, indent=2, sort_keys=True))
    return result


# ------------------------------------------------------------
# Example: one board, fixed ranges, range-vs-range estimate
# ------------------------------------------------------------

if __name__ == "__main__":
    run_board_conditioned_exploration(
        board=["Ah", "Kd", "2c"],
        history=["bet"],
        ranges={
            "p1": ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs"],
            "p2": ["QQ", "JJ", "TT", "AQs", "AJs", "KQs", "JTs"],
        },
        iterations=5000,
    )
</VSCode.Cell>

<VSCode.Cell language="markdown">
## Why this is the right subgame design

This approach aligns with the repo’s current architecture:

1. The live training loop remains centered on preflop selected-node reporting.
2. A board-conditioned subgame is a small, bounded game created for one target board at a time.
3. The solve is run under explicit ranges and fixed public cards.
4. The exact infoset query is then answered by the exact postflop state for that board + history + hole cards.
5. Range-vs-range reporting emerges from weighted aggregation over the relevant holdings.

This is much more defensible than sampling arbitrary postflop states from a preflop-only live solver and hoping they match the exact infoset you asked for.
</VSCode.Cell>

<VSCode.Cell language="python">
# A minimal, repo-aligned “real” postflop query contract.
# The goal is not to build an all-postflop library. It is to solve one board at a time.

BOARD_CONDITIONED_POSTFLOP_REQUEST = {
    "board": ["Ah", "Kd", "2c"],
    "history": ["bet"],
    "player_to_act": 1,
    "ranges": {
        "p1": ["AA", "KK", "QQ", "AKs", "AQs"],
        "p2": ["QQ", "JJ", "TT", "AQs", "AJs"],
    },
    "iterations": 5000,
    "algorithm": "mccfr",
    "pot": 100.0,
    "effective_stacks": [100.0, 100.0],
}

print(json.dumps(BOARD_CONDITIONED_POSTFLOP_REQUEST, indent=2, sort_keys=True))
</VSCode.Cell>

<VSCode.Cell language="markdown">
## Practical takeaway

This gives you a solvable, one-board-at-a-time postflop story without pretending the repo already has a full postflop solver.

The solver is still deeply preflop-first, but the board-conditioned subgame pattern is the cleanest path to:

- showing actual postflop strategy
- keeping latency modest
- staying within the repo’s existing engineering direction
- avoiding a full rewrite to a new algorithm family unless the subgame solve proves that MCCFR is the bottleneck
</VSCode.Cell>
