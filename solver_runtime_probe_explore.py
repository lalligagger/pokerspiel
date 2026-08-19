# %%
"""Small runtime-matching probe explorer for the HULH solver.

This script is designed to behave like a notebook cell workflow in VS Code while
still being executable directly as a Python file.

It intentionally mirrors the app's runtime structure without the checkpointing
logic:
- build the same HULH game as the app
- create the same MCCFR solver family used by the app
- run a single training iteration
- sample preflop node probes using the same selected-node helper
- sample postflop states in the same style as the live service
- print compact summaries, but do not trigger checkpointing or persistence

Usage:
    python solver_runtime_probe_explore.py
    # or open in VS Code and run the cells with "Run Cell"
"""

# %%
import argparse
import json
import time
from pprint import pprint

import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

from app_solver import (
    GAME_CONFIGS,
    make_solver,
    prepare_selected_node_probes,
    resolve_node_specs,
    snapshot_probe_states,
    aggregate_selected_node_ranges,
    policy_snapshot_record,
    resolve_actor_state,
    infer_state_context,
)

# %%

CHECKPOINT_EVERY = None
# Intentionally disabled here; this script models the live runtime probe surface,
# not the checkpoint reporting layer.

# %%
def build_hulh_game():
    """Build the same wrapped HULH game used by the service."""
    return pyspiel.load_game("python_pokerkit_wrapper", GAME_CONFIGS["hulh"])


# %%
def build_solver(game, solver_name="external"):
    if solver_name == "cfr+":
        try:
            return pyspiel.CFRPlusSolver(game)
        except AttributeError:
            raise ValueError("CFRPlusSolver is not available in this OpenSpiel build")
    return make_solver(game, solver_name)


# %%
def build_debug_solver(game=None, solver_name="external"):
    """Create a standalone solver instance for notebook/debug work.

    This mirrors the runtime construction path used by the live service but does
    not participate in the service lifecycle or readiness gating.
    """
    if game is None:
        game = build_hulh_game()
    solver = build_solver(game, solver_name=solver_name)
    return game, solver


# %%
def setup_runtime(solver_name="external"):
    game, solver = build_debug_solver(solver_name=solver_name)
    return game, solver, solver_name


# Shared notebook/script state so the cells are self-contained and do not depend on
# a prior cell calling main().
game, solver, solver_name = setup_runtime(solver_name="external")


# %%
# Raw 100-iteration speed test with no probing/checkpointing.
print("=== raw 100 iteration speed test (no probes) ===")
raw_start = time.perf_counter()
for _ in range(100):
    solver.run_iteration()
raw_elapsed = time.perf_counter() - raw_start
print(f"raw_100_elapsed_seconds={raw_elapsed:.4f}")
print(f"raw_100_iters_per_second={100 / raw_elapsed:.2f}")


# %%
# Quick external-vs-CFR+ timing comparison for the same game.
for name in ["external", "cfr+"]:
    try:
        local_game = build_hulh_game()
        local_solver = build_solver(local_game, solver_name=name)
        loop_start = time.perf_counter()
        for _ in range(100):
            local_solver.run_iteration()
        loop_elapsed = time.perf_counter() - loop_start
        print(f"{name}: 100 iterations in {loop_elapsed:.4f}s ({100 / loop_elapsed:.2f} it/s)")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")


# %%
def run_single_iteration(game, solver_name="external"):
    """Run a single solver iteration without using the service-owned runtime.

    This is the notebook/debug path: one fresh solver instance, one iteration,
    then inspect average_policy(). It intentionally avoids the live API/thread
    state machine so it can be used in analysis work.
    """
    _, solver = build_debug_solver(game=game, solver_name=solver_name)
    start = time.perf_counter()
    solver.run_iteration()
    elapsed = time.perf_counter() - start
    return solver, elapsed


# %%
def sample_postflop_states_like_runtime(
    game,
    *,
    board=None,
    history=None,
    hole_cards=None,
    player=None,
    samples=8,
):
    """Approximate the service's postflop sampling behavior without checkpointing.

    This intentionally matches the runtime style used by the app service:
    - start from a fresh state
    - walk a candidate action history
    - optionally filter by board / hole-card match
    - return a small sample batch for policy inspection
    """
    board = list(board or [])
    history = list(history or [])
    hole_cards = list(hole_cards or [])
    sampled = []
    attempts = 0
    max_attempts = max(samples * 25, 200)

    while len(sampled) < max(samples, 1) and attempts < max_attempts:
        attempts += 1
        state = game.new_initial_state()
        wrapped = getattr(state, "_wrapped_state", None)
        if wrapped is None:
            continue

        actual_board = [str(card) for card in getattr(wrapped, "board_cards", []) or []]
        if board and sorted(actual_board) != sorted(board):
            continue

        if hole_cards:
            actual_hole = getattr(wrapped, "hole_cards", []) or []
            player_hole = []
            if player is not None and player < len(actual_hole):
                player_hole = [str(card) for card in actual_hole[player]]
            elif actual_hole:
                player_hole = [str(card) for card in actual_hole[0]]
            if player_hole and sorted(player_hole) != sorted(hole_cards):
                continue

        resolved = state
        for action in history:
            legal = list(resolved.legal_actions())
            if not legal:
                resolved = None
                break
            normalized = str(action).lower()
            if normalized in {"check", "call"}:
                chosen = 1 if 1 in legal else legal[0]
            elif normalized in {"bet", "raise"}:
                chosen = 4 if 4 in legal else legal[0]
            elif normalized == "fold":
                chosen = 0 if 0 in legal else legal[0]
            else:
                try:
                    chosen = int(action)
                except Exception:
                    chosen = legal[0]
                if chosen not in legal:
                    chosen = legal[0]
            resolved = resolved.child(chosen)
            if resolved is None:
                break

        if resolved is not None and resolve_actor_state(resolved) is not None:
            sampled.append(resolved)

    return sampled


# %%
def summarize_probe_records(records):
    """Print a compact summary for selected node records."""
    if not records:
        print("  no probe records")
        return
    for record in records[:5]:
        print(
            "  node=%s history=%s player=%s actions=%s" % (
                record.get("label"),
                record.get("history"),
                record.get("player"),
                record.get("action_probabilities"),
            )
        )


# %%
def summarize_postflop_records(records):
    if not records:
        print("  no postflop records")
        return
    for item in records[:3]:
        print(
            "  state=%s policy=%s" % (
                item.get("label"),
                item.get("action_probabilities"),
            )
        )


# %%
def main():
    parser = argparse.ArgumentParser(description="Runtime-like HULH probe explorer")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="number of MCCFR iterations to run before showing probes")
    args = parser.parse_args()

    print("=== HULH runtime probe explorer ===")
    print("Checkpointing is intentionally disabled in this script.")
    print(f"iterations={args.iterations}")

    global game, solver, solver_name
    game, solver, solver_name = setup_runtime(solver_name="external")

    # %%
    print("\n--- Solver iterations ---")
    for iteration_index in range(args.iterations):
        solver, elapsed = run_single_iteration(game, solver_name=solver_name)
        print(f"iteration {iteration_index + 1}: {elapsed:.4f}s")

    print(f"completed {args.iterations} solver iterations")

    # %%
    print("\n--- Preflop probes (same selected-node helper as runtime) ---")
    specs = resolve_node_specs("hulh-preflop", ())
    preflop_probes = prepare_selected_node_probes(
        game,
        specs,
        samples_per_node=2,
        max_attempts=200,
        dedupe=False,
    )
    preflop_records = snapshot_probe_states(solver.average_policy(), preflop_probes)
    aggregate = aggregate_selected_node_ranges(preflop_records)
    print(f"preflop probes: {len(preflop_probes)}")
    pprint(aggregate.get("nodes", [])[:3])

    # %%
    print("\n--- Postflop probes (runtime-shaped sampling, no checkpointing) ---")
    # Sample a realistic postflop board and a candidate action history from the live runtime pattern.
    example_board = ["Ah", "Kd", "2c"]
    example_history = ["bet", "call"]
    example_hole_cards = ["As", "Qs"]

    postflop_states = sample_postflop_states_like_runtime(
        game,
        board=example_board,
        history=example_history,
        hole_cards=example_hole_cards,
        player=0,
        samples=6,
    )
    print(f"postflop states sampled: {len(postflop_states)}")

    postflop_records = []
    for idx, state in enumerate(postflop_states[:3]):
        resolved = resolve_actor_state(state)
        if resolved is None:
            continue
        info = infer_state_context(resolved)
        rec = policy_snapshot_record(
            solver.average_policy(),
            resolved,
            history=list(info.get("history") or example_history),
            label=f"postflop_{idx}",
        )
        if rec is not None:
            postflop_records.append(rec)

    summarize_postflop_records(postflop_records)

    # %%
    print("\n--- Final status summary ---")
    print(json.dumps({
        "solver_name": solver_name,
        "checkpoint_every": CHECKPOINT_EVERY,
        "preflop_probe_count": len(preflop_probes),
        "postflop_state_count": len(postflop_states),
        "postflop_record_count": len(postflop_records),
    }, indent=2))


# %%
if __name__ == "__main__":
    main()
