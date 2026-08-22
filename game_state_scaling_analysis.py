# %%
"""Game-state scaling and single-iteration sampler for poker solver variants.

This is a dedicated analysis notebook/script meant to compare:
- full-game combinatorial scale
- single-iteration traversal footprint
- approximate infoset counts actually touched by a one-iteration run

It intentionally avoids checkpointing and live probe work. It is only for reasoning
about the state-space and runtime cost of the solver family.
"""

# %%
import math
from collections import defaultdict

import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

from app_solver import GAME_CONFIGS, make_solver

# %%


def n_choose_k(n, k):
    return math.comb(n, k)


# %%
def combinatorial_state_summary():
    data = {}

    # Toy and benchmark games.
    data["Kuhn Poker"] = {
        "players": 2,
        "deck_size": 3,
        "hole_cards_per_player": 1,
        "possible_hole_card_assignments": n_choose_k(3, 1) * n_choose_k(2, 1),
        "state_size_note": "tiny benchmark game with 3 cards and 1-card hands",
    }

    data["Leduc Poker"] = {
        "players": 2,
        "deck_size": 6,
        "hole_cards_per_player": 1,
        "possible_hole_card_assignments": n_choose_k(6, 2) * n_choose_k(4, 2),
        "state_size_note": "small game, but still much larger than Kuhn",
    }

    # HULH and no-limit family.
    data["HULH (full 52-card deck)"] = {
        "players": 2,
        "deck_size": 52,
        "hole_cards_per_player": 2,
        "possible_hole_card_assignments": n_choose_k(52, 4) * 3,
        "board_cards_per_street": {"flop": 3, "turn": 1, "river": 1},
        "state_size_note": "full 52-card HULH has explosive reachability once betting history and board textures are included",
    }

    data["HUNL (full no-limit, not exhaustive)"] = {
        "players": 2,
        "deck_size": 52,
        "hole_cards_per_player": 2,
        "possible_hole_card_assignments": n_choose_k(52, 4) * 3,
        "betting_actions_note": "No-limit action space is much larger because bet sizes are unbounded and history-dependent",
        "state_size_note": "not feasible to enumerate exhaustively; the combinatorial growth is far beyond HULH",
    }

    return data


# %%
def pretty_count(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


# %%
def print_scaling_summary():
    summary = combinatorial_state_summary()
    for name, payload in summary.items():
        print(f"\n{name}")
        for key, value in payload.items():
            if key == "state_size_note":
                print(f"  - {value}")
            elif key == "betting_actions_note":
                print(f"  - {value}")
            elif key == "possible_hole_card_assignments":
                print(f"  - possible hole-card assignments: {pretty_count(value)}")
            else:
                print(f"  - {key}: {value}")


# %%
def build_game(name):
    if name == "kuhn":
        return pyspiel.load_game("kuhn_poker")
    if name == "leduc":
        return pyspiel.load_game("leduc_poker")
    if name == "hulh":
        return pyspiel.load_game("python_pokerkit_wrapper", GAME_CONFIGS["hulh"])
    raise ValueError(f"unsupported game '{name}'")


# %%
def build_solver(game, solver_name):
    try:
        return make_solver(game, solver_name)
    except ValueError as exc:
        if solver_name == "cfr+":
            try:
                return pyspiel.CFRPlusSolver(game)
            except AttributeError:
                raise ValueError("CFRPlusSolver is unavailable in this OpenSpiel build") from exc
        raise


# %%
def walk_infoset_sample(game, max_depth=4, max_states=20000):
    """Depth-limited tree walk that approximates what a single iteration touches.

    This is intentionally not exhaustive. The point is to estimate the footprint of a
    one-iteration traversal and compare the rough scale across game families.
    """
    seen = set()
    queue = [(game.new_initial_state(), 0)]
    visited = 0

    while queue and visited < max_states:
        state, depth = queue.pop(0)
        visited += 1
        if state.is_terminal() or depth >= max_depth:
            continue

        if not state.is_chance_node():
            player = state.current_player()
            info_state = state.information_state_string(player)
            seen.add((player, info_state))

        for action in state.legal_actions():
            child = state.child(action)
            if child is not None:
                queue.append((child, depth + 1))

    return len(seen), visited


# %%
def single_iteration_sampler(game_name, solver_name, max_depth=4, max_states=20000):
    game = build_game(game_name)
    solver = build_solver(game, solver_name)
    before = walk_infoset_sample(game, max_depth=max_depth, max_states=max_states)
    solver.run_iteration()
    after = walk_infoset_sample(game, max_depth=max_depth, max_states=max_states)
    return {
        "game": game_name,
        "solver": solver_name,
        "before_iteration_sampled_infosets": before[0],
        "before_iteration_visited_states": before[1],
        "after_iteration_sampled_infosets": after[0],
        "after_iteration_visited_states": after[1],
    }


# %%
def compare_algorithms_for_one_iteration():
    results = []
    for game_name, solver_name in [
        ("kuhn", "external"),
        ("kuhn", "outcome"),
        ("kuhn", "cfr+"),
        ("leduc", "external"),
        ("leduc", "outcome"),
        ("leduc", "cfr+"),
    ]:
        try:
            result = single_iteration_sampler(game_name, solver_name, max_depth=4, max_states=20000)
            results.append(result)
        except Exception as exc:
            print(f"{game_name}/{solver_name}: unavailable ({exc})")
    return results


# %%
print("=== combinatorial game-state scaling ===")
print_scaling_summary()

# %%
print("\n=== one-iteration sampled footprint ===")
for result in compare_algorithms_for_one_iteration():
    print(result)

# %%
# Conceptual interpretation.
print("\nInterpretation:")
print("- toy games like Kuhn and Leduc are small enough to inspect directly")
print("- HULH and no-limit explode combinatorially as betting histories, board textures, and hidden card assignments accumulate")
print("- the sampled infoset footprint per one-iteration run is only a small, depth-limited slice of the full game")
print("- this is why full-game CFR+ is dramatically slower than small toy benchmarks, even before checkpointing or probes")
