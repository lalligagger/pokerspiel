import json
import logging
import os
import random
import resource
import statistics
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from absl import logging as absl_logging
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

absl_logging.set_verbosity(absl_logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from range_export import (
    aggregate_range_profiles,
    aggregate_selected_node_ranges,
    export_range_dump,
)


GAME_CONFIGS = {
    "hulh": {
        "variant": "FixedLimitTexasHoldem",
        "num_players": 2,
        "blinds": "1 2",
        "stack_sizes": "200 200",
        "antes": "0 0",
        "num_streets": 4,
        "small_bet": 2,
        "big_bet": 4,
    },
    "sdhunl": {
        "variant": "NoLimitShortDeckHoldem",
        "num_players": 2,
        "blinds": "1 2",
        "stack_sizes": "200 200",
        "antes": "0 0",
        "num_streets": 4,
    },
}

ACTION_NAME_TO_ID = {
    "fold": 0,
    "check": 1,
    "call": 1,
    "check_call": 1,
    "bet": 4,
    "raise": 4,
    "bet_raise": 4,
}

HULH_ACTION_SEQUENCE_TO_HISTORY = {
    "f": ["fold"],
    "c": ["call"],
    "cx": ["call", "check"],
    "cr": ["call", "bet"],
    "rf": ["bet", "fold"],
    "rc": ["bet", "call"],
    "rr": ["bet", "bet"],
    "rrf": ["bet", "bet", "fold"],
    "rrc": ["bet", "bet", "call"],
    "rrr": ["bet", "bet", "raise"],
    "rrrf": ["bet", "bet", "raise", "fold"],
    "rrrc": ["bet", "bet", "raise", "call"],
}

HULH_NODE_LABELS = {
    "first_to_act": "first_to_act",
    "response_to_limp": "response_to_limp",
    "respond_to_limp": "response_to_limp",
    "response_to_open": "response_to_open",
    "respond_to_open": "response_to_open",
    "response_to_limp_raise": "response_to_limp_raise",
    "respond_to_limp_raise": "response_to_limp_raise",
    "response_to_limp_reraise": "response_to_limp_reraise",
    "respond_to_limp_reraise": "response_to_limp_reraise",
    "response_to_open_3bet": "response_to_open_3bet",
    "response_to_open_4bet": "response_to_open_4bet",
    "response_to_open_5bet": "response_to_open_5bet",
    "opener_response_to_3bet": "response_to_open_3bet",
    "opener_response_to_4bet": "response_to_open_4bet",
    "opener_response_to_5bet": "response_to_open_5bet",
}


def format_hulh_history_label(history):
    """Map an action history to the HULH display label used in reporting."""
    normalized = []
    for item in history or []:
        if isinstance(item, str):
            normalized.append(item.strip().lower())
        else:
            normalized.append(str(item))

    if not normalized:
        return "first_to_act"
    if normalized == ["call"]:
        return "response_to_limp"
    if normalized == ["bet"]:
        return "response_to_open"
    if normalized == ["call", "bet"]:
        return "response_to_limp_raise"
    if normalized == ["call", "raise"]:
        return "response_to_limp_raise"
    if normalized == ["call", "raise", "raise"]:
        return "response_to_limp_reraise"
    if normalized == ["bet", "bet"]:
        return "response_to_open_3bet"
    if normalized == ["raise", "raise"]:
        return "response_to_open_3bet"
    if normalized == ["bet", "bet", "fold"]:
        return "response_to_open_3bet_fold"
    if normalized == ["bet", "bet", "call"]:
        return "response_to_open_3bet_call"
    if normalized == ["bet", "bet", "raise"]:
        return "response_to_open_4bet"
    if normalized == ["raise", "raise", "raise"]:
        return "response_to_open_4bet"
    if normalized == ["bet", "bet", "raise", "raise"]:
        return "response_to_open_5bet"
    if normalized == ["raise", "raise", "raise", "raise"]:
        return "response_to_open_5bet"
    if normalized == ["bet", "bet", "raise", "fold"]:
        return "response_to_open_4bet_fold"
    if normalized == ["bet", "bet", "raise", "call"]:
        return "response_to_open_4bet_call"
    if normalized == ["bet", "bet", "raise", "raise", "fold"]:
        return "response_to_open_5bet_fold"
    if normalized == ["bet", "bet", "raise", "raise", "call"]:
        return "response_to_open_5bet_call"
    return "history_" + "_".join(normalized)


NODE_PRESETS = {
    "hulh-preflop": [
        {"name": "first_to_act", "history": []},
        {"name": "response_to_limp", "history": ["call"]},
        {"name": "response_to_open", "history": ["bet"]},
        {"name": "response_to_limp_raise", "history": ["call", "bet"]},
        {"name": "response_to_open_3bet", "history": ["bet", "bet"]},
        {"name": "response_to_open_4bet", "history": ["bet", "bet", "raise"]},
        {"name": "response_to_open_5bet", "history": ["bet", "bet", "raise", "raise"]},
    ],
    "hulh-preflop-lw": [
        {"name": "first_to_act", "history": []},
        {"name": "response_to_limp", "history": ["call"]},
        {"name": "response_to_open", "history": ["bet"]},
        {"name": "response_to_limp_reraise", "history": ["call", "raise", "raise"]},
    ],
    "root": [{"name": "first_to_act", "history": []}],
}


def resolve_actor_state(state):
    """Return the first state in the tree whose current_player() is a real actor."""
    seen = set()

    def walk(cur):
        key = cur.serialize()
        if key in seen:
            return None
        seen.add(key)
        if cur.current_player() >= 0:
            return cur
        for action in cur.legal_actions():
            child = cur.child(action)
            result = walk(child)
            if result is not None:
                return result
        return None

    return walk(state)


def timed(label: str, fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
    return out


def snapshot_policy(policy, state):
    resolved = resolve_actor_state(state)
    if resolved is None:
        print("snapshot_policy: no valid actor state found")
        return

    player = resolved.current_player()
    pol = policy.get_state_policy(resolved, player)
    legal = resolved.legal_actions()
    print(
        f"policy_snapshot: player={player} legal_count={len(legal)} "
        f"sample={pol[:10]} prob_sum={sum(p for _, p in pol):.6f}"
    )


def sample_actor_state_histories(root_state, max_depth: int = 3, max_states: int = 6):
    """Return a handful of meaningful actor states sampled from actual action histories."""
    queue = [(root_state, [])]
    results = []
    seen = set()

    while queue and len(results) < max_states:
        state, history = queue.pop(0)
        resolved = resolve_actor_state(state)
        if resolved is not None:
            info = infer_state_context(resolved, history=history)
            signature = (
                info.get("street"),
                info.get("pot_context"),
                tuple(sorted(info.get("legal_actions") or [])),
                tuple(history),
            )
            if is_meaningful_state(resolved, history=history) and signature not in seen:
                seen.add(signature)
                results.append((history, resolved))

        if len(history) >= max_depth:
            continue

        actions = state.legal_actions()
        if not actions:
            continue

        picks = actions[: min(4, len(actions))]
        for action in picks:
            child = state.child(action)
            if child is not None:
                queue.append((child, history + [action]))

    return results


def sample_street_boundary_states(root_state, max_states: int = 6):
    """Sample meaningful valid actor states at distinct street boundaries from real action histories."""
    queue = [(root_state, [])]
    samples = []
    seen = set()

    while queue and len(samples) < max_states:
        state, history = queue.pop(0)
        resolved = resolve_actor_state(state)
        if resolved is not None:
            info = infer_state_context(resolved, history=history)
            signature = (
                info.get("street"),
                info.get("pot_context"),
                tuple(sorted(info.get("legal_actions") or [])),
                tuple(history),
            )
            if is_meaningful_state(resolved, history=history) and signature not in seen:
                seen.add(signature)
                samples.append((history, resolved))

        actions = state.legal_actions()
        if not actions:
            continue

        for action in actions[: min(3, len(actions))]:
            child = state.child(action)
            if child is not None:
                queue.append((child, history + [action]))

    return samples


def snapshot_policy_histories(policy, root_state, max_depth: int = 3, max_states: int = 6):
    samples = sample_actor_state_histories(root_state, max_depth=max_depth, max_states=max_states)
    if not samples:
        samples = sample_street_boundary_states(root_state, max_states=max_states)
    if not samples:
        print("snapshot_policy_histories: no valid actor states found")
        return

    for history, resolved in samples:
        player = resolved.current_player()
        pol = policy.get_state_policy(resolved, player)
        legal = resolved.legal_actions()
        print(
            f"policy_snapshot history={history} player={player} legal_count={len(legal)} "
            f"sample={pol[:10]} prob_sum={sum(p for _, p in pol):.6f}"
        )


def snapshot_street_boundaries(policy, root_state, max_states: int = 6):
    samples = sample_street_boundary_states(root_state, max_states=max_states)
    if not samples:
        samples = sample_actor_state_histories(root_state, max_depth=3, max_states=max_states)
    if not samples:
        print("snapshot_street_boundaries: no valid street-boundary states found")
        return

    for history, resolved in samples:
        player = resolved.current_player()
        pol = policy.get_state_policy(resolved, player)
        legal = resolved.legal_actions()
        print(
            f"street_boundary history={history} player={player} legal_count={len(legal)} "
            f"sample={pol[:10]} prob_sum={sum(p for _, p in pol):.6f}"
        )


def _checkpoint_key_for_record(record):
    for key in (
        "exact_infoset_key",
        "state_serialize",
        "summary",
        "label",
        "history",
        "street",
        "pot_context",
        "player",
    ):
        value = record.get(key)
        if value is not None:
            return (key, value)
    return ("fallback", json.dumps(record, sort_keys=True, default=str))


def summarize_checkpoint_stability(current_records, previous_records=None):
    """Compute a compact checkpoint-to-checkpoint stability summary."""
    if not current_records:
        return {"sample_count": 0, "avg_l1_delta": 0.0, "max_l1_delta": 0.0, "top_moving": []}

    current_by_key = {_checkpoint_key_for_record(item): item for item in current_records}
    previous_by_key = {_checkpoint_key_for_record(item): item for item in (previous_records or [])}

    deltas = []
    moving = []
    for key, current in current_by_key.items():
        prev = previous_by_key.get(key)
        if prev is None:
            continue
        prev_policy = {
            str(entry["action"]): float(entry["probability"])
            for entry in prev.get("action_probabilities", [])
        }
        curr_policy = {
            str(entry["action"]): float(entry["probability"])
            for entry in current.get("action_probabilities", [])
        }
        actions = sorted(set(prev_policy) | set(curr_policy), key=lambda value: int(value) if value.isdigit() else value)
        delta = sum(abs(curr_policy.get(action, 0.0) - prev_policy.get(action, 0.0)) for action in actions)
        deltas.append(delta)
        moving.append({
            "key": key,
            "label": current.get("label", "record"),
            "history": current.get("history", []),
            "street": current.get("street"),
            "pot_context": current.get("pot_context"),
            "delta": float(delta),
        })

    moving.sort(key=lambda item: item["delta"], reverse=True)
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    max_delta = max(deltas) if deltas else 0.0
    return {
        "sample_count": len(current_by_key),
        "matched_records": len(deltas),
        "avg_l1_delta": float(avg_delta),
        "max_l1_delta": float(max_delta),
        "top_moving": moving[:5],
    }


def write_checkpoint_payload(output_json_path: str, iteration: int, payload: dict, stability_payload: dict):
    if output_json_path is None:
        return None

    base_dir = os.path.dirname(output_json_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(output_json_path))[0]
    checkpoint_path = os.path.join(base_dir, f"{basename}_checkpoint_{iteration:06d}.json")

    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"checkpoint_path": checkpoint_path, "stability_payload": stability_payload}


def write_run_manifest(
    report_path: str,
    range_path: str = None,
    stability_path: str = None,
    summary_path: str = None,
    checkpoint_paths: Iterable[str] = (),
):
    if report_path is None:
        return None

    base_dir = os.path.dirname(report_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(report_path))[0]
    manifest_path = os.path.join(base_dir, f"{basename}_manifest.json")

    artifact_paths = [report_path]
    if range_path:
        artifact_paths.append(range_path)
    if stability_path:
        artifact_paths.append(stability_path)
    if summary_path:
        artifact_paths.append(summary_path)
    artifact_paths.extend(checkpoint_paths)
    artifact_paths = sorted({os.path.abspath(path) for path in artifact_paths if path})

    manifest = {
        "report_path": os.path.abspath(report_path),
        "range_path": os.path.abspath(range_path) if range_path else None,
        "checkpoint_stability_path": os.path.abspath(stability_path) if stability_path else None,
        "summary_path": os.path.abspath(summary_path) if summary_path else None,
        "selected_node_summary_path": os.path.abspath(summary_path) if summary_path else None,
        "checkpoint_files": [os.path.abspath(path) for path in checkpoint_paths if path],
        "artifacts": artifact_paths,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def infer_state_context(state, history=None):
    """Infer street and pot context from the wrapped PokerKit state and action history.

    The OpenSpiel serialized form is intentionally opaque and not a reliable source of
    semantic metadata, so we prefer the actual PokerKit-backed wrapper object when it exists.
    """
    history_list = list(history or [])
    legal_actions = []

    if state is not None:
        try:
            history_list = list(state.history())
        except Exception:
            pass
        try:
            legal_actions = list(state.legal_actions())
        except Exception:
            legal_actions = []

    wrapped = getattr(state, "_wrapped_state", None)
    street = "unknown"
    pot_context = None

    if wrapped is not None:
        try:
            pot_context = int(wrapped.total_pot_amount)
        except Exception:
            pot_context = None

        try:
            board_count = int(getattr(wrapped, "board_count", 0) or 0)
            if board_count in (0, 3, 4, 5):
                street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[board_count]
        except Exception:
            board_count = None

        try:
            street_index = int(getattr(wrapped, "street_index", -1))
            if street_index in range(4):
                street = ["preflop", "flop", "turn", "river"][street_index]
        except Exception:
            street_index = None

        try:
            street_obj = getattr(wrapped, "street", None)
            board_dealing_count = getattr(street_obj, "board_dealing_count", None)
            if board_dealing_count is not None:
                street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
                    int(board_dealing_count), street
                )
        except Exception:
            pass

    if street == "unknown":
        state_text = state.serialize() if state is not None else ""
        lower_text = state_text.lower()
        if "street=0" in lower_text or "preflop" in lower_text:
            street = "preflop"
        elif "street=1" in lower_text or "flop" in lower_text:
            street = "flop"
        elif "street=2" in lower_text or "turn" in lower_text:
            street = "turn"
        elif "street=3" in lower_text or "river" in lower_text:
            street = "river"

    if pot_context is None and state is not None:
        try:
            if hasattr(state, "pot"):
                pot_context = int(state.pot())
        except Exception:
            pot_context = None

    if pot_context is None and state is not None:
        lower_text = state.serialize().lower()
        for token in ("pot=", "pot "):
            idx = lower_text.find(token)
            if idx != -1:
                tail = lower_text[idx + len(token):]
                digits = "".join(ch for ch in tail if ch.isdigit())
                if digits:
                    pot_context = int(digits)
                    break

    return {
        "history": history_list,
        "street": street,
        "pot_context": pot_context,
        "legal_actions": legal_actions,
    }


def is_meaningful_state(state, history=None):
    """Return True only for states that are not just repeated root-family preflop nodes."""
    if state is None:
        return False

    info = infer_state_context(state, history=history)
    history_list = list(history or info.get("history") or [])
    legal_actions = list(info.get("legal_actions") or [])
    street = info.get("street")
    pot_context = info.get("pot_context")

    if not history_list:
        return False
    if street != "preflop":
        return True
    if pot_context not in (None, 3):
        return True
    if sorted(legal_actions) != [0, 1, 4]:
        return True
    return False


def infer_action_mapping(state):
    """Map raw OpenSpiel integer legal actions to visible PokerKit action labels.

    At a real acting node, the OpenSpiel wrapper exposes a compact integer action set and
    the underlying PokerKit state exposes the human-readable legal family. We preserve the
    integer IDs as the solver-native contract but attach the readable labels alongside them.
    """
    mapping = {}
    if state is None:
        return mapping

    raw_actions = list(getattr(state, "legal_actions", lambda: [])())
    wrapped = getattr(state, "_wrapped_state", None)
    pokerkit_actions = []
    if wrapped is not None:
        try:
            from pokerkit_poc import legal_actions_for_state

            pokerkit_actions = legal_actions_for_state(wrapped)
        except Exception:
            pokerkit_actions = []

    if raw_actions and pokerkit_actions:
        for raw_action, (name, amount) in zip(raw_actions, pokerkit_actions):
            mapping[int(raw_action)] = {
                "label": str(name),
                "amount": int(amount) if amount is not None else None,
            }
        return mapping

    for raw_action in raw_actions:
        mapping[int(raw_action)] = {"label": "unknown", "amount": None}
    return mapping


def exact_infoset_key_for_state(state, history=None):
    """Build a canonical exact infoset key from the PokerKit wrapper state."""
    if state is None:
        return "unknown"

    inferred = infer_state_context(state, history=history)
    history_list = list(inferred.get("history") or [])
    street = inferred.get("street", "unknown")
    player = int(state.current_player())
    wrapped = getattr(state, "_wrapped_state", None)

    hole_cards = []
    board_cards = []
    if wrapped is not None:
        try:
            hole_cards = list(getattr(wrapped, "hole_cards", []) or [])
            if player < len(hole_cards):
                hole_cards = list(hole_cards[player])
        except Exception:
            hole_cards = []
        try:
            board_cards = list(getattr(wrapped, "board_cards", []) or [])
        except Exception:
            board_cards = []

    return (
        f"street={street}|player={player}|hole={sorted(str(card) for card in hole_cards)}"
        f"|board={sorted(str(card) for card in board_cards)}|hist={history_list}"
    )


def policy_snapshot_record(policy, resolved, history=None, label: str = "state"):
    """Return a JSON-friendly snapshot of the policy at a valid acting state."""
    if resolved is None:
        return None

    player = resolved.current_player()
    legal = list(resolved.legal_actions())
    raw_policy = policy.get_state_policy(resolved, player)
    entries = [
        {"action": int(action), "probability": float(probability)}
        for action, probability in raw_policy
    ]
    action_lookup = infer_action_mapping(resolved)

    inferred = infer_state_context(resolved, history=history)
    history_list = inferred["history"]
    state_text = resolved.serialize()
    street = inferred["street"]
    pot_context = inferred["pot_context"]

    wrapped = getattr(resolved, "_wrapped_state", None)
    hole_cards = []
    if wrapped is not None:
        try:
            hole_cards = list(getattr(wrapped, "hole_cards", [])[player])
        except Exception:
            try:
                hole_cards = list(getattr(wrapped, "hole_cards", []) or [])
            except Exception:
                hole_cards = []

    exact_infoset_key = exact_infoset_key_for_state(resolved, history=history)

    legal_action_labels = [
        {
            "raw_action": int(action),
            "label": action_lookup.get(int(action), {}).get("label", "unknown"),
            "amount": action_lookup.get(int(action), {}).get("amount"),
        }
        for action in legal
    ]

    summary = {
        "label": label,
        "history": history_list,
        "player": int(player),
        "legal_actions": [int(action) for action in legal],
        "legal_action_labels": legal_action_labels,
        "action_probabilities": entries,
        "prob_sum": float(sum(item["probability"] for item in entries)),
        "state_serialize": state_text,
        "street": street,
        "pot_context": pot_context,
        "hole_cards": [str(card) for card in hole_cards],
        "exact_infoset_key": exact_infoset_key,
        "summary": f"player={player}; street={street}; pot={pot_context}; history={history_list}; actions={[int(a) for a in legal]}",
    }
    return summary


def summarize_policy_profiles(snapshots):
    """Summarize the report by profiling how often the same policy appears and where it sits semantically."""
    unique_profiles = set()
    repeated_same_family_preflop = 0
    deeper_non_preflop = 0
    seen_same_family = set()

    for snapshot in snapshots or []:
        probs = tuple(
            round(float(item["probability"]), 8)
            for item in snapshot.get("action_probabilities", [])
        )
        if probs:
            unique_profiles.add(probs)

        street = snapshot.get("street")
        pot_context = snapshot.get("pot_context")
        history = tuple(snapshot.get("history") or [])
        family_key = (street, pot_context, tuple(sorted(snapshot.get("legal_actions") or [])), tuple(history[:4]))

        if street == "preflop" and pot_context == 3:
            if family_key not in seen_same_family:
                seen_same_family.add(family_key)
            else:
                repeated_same_family_preflop += 1
            continue

        if street != "preflop":
            deeper_non_preflop += 1

    return {
        "unique_policy_profiles": len(unique_profiles),
        "repeated_same_family_preflop_states": repeated_same_family_preflop,
        "deeper_non_preflop_states": deeper_non_preflop,
    }


def deal_budget_for_iterations(total_iterations: int) -> int:
    """How many fresh deal states to sample in the reporting layer for a single solver run."""
    if total_iterations <= 100:
        return 3
    if total_iterations <= 500:
        return 5
    if total_iterations <= 2500:
        return 8
    if total_iterations <= 10000:
        return 12
    return 16


def exact_hole_board_signature(state):
    """Return a stable exact private-card plus board signature for a state."""
    wrapped = getattr(state, "_wrapped_state", None)
    if wrapped is None:
        return "unknown"

    try:
        player = int(state.current_player())
        hole_cards = list(getattr(wrapped, "hole_cards", []) or [])
        board_cards = list(getattr(wrapped, "board_cards", []) or [])
        if hole_cards and player < len(hole_cards):
            hole = sorted(str(card) for card in hole_cards[player])
        else:
            hole = []
        board = sorted(str(card) for card in board_cards)
        return f"player={player}|hole={hole}|board={board}"
    except Exception:
        return "unknown"


def sample_random_actor_state(game):
    """Follow a random chance path until reaching the first real acting state."""
    state = game.new_initial_state()
    for _ in range(200):
        if state is None or state.is_terminal():
            return None
        if state.current_player() >= 0:
            return state
        actions = list(state.legal_actions())
        if not actions:
            return None
        state = state.child(random.choice(actions))
    return state if state is not None and state.current_player() >= 0 else None


def sample_distinct_deal_states(game, target_count: int, max_attempts: int = 200, dedupe: bool = False):
    """Return fresh root states for reporting.

    We intentionally disable hole-card/board deduplication by default because
    deduping changes the sampling distribution and biases action-frequency
    estimates. The opt-in `dedupe=True` mode remains available for diversity-only
    coverage experiments.
    """
    states = []
    seen = set()
    attempts = 0

    while len(states) < target_count and attempts < max_attempts:
        attempts += 1
        resolved = sample_random_actor_state(game)
        if resolved is None:
            continue
        if dedupe:
            signature = exact_hole_board_signature(resolved)
            if signature in seen:
                continue
            seen.add(signature)
        states.append(resolved)

    return states


def accumulate_global_infoset_policy(accumulator, state, policy):
    """Accumulate policy mass for a running full-run infoset summary inside the solver loop."""
    if state is None:
        return

    resolved = resolve_actor_state(state)
    if resolved is None:
        return

    player = int(resolved.current_player())
    legal = list(resolved.legal_actions())
    if not legal:
        return

    raw_policy = policy.get_state_policy(resolved, player)
    entries = [
        {"action": int(action), "probability": float(probability)}
        for action, probability in raw_policy
    ]
    if not entries:
        return

    info = infer_state_context(resolved)
    exact_key = exact_infoset_key_for_state(resolved, history=info.get("history") or [])
    bucket = accumulator.setdefault(
        exact_key,
        {
            "street": info.get("street"),
            "pot_context": info.get("pot_context"),
            "player": player,
            "legal_actions": [int(action) for action in legal],
            "hole_cards": [],
            "record_count": 0,
            "action_totals": defaultdict(float),
            "history": list(info.get("history") or []),
            "exact_infoset_key": exact_key,
        },
    )
    bucket["record_count"] += 1
    for entry in entries:
        bucket["action_totals"][int(entry["action"])] += float(entry["probability"])

    wrapped = getattr(resolved, "_wrapped_state", None)
    if wrapped is not None:
        hole_cards = getattr(wrapped, "hole_cards", []) or []
        if player < len(hole_cards):
            bucket["hole_cards"] = [str(card) for card in hole_cards[player]]


def collect_checkpoint_snapshots(
    policy,
    state,
    history_samples: int = 0,
    history_depth: int = 3,
    street_samples: int = 0,
    deal_samples: int = 0,
    deal_game=None,
):
    """Collect a structured set of policy snapshots for a checkpoint.

    The returned payload can be persisted to JSON and later used as a warm-start
    seed for custom MCCFR or policy initialization logic. We keep a single MCCFR run,
    but we sample several fresh deal states in the reporting layer to expose hole-card
    and board diversity without changing the solver's training loop.
    """
    snapshots = []

    if deal_samples > 0 and deal_game is not None:
        deal_states = sample_distinct_deal_states(deal_game, target_count=deal_samples)
        for resolved in deal_states:
            record = policy_snapshot_record(policy, resolved, history=[], label="deal_sample")
            if record is not None:
                snapshots.append(record)
        if snapshots:
            return snapshots

    if street_samples > 0:
        for history, resolved in sample_street_boundary_states(state, max_states=street_samples):
            record = policy_snapshot_record(policy, resolved, history=history, label="street_boundary")
            if record is not None:
                snapshots.append(record)
        return snapshots

    if history_samples > 0:
        for history, resolved in sample_actor_state_histories(
            state,
            max_depth=history_depth,
            max_states=history_samples,
        ):
            record = policy_snapshot_record(policy, resolved, history=history, label="history")
            if record is not None:
                snapshots.append(record)
        if snapshots:
            return snapshots

    resolved = resolve_actor_state(state)
    if resolved is not None:
        record = policy_snapshot_record(policy, resolved, history=[], label="resolved_state")
        if record is not None:
            snapshots.append(record)
    return snapshots


def make_solver(game, solver_name: str):
    solver_name = solver_name.lower()
    if solver_name == "external":
        return pyspiel.ExternalSamplingMCCFRSolver(game)
    if solver_name == "outcome":
        return pyspiel.OutcomeSamplingMCCFRSolver(game)
    raise ValueError(f"unsupported solver '{solver_name}'. choose 'external' or 'outcome'")


def parse_node_selector(value: str) -> dict:
    """Parse NAME=ACTION,ACTION selectors for selected node presets."""
    name, separator, raw_history = value.partition("=")
    if not separator or not name.strip():
        raise ValueError("node selector must use NAME=ACTION,ACTION")

    history = []
    for token in (item.strip().lower() for item in raw_history.split(",")):
        if not token:
            continue
        if token in HULH_ACTION_SEQUENCE_TO_HISTORY:
            history.extend(HULH_ACTION_SEQUENCE_TO_HISTORY[token])
            continue
        if token in {"fold", "check", "call", "check_call", "bet", "raise", "bet_raise"}:
            history.append(token)
            continue
        try:
            history.append(int(token))
        except ValueError as exc:
            raise ValueError(f"unknown action '{token}' in node selector '{value}'") from exc

    return {"name": name.strip(), "history": history}


def resolve_node_specs(preset: str = None, node_selectors=()):
    """Return the selected node specs to sample and report."""
    preset_name = preset or "root"
    if preset_name not in NODE_PRESETS:
        raise ValueError(f"unsupported node preset '{preset_name}'")

    specs = [dict(item) for item in NODE_PRESETS[preset_name]]
    specs.extend(parse_node_selector(value) for value in (node_selectors or []))
    if not specs:
        raise ValueError("at least one selected node is required")

    seen_names = set()
    for spec in specs:
        spec["normalized_name"] = HULH_NODE_LABELS.get(spec["name"], spec["name"])
        spec["display_name"] = format_hulh_history_label(spec["history"])
        if spec["name"] in seen_names:
            raise ValueError(f"duplicate selected node name '{spec['name']}'")
        seen_names.add(spec["name"])
    return specs


def state_after_history(game, history):
    """Sample a random deal and follow the requested action history."""
    state = sample_random_actor_state(game)
    if state is None:
        return None

    for requested in history:
        legal = list(state.legal_actions())
        if isinstance(requested, str):
            chosen = choose_action_from_family(legal, requested)
        else:
            chosen = int(requested) if int(requested) in legal else None
        if chosen is None:
            return None
        state = state.child(chosen)
        if state is None:
            return None

    return resolve_actor_state(state)


def canonical_action_family(action: int) -> str:
    if int(action) == 0:
        return "fold"
    if int(action) == 1:
        return "check_call"
    return "bet_raise"


def choose_action_from_family(legal_actions, family: str):
    normalized = family.lower()
    if normalized in {"check", "call"}:
        normalized = "check_call"
    if normalized in {"bet", "raise"}:
        normalized = "bet_raise"

    target = ACTION_NAME_TO_ID.get(normalized)
    if target in legal_actions:
        return target

    matching = [action for action in legal_actions if canonical_action_family(action) == normalized]
    if not matching:
        return None
    return sorted(matching)[0]


def prepare_selected_node_probes(game, node_specs, samples_per_node: int, max_attempts: int = None, dedupe: bool = False):
    """Sample deal states for each selected node.

    We intentionally keep the default path unbiased: one random deal is one sample.
    Dedupe-by-signature remains available as an explicit `dedupe=True` option for
    diversity-only coverage studies, but it is not used in the normal export path.
    """
    if max_attempts is None:
        max_attempts = max(samples_per_node * 20, 2000)

    probes = []
    for spec in node_specs:
        seen = set()
        attempts = 0
        no_progress_rounds = 0
        while len(probes) < len(node_specs) * samples_per_node and attempts < max_attempts:
            attempts += 1
            state = state_after_history(game, spec["history"])
            if state is None:
                continue
            if dedupe:
                signature = exact_hole_board_signature(state)
                if signature in seen:
                    no_progress_rounds += 1
                    if len(seen) > 0 and no_progress_rounds >= max(50, min(250, samples_per_node // 2)):
                        break
                    continue
                seen.add(signature)
                no_progress_rounds = 0
            probes.append({"node_name": spec["name"], "history": list(spec["history"]), "state": state})

        if dedupe and len(seen) < samples_per_node:
            warnings.warn(
                f"only sampled {len(seen)} distinct deals for selected node '{spec['name']}' (requested {samples_per_node}); continuing with the available states",
                UserWarning,
                stacklevel=2,
            )
    return probes


def snapshot_probe_states(policy, probes):
    records = []
    for probe in probes:
        record = policy_snapshot_record(
            policy,
            probe["state"],
            history=probe["history"],
            label=probe["node_name"],
        )
        if record is not None:
            record["node_name"] = probe["node_name"]
            record["normalized_name"] = HULH_NODE_LABELS.get(probe["node_name"], probe["node_name"])
            record["display_name"] = format_hulh_history_label(probe["history"])
            record["selected_history"] = list(probe["history"])
            records.append(record)
    return records


def summarize_selected_node_stability(current_ranges, previous_ranges=None, threshold: float = 0.01):
    """Summarize checkpoint-to-checkpoint drift for selected-node range policies."""
    current_nodes = {
        node["name"]: node
        for node in (current_ranges or {}).get("nodes", [])
        if node.get("name") is not None
    }
    previous_nodes = {
        node["name"]: node
        for node in (previous_ranges or {}).get("nodes", [])
        if node.get("name") is not None
    }

    deltas = []
    moving = []
    for name, current in current_nodes.items():
        prev = previous_nodes.get(name)
        if prev is None:
            continue

        current_actions = current.get("action_frequencies") or {}
        previous_actions = prev.get("action_frequencies") or {}
        node_action_deltas = []
        for action in sorted(set(current_actions) | set(previous_actions)):
            delta = abs(float(current_actions.get(action, 0.0)) - float(previous_actions.get(action, 0.0)))
            deltas.append(delta)
            node_action_deltas.append(delta)

        current_hands = {hand["hand"]: hand for hand in current.get("hands", []) if hand.get("hand")}
        previous_hands = {hand["hand"]: hand for hand in prev.get("hands", []) if hand.get("hand")}
        for hand_name, hand in current_hands.items():
            prev_hand = previous_hands.get(hand_name)
            if prev_hand is None:
                continue
            for action in sorted(set(hand.get("policy") or {}) | set(prev_hand.get("policy") or {})):
                delta = abs(
                    float((hand.get("policy") or {}).get(action, 0.0))
                    - float((prev_hand.get("policy") or {}).get(action, 0.0))
                )
                deltas.append(delta)
                node_action_deltas.append(delta)

        moving.append(
            {
                "node_name": name,
                "max_delta": max(node_action_deltas) if node_action_deltas else 0.0,
            }
        )

    moving.sort(key=lambda item: item["max_delta"], reverse=True)
    max_delta = max(deltas) if deltas else None
    return {
        "sample_count": len(current_nodes),
        "matched_nodes": len([node for node in current_nodes if node in previous_nodes]),
        "avg_abs_delta": statistics.fmean(deltas) if deltas else None,
        "max_abs_delta": max_delta,
        "top_moving": moving[:5],
        "threshold": threshold,
        "passed": max_delta is not None and max_delta <= threshold,
    }


def summarize_durations(durations):
    """Return stable timing statistics for a sequence of durations in seconds."""
    values = [float(value) for value in durations]
    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "stddev_seconds": 0.0,
            "iterations_per_second": 0.0,
        }

    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    total = sum(values)
    return {
        "count": len(values),
        "total_seconds": total,
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[p95_index],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "stddev_seconds": statistics.pstdev(values),
        "iterations_per_second": len(values) / total if total else 0.0,
    }


def build_selected_node_summary(range_payload):
    summary_rows = []
    for node in (range_payload or {}).get("nodes", []):
        summary_rows.append(
            {
                "node_name": node.get("name"),
                "display_name": node.get("display_name") or node.get("history_label") or node.get("name"),
                "history": list(node.get("history") or []),
                "sample_count": node.get("sample_count"),
                "action_frequencies": dict(node.get("action_frequencies") or {}),
            }
        )
    return summary_rows


def compact_checkpoint_payload(payload):
    return {
        "mode": payload.get("mode"),
        "solver": payload.get("solver"),
        "iteration": payload.get("iteration"),
        "checkpoint_every": payload.get("checkpoint_every"),
        "selected_node_summary": build_selected_node_summary(payload.get("range_policies") or {}),
        "stability": payload.get("stability"),
    }


def compact_report_payload(report):
    return {
        "schema_version": report.get("schema_version"),
        "mode": report.get("mode"),
        "game_parameters": report.get("game_parameters"),
        "solver": report.get("solver"),
        "iterations": report.get("iterations"),
        "completed_iterations": report.get("completed_iterations"),
        "stopped_early": report.get("stopped_early"),
        "checkpoint_every": report.get("checkpoint_every"),
        "samples_per_node": report.get("samples_per_node"),
        "selected_node_preset": report.get("selected_node_preset"),
        "selected_nodes": report.get("selected_nodes"),
        "selected_node_summary": report.get("selected_node_summary"),
        "stop_policy": report.get("stop_policy"),
        "performance": report.get("performance"),
        "checkpoint_history": [
            compact_checkpoint_payload(item) for item in (report.get("checkpoint_history") or [])
        ],
    }


def format_selected_node_summary_table(selected_summary, iteration=None, stability=None, heading=None):
    lines = []
    title = heading or "selected node summary"
    if iteration is not None:
        title = f"{title} @ iter {iteration}"
    lines.append(title)
    headers = ("node", "samples", "fold", "check/call", "bet/raise")
    rows = []
    for item in selected_summary or []:
        freqs = item.get("action_frequencies") or {}
        rows.append(
            (
                str(item.get("display_name") or item.get("node_name") or "node"),
                str(item.get("sample_count") or 0),
                f"{float(freqs.get('fold', 0.0)):.3f}",
                f"{float(freqs.get('check_call', 0.0)):.3f}",
                f"{float(freqs.get('bet_raise', 0.0)):.3f}",
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    lines.append(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    if stability is not None:
        lines.append(
            "stability: "
            f"passed={stability.get('passed')} "
            f"avg_abs_delta={float(stability.get('avg_abs_delta') or 0.0):.4f} "
            f"max_abs_delta={float(stability.get('max_abs_delta') or 0.0):.4f} "
            f"matched_nodes={int(stability.get('matched_nodes') or 0)}"
        )
    return "\n".join(lines)


def max_rss_mb():
    """Return the process max RSS in MiB, normalized across Linux/macOS."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = getattr(usage, "ru_maxrss", 0)
    except Exception:
        return {"max_rss_mb": None, "unit": "MiB", "available": False}

    if sys.platform == "darwin":
        rss_mb = float(max_rss) / (1024.0 * 1024.0)
    else:
        rss_mb = float(max_rss) / 1024.0

    return {"max_rss_mb": rss_mb, "unit": "MiB", "available": True}


def filter_recent_iteration_records(records, last_n_iterations: int | None):
    """Return records from the final N iteration buckets, preserving all records when unset."""
    if last_n_iterations is None or int(last_n_iterations) <= 0:
        return list(records or [])

    last_n_iterations = int(last_n_iterations)
    records = list(records or [])
    if not records:
        return []

    iterated = [(record, int(record.get("iteration"))) for record in records if record.get("iteration") is not None]
    if not iterated:
        return records

    recent_iterations = sorted({iteration for _, iteration in iterated}, reverse=True)[:last_n_iterations]
    recent_set = set(recent_iterations)
    filtered = []
    for record in records:
        iteration = record.get("iteration")
        if iteration is None:
            filtered.append(record)
        elif int(iteration) in recent_set:
            filtered.append(record)
    return filtered


def write_run_artifacts(output_json_path, report, range_records, checkpoint_paths, artifact_mode: str = "standard"):
    """Write each final run artifact exactly once and return its manifest path."""
    if output_json_path is None:
        return None

    base_dir = os.path.dirname(output_json_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(output_json_path))[0]
    range_path = os.path.join(base_dir, f"{basename}_ranges.json")
    stability_path = os.path.join(base_dir, f"{basename}_checkpoint_stability.json")
    summary_path = os.path.join(base_dir, f"{basename}_selected_node_summary.txt")

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    export_range_dump(range_records, range_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(
            format_selected_node_summary_table(
                report.get("selected_node_summary") or [],
                iteration=report.get("completed_iterations"),
                stability=((report.get("stop_policy") or {}).get("strategy_stability") or {}).get("latest"),
                heading="final selected node summary",
            )
        )
        handle.write("\n")

    checkpoint_history = report.get("checkpoint_history", [])
    if checkpoint_history and artifact_mode != "lightweight":
        with open(stability_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint_history, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        stability_path = None

    return write_run_manifest(
        output_json_path,
        range_path=range_path,
        stability_path=stability_path,
        summary_path=summary_path,
        checkpoint_paths=[] if artifact_mode == "lightweight" else checkpoint_paths,
    )


def _latest_checkpoint_stability(checkpoint_history):
    """Return the latest checkpoint stability object, if one exists."""
    if not checkpoint_history:
        return None
    return checkpoint_history[-1].get("stability")


def build_runtime_state(iteration: int, checkpoint_history: list, stopped_early: bool, output_json_path: str | None):
    """Describe the solver's in-memory/live state for a queryable API layer.

    This intentionally keeps the legacy report data intact while exposing a compact,
    API-friendly snapshot of the current solver readiness and convergence state.
    """
    latest_stability = _latest_checkpoint_stability(checkpoint_history)
    stable = bool(latest_stability and latest_stability.get("passed"))

    if stopped_early and stable:
        state = "queryable"
    elif stopped_early and not stable:
        state = "paused"
    else:
        state = "stable" if stable else "running"

    return {
        "state": state,
        "stable": stable,
        "ready_for_queries": stable or state == "queryable",
        "current_policy_in_memory": True,
        "latest_stable_snapshot": latest_stability,
        "checkpoint_on_disk": bool(output_json_path),
        "iteration": iteration,
        "last_probe_at": checkpoint_history[-1].get("iteration") if checkpoint_history else None,
    }


def build_selected_node_export(records, range_last_n: int | None):
    """Return the filtered records used for final selected-node export and the associated aggregate."""
    filtered_records = filter_recent_iteration_records(records, range_last_n)
    export_records = filtered_records if range_last_n is not None else records
    return export_records, aggregate_selected_node_ranges(export_records)


def profile_variant(
    name: str,
    params: Dict[str, object],
    iterations: int = 10,
    stability_checkpoint: int = 0,
    solver_name: str = "external",
    history_samples: int = 0,
    history_depth: int = 3,
    street_samples: int = 0,
    report_mode: str = "policy",
    output_json_path: str = None,
    range_samples: int | None = None,
    postflop_samples: int | None = None,
    heartbeat_seconds: float = 10.0,
    node_preset: str = None,
    node_selectors=(),
    stability_threshold: float = 0.01,
    stop_patience: int = 3,
    min_iterations: int = 1_000_000,
    range_last_n: int | None = None,
    artifact_mode: str = "standard",
    checkpoint_history_limit: int | None = None,
    checkpoint_every: int | None = None,
    samples: int | None = None,
):
    if checkpoint_every is not None:
        if stability_checkpoint != 0 and stability_checkpoint != checkpoint_every:
            raise ValueError("stability_checkpoint and checkpoint_every must match when both are specified")
        stability_checkpoint = checkpoint_every
    if samples is not None:
        if range_samples is not None and range_samples != samples:
            raise ValueError("range_samples and samples must match when both are specified")
        range_samples = samples
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if stability_checkpoint < 0:
        raise ValueError("stability_checkpoint cannot be negative")
    if any(value < 0 for value in (history_samples, history_depth, street_samples)):
        raise ValueError("sample counts and history depth cannot be negative")
    if range_samples is not None and range_samples < 0:
        raise ValueError("range_samples cannot be negative")
    if postflop_samples is not None and postflop_samples < 0:
        raise ValueError("postflop_samples cannot be negative")
    if heartbeat_seconds < 0:
        raise ValueError("heartbeat_seconds cannot be negative")
    if stability_threshold <= 0:
        raise ValueError("stability_threshold must be greater than zero")
    if stop_patience <= 0:
        raise ValueError("stop_patience must be greater than zero")
    if min_iterations < 0:
        raise ValueError("min_iterations cannot be negative")
    if artifact_mode not in {"standard", "lightweight"}:
        raise ValueError("artifact_mode must be 'standard' or 'lightweight'")
    if checkpoint_history_limit is not None and checkpoint_history_limit < 0:
        raise ValueError("checkpoint_history_limit cannot be negative")

    report_snapshots = []
    checkpoint_history = []
    previous_checkpoint_ranges = None
    checkpoint_file_paths = []
    iteration_durations = []
    checkpoint_durations = []
    selected_node_specs = resolve_node_specs(
        node_preset or ("hulh-preflop" if name == "hulh" else "root"),
        node_selectors,
    )
    stop_policy_state = {
        "consecutive_stable_checkpoints": 0,
        "stopped_early": False,
    }
    print(f"\n=== {name.upper()} profile ===")
    print(f"solver={solver_name}")

    start = time.perf_counter()
    game = pyspiel.load_game("python_pokerkit_wrapper", params)
    load_elapsed = time.perf_counter() - start
    print(f"load_game: {load_elapsed:.4f}s")

    start = time.perf_counter()
    state = game.new_initial_state()
    initial_state_elapsed = time.perf_counter() - start
    print(f"new_initial_state: {initial_state_elapsed:.4f}s")

    start = time.perf_counter()
    legal = state.legal_actions()
    legal_elapsed = time.perf_counter() - start
    print(f"initial_legal_actions: {legal_elapsed:.4f}s len={len(legal)} sample={legal[:10]}")

    start = time.perf_counter()
    solver = make_solver(game, solver_name)
    solver_elapsed = time.perf_counter() - start
    print(f"solver_construct: {solver_elapsed:.4f}s")

    probe_count = deal_budget_for_iterations(iterations) if range_samples is None else range_samples
    start = time.perf_counter()
    probes = prepare_selected_node_probes(
        game,
        selected_node_specs,
        samples_per_node=probe_count,
    )
    probe_preparation_elapsed = time.perf_counter() - start
    print(f"probe_preparation: {probe_preparation_elapsed:.4f}s count={len(probes)}")

    last_heartbeat = time.perf_counter()
    training_wall_start = time.perf_counter()
    for idx in range(iterations):
        iteration_start = time.perf_counter()
        solver.run_iteration()
        iteration_durations.append(time.perf_counter() - iteration_start)

        if stability_checkpoint and (idx + 1) % stability_checkpoint == 0:
            checkpoint_start = time.perf_counter()
            policy = solver.average_policy()
            checkpoint_records = snapshot_probe_states(policy, probes)
            for record in checkpoint_records:
                record["iteration"] = idx + 1
                report_snapshots.append(record)

            checkpoint_ranges = aggregate_selected_node_ranges(checkpoint_records)
            checkpoint_summary = summarize_selected_node_stability(
                checkpoint_ranges,
                previous_checkpoint_ranges,
                threshold=stability_threshold,
            )
            if checkpoint_summary["passed"]:
                stop_policy_state["consecutive_stable_checkpoints"] += 1
            else:
                stop_policy_state["consecutive_stable_checkpoints"] = 0
            checkpoint_payload = {
                "mode": name,
                "solver": solver_name,
                "iteration": idx + 1,
                "stability_checkpoint": stability_checkpoint,
                "records": checkpoint_records,
                "range_policies": checkpoint_ranges,
                "stability": checkpoint_summary,
            }
            checkpoint_history.append(checkpoint_payload)
            if checkpoint_history_limit is not None:
                checkpoint_history = checkpoint_history[-checkpoint_history_limit:]
            if artifact_mode != "lightweight":
                checkpoint_artifact = write_checkpoint_payload(
                    output_json_path,
                    idx + 1,
                    checkpoint_payload,
                    {
                        "iteration": idx + 1,
                        "stability_checkpoint": stability_checkpoint,
                        "stability": checkpoint_summary,
                    },
                )
                if checkpoint_artifact:
                    checkpoint_file_paths.append(checkpoint_artifact["checkpoint_path"])
            previous_checkpoint_ranges = checkpoint_ranges
            checkpoint_durations.append(time.perf_counter() - checkpoint_start)
            if (
                idx + 1 >= min_iterations
                and stop_policy_state["consecutive_stable_checkpoints"] >= stop_patience
            ):
                stop_policy_state["stopped_early"] = True
                break
        now = time.perf_counter()
        if heartbeat_seconds and now - last_heartbeat >= heartbeat_seconds:
            elapsed = time.perf_counter() - training_wall_start
            remaining = max(iterations - idx - 1, 0)
            print(
                f"heartbeat: completed={idx + 1}/{iterations} "
                f"remaining={remaining} elapsed={elapsed:.2f}s"
            )
            last_heartbeat = now
    training_wall_elapsed = time.perf_counter() - training_wall_start
    iteration_timing = summarize_durations(iteration_durations)
    checkpoint_timing = summarize_durations(checkpoint_durations)
    print(
        f"{iterations}_iterations: solver={iteration_timing['total_seconds']:.4f}s "
        f"wall={training_wall_elapsed:.4f}s rate={iteration_timing['iterations_per_second']:.2f}/s"
    )

    start = time.perf_counter()
    policy = solver.average_policy()
    avg_policy_elapsed = time.perf_counter() - start
    print(f"average_policy: {avg_policy_elapsed:.4f}s")

    start = time.perf_counter()
    resolved = resolve_actor_state(state)
    resolved_elapsed = time.perf_counter() - start
    print(f"resolve_actor_state: {resolved_elapsed:.4f}s")
    if resolved is None:
        print("no valid actor state found")
        return

    player = resolved.current_player()
    print(f"resolved_player={player} legal_count={len(resolved.legal_actions())}")

    start = time.perf_counter()
    final_records = snapshot_probe_states(policy, probes)
    for record in final_records:
        record["iteration"] = iterations
    final_snapshot_elapsed = time.perf_counter() - start

    warm_start_infosets = []
    if report_mode in {"warm_start", "all"}:
        warm_start_infosets = [
            {
                "state_serialize": item["state_serialize"],
                "player": item["player"],
                "legal_actions": item["legal_actions"],
                "current_policy": {
                    str(entry["action"]): float(entry["probability"])
                    for entry in item["action_probabilities"]
                },
                "label": item["label"],
                "history": item["history"],
                "iteration": item.get("iteration"),
                "prob_sum": item.get("prob_sum"),
                "street": item.get("street"),
                "pot_context": item.get("pot_context"),
                "summary": item.get("summary"),
            }
            for item in final_records
        ]

    include_full_policy = report_mode in {"policy", "all"}
    include_summary_only = report_mode == "summary"

    reporting_start = time.perf_counter()
    summary = summarize_policy_profiles(final_records)
    range_export_records, final_ranges = build_selected_node_export(final_records, range_last_n)
    selected_summary = build_selected_node_summary(final_ranges)
    reporting_elapsed = time.perf_counter() - reporting_start
    print(
        "policy_profile_summary: "
        f"unique_profiles={summary['unique_policy_profiles']} "
        f"repeated_same_family_preflop_states={summary['repeated_same_family_preflop_states']} "
        f"deeper_non_preflop_states={summary['deeper_non_preflop_states']}"
    )
    memory_stats = max_rss_mb()
    print(f"memory: max_rss_mb={memory_stats['max_rss_mb']:.2f} if memory_stats['max_rss_mb'] is not None else 'unavailable'")

    report = {
        "schema_version": 2,
        "artifact_mode": artifact_mode,
        "mode": name,
        "game_parameters": params,
        "solver": solver_name,
        "iterations": iterations,
        "completed_iterations": len(iteration_durations),
        "stopped_early": stop_policy_state["stopped_early"],
        "stability_checkpoint": stability_checkpoint,
        "history_samples": history_samples,
        "history_depth": history_depth,
        "street_samples": street_samples,
        "range_samples": range_samples,
        "postflop_samples": max(int(postflop_samples if postflop_samples is not None else 32), 1),
        "samples_per_node": probe_count,
        "selected_node_preset": node_preset or ("hulh-preflop" if name == "hulh" else "root"),
        "selected_nodes": selected_node_specs,
        "report_mode": report_mode,
        "range_last_n_iterations": range_last_n,
        "range_export_window": {
            "requested": range_last_n,
            "applied": range_last_n if range_last_n is not None else None,
            "records_in_window": len(range_export_records),
            "full_run_records": len(final_records),
            "iteration_min": min((record.get("iteration") for record in range_export_records if record.get("iteration") is not None), default=None),
            "iteration_max": max((record.get("iteration") for record in range_export_records if record.get("iteration") is not None), default=None),
        },
        "snapshots": report_snapshots if include_full_policy else [],
        "final_policy_records": final_records if include_full_policy else [],
        "selected_node_records": final_records if include_full_policy else [],
        "range_policies": final_ranges,
        "selected_node_ranges": final_ranges,
        "selected_node_summary": selected_summary,
        "full_run_ranges": aggregate_selected_node_ranges(final_records),
        "warm_start": {"infosets": warm_start_infosets} if report_mode in {"warm_start", "all"} else {"infosets": []},
        "checkpoint_history": checkpoint_history,
        "policy_profile_summary": summary,
        "stop_policy": {
            "recommendation": (
                "stop"
                if stop_policy_state["consecutive_stable_checkpoints"] >= stop_patience
                else "continue"
            ),
            "strategy_stability": {
                "required_consecutive_checkpoints": stop_patience,
                "consecutive_passing_checkpoints": stop_policy_state["consecutive_stable_checkpoints"],
                "max_action_delta_threshold": stability_threshold,
                "latest": checkpoint_history[-1]["stability"] if checkpoint_history else None,
            },
            "average_strategy_exported": True,
            "exploitability": {
                "status": "not_measured",
                "reason": "exact exploitability would dominate the profiling run",
            },
            "ev_stability": {
                "status": "not_measured",
                "reason": "selected-node range reporting does not compute exact branch EV",
            },
            "average_vs_last_iterate": {
                "status": "not_available",
                "reason": "the solver binding exposes average policy rather than last iterate policy",
            },
            "out_of_sample_robustness": {
                "status": "requires_separate_seeded_run",
            },
        },
        "performance": {
            "setup": {
                "load_game_seconds": load_elapsed,
                "initial_state_seconds": initial_state_elapsed,
                "initial_legal_actions_seconds": legal_elapsed,
                "solver_construct_seconds": solver_elapsed,
                "probe_preparation_seconds": probe_preparation_elapsed,
            },
            "training": {
                **iteration_timing,
                "wall_seconds": training_wall_elapsed,
                "measurement_scope": "solver.run_iteration only",
            },
            "checkpoint_overhead": {
                **checkpoint_timing,
                "measurement_scope": "average policy, selected-node snapshots, stability, and checkpoint file",
            },
            "finalization": {
                "average_policy_seconds": avg_policy_elapsed,
                "resolve_actor_state_seconds": resolved_elapsed,
                "snapshot_seconds": final_snapshot_elapsed,
                "report_aggregation_seconds": reporting_elapsed,
            },
            "memory": {
                **memory_stats,
                "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            },
        },
    }
    report["runtime_state"] = build_runtime_state(
        iteration=len(iteration_durations),
        checkpoint_history=checkpoint_history,
        stopped_early=stop_policy_state["stopped_early"],
        output_json_path=output_json_path,
    )

    write_run_artifacts(output_json_path, report, final_records, checkpoint_file_paths, artifact_mode=artifact_mode)
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark MCCFR training and export selected-node range policies."
    )
    parser.add_argument("mode", nargs="?", default="hulh", choices=sorted(GAME_CONFIGS), help="game mode to profile")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="number of measured MCCFR iterations to run (default: 100)",
    )
    parser.add_argument(
        "--stability-checkpoint",
        "--checkpoint-every",
        dest="stability_checkpoint",
        type=int,
        default=0,
        help="evaluate policy stability every N iterations; 0 disables stability checkpoints; --checkpoint-every is kept as a compatibility alias",
    )
    parser.add_argument(
        "--solver",
        choices=["external", "outcome"],
        default="external",
        help="MCCFR variant to use: external or outcome (default: external)",
    )
    parser.add_argument(
        "--range-samples",
        type=int,
        default=1000,
        help="distinct deals sampled per selected node for the final range export (default: 1000)",
    )
    parser.add_argument(
        "--postflop-samples",
        type=int,
        default=32,
        help="fallback sample count for query-time postflop exact/range lookups (default: 32)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(NODE_PRESETS),
        default=None,
        help="named selected-node preset (default: hulh-preflop for HULH, root otherwise)",
    )
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        metavar="NAME=ACTION,ACTION",
        help="add a selected node selector using fold/call/bet names or raw action IDs",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.01,
        help="max action-frequency delta for a stable checkpoint (default: 0.01)",
    )
    parser.add_argument(
        "--stop-patience",
        type=int,
        default=3,
        help="number of consecutive stable checkpoints before stopping early (default: 3)",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=1_000_000,
        help="minimum iterations before early stop can trigger (default: 1000000)",
    )
    parser.add_argument(
        "--report-mode",
        choices=["policy", "warm_start", "summary", "all"],
        default="policy",
        help="choose the JSON report shape: policy (snapshot-only), warm_start (seedable checkpoint payload), summary (compact summary only), or all (both)",
    )
    parser.add_argument(
        "--range-last-n",
        type=int,
        default=None,
        help="export recent range data only from the final N iterations; defaults to the full run",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=["standard", "lightweight"],
        default="standard",
        help="standard keeps every checkpoint file; lightweight keeps only final compact outputs and recent stability history",
    )
    parser.add_argument(
        "--checkpoint-history-limit",
        type=int,
        default=None,
        help="when using lightweight mode, keep only the most recent N checkpoint stability entries in memory for the report",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="write the report and sibling artifacts using this JSON path as the basename",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=10.0,
        help="progress interval in seconds; 0 disables progress output (default: 10)",
    )
    args = parser.parse_args()

    profile_variant(
        args.mode,
        GAME_CONFIGS[args.mode],
        iterations=args.iterations,
        stability_checkpoint=args.stability_checkpoint,
        solver_name=args.solver,
        report_mode=args.report_mode,
        output_json_path=args.output_json,
        range_samples=args.range_samples,
        postflop_samples=args.postflop_samples,
        heartbeat_seconds=args.heartbeat_seconds,
        node_preset=args.preset,
        node_selectors=args.node,
        stability_threshold=args.stability_threshold,
        stop_patience=args.stop_patience,
        min_iterations=args.min_iterations,
        range_last_n=args.range_last_n,
        artifact_mode=args.artifact_mode,
        checkpoint_history_limit=args.checkpoint_history_limit,
    )


if __name__ == "__main__":
    main()
