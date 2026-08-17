import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from absl import logging as absl_logging
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

absl_logging.set_verbosity(absl_logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from range_export import aggregate_range_profiles, export_range_dump


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
    stability_path = os.path.join(base_dir, f"{basename}_checkpoint_stability.json")

    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    existing = []
    if os.path.exists(stability_path):
        try:
            with open(stability_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except Exception:
            existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append(stability_payload)
    with open(stability_path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {"checkpoint_path": checkpoint_path, "stability_path": stability_path}


def write_run_manifest(report_path: str, range_path: str = None, stability_path: str = None, checkpoint_paths: Iterable[str] = ()):
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
    artifact_paths.extend(checkpoint_paths)
    artifact_paths = sorted({os.path.abspath(path) for path in artifact_paths if path})

    manifest = {
        "report_path": os.path.abspath(report_path),
        "range_path": os.path.abspath(range_path) if range_path else None,
        "checkpoint_stability_path": os.path.abspath(stability_path) if stability_path else None,
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


def sample_distinct_deal_states(game, target_count: int, max_attempts: int = 200):
    """Return fresh root states with diverse private-card/board signatures for reporting."""
    states = []
    seen = set()
    attempts = 0

    while len(states) < target_count and attempts < max_attempts:
        attempts += 1
        resolved = sample_random_actor_state(game)
        if resolved is None:
            continue
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


def profile_variant(
    name: str,
    params: Dict[str, object],
    iterations: int = 10,
    checkpoint_every: int = 0,
    solver_name: str = "external",
    history_samples: int = 0,
    history_depth: int = 3,
    street_samples: int = 0,
    report_mode: str = "policy",
    output_json_path: str = None,
):
    report_snapshots = []
    full_run_records = []
    checkpoint_history = []
    previous_checkpoint_records = []
    global_infoset_accumulator = {}
    checkpoint_file_paths = []
    print(f"\n=== {name.upper()} profile ===")
    print(f"solver={solver_name}")

    start = time.perf_counter()
    game = pyspiel.load_game("python_pokerkit_wrapper", params)
    load_elapsed = time.perf_counter() - start
    print(f"load_game: {load_elapsed:.4f}s")

    report_budget = deal_budget_for_iterations(iterations)
    print(f"report_budget={report_budget} for total_iterations={iterations}")

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

    heartbeat_interval = 10.0
    last_heartbeat = time.perf_counter()
    start = time.perf_counter()
    for idx in range(iterations):
        solver.run_iteration()
        policy = solver.average_policy()
        sampled_state = sample_random_actor_state(game)
        if sampled_state is not None:
            accumulate_global_infoset_policy(global_infoset_accumulator, sampled_state, policy)

        if checkpoint_every and (idx + 1) % checkpoint_every == 0:
            if street_samples > 0:
                snapshot_street_boundaries(policy, state, max_states=street_samples)
            elif history_samples > 0:
                snapshot_policy_histories(policy, state, max_depth=history_depth, max_states=history_samples)
            else:
                snapshot_policy(policy, state)
            checkpoint_records = collect_checkpoint_snapshots(
                policy,
                state,
                history_samples=history_samples,
                history_depth=history_depth,
                street_samples=street_samples,
                deal_samples=report_budget,
                deal_game=game,
            )
            for record in checkpoint_records:
                record["iteration"] = idx + 1
                report_snapshots.append(record)
                full_run_records.append(record)

            checkpoint_summary = summarize_checkpoint_stability(checkpoint_records, previous_checkpoint_records)
            checkpoint_payload = {
                "mode": name,
                "solver": solver_name,
                "iteration": idx + 1,
                "checkpoint_every": checkpoint_every,
                "records": checkpoint_records,
                "stability": checkpoint_summary,
            }
            checkpoint_history.append(checkpoint_payload)
            checkpoint_artifact = write_checkpoint_payload(
                output_json_path,
                idx + 1,
                checkpoint_payload,
                {
                    "iteration": idx + 1,
                    "checkpoint_every": checkpoint_every,
                    "stability": checkpoint_summary,
                },
            )
            if checkpoint_artifact:
                checkpoint_file_paths.append(checkpoint_artifact["checkpoint_path"])
            previous_checkpoint_records = checkpoint_records
        now = time.perf_counter()
        if now - last_heartbeat >= heartbeat_interval:
            elapsed = time.perf_counter() - start
            remaining = max(iterations - idx - 1, 0)
            print(
                f"heartbeat: completed={idx + 1}/{iterations} "
                f"remaining={remaining} elapsed={elapsed:.2f}s"
            )
            last_heartbeat = now
    iteration_elapsed = time.perf_counter() - start
    print(f"{iterations}_iterations: {iteration_elapsed:.4f}s")

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
    pol = policy.get_state_policy(resolved, player)
    lookup_elapsed = time.perf_counter() - start
    print(f"policy_get_state_policy: {lookup_elapsed:.4f}s sample={pol[:10]} prob_sum={sum(p for _, p in pol):.6f}")

    final_record = policy_snapshot_record(policy, resolved, history=[], label="final_state")
    if final_record is not None:
        final_record_with_iteration = {**final_record, "iteration": iterations}
        report_snapshots.append(final_record_with_iteration)
        full_run_records.append(final_record_with_iteration)

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
            for item in report_snapshots
        ]

    global_run_records = []
    for bucket in global_infoset_accumulator.values():
        policy_payload = [
            {"action": int(action), "probability": float(probability / max(bucket["record_count"], 1))}
            for action, probability in sorted(bucket["action_totals"].items())
        ]
        global_run_records.append(
            {
                "label": "global_accumulator",
                "history": bucket.get("history", []),
                "player": bucket.get("player"),
                "legal_actions": bucket.get("legal_actions", []),
                "action_probabilities": policy_payload,
                "prob_sum": float(sum(item["probability"] for item in policy_payload)),
                "street": bucket.get("street"),
                "pot_context": bucket.get("pot_context"),
                "hole_cards": bucket.get("hole_cards", []),
                "exact_infoset_key": bucket.get("exact_infoset_key"),
                "record_count": bucket.get("record_count", 0),
            }
        )

    summary = summarize_policy_profiles(full_run_records + global_run_records)
    full_run_ranges = aggregate_range_profiles(full_run_records + global_run_records)
    print(
        "policy_profile_summary: "
        f"unique_profiles={summary['unique_policy_profiles']} "
        f"repeated_same_family_preflop_states={summary['repeated_same_family_preflop_states']} "
        f"deeper_non_preflop_states={summary['deeper_non_preflop_states']}"
    )
    report = {
        "mode": name,
        "solver": solver_name,
        "iterations": iterations,
        "checkpoint_every": checkpoint_every,
        "history_samples": history_samples,
        "history_depth": history_depth,
        "street_samples": street_samples,
        "report_mode": report_mode,
        "snapshots": report_snapshots if report_mode in {"policy", "all"} else [],
        "full_run_records": full_run_records if report_mode in {"policy", "all"} else [],
        "global_infoset_accumulator": global_run_records,
        "full_run_ranges": full_run_ranges,
        "warm_start": {"infosets": warm_start_infosets},
        "checkpoint_history": checkpoint_history,
        "policy_profile_summary": summary,
    }

    if output_json_path is not None:
        base_dir = os.path.dirname(output_json_path) or "."
        os.makedirs(base_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        range_path = os.path.join(base_dir, f"{os.path.splitext(os.path.basename(output_json_path))[0]}_ranges.json")
        export_range_dump(full_run_records, range_path)
        stability_path = os.path.join(base_dir, f"{os.path.splitext(os.path.basename(output_json_path))[0]}_checkpoint_stability.json")
        if checkpoint_history:
            with open(stability_path, "w", encoding="utf-8") as handle:
                json.dump(checkpoint_history, handle, indent=2, sort_keys=True)
                handle.write("\n")
        write_run_manifest(output_json_path, range_path=range_path, stability_path=stability_path, checkpoint_paths=checkpoint_file_paths)
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile the PokerKit OpenSpiel wrapper solver.")
    parser.add_argument("mode", nargs="?", default="hulh", choices=sorted(GAME_CONFIGS), help="game mode to profile")
    parser.add_argument(
        "-n",
        "--niterations",
        type=int,
        default=10,
        help="number of MCCFR iterations to run (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="print the average-policy snapshot every N iterations; 0 disables (default)",
    )
    parser.add_argument(
        "--solver",
        choices=["external", "outcome"],
        default="external",
        help="MCCFR variant to use: external or outcome (default: external)",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=0,
        help="sample this many real action-history actor states at each checkpoint; 0 uses the first valid actor state only",
    )
    parser.add_argument(
        "--history-depth",
        type=int,
        default=3,
        help="max action-history depth when sampling history states (default: 3)",
    )
    parser.add_argument(
        "--street-samples",
        type=int,
        default=0,
        help="sample this many valid states at street boundaries (preflop/flop/turn/river) during each checkpoint; 0 disables",
    )
    parser.add_argument(
        "--report-mode",
        choices=["policy", "warm_start", "all"],
        default="policy",
        help="choose the JSON report shape: policy (snapshot-only), warm_start (seedable checkpoint payload), or all (both)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="write a JSON summary report to this path (for example: /app/report.json)",
    )
    args = parser.parse_args()

    output_json_path = args.output_json
    report = profile_variant(
        args.mode,
        GAME_CONFIGS[args.mode],
        iterations=args.niterations,
        checkpoint_every=args.checkpoint_every,
        solver_name=args.solver,
        history_samples=args.history_samples,
        history_depth=args.history_depth,
        street_samples=args.street_samples,
        report_mode=args.report_mode,
        output_json_path=args.output_json,
    )

    if output_json_path:
        out_dir = os.path.dirname(output_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        base_name = os.path.splitext(os.path.basename(output_json_path))[0]
        range_path = os.path.join(os.path.dirname(output_json_path), f"{base_name}_ranges.json")
        export_range_dump(report.get("full_run_records", report.get("snapshots", [])), range_path)
        stability_path = os.path.join(os.path.dirname(output_json_path), f"{base_name}_checkpoint_stability.json")
        if report.get("checkpoint_history"):
            with open(stability_path, "w", encoding="utf-8") as handle:
                json.dump(report["checkpoint_history"], handle, indent=2)
                handle.write("\n")
        checkpoint_paths = []
        for path in sorted(os.listdir(out_dir or ".")):
            if path.startswith(f"{base_name}_checkpoint_") and path.endswith(".json"):
                checkpoint_paths.append(os.path.join(out_dir, path))
        write_run_manifest(output_json_path, range_path=range_path, stability_path=stability_path if os.path.exists(stability_path) else None, checkpoint_paths=checkpoint_paths)


if __name__ == "__main__":
    main()
