import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


_CARD_TOKEN_RE = re.compile(r"(?<![A-Z])(?:10|[2-9TJQKA])[cdhs](?![A-Z])", re.IGNORECASE)


def parse_pokerkit_hole_cards(value: Any) -> List[str]:
    """Convert PokerKit hole-card payloads into canonical compact tokens like ['Ac', 'Ks'].

    Accepts raw strings such as:
    - 'Ac|Ks'
    - 'ACE OF CLUBS (Ac)|KING OF SPADES (Ks)'
    - 'AsKs'
    - ['ACE OF CLUBS (Ac)', 'FIVE OF DIAMONDS (5d)']
    - [Card(...), Card(...)]
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        tokens: List[str] = []
        for item in value:
            tokens.extend(parse_pokerkit_hole_cards(item))
        return tokens

    if not isinstance(value, str):
        return parse_pokerkit_hole_cards(str(value))

    text = value.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        parts = []
        for fragment in re.split(r"\s*,\s*", inner):
            fragment = fragment.strip().strip("\"'")
            if fragment:
                parts.append(fragment)
        tokens: List[str] = []
        for fragment in parts:
            tokens.extend(parse_pokerkit_hole_cards(fragment))
        return tokens

    # Prefer explicit PokerKit forms like "ACE OF CLUBS (Ac)" or "Ac|Ks".
    compact_tokens = [token.upper() if len(token) == 2 else token for token in _CARD_TOKEN_RE.findall(text)]
    if compact_tokens:
        return [token[:1].upper() + token[1:].lower() if len(token) == 2 else token for token in compact_tokens]

    # Handle explicit parenthesized tokens when regex misses due to surrounding text.
    explicit = re.findall(r"\(([2-9TJQKA]|10)[cdhs]\)", text, flags=re.IGNORECASE)
    if explicit:
        return [token.upper() + suit.lower() for token, suit in []]

    # Fallback for compact strings without separators, e.g. 'AsKs' or 'QdJh'.
    compact = re.sub(r"[^2-9TJQKAcdhs]", "", text, flags=re.IGNORECASE)
    if len(compact) >= 2 and len(compact) % 2 == 0:
        return [compact[i : i + 2].upper()[:1] + compact[i : i + 2].lower()[1:] for i in range(0, len(compact), 2)]

    return []


def canonical_cards(cards: Iterable[Any]) -> str:
    """Normalize a list of cards to a stable string key."""
    normalized = []
    for token in parse_pokerkit_hole_cards(cards):
        normalized.append(token)
    return "|".join(sorted(normalized))


def extract_private_holding_key(state, player_index: int | None = None) -> str:
    """Return the exact private holding key for the acting player if available."""
    wrapped = getattr(state, "_wrapped_state", None) or state
    if wrapped is None:
        return "unknown"

    player = int(player_index if player_index is not None else getattr(state, "current_player", lambda: 0)())
    hole_cards = getattr(wrapped, "hole_cards", None)
    if hole_cards is None:
        return "unknown"

    try:
        actor_hole = list(hole_cards[player])
    except Exception:
        try:
            actor_hole = list(hole_cards)
        except Exception:
            return "unknown"

    if not actor_hole:
        return "unknown"
    return canonical_cards(actor_hole)


def extract_exact_infoset_key(state, history=None) -> str:
    """Build a stable exact infoset key including private cards and betting context."""
    if state is None:
        return "unknown"

    wrapped = getattr(state, "_wrapped_state", None) or state
    player = int(getattr(state, "current_player", lambda: -1)())
    street = "unknown"
    try:
        board_count = int(getattr(wrapped, "board_count", 0) or 0)
        if board_count in (0, 3, 4, 5):
            street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[board_count]
    except Exception:
        pass

    board_cards = getattr(wrapped, "board_cards", []) or []
    board_key = canonical_cards(board_cards) if board_cards else "empty"
    history_key = "|".join(str(item) for item in (history or []))
    hole_key = extract_private_holding_key(state, player_index=player)
    return f"street={street}|player={player}|hole={hole_key}|board={board_key}|hist={history_key}"


def _card_rank(card: Any) -> int | None:
    """Extract the rank value for a card-like object, supporting PokerKit card strings."""
    tokens = parse_pokerkit_hole_cards(card)
    if not tokens:
        return None
    token = tokens[0]
    rank = token[0].upper()
    rank_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    return rank_map.get(rank)


def _rank_label(rank: int | None) -> str:
    if rank is None:
        return "?"
    label_map = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}
    return label_map.get(rank, str(rank))


def _card_suit(card: Any) -> str | None:
    tokens = parse_pokerkit_hole_cards(card)
    if not tokens:
        return None
    token = tokens[0]
    if len(token) < 2:
        return None
    suit = token[-1].lower()
    return suit if suit in {"c", "d", "h", "s"} else None


def canonical_preflop_label(hole_cards: Iterable[Any] | Any) -> str | None:
    """Return a compact label such as AKs, AKo, QQ using rank/suitedness semantics."""
    cards = parse_pokerkit_hole_cards(hole_cards)
    if len(cards) != 2:
        return None
    ranks = sorted([_card_rank(card) for card in cards])
    if any(rank is None for rank in ranks):
        return None
    lo, hi = ranks
    is_pair = lo == hi
    suited = _card_suit(cards[0]) == _card_suit(cards[1])
    if is_pair:
        return f"{_rank_label(lo)}{_rank_label(hi)}"
    first = _rank_label(hi)
    second = _rank_label(lo)
    return f"{first}{second}{'s' if suited else 'o'}"


def infer_rank_order_for_snapshots(snapshots: Iterable[Dict[str, Any]]) -> List[int]:
    """Infer the active rank set from snapshot metadata, defaulting to a standard deck when unspecified."""
    variant_hint = None
    for snapshot in snapshots or []:
        candidate = snapshot.get("variant") or snapshot.get("game_variant") or snapshot.get("deck_variant")
        if candidate:
            variant_hint = str(candidate)
            break
        deck_ranks = snapshot.get("deck_ranks")
        if isinstance(deck_ranks, list) and deck_ranks:
            return [int(rank) for rank in deck_ranks]

    if variant_hint:
        return list(range(6, 15)) if "short" in variant_hint.lower() else list(range(2, 15))

    ranks = set()
    for snapshot in snapshots or []:
        for card in snapshot.get("hole_cards") or []:
            rank = _card_rank(card)
            if rank is not None:
                ranks.add(rank)
    if not ranks:
        return list(range(2, 15))

    # Heuristic fallback only for deeply ambiguous cases where the caller has no variant metadata.
    ordered = sorted(ranks)
    if ordered and ordered[0] >= 6 and ordered[-1] <= 14 and all(rank >= 6 for rank in ordered):
        return list(range(6, 15))
    return list(range(2, 15))


def flatten_preflop_bucket(hole_cards: Iterable[Any] | Any, rank_order: Iterable[int] | None = None):
    """Flatten a 2-card preflop hold into a canonical matrix-slot keyed by rank pair and suit class."""
    cards = parse_pokerkit_hole_cards(hole_cards)
    if len(cards) != 2:
        return None

    rank_order = list(rank_order or infer_rank_order_for_snapshots([]))
    ranks = sorted([_card_rank(card) for card in cards])
    if any(rank is None for rank in ranks):
        return None

    lo, hi = ranks
    low_idx = rank_order.index(lo)
    high_idx = rank_order.index(hi)
    kind = "pair" if lo == hi else ("suited" if _card_suit(cards[0]) == _card_suit(cards[1]) else "offsuit")
    matrix_index = low_idx * len(rank_order) + high_idx
    return {
        "kind": kind,
        "rank_pair": [lo, hi],
        "matrix_coords": [low_idx, high_idx],
        "matrix_index": matrix_index,
        "matrix_size": len(rank_order),
        "label": canonical_preflop_label(cards) or (f"{_rank_label(lo)}{_rank_label(hi)}" if lo == hi else f"{lo}-{hi}-{kind}"),
    }


def aggregate_flattened_preflop_ranges(snapshots: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate preflop snapshots into a rank matrix suitable for 13x13 or short-deck 9x9 heatmaps."""
    snapshots = list(snapshots or [])
    rank_order = infer_rank_order_for_snapshots(snapshots)
    matrix = defaultdict(float)
    labels = {}

    for snapshot in snapshots:
        if (snapshot.get("street") or "").lower() != "preflop":
            continue
        hole_cards = snapshot.get("hole_cards") or []
        flat = flatten_preflop_bucket(hole_cards, rank_order=rank_order)
        if flat is None:
            continue
        matrix[flat["matrix_index"]] += 1.0
        labels.setdefault(flat["matrix_index"], canonical_preflop_label(hole_cards) or flat["label"])

    cells: List[Dict[str, Any]] = []
    for index, count in sorted(matrix.items()):
        row = index // len(rank_order)
        col = index % len(rank_order)
        cells.append(
            {
                "matrix_coords": [row, col],
                "rank_pair": [rank_order[row], rank_order[col]],
                "range_label": labels.get(index),
                "count": count,
            }
        )
    return {
        "deck_ranks": rank_order,
        "matrix_size": len(rank_order),
        "cells": cells,
    }


def canonical_range_bucket(snapshot: Dict[str, Any]) -> str:
    """Return a compact bucket key that ignores action-history noise for range aggregation."""
    if not snapshot:
        return "unknown"

    street = str(snapshot.get("street") or "unknown").lower()
    player = snapshot.get("player")
    hole_cards = snapshot.get("hole_cards") or []
    compact_label = canonical_preflop_label(hole_cards) if street == "preflop" else None
    if compact_label is not None:
        return f"street={street}|player={player}|bucket={compact_label}"

    board_key = canonical_cards(snapshot.get("board_cards") or [])
    return f"street={street}|player={player}|board={board_key}|hole={canonical_cards(hole_cards)}"


def classify_preflop_context(snapshot: Dict[str, Any]) -> str | None:
    """Classify a preflop sample into a compact action-context bucket.

    This is intentionally lightweight: it is meant to expose common response ranges
    such as 'facing open' and 'responding to 3-bet' without requiring a full custom
    game abstraction layer.
    """
    if not snapshot:
        return None
    if str(snapshot.get("street") or "").lower() != "preflop":
        return None

    player = int(snapshot.get("player", -1))
    history = snapshot.get("history") or []
    history_ids = []
    for item in history:
        try:
            history_ids.append(int(item))
        except Exception:
            continue

    if not history_ids:
        return f"preflop_open|player={player}"

    if player == 1:
        return f"preflop_facing_open|player={player}|history_len={len(history_ids)}"

    if player == 0 and len(history_ids) >= 2:
        return f"preflop_respond_to_3bet|player={player}|history_len={len(history_ids)}"

    return f"preflop_context|player={player}|history_len={len(history_ids)}"


def canonical_action_bucket(action: Any) -> int:
    """Collapse raw PokerKit action IDs to the compact solver-facing family: fold, check/call, bet/raise."""
    action_id = int(action)
    if action_id == 0:
        return 0
    if action_id == 1:
        return 1
    return 4


def canonical_action_name(action: Any) -> str:
    return {0: "fold", 1: "check_call", 4: "bet_raise"}[canonical_action_bucket(action)]


def format_node_label(node_name: Any, history: Iterable[Any] = ()) -> str:
    """Return the canonical HULH reporting label for a selected node history."""
    name = str(node_name or "")
    history_tokens = []
    for item in history or ():
        if item is None:
            continue
        history_tokens.append(str(item).strip().lower())

    if name in {"first_to_act", "root"} or not history_tokens:
        return "first_to_act"
    if name in {"response_to_limp", "c"} or history_tokens == ["call"]:
        return "response_to_limp"
    if name in {"response_to_open", "rf", "rc", "rr"} or history_tokens == ["bet"]:
        return "response_to_open"
    if name in {"response_to_limp_raise", "cr"} or history_tokens == ["call", "bet"]:
        return "response_to_limp_raise"
    if name in {"response_to_open_3bet", "opener_response_to_3bet"} or history_tokens == ["bet", "bet"]:
        return "response_to_open_3bet"
    if name in {"response_to_open_4bet", "opener_response_to_4bet"} or history_tokens == ["bet", "bet", "raise"]:
        return "response_to_open_4bet"
    if name in {"response_to_open_5bet", "opener_response_to_5bet"} or history_tokens == ["bet", "bet", "raise", "raise"]:
        return "response_to_open_5bet"
    return name or "selected_node"


def _normalize_history_token(token: Any) -> str | None:
    """Normalize raw action IDs and display aliases into a canonical HULH family token."""
    if token is None:
        return None
    if isinstance(token, (int, float)):
        action_id = int(token)
        if action_id == 0:
            return "fold"
        if action_id in {1, 2, 3}:
            return "call"
        if action_id >= 4:
            return "bet"
        return str(action_id)

    normalized = str(token).strip().lower()
    if normalized in {"check", "call"}:
        return "call"
    if normalized in {"bet", "raise"}:
        return "bet"
    if normalized == "fold":
        return "fold"
    return normalized


def replay_history_matches_spot(history: Iterable[Any], spot_name: Any) -> bool:
    """Return True when the path actually reaches the selected HULH spot.

    The root node is not a branch history; it is the base preflop acting state. Exact
    matching for deeper response paths is still preserved, but the root node must never be
    filtered out before the checkpoint aggregate is built.
    """
    spot_key = str(spot_name or "")
    if spot_key in {"first_to_act", "root"}:
        return True

    normalized = [
        item
        for item in (_normalize_history_token(entry) for entry in (history or []))
        if item is not None
    ]
    expected = {
        "first_to_act": [],
        "response_to_limp": ["call"],
        "response_to_open": ["bet"],
        "response_to_limp_raise": ["call", "bet"],
        "response_to_open_3bet": ["bet", "bet"],
        "response_to_open_4bet": ["bet", "bet", "bet"],
        "response_to_open_5bet": ["bet", "bet", "bet", "bet"],
    }
    target = expected.get(spot_key)
    if target is None:
        return True
    return normalized == target


def aggregate_selected_node_ranges(snapshots: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate average-strategy samples by selected node, regardless of exact hand infoset.

    Each selected node is an atomic reporting spot. We only roll up a sample if it
    actually reaches that spot's historical betting path; states that folded earlier
    or diverged from the expected action sequence are discarded before aggregation.
    """
    snapshots = list(snapshots or [])
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    node_metadata: Dict[str, Dict[str, Any]] = {}

    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue

        node_name = str(snapshot.get("node_name") or snapshot.get("normalized_name") or snapshot.get("label") or "node")
        history = list(snapshot.get("selected_history") or snapshot.get("history") or [])
        if not replay_history_matches_spot(history, node_name):
            continue
        label = format_node_label(node_name, history)
        compact_label = canonical_preflop_label(snapshot.get("hole_cards") or [])
        player = snapshot.get("player")
        node_key = f"{node_name}|player={player}" if player is not None else node_name

        grouped[node_key][compact_label or "unknown"].append(snapshot)
        node_metadata.setdefault(
            node_key,
            {
                "name": node_name,
                "display_name": label,
                "history": history,
                "history_label": label,
                "player": player,
                "street": snapshot.get("street"),
            },
        )

    nodes = []
    for node_name, hands in sorted(grouped.items()):
        hand_rows = []
        node_action_totals: Dict[str, float] = defaultdict(float)
        node_sample_count = 0

        for compact_label, items in sorted(hands.items()):
            if compact_label == "unknown":
                continue
            action_totals: Dict[str, float] = defaultdict(float)
            for item in items:
                for entry in item.get("action_probabilities") or []:
                    action_totals[canonical_action_name(entry["action"])] += float(entry["probability"])
            sample_count = len(items)
            node_sample_count += sample_count
            for action, total in action_totals.items():
                node_action_totals[action] += total
            hand_rows.append(
                {
                    "hand": compact_label,
                    "sample_count": sample_count,
                    "policy": {
                        action: total / sample_count
                        for action, total in sorted(action_totals.items())
                    },
                }
            )

        if node_sample_count == 0:
            continue

        metadata = node_metadata[node_name]
        nodes.append(
            {
                **metadata,
                "sample_count": node_sample_count,
                "hand_count": len(hand_rows),
                "action_frequencies": {
                    action: total / node_sample_count
                    for action, total in sorted(node_action_totals.items())
                },
                "hands": hand_rows,
            }
        )

    return {
        "action_families": ["fold", "check_call", "bet_raise"],
        "nodes": nodes,
    }


def aggregate_range_profiles(snapshots: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate exact-policy samples into a compact range dump keyed by canonical bucket, not action-history noise."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots or []:
        if not snapshot:
            continue
        key = canonical_range_bucket(snapshot)
        grouped[key].append(snapshot)

    rows: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        policy_totals: Dict[int, float] = defaultdict(float)
        street = items[0].get("street", "unknown")
        pot_context = items[0].get("pot_context")
        player = items[0].get("player")
        legal_actions = items[0].get("legal_actions") or []
        for item in items:
            for entry in item.get("action_probabilities") or []:
                action = canonical_action_bucket(entry.get("action"))
                policy_totals[action] += float(entry.get("probability", 0.0))
        total = sum(policy_totals.values()) or 1.0
        hole_cards = items[0].get("hole_cards") or []
        compact_label = canonical_preflop_label(hole_cards) if street == "preflop" else None
        canonical_legal_actions = [
            canonical_action_bucket(action)
            for action in legal_actions
        ]
        unique_actions = []
        for action in canonical_legal_actions:
            if action not in unique_actions:
                unique_actions.append(action)
        rows.append(
            {
                "infoset_key": key,
                "compact_label": compact_label,
                "sample_count": len(items),
                "street": street,
                "pot_context": pot_context,
                "player": player,
                "legal_actions": [int(a) for a in unique_actions],
                "policy": {
                    str(action): float(value / total)
                    for action, value in sorted(policy_totals.items())
                },
            }
        )
    return rows


def aggregate_preflop_context_profiles(snapshots: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate a few coarse action-context buckets for preflop response ranges.

    This exposes the common response families without mirroring the full exact-state
    tree: facing-open, responding-to-3bet, and the generic preflop open bucket.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots or []:
        if not snapshot:
            continue
        context_key = classify_preflop_context(snapshot)
        if context_key is None:
            continue
        grouped[context_key].append(snapshot)

    rows: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        policy_totals: Dict[int, float] = defaultdict(float)
        for item in items:
            for entry in item.get("action_probabilities") or []:
                action = canonical_action_bucket(entry.get("action"))
                policy_totals[action] += float(entry.get("probability", 0.0))
        total = sum(policy_totals.values()) or 1.0
        first = items[0]
        rows.append(
            {
                "context_bucket": key,
                "street": first.get("street", "preflop"),
                "player": first.get("player"),
                "sample_count": len(items),
                "legal_actions": sorted({canonical_action_bucket(action) for action in (first.get("legal_actions") or [])}),
                "policy": {
                    str(action): float(value / total)
                    for action, value in sorted(policy_totals.items())
                },
            }
        )
    return rows


def export_range_dump(snapshots: Iterable[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
    """Write the full-run cumulative range summary keyed by exact private state + board context."""
    rows = aggregate_range_profiles(snapshots)
    flattened_preflop = aggregate_flattened_preflop_ranges(snapshots)
    context_rows = aggregate_preflop_context_profiles(snapshots)
    selected_node_ranges = aggregate_selected_node_ranges(snapshots)
    payload = {
        "range_rows": rows,
        "range_count": len(rows),
        "unique_infosets": len(rows),
        "preflop_flattened": flattened_preflop,
        "preflop_context_rows": context_rows,
        "selected_node_ranges": selected_node_ranges,
        "selected_node_count": len(selected_node_ranges.get("nodes", [])),
        "facing_open_ranges": [row for row in context_rows if "facing_open" in row.get("context_bucket", "")],
        "respond_to_3bet_ranges": [row for row in context_rows if "respond_to_3bet" in row.get("context_bucket", "")],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload
