# %%
"""Board-conditioned subgame explorer v1.

This version is designed to reflect the model you described:

- there is preflop betting history
- from that history we derive a range for each player
- the board is fixed
- each hand has private info relative to the range (strength, equity, draw bucket)
- a training loop updates action weights for that exact subgame

This is still a lightweight, repo-local demonstration, but it is conceptually much
closer to a true subgame solve than the v0 version.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# %%
BOARD_EXAMPLE_1 = ["Ah", "Kd", "2c"]
BOARD_EXAMPLE_3 = ["Qs", "Jd", "7c"]

PRE_FLOP_HISTORY_EXAMPLES = {
    "bet_call": ["bet", "call"],
    "bet_raise_call": ["bet", "raise", "call"],
    "check_call": ["check", "call"],
}

# Heuristic preflop range model. Stronger hands are favored by default.
# This mirrors the idea that aces / pocket pairs / suited broadways > weak offsuit hands.
RANGE_TEMPLATES = {
    "bet_call": {
        "p1": ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs", "QJs", "JJ", "TT", "99", "88"],
        "p2": ["QQ", "JJ", "TT", "99", "AQs", "AJs", "KQs", "QJs", "JTs", "T9s", "98s"],
    },
    "bet_raise_call": {
        "p1": ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs", "JJ", "TT", "99", "AKo", "AQo"],
        "p2": ["QQ", "JJ", "TT", "99", "88", "AQs", "AJs", "KQs", "QJs", "JTs", "T9s", "98s"],
    },
    "check_call": {
        "p1": ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs", "QJs", "JJ", "TT", "99", "88", "77"],
        "p2": ["QQ", "JJ", "TT", "99", "88", "77", "AQs", "AJs", "KQs", "QJs", "JTs", "T9s", "98s"],
    },
}

# %%
def canonicalize_card(token: str) -> str:
    text = str(token).strip().replace(" ", "").replace("|", "")
    return text if text else ""


def canonicalize_board(board: Sequence[str]) -> List[str]:
    return [canonicalize_card(card) for card in board if canonicalize_card(card)]


def normalize_range_entry(entry: str) -> List[str]:
    text = str(entry).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("("):
        try:
            parts = [p.strip() for p in text.strip("[]() ").split(",")]
        except Exception:
            parts = []
        if parts:
            return [p for p in parts if p]
    return [text]


def normalize_ranges(ranges: Optional[Dict[str, Sequence[str]]]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for player, entries in (ranges or {}).items():
        flattened: List[str] = []
        for entry in entries or []:
            flattened.extend(normalize_range_entry(entry))
        cleaned[str(player)] = sorted({item for item in flattened if item})
    return cleaned


def _rank_value(card: str) -> int:
    text = str(card).strip()
    if not text:
        return 0
    rank = text[:-1]
    values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    return values.get(rank.upper(), 0)


def _suit_value(card: str) -> str:
    text = str(card).strip()
    return text[-1].upper() if text else ""


def _suit_is_same(a: str, b: str) -> bool:
    return bool(a and b and _suit_value(a) == _suit_value(b))


def _parse_hand_label(hand_label: str) -> Tuple[int, int, bool]:
    text = str(hand_label).strip()
    if not text:
        return 0, 0, False
    if len(text) == 2:
        return _rank_value(text[0]), _rank_value(text[1]), False
    if len(text) >= 3:
        ranks = []
        for ch in text:
            if ch.upper() in {"A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"}:
                ranks.append(_rank_value(ch))
        suited = text.endswith("s")
        if len(ranks) >= 2:
            return ranks[0], ranks[1], suited
    return _rank_value(text[0]) if text else 0, _rank_value(text[-1]) if text else 0, False


def _hand_rank_bucket(hole_cards: Sequence[str], board: Sequence[str]) -> str:
    cards = [str(c) for c in list(hole_cards) + list(board)]
    if not cards:
        return "weak"

    if len(hole_cards) == 1:
        label = str(hole_cards[0])
        r1, r2, suited = _parse_hand_label(label)
        if r1 == r2:
            return "premium_pair" if r1 >= 10 else "pair"
        if {r1, r2} >= {14, 13}:
            return "premium_connectors"
        if abs(r1 - r2) <= 2 and suited:
            return "suited_connector"
        if abs(r1 - r2) <= 2:
            return "broadway_gap"
        if suited:
            return "suited"
        return "premium_highcard" if max(r1, r2) >= 12 else "strong_highcard" if max(r1, r2) >= 9 else "weak"

    ranks = sorted({_rank_value(c) for c in cards if _rank_value(c) > 0})
    high = max(_rank_value(c) for c in hole_cards if _rank_value(c) > 0) if hole_cards else 0
    if len(hole_cards) >= 2:
        a, b = hole_cards[0], hole_cards[1]
        r1, r2 = _rank_value(a), _rank_value(b)
        if r1 == r2:
            if r1 >= 10:
                return "premium_pair"
            return "pair"
        if {r1, r2} >= {14, 13}:
            return "premium_connectors"
        if abs(r1 - r2) <= 2 and _suit_is_same(a, b):
            return "suited_connector"
        if abs(r1 - r2) <= 2:
            return "broadway_gap"
        if _suit_is_same(a, b):
            return "suited"
    if high >= 12:
        return "premium_highcard"
    if high >= 9:
        return "strong_highcard"
    return "weak"


def _draw_bucket(hole_cards: Sequence[str], board: Sequence[str]) -> str:
    if not board:
        return "none"
    if len(hole_cards) == 1:
        label = str(hole_cards[0]).strip()
        r1, r2, suited = _parse_hand_label(label)
        if abs(r1 - r2) <= 2 and suited:
            return "suited_connector_draw"
        return "none"
    if len(hole_cards) < 2 or len(board) < 3:
        return "none"
    ranks = [_rank_value(c) for c in board if _rank_value(c) > 0]
    r1, r2 = _rank_value(hole_cards[0]), _rank_value(hole_cards[1])
    if {r1, r2} & {10, 11, 12, 13, 14}:
        if any(r in ranks for r in [8, 9, 10, 11, 12, 13]):
            return "broadway_draw"
    if abs(r1 - r2) <= 2 and _suit_is_same(hole_cards[0], hole_cards[1]):
        return "suited_connector_draw"
    if len(set(ranks)) <= 2:
        return "paired_board_draw"
    return "none"


def _estimate_equity_vs_range(hole_cards: Sequence[str], board: Sequence[str], villain_range: Sequence[str]) -> float:
    """Simple deterministic proxy for equity against villain range.

    Higher value means more equity. This is intentionally lightweight and should be interpreted as
    an infoset feature rather than a true poker equity engine.
    """
    if not hole_cards:
        return 0.5

    if len(hole_cards) == 1:
        label = str(hole_cards[0]).strip()
        r1, r2, _ = _parse_hand_label(label)
        hand_strength = (r1 + r2) / 2.0
    else:
        hand_strength = 0.0
        for card in hole_cards:
            hand_strength += _rank_value(card)
        hand_strength /= max(len(hole_cards), 1)

    board_strength = sum(_rank_value(c) for c in board if _rank_value(c) > 0) / max(len(board), 1)
    range_strength = 0.0
    for entry in villain_range:
        if isinstance(entry, str):
            label = str(entry).strip()
            if label:
                r1, r2, _ = _parse_hand_label(label)
                range_strength += (r1 + r2) / 2.0
    if villain_range:
        range_strength /= max(len(villain_range), 1)

    equity = 0.5 + 0.25 * (hand_strength / 14.0) + 0.20 * (board_strength / 14.0) - 0.20 * (range_strength / 14.0)
    return max(0.05, min(0.95, equity))


@dataclass
class SubgameState:
    board: List[str]
    history: List[str]
    player_to_act: int
    hole_cards: List[str]
    villain_range: List[str]
    hand_strength: float
    equity_vs_range: float
    hand_bucket: str
    draw_bucket: str
    action_counts: Dict[str, float] = field(default_factory=lambda: {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0})


# %%
def make_range_from_history(history: Sequence[str], player: str = "p1") -> List[str]:
    key = "_".join(str(x).lower() for x in history if str(x).strip()) or "bet_call"
    if key not in PRE_FLOP_HISTORY_EXAMPLES:
        key = "bet_call"
    template = RANGE_TEMPLATES.get(key, RANGE_TEMPLATES["bet_call"])
    return list(template.get(player, template["p1"]))


def build_training_rows(
    board: Sequence[str],
    history: Sequence[str],
    ranges: Optional[Dict[str, Sequence[str]]] = None,
    player_to_act: int = 1,
) -> List[SubgameState]:
    board_list = canonicalize_board(board)
    range_map = normalize_ranges(ranges or {
        "p1": make_range_from_history(history, "p1"),
        "p2": make_range_from_history(history, "p2"),
    })

    rows: List[SubgameState] = []
    for player, hand_set in range_map.items():
        player_index = int(str(player).replace("p", "")) if str(player).startswith("p") else player_to_act
        villain_key = "p2" if player == "p1" else "p1"
        villain_range = range_map.get(villain_key, [])
        for hand in hand_set:
            hole_cards = [part.strip() for part in str(hand).split(",") if part.strip()]
            if not hole_cards:
                continue
            if len(hole_cards) == 1:
                hole_cards = [str(hand).strip()]
            elif len(hole_cards) > 2:
                hole_cards = hole_cards[:2]
            hand_bucket = _hand_rank_bucket(hole_cards, board_list)
            draw_bucket = _draw_bucket(hole_cards, board_list)
            strength = _estimate_equity_vs_range(hole_cards, board_list, villain_range)
            rows.append(SubgameState(
                board=list(board_list),
                history=[str(x).strip() for x in history if str(x).strip()],
                player_to_act=player_index,
                hole_cards=hole_cards,
                villain_range=list(villain_range),
                hand_strength=strength,
                equity_vs_range=strength,
                hand_bucket=hand_bucket,
                draw_bucket=draw_bucket,
            ))
    return rows


def train_subgame_state(rows: List[SubgameState], iterations: int = 500) -> List[Dict[str, object]]:
    for _ in range(max(1, int(iterations))):
        for row in rows:
            strength = row.equity_vs_range
            draw_bonus = 0.1 if row.draw_bucket != "none" else 0.0
            bucket_bonus = {
                "premium_pair": 0.7,
                "suited_connector": 0.45,
                "premium_connectors": 0.6,
                "premium_highcard": 0.5,
                "strong_highcard": 0.25,
                "suited": 0.2,
                "pair": 0.15,
                "weak": -0.1,
            }.get(row.hand_bucket, 0.0)

            target = {
                "fold": 0.25 + (0.5 - strength) + (0.1 if row.hand_bucket == "weak" else 0.0),
                "check_call": 0.45 + 0.25 * strength + draw_bonus + 0.12 * max(bucket_bonus, 0.0),
                "bet_raise": 0.30 + 0.55 * strength + draw_bonus + 0.20 * max(bucket_bonus, 0.0),
            }
            total = sum(target.values())
            target = {k: v / total for k, v in target.items()}

            for action in ("fold", "check_call", "bet_raise"):
                row.action_counts[action] += max(0.01, target[action] * (1.0 + bucket_bonus))

    outputs: List[Dict[str, object]] = []
    for row in rows:
        total = sum(row.action_counts.values())
        policy = {k: v / total for k, v in row.action_counts.items()} if total else {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0}
        outputs.append({
            "board": row.board,
            "history": row.history,
            "player_to_act": row.player_to_act,
            "hole_cards": row.hole_cards,
            "villain_range": row.villain_range,
            "hand_bucket": row.hand_bucket,
            "draw_bucket": row.draw_bucket,
            "equity_vs_range": row.equity_vs_range,
            "policy": policy,
        })
    return outputs


def train_board_conditioned_subgame(
    board: Sequence[str],
    history: Sequence[str],
    ranges: Optional[Dict[str, Sequence[str]]] = None,
    iterations: int = 500,
    player_to_act: int = 1,
) -> Dict[str, object]:
    rows = build_training_rows(board=board, history=history, ranges=ranges, player_to_act=player_to_act)
    trained = train_subgame_state(rows=rows, iterations=iterations)
    aggregate = {action: sum(item["policy"].get(action, 0.0) for item in trained) / max(len(trained), 1) for action in ("fold", "check_call", "bet_raise")}
    total = sum(aggregate.values())
    if total:
        aggregate = {k: v / total for k, v in aggregate.items()}

    return {
        "board": canonicalize_board(board),
        "history": [str(x).strip() for x in history if str(x).strip()],
        "player_to_act": int(player_to_act),
        "iteration": int(iterations),
        "ranges": normalize_ranges(ranges or {
            "p1": make_range_from_history(history, "p1"),
            "p2": make_range_from_history(history, "p2"),
        }),
        "aggregate_policy": aggregate,
        "sample_states": trained[:12],
    }


def demo(iterations: int = 500) -> List[Dict[str, object]]:
    scenarios = [
        ("board_1_bet_call", BOARD_EXAMPLE_1, ["bet", "call"]),
        ("board_1_bet_raise_call", BOARD_EXAMPLE_1, ["bet", "raise", "call"]),
        ("board_3_bet_call", BOARD_EXAMPLE_3, ["bet", "call"]),
    ]
    output = []
    for label, board, history in scenarios:
        record = train_board_conditioned_subgame(
            board=board,
            history=history,
            ranges={
                "p1": make_range_from_history(history, "p1"),
                "p2": make_range_from_history(history, "p2"),
            },
            iterations=iterations,
            player_to_act=1,
        )
        output.append({"label": label, **record})
        print(f"\n=== {label} ===")
        print(json.dumps(record, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Board-conditioned subgame explorer v1")
    parser.add_argument("--iterations", type=int, default=500, help="Number of training rounds per subgame")
    args = parser.parse_args()
    demo(iterations=args.iterations)
# %%
