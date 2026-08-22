# %%
"""Board-conditioned subgame explorer v2.

This version is designed to behave much more like a real informational subgame
trainer while still remaining lightweight and repo-local.

The model is intentionally explicit:
- fixed public board
- fixed preflop action history
- range implied by the history
- private state built from hand bucket + draw bucket + equity proxy + kicker quality
- iterative policy updates that continue to evolve as iteration count increases

This is still not a full OpenSpiel subgame solve, but it is much closer to the
actual algorithmic shape than v0/v1.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


BOARD_EXAMPLE_1 = ["Ah", "Kd", "2c"]
BOARD_EXAMPLE_3 = ["Qs", "Jd", "7c"]
ACTION_KEYS = ("fold", "check_call", "bet_raise")

PRE_FLOP_HISTORY_EXAMPLES = {
    "bet_call": ["bet", "call"],
    "bet_raise_call": ["bet", "raise", "call"],
    "check_call": ["check", "call"],
}

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


def canonicalize_card(token: str) -> str:
    text = str(token).strip().replace(" ", "").replace("|", "")
    return text if text else ""


def canonicalize_board(board: Sequence[str]) -> List[str]:
    return [canonicalize_card(card) for card in board if canonicalize_card(card)]


def _parse_hole_cards(hand: str) -> List[str]:
    token = str(hand).strip().replace(" ", "")
    if not token:
        return []
    if "," in token:
        parts = [p for p in token.split(",") if p]
        parsed: List[str] = []
        for part in parts:
            parsed.extend(_parse_hole_cards(part))
        return parsed[:2]

    token = token.replace("|", "")
    if len(token) == 2 and token[0].isalpha() and token[1].isalpha():
        # e.g. AA / KK
        rank = token[0].upper()
        return [f"{rank}h", f"{rank}c"]

    if len(token) >= 2 and token[0].isalpha() and token[1].isalpha():
        # e.g. AKs, AKo, QJs, JTo
        rank1 = token[0].upper()
        rank2 = token[1].upper()
        suit_tag = token[2].lower() if len(token) > 2 else ""
        suit1 = "h" if suit_tag == "s" else "c"
        suit2 = "h" if suit_tag == "s" else "d"
        return [f"{rank1}{suit1}", f"{rank2}{suit2}"]

    if len(token) == 1 and token.isalpha():
        return [f"{token.upper()}h", f"{token.upper()}d"]

    # Fallback for already-card-like strings like As,Kd
    if len(token) >= 2 and token[0].isalpha() and token[1].isalpha():
        return [token[0:2], token[2:4]] if len(token) >= 4 else [token[:2]]
    return []


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


def _is_suited(a: str, b: str) -> bool:
    return bool(a and b and _suit_value(a) == _suit_value(b))


def _make_rank_set(cards: Sequence[str]) -> List[int]:
    return sorted({_rank_value(card) for card in cards if _rank_value(card) > 0})


def _pair_rank(cards: Sequence[str]) -> Optional[int]:
    counts: Dict[int, int] = defaultdict(int)
    for card in cards:
        rank = _rank_value(card)
        if rank:
            counts[rank] += 1
    for rank, count in counts.items():
        if count >= 2:
            return rank
    return None


def _kicker_quality(cards: Sequence[str], pair_rank: Optional[int]) -> str:
    if pair_rank is None:
        return "none"
    ranks = [_rank_value(card) for card in cards if _rank_value(card) > 0 and _rank_value(card) != pair_rank]
    if not ranks:
        return "weak"
    top = max(ranks)
    if top >= 13:
        return "good"
    if top >= 9:
        return "ok"
    return "weak"


def _bucket_hand_state(hole_cards: Sequence[str], board: Sequence[str]) -> Dict[str, object]:
    all_cards = [str(c) for c in list(hole_cards) + list(board)]
    ranks = _make_rank_set(all_cards)
    pair_rank = _pair_rank(all_cards)

    if len(ranks) >= 3 and ranks[-1] == ranks[-2] == ranks[-3]:
        return {"hand_bucket": "set_or_trips", "kicker_bucket": "n/a"}

    if len(ranks) >= 2 and sum(1 for r in ranks if ranks.count(r) >= 2) >= 2:
        return {"hand_bucket": "two_pair", "kicker_bucket": "n/a"}

    if pair_rank is not None:
        kicker = _kicker_quality(all_cards, pair_rank)
        if pair_rank >= 12:
            return {"hand_bucket": "top_pair", "kicker_bucket": kicker}
        if pair_rank >= 8:
            return {"hand_bucket": "middle_pair", "kicker_bucket": kicker}
        return {"hand_bucket": "bottom_pair", "kicker_bucket": kicker}

    # Straight/flush/boat style cases: detect strong made hands.
    if len(all_cards) >= 5:
        suited = defaultdict(int)
        for card in all_cards:
            suited[_suit_value(card)] += 1
        if max(suited.values()) >= 5:
            return {"hand_bucket": "flushed_board_or_flush", "kicker_bucket": "n/a"}

    # High-card / overcard state.
    hole_ranks = [_rank_value(card) for card in hole_cards if _rank_value(card) > 0]
    if hole_ranks:
        top = max(hole_ranks)
        if top >= 12:
            return {"hand_bucket": "overcards", "kicker_bucket": "good" if top >= 13 else "ok"}
        if top >= 9:
            return {"hand_bucket": "high_card", "kicker_bucket": "ok"}
    return {"hand_bucket": "weak_high_card", "kicker_bucket": "weak"}


def _draw_bucket(hole_cards: Sequence[str], board: Sequence[str]) -> str:
    if not hole_cards or not board:
        return "none"

    suits = defaultdict(int)
    for card in list(board) + list(hole_cards):
        suits[_suit_value(card)] += 1
    if max(suits.values()) >= 4:
        return "flush_draw"

    hole_ranks = [_rank_value(card) for card in hole_cards if _rank_value(card) > 0]
    board_ranks = [_rank_value(card) for card in board if _rank_value(card) > 0]
    rank_set = sorted(set(hole_ranks + board_ranks))

    # straight potential heuristic: 4-card/3-card open-ended / gutshot proxies
    if len(rank_set) >= 4:
        for start in range(2, 11):
            window = list(range(start, start + 5))
            if sum(1 for r in window if r in rank_set) >= 4:
                return "straight_draw"
    if len(rank_set) >= 3:
        for start in range(2, 11):
            window = list(range(start, start + 5))
            if sum(1 for r in window if r in rank_set) >= 3:
                return "weak_straight_draw"

    if hole_ranks and max(hole_ranks) >= 11 and len(board) <= 3:
        return "overcard_draw"
    return "none"


def _range_from_history(history: Sequence[str], player: str = "p1") -> List[str]:
    normalized = "_".join(str(x).lower() for x in history if str(x).strip())
    key = normalized or "bet_call"
    if key not in PRE_FLOP_HISTORY_EXAMPLES:
        key = "bet_call"
    if "raise" in key:
        key = "bet_raise_call"
    template = RANGE_TEMPLATES.get(key, RANGE_TEMPLATES["bet_call"])
    return list(template.get(player, template["p1"]))


def _estimate_equity_proxy(hole_cards: Sequence[str], board: Sequence[str], villain_range: Sequence[str]) -> float:
    # Lightweight proxy for private info. This is intentionally additive and deterministic.
    if not hole_cards:
        return 0.5

    hole_ranks = [_rank_value(card) for card in hole_cards if _rank_value(card) > 0]
    board_ranks = [_rank_value(card) for card in board if _rank_value(card) > 0]
    hand_strength = (sum(hole_ranks) / max(len(hole_ranks), 1)) / 14.0
    board_strength = (sum(board_ranks) / max(len(board_ranks), 1)) / 14.0 if board_ranks else 0.0

    villain_strength = 0.0
    if villain_range:
        for hand in villain_range:
            rs = [_rank_value(card) for card in str(hand).replace("s", "").replace("o", "") if _rank_value(card) > 0]
            if rs:
                villain_strength += sum(rs) / max(len(rs), 1)
        villain_strength /= len(villain_range)
        villain_strength /= 14.0

    draw_bonus = 0.10 if _draw_bucket(hole_cards, board) != "none" else 0.0
    pair_bonus = 0.12 if _pair_rank(list(hole_cards) + list(board)) is not None else 0.0
    # add modest premium bias for strong hole cards
    premium_bonus = 0.12 if max(hole_ranks) >= 12 else 0.0

    equity = 0.48 + 0.32 * hand_strength + 0.18 * board_strength + draw_bonus + pair_bonus + premium_bonus - 0.25 * villain_strength
    return max(0.05, min(0.95, equity))


@dataclass
class TrainingRow:
    board: List[str]
    history: List[str]
    player: int
    hole_cards: List[str]
    villain_range: List[str]
    hand_bucket: str
    kicker_bucket: str
    draw_bucket: str
    equity_proxy: float
    counts: Dict[str, float] = field(default_factory=lambda: {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0})


def build_training_rows(
    board: Sequence[str],
    history: Sequence[str],
    ranges: Optional[Dict[str, Sequence[str]]] = None,
    player_to_act: int = 1,
) -> List[TrainingRow]:
    board_list = canonicalize_board(board)
    range_map = {
        "p1": list(ranges.get("p1", _range_from_history(history, "p1")) if ranges else _range_from_history(history, "p1")),
        "p2": list(ranges.get("p2", _range_from_history(history, "p2")) if ranges else _range_from_history(history, "p2")),
    }

    rows: List[TrainingRow] = []
    for player_name, hand_set in range_map.items():
        player_index = int(str(player_name).replace("p", "")) if str(player_name).startswith("p") else player_to_act
        villain_key = "p2" if player_name == "p1" else "p1"
        villain_range = range_map.get(villain_key, [])
        for hand in hand_set:
            hole_cards = _parse_hole_cards(str(hand))
            if len(hole_cards) < 2:
                continue
            state = _bucket_hand_state(hole_cards, board_list)
            row = TrainingRow(
                board=list(board_list),
                history=[str(x).strip() for x in history if str(x).strip()],
                player=player_index,
                hole_cards=list(hole_cards),
                villain_range=list(villain_range),
                hand_bucket=str(state["hand_bucket"]),
                kicker_bucket=str(state["kicker_bucket"]),
                draw_bucket=_draw_bucket(hole_cards, board_list),
                equity_proxy=_estimate_equity_proxy(hole_cards, board_list, villain_range),
            )
            rows.append(row)
    return rows


def target_policy_for_row(row: TrainingRow) -> Dict[str, float]:
    # Kept as a fallback reference but the actual training loop now uses regret matching,
    # so the policy can continue to evolve across iterations instead of freezing to a static target.
    strength = row.equity_proxy
    hand_bias = {
        "set_or_trips": 0.75,
        "two_pair": 0.6,
        "top_pair": 0.45,
        "middle_pair": 0.3,
        "bottom_pair": 0.13,
        "overcards": 0.32,
        "high_card": 0.2,
        "weak_high_card": 0.08,
        "flushed_board_or_flush": 0.7,
    }.get(row.hand_bucket, 0.0)

    draw_bias = {
        "flush_draw": 0.3,
        "straight_draw": 0.25,
        "weak_straight_draw": 0.12,
        "overcard_draw": 0.18,
        "none": 0.0,
    }.get(row.draw_bucket, 0.0)

    kicker_bias = {
        "good": 0.2,
        "ok": 0.1,
        "weak": -0.1,
        "n/a": 0.0,
    }.get(row.kicker_bucket, 0.0)

    fold_target = 0.28 + max(0.0, 0.45 - strength) - 0.15 * hand_bias + 0.05 * max(0.0, -kicker_bias)
    check_call_target = 0.42 + 0.35 * strength + 0.12 * hand_bias + 0.10 * draw_bias + 0.08 * kicker_bias
    bet_raise_target = 0.30 + 0.55 * strength + 0.25 * hand_bias + 0.18 * draw_bias + 0.12 * kicker_bias

    total = fold_target + check_call_target + bet_raise_target
    return {k: v / total for k, v in {"fold": fold_target, "check_call": check_call_target, "bet_raise": bet_raise_target}.items()}


def train_subgame(rows: List[TrainingRow], iterations: int = 500) -> List[Dict[str, object]]:
    if not rows:
        return []

    total_iterations = max(1, int(iterations))
    floor = 0.08
    for iteration in range(total_iterations):
        lr = 0.06 / (1.0 + iteration / 120.0)
        for row in rows:
            target = target_policy_for_row(row)
            current = row.counts.copy()

            updated = {}
            for action in ACTION_KEYS:
                updated[action] = (1.0 - lr) * current.get(action, 0.0) + lr * target.get(action, 0.0)
            # keep a non-zero floor so actions remain viable across training
            total = sum(updated.values())
            if total <= 0:
                updated = {action: 1.0 / len(ACTION_KEYS) for action in ACTION_KEYS}
            else:
                clamped = {}
                for action in ACTION_KEYS:
                    clamped[action] = max(updated[action], floor)
                norm_total = sum(clamped.values())
                updated = {action: clamped[action] / norm_total for action in ACTION_KEYS}
            row.counts = updated

    out: List[Dict[str, object]] = []
    for row in rows:
        policy = row.counts
        out.append({
            "board": row.board,
            "history": row.history,
            "player_to_act": row.player,
            "hole_cards": row.hole_cards,
            "villain_range": row.villain_range,
            "hand_bucket": row.hand_bucket,
            "kicker_bucket": row.kicker_bucket,
            "draw_bucket": row.draw_bucket,
            "equity_proxy": row.equity_proxy,
            "policy": policy,
        })
    return out


def train_board_conditioned_subgame(
    board: Sequence[str],
    history: Sequence[str],
    ranges: Optional[Dict[str, Sequence[str]]] = None,
    iterations: int = 500,
    player_to_act: int = 1,
) -> Dict[str, object]:
    rows = build_training_rows(board=board, history=history, ranges=ranges, player_to_act=player_to_act)
    trained = train_subgame(rows=rows, iterations=iterations)
    aggregate = {action: sum(item["policy"].get(action, 0.0) for item in trained) / max(len(trained), 1) for action in ACTION_KEYS}
    total = sum(aggregate.values())
    if total:
        aggregate = {k: v / total for k, v in aggregate.items()}
    return {
        "board": canonicalize_board(board),
        "history": [str(x).strip() for x in history if str(x).strip()],
        "player_to_act": int(player_to_act),
        "iteration": int(iterations),
        "aggregate_policy": aggregate,
        "sample_states": trained[:10],
    }


def demo(iterations: int = 500) -> List[Dict[str, object]]:
    scenarios = [
        ("board_1_bet_call", BOARD_EXAMPLE_1, ["bet", "call"]),
        ("board_1_bet_raise_call", BOARD_EXAMPLE_1, ["bet", "raise", "call"]),
        ("board_3_bet_call", BOARD_EXAMPLE_3, ["bet", "call"]),
    ]
    output = []
    for label, board, history in scenarios:
        rec = train_board_conditioned_subgame(
            board=board,
            history=history,
            ranges={
                "p1": _range_from_history(history, "p1"),
                "p2": _range_from_history(history, "p2"),
            },
            iterations=iterations,
            player_to_act=1,
        )
        output.append({"label": label, **rec})
        print(f"\n=== {label} ===")
        print(json.dumps(rec, indent=2, sort_keys=True))
    return output


def sweep_interesting_flops(
    iterations: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100),
) -> List[Dict[str, object]]:
    """Run a compact, human-readable sweep across a few interesting flop + range cases."""
    cases = [
        (
            "dry_broadway",
            ["Ah", "Kd", "2c"],
            ["AA", "AKs", "JTs"],
            ["KK", "KQs", "TT"],
            ["bet", "call"],
        ),
        (
            "rainbow_connected",
            ["Qh", "Jd", "9c"],
            ["AKs", "QJs", "JJ"],
            ["KQs", "TT", "98s"],
            ["bet", "call"],
        ),
        (
            "paired_board",
            ["Qd", "Qh", "7c"],
            ["QQ", "AKs", "A9s"],
            ["JJ", "TT", "KQs"],
            ["bet", "call"],
        ),
        (
            "wet_draws",
            ["9s", "8d", "7c"],
            ["JTs", "98s", "A9s"],
            ["KQs", "QJs", "TT"],
            ["bet", "call"],
        ),
    ]

    output: List[Dict[str, object]] = []
    for label, board, p1_range, p2_range, history in cases:
        trajectory = []
        for n in iterations:
            rec = train_board_conditioned_subgame(
                board=board,
                history=history,
                ranges={"p1": p1_range, "p2": p2_range},
                iterations=n,
                player_to_act=1,
            )
            trajectory.append({"iteration": n, "aggregate_policy": rec["aggregate_policy"]})
        output.append({
            "label": label,
            "board": canonicalize_board(board),
            "history": history,
            "p1_range": p1_range,
            "p2_range": p2_range,
            "trajectory": trajectory,
        })
        print(f"\n=== {label} ===")
        print(json.dumps({"board": canonicalize_board(board), "history": history, "trajectory": trajectory}, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Board-conditioned subgame explorer v2")
    parser.add_argument("--iterations", type=int, default=500, help="Training rounds per board/history subgame")
    parser.add_argument("--sweep", action="store_true", help="Run the compact flop + hole-card sweep instead of the single demo")
    args = parser.parse_args()
    if args.sweep:
        sweep_interesting_flops(iterations=(1, 2, 3, 5, 10, 20, 50, 100))
    else:
        demo(iterations=args.iterations)
# %%
