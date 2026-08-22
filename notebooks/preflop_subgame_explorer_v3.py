# %%
"""Board-conditioned subgame explorer v3.

This version is designed to respect the structure the repo already uses:
- all reporting is scoped to an explicit decision context
- action-family probabilities are normalized only over legal families for that context
- if checking is legal, we do not allow fold in that decision context
- if a bet/raise was made, the response context allows fold/check_call/bet_raise

The goal is to avoid mixing open frequencies with responding-to-bet frequencies in the same report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BOARD_EXAMPLE_1 = ["Ah", "Kd", "2c"]
BOARD_EXAMPLE_2 = ["Qh", "Jd", "9c"]
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


@dataclass(frozen=True)
class DecisionContext:
    name: str
    legal_families: Tuple[str, ...]
    description: str

    def normalize_policy(self, raw_policy: Dict[str, float]) -> Dict[str, float]:
        allowed = set(self.legal_families)
        filtered = {fam: float(raw_policy.get(fam, 0.0)) for fam in ACTION_KEYS if fam in allowed}
        total = sum(filtered.values())
        if total <= 0.0:
            if not filtered:
                return {fam: 0.0 for fam in ACTION_KEYS}
            neutral = 1.0 / len(filtered)
            return {fam: neutral if fam in filtered else 0.0 for fam in ACTION_KEYS}
        return {fam: filtered[fam] / total for fam in ACTION_KEYS if fam in allowed}


CONTEXTS: Dict[str, DecisionContext] = {
    "flop_open": DecisionContext(
        name="flop_open",
        legal_families=("check_call", "bet_raise"),
        description="player acts first on a checked flop; fold is not legal",
    ),
    "flop_check": DecisionContext(
        name="flop_check",
        legal_families=("check_call", "bet_raise"),
        description="player can either check or bet; fold not legal",
    ),
    "response_to_bet": DecisionContext(
        name="response_to_bet",
        legal_families=("fold", "check_call", "bet_raise"),
        description="player is facing a bet and may fold, call, or raise",
    ),
    "response_to_raise": DecisionContext(
        name="response_to_raise",
        legal_families=("fold", "check_call", "bet_raise"),
        description="player is facing a raise and may fold, call, or re-raise",
    ),
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
        rank = token[0].upper()
        return [f"{rank}h", f"{rank}c"]
    if len(token) >= 2 and token[0].isalpha() and token[1].isalpha():
        rank1 = token[0].upper()
        rank2 = token[1].upper()
        suit_tag = token[2].lower() if len(token) > 2 else ""
        suit1 = "h" if suit_tag == "s" else "c"
        suit2 = "h" if suit_tag == "s" else "d"
        return [f"{rank1}{suit1}", f"{rank2}{suit2}"]
    if len(token) == 1 and token.isalpha():
        return [f"{token.upper()}h", f"{token.upper()}d"]
    if len(token) >= 4 and token[0].isalpha() and token[1].isalpha() and token[2].isalpha() and token[3].isalpha():
        return [token[:2], token[2:4]]
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
    if len(all_cards) >= 5:
        suited = defaultdict(int)
        for card in all_cards:
            suited[_suit_value(card)] += 1
        if max(suited.values()) >= 5:
            return {"hand_bucket": "flushed_board_or_flush", "kicker_bucket": "n/a"}
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
    context: DecisionContext
    counts: Dict[str, float] = field(default_factory=lambda: {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0})


def decision_context_for_history(history: Sequence[str]) -> DecisionContext:
    tokens = [str(x).strip().lower() for x in history if str(x).strip()]
    if not tokens:
        return CONTEXTS["flop_open"]
    if tokens[-1] in {"check"}:
        return CONTEXTS["flop_check"]
    if tokens[-1] in {"bet", "raise"}:
        return CONTEXTS["response_to_bet"]
    if tokens[-1] in {"call"}:
        return CONTEXTS["response_to_bet"]
    return CONTEXTS["flop_open"]


def build_training_rows(
    board: Sequence[str],
    history: Sequence[str],
    ranges: Optional[Dict[str, Sequence[str]]] = None,
    player_to_act: int = 1,
    context: Optional[DecisionContext] = None,
) -> List[TrainingRow]:
    board_list = canonicalize_board(board)
    context_obj = context or decision_context_for_history(history)
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
                context=context_obj,
            )
            rows.append(row)
    return rows


def target_policy_for_row(row: TrainingRow) -> Dict[str, float]:
    strength = row.equity_proxy
    hand_bias = {
        "set_or_trips": 0.75,
        "two_pair": 0.60,
        "top_pair": 0.45,
        "middle_pair": 0.30,
        "bottom_pair": 0.13,
        "overcards": 0.32,
        "high_card": 0.20,
        "weak_high_card": 0.08,
        "flushed_board_or_flush": 0.70,
    }.get(row.hand_bucket, 0.0)

    draw_bias = {
        "flush_draw": 0.30,
        "straight_draw": 0.25,
        "weak_straight_draw": 0.12,
        "overcard_draw": 0.18,
        "none": 0.0,
    }.get(row.draw_bucket, 0.0)

    kicker_bias = {
        "good": 0.20,
        "ok": 0.10,
        "weak": -0.10,
        "n/a": 0.0,
    }.get(row.kicker_bucket, 0.0)

    if row.context.name in {"flop_open", "flop_check"}:
        fold_target = 0.0
        check_call_target = 0.46 + 0.22 * strength + 0.18 * hand_bias + 0.10 * draw_bias + 0.08 * kicker_bias
        bet_raise_target = 0.54 + 0.48 * strength + 0.26 * hand_bias + 0.18 * draw_bias + 0.12 * kicker_bias
    else:
        fold_target = 0.22 + max(0.0, 0.42 - strength) - 0.12 * hand_bias + 0.04 * max(0.0, -kicker_bias)
        check_call_target = 0.42 + 0.32 * strength + 0.12 * hand_bias + 0.10 * draw_bias + 0.08 * kicker_bias
        bet_raise_target = 0.36 + 0.54 * strength + 0.22 * hand_bias + 0.18 * draw_bias + 0.12 * kicker_bias

    raw = {"fold": fold_target, "check_call": check_call_target, "bet_raise": bet_raise_target}
    allowed = set(row.context.legal_families)
    for fam, val in list(raw.items()):
        if fam not in allowed:
            raw[fam] = 0.0
    total = sum(raw.values())
    if total <= 0.0:
        return {fam: 1.0 / len(row.context.legal_families) if fam in row.context.legal_families else 0.0 for fam in ACTION_KEYS}
    return {fam: raw[fam] / total for fam in ACTION_KEYS}


def train_subgame(rows: List[TrainingRow], iterations: int = 500) -> List[Dict[str, object]]:
    if not rows:
        return []

    total_iterations = max(1, int(iterations))
    floor = 0.05
    for iteration in range(total_iterations):
        lr = 0.05 / (1.0 + iteration / 100.0)
        for row in rows:
            target = target_policy_for_row(row)
            current = row.counts.copy()
            next_counts = {}
            for action in ACTION_KEYS:
                if action not in row.context.legal_families:
                    next_counts[action] = 0.0
                else:
                    next_counts[action] = (1.0 - lr) * current.get(action, 0.0) + lr * target.get(action, 0.0)
            # enforce a legal-action-only normalization with a modest floor for valid actions
            legal = [action for action in ACTION_KEYS if action in row.context.legal_families]
            if legal:
                clamped = {action: max(next_counts.get(action, 0.0), floor if action in legal else 0.0) for action in ACTION_KEYS}
                norm_total = sum(clamped[action] for action in legal)
                for action in ACTION_KEYS:
                    if action in legal:
                        clamped[action] = clamped[action] / norm_total
                    else:
                        clamped[action] = 0.0
                row.counts = clamped
            else:
                row.counts = {action: 0.0 for action in ACTION_KEYS}

    out: List[Dict[str, object]] = []
    for row in rows:
        policy = {action: row.counts.get(action, 0.0) for action in ACTION_KEYS}
        out.append({
            "board": row.board,
            "history": row.history,
            "decision_context": row.context.name,
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
    context: Optional[str] = None,
) -> Dict[str, object]:
    context_obj = CONTEXTS.get(context or decision_context_for_history(history).name, decision_context_for_history(history))
    rows = build_training_rows(board=board, history=history, ranges=ranges, player_to_act=player_to_act, context=context_obj)
    trained = train_subgame(rows=rows, iterations=iterations)
    aggregate = {action: sum(item["policy"].get(action, 0.0) for item in trained) / max(len(trained), 1) for action in ACTION_KEYS}
    for action in list(aggregate.keys()):
        if action not in context_obj.legal_families:
            aggregate[action] = 0.0
    total = sum(aggregate.values())
    if total:
        aggregate = {k: v / total for k, v in aggregate.items()}
    return {
        "board": canonicalize_board(board),
        "history": [str(x).strip() for x in history if str(x).strip()],
        "decision_context": context_obj.name,
        "player_to_act": int(player_to_act),
        "iteration": int(iterations),
        "aggregate_policy": aggregate,
        "sample_states": trained[:8],
    }


def demo(iterations: int = 500) -> List[Dict[str, object]]:
    cases = [
        ("dry_broadway_open", BOARD_EXAMPLE_1, [], "flop_open"),
        ("dry_broadway_response_to_bet", BOARD_EXAMPLE_1, ["bet"], "response_to_bet"),
        ("rainbow_open", BOARD_EXAMPLE_2, [], "flop_open"),
        ("rainbow_response_to_bet", BOARD_EXAMPLE_2, ["bet"], "response_to_bet"),
        ("paired_board_check", BOARD_EXAMPLE_3, ["check"], "flop_check"),
    ]
    out = []
    for label, board, history, context in cases:
        rec = train_board_conditioned_subgame(
            board=board,
            history=history,
            ranges={
                "p1": _range_from_history(history, "p1"),
                "p2": _range_from_history(history, "p2"),
            },
            iterations=iterations,
            player_to_act=1,
            context=context,
        )
        out.append({"label": label, **rec})
        print(f"\n=== {label} ===")
        print(json.dumps(rec, indent=2, sort_keys=True))
    return out


def sweep_contexts_and_flops(
    iterations: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100),
    context_names: Sequence[str] = ("flop_open", "flop_check", "response_to_bet"),
) -> List[Dict[str, object]]:
    cases = [
        ("dry_broadway", BOARD_EXAMPLE_1, ["AA", "AKs", "JTs"], ["KK", "KQs", "TT"], ["bet"]),
        ("rainbow_connected", BOARD_EXAMPLE_2, ["AKs", "QJs", "JJ"], ["KQs", "TT", "98s"], ["bet"]),
        ("paired_board", BOARD_EXAMPLE_3, ["QQ", "AKs", "A9s"], ["JJ", "TT", "KQs"], ["bet"]),
    ]
    out = []
    for label, board, p1_range, p2_range, history in cases:
        for context_name in context_names:
            trajectory = []
            for n in iterations:
                rec = train_board_conditioned_subgame(
                    board=board,
                    history=history,
                    ranges={"p1": p1_range, "p2": p2_range},
                    iterations=n,
                    player_to_act=1,
                    context=context_name,
                )
                trajectory.append({"iteration": n, "aggregate_policy": rec["aggregate_policy"]})
            out.append({
                "label": label,
                "context": context_name,
                "board": canonicalize_board(board),
                "history": history,
                "trajectory": trajectory,
            })
            print(f"\n=== {label} :: {context_name} ===")
            print(json.dumps({"board": canonicalize_board(board), "history": history, "context": context_name, "trajectory": trajectory}, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Board-conditioned subgame explorer v3")
    parser.add_argument("--iterations", type=int, default=500, help="Training rounds per board/history subgame")
    parser.add_argument("--sweep", action="store_true", help="Run context-separated flop + hole-card sweep")
    args = parser.parse_args()
    if args.sweep:
        sweep_contexts_and_flops(iterations=(1, 2, 3, 5, 10, 20, 50, 100))
    else:
        demo(iterations=args.iterations)
# %%
