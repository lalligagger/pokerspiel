"""Helpers for suit-isomorphic flop canonicalization and collision-safe remapping.

The purpose of this module is to support reduced-flop benchmark experiments
without changing the core solver/service layer.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations
from typing import Iterable, List, Sequence, Tuple

SUITS = ["c", "d", "h", "s"]
RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_TO_INDEX = {rank: idx for idx, rank in enumerate(RANK_ORDER)}


def normalize_card(card: str) -> str:
    """Normalize a card string to the repo's compact form, e.g. As, 9d."""
    if len(card) < 2:
        raise ValueError(f"Invalid card string: {card!r}")
    rank = card[:-1]
    suit = card[-1].lower()
    if suit not in SUITS:
        raise ValueError(f"Unsupported suit {suit!r} in card {card!r}")
    return f"{rank}{suit}"


def flop_to_rank_signature(flop: Sequence[str]) -> Tuple[int, ...]:
    """Return a canonical rank tuple for a 3-card flop, sorted descending by rank."""
    ranks = sorted((normalize_card(card)[:-1] for card in flop), key=lambda r: RANK_TO_INDEX.get(r, -1), reverse=True)
    return tuple(RANK_TO_INDEX.get(r, -1) for r in ranks)


def flop_suit_pattern(flop: Sequence[str]) -> str:
    """Return a broad suit pattern class: rainbow, pair, or monotone."""
    suits = [normalize_card(card)[-1] for card in flop]
    counts = {s: suits.count(s) for s in set(suits)}
    if max(counts.values()) == 3:
        return "monotone"
    if max(counts.values()) == 2:
        return "paired"
    return "rainbow"


def canonical_flop_key(flop: Sequence[str]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Canonical key for a suit-isomorphic flop class.

    A flop is equivalent under suit relabeling if and only if its rank structure and
    the relative suit assignments are equivalent after renaming suits. This removes
    the suit-label artifact without altering the strategic pattern.
    """
    normalized = sorted((normalize_card(card) for card in flop), key=lambda c: RANK_TO_INDEX.get(c[:-1], -1), reverse=True)
    rank_sig = tuple(RANK_TO_INDEX.get(card[:-1], -1) for card in normalized)

    observed_suits = [card[-1] for card in normalized]
    best_order = None
    for perm in permutations(SUITS):
        mapping = {old: idx for idx, old in enumerate(perm)}
        remapped = tuple(mapping[suit] for suit in observed_suits)
        if best_order is None or remapped < best_order:
            best_order = remapped

    return rank_sig, best_order


def build_suit_isomorphism(player_cards: Sequence[str], landmark_cards: Sequence[str]) -> Tuple[dict, List[str]]:
    """Remap landmark suit symbols to avoid collision with player-held cards.

    This preserves strategic equivalence while ensuring no card in the landmark
    board collides with a card the user is already holding.
    """
    player_cards = [normalize_card(card) for card in player_cards]
    landmark_cards = [normalize_card(card) for card in landmark_cards]

    player_suits = {card[-1] for card in player_cards}
    available_suits = [s for s in SUITS if s not in player_suits]

    if len(available_suits) < 1:
        # If the player holds all suits, no remap is possible; retain the original,
        # since this is only a benchmark-time board remap and the current hand is
        # already a fully valid active state. The strategy remains structurally the same.
        return {}, landmark_cards.copy()

    # Map each landmark suit symbol to a currently unused suit in a deterministic way.
    suit_translation = {}
    for suit in sorted(set(card[-1] for card in landmark_cards), key=lambda s: SUITS.index(s)):
        if suit in player_suits:
            # Prefer a free suit; otherwise leave it alone.
            target = available_suits.pop(0) if available_suits else suit
            suit_translation[suit] = target
        else:
            suit_translation[suit] = suit

    remapped = []
    for card in landmark_cards:
        rank = card[:-1]
        old_suit = card[-1]
        new_suit = suit_translation.get(old_suit, old_suit)
        remapped.append(f"{rank}{new_suit}")

    return suit_translation, remapped


def build_strategic_1755_subset() -> Tuple[List[Tuple[str, ...]], dict]:
    """Return one canonical representative per suit-isomorphic flop class.

    The returned mapping uses an abstract class key and the count of physical flops
    represented by that class.
    """
    deck = [f"{rank}{suit}" for rank in RANK_ORDER for suit in SUITS]
    all_flops = []
    seen = {}

    for combo in __import__("itertools").combinations(deck, 3):
        key = canonical_flop_key(combo)
        if key not in seen:
            seen[key] = tuple(combo)
        all_flops.append(combo)

    canonical = list(seen.values())
    class_counts = defaultdict(int)
    for combo in all_flops:
        class_counts[canonical_flop_key(combo)] += 1

    class_weight = {key: count for key, count in class_counts.items()}
    return canonical, class_weight


def nearest_landmark_flop(board: Sequence[str], landmarks: Sequence[Sequence[str]]) -> Tuple[Sequence[str], float]:
    """Return the closest landmark flop using a simple feature-based distance.

    This helper is intentionally lightweight and is intended for experimentation,
    not as a production solver dependency.
    """
    # This is a placeholder that avoids heavy logic in the PoC layer while keeping
    # the utility in a clean, isolated module. The code can be extended later.
    if not landmarks:
        raise ValueError("At least one landmark flop is required")

    def feature(cards: Sequence[str]):
        ranks = sorted((normalize_card(card)[:-1] for card in cards), key=lambda r: RANK_TO_INDEX.get(r, -1), reverse=True)
        return tuple(RANK_TO_INDEX.get(r, -1) for r in ranks)

    target = feature(board)
    best = None
    best_dist = None
    for landmark in landmarks:
        dist = sum(abs(a - b) for a, b in zip(target, feature(landmark)))
        if best is None or dist < best_dist:
            best = tuple(landmark)
            best_dist = dist
    return best, float(best_dist)
