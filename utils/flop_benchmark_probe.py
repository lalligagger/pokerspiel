"""Quick reduced-flop benchmark probe.

This is intentionally not a solver run. It checks that:
- the full deck decomposes to 22,100 physical flops
- the suit-isomorphic reduction yields a reduced strategic set
- a small landmark set can be matched against the reduced class space

This gives a fast validation of the remapping/benchmark logic before any full
training run is attempted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations

from utils.flop_isomorphism import build_strategic_1755_subset, build_suit_isomorphism
from utils.flop_sampler import all_physical_flops

# https://justpaste.it/8yjcr
LANDMARKS_25 = [
    'KhKdTs',
    'Kh8s7s',
    '9s8s5h',
    'AsQs9h',
    'KsTsTh',
    'Js5h4d',
    '9s6s2s',
    'AsQhQd',
    '6s4h2d',
    '8s6h2d',
    'Ts8h3s',
    'As6s5s',
    'AsAhKs',
    '7s5h2s',
    'QhJs8s',
    'Ks9h3d',
    'As7h3d',
    'Ah5s4s',
    'Js9h6s',
    'QsTs2h',
    '9s7h4d',
    '8h4s3s',
    'Qs7h5d',
    'JhTs6s',
    'Js4h3s',
]

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']


def as_3card_tuple(flop_str: str):
    if len(flop_str) != 6:
        raise ValueError(f'Expected 3-card flop string like AsKdQh, got {flop_str!r}')
    return [flop_str[i:i+2] for i in range(0, 6, 2)]


def sample_landmarks(n: int):
    """Return a deterministic sample of representative flops from the strategic set."""
    canonical, _ = build_strategic_1755_subset()
    if n <= 0:
        return []
    selected = canonical[: min(n, len(canonical))]
    return [''.join(card for card in flop) for flop in selected]


def build_dataset(mode: str, landmark_count: int):
    if mode == 'full':
        return {'name': 'full', 'members': all_physical_flops(), 'count': len(all_physical_flops())}

    if mode == 'strategic_1755':
        canonical, counts = build_strategic_1755_subset()
        return {'name': 'strategic_1755', 'members': canonical, 'count': len(canonical), 'weights': counts}

    if mode == 'landmark':
        landmark_members = [as_3card_tuple(flop) for flop in (LANDMARKS_25[:landmark_count] if landmark_count else LANDMARKS_25)]
        return {'name': f'landmark_{landmark_count}', 'members': landmark_members, 'count': len(landmark_members)}

    raise ValueError(f'Unsupported mode: {mode!r}')


def main():
    parser = argparse.ArgumentParser(description='Reduced-flop benchmark probe for HULH flop sampling experiments.')
    parser.add_argument('--mode', choices=['full', 'strategic_1755', 'landmark'], default='strategic_1755',
                        help='flop distribution to benchmark: full deck, strategic 1755 classes, or landmark subset')
    parser.add_argument('--landmark-count', type=int, default=25, help='number of landmark flops to use when mode=landmark')
    parser.add_argument('--show-remap-samples', action='store_true', help='print a few suit-remapping examples')
    args = parser.parse_args()

    physical = all_physical_flops()
    canonical, counts = build_strategic_1755_subset()
    dataset = build_dataset(args.mode, args.landmark_count)

    print(f'mode={args.mode}')
    print(f'physical_flops={len(physical)}')
    print(f'canonical_classes={len(canonical)}')
    print(f'sum_class_weight={sum(counts.values())}')
    print(f'selected_count={dataset["count"]}')
    print(f'selected_name={dataset["name"]}')

    if args.mode == 'strategic_1755':
        print(f'class_weight_min_max={min(counts.values())} {max(counts.values())}')

    if args.mode == 'landmark':
        first = dataset['members'][:5]
        print('landmarks_preview=' + ', '.join(''.join(card for card in flop) for flop in first))

    if args.show_remap_samples:
        examples = [
            (['As', 'Kd'], ['As', 'Jh', '4d']),
            (['Ah', 'Ks'], ['As', '9h', '7d']),
            (['Qd', '7s'], ['Qs', '8h', '2c']),
        ]
        for player_cards, landmark in examples:
            translation, remapped = build_suit_isomorphism(player_cards, landmark)
            print('remap_example', {'player': player_cards, 'landmark': landmark, 'translation': translation, 'remapped': remapped})

    # Optional diagnostic: compare the sample landmark set to the canonical set size.
    if args.mode == 'landmark':
        print(f'landmark_ratio={len(dataset["members"]) / len(canonical):.4f}')


if __name__ == '__main__':
    main()
