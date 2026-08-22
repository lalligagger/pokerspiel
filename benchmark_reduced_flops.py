#!/usr/bin/env python3
"""Benchmark harness for reduced-flop training experiments.

This script is intentionally benchmark-only and does not change the production
solver or API layer. It compares these modes over the same 10k iteration budget:

- full: all physical flops
- reduced_1911: suit-isomorphic reduced set
- landmark_25: a compact landmark representative set

It is designed to run under the repo's Docker environment and reports only final
summary metrics, not intermediate checkpoint snapshots.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from utils.flop_isomorphism import build_strategic_1755_subset, build_suit_isomorphism
from utils.flop_sampler import all_physical_flops

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']

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


def parse_flop_string(flop: str):
    if len(flop) != 6:
        raise ValueError(f'Expected a 3-card flop string, got {flop!r}')
    return [flop[i:i+2] for i in range(0, 6, 2)]


def sample_full_distribution(iterations: int):
    physical = all_physical_flops()
    if not physical:
        raise RuntimeError('No physical flops generated')
    return {'mode': 'full', 'members': physical, 'weights': {board: 1.0 for board in physical}, 'iterations': iterations}


def sample_reduced_1911_distribution(iterations: int):
    canonical, weights = build_strategic_1755_subset()
    # preserve the full physical count in the benchmark distribution
    return {'mode': 'reduced_1911', 'members': canonical, 'weights': weights, 'iterations': iterations}


def sample_landmark_distribution(iterations: int, landmark_count: int = 25):
    members = [parse_flop_string(flop) for flop in LANDMARKS_25[:landmark_count]]
    weights = {tuple(member): 1.0 for member in members}
    return {'mode': f'landmark_{landmark_count}', 'members': members, 'weights': weights, 'iterations': iterations}


def random_from_distribution(dist, rng):
    members = list(dist['members'])
    weights = [dist['weights'].get(tuple(board), 1.0) for board in members]
    choice = rng.choices(members, weights=weights, k=1)[0]
    return tuple(choice)


def check_remap_examples():
    examples = [
        (['As', 'Kd'], ['As', 'Jh', '4d']),
        (['Ah', 'Ks'], ['As', '9h', '7d']),
        (['Qd', '7s'], ['Qs', '8h', '2c']),
    ]
    rows = []
    for player_cards, landmark in examples:
        translation, remapped = build_suit_isomorphism(player_cards, landmark)
        rows.append({
            'player_cards': player_cards,
            'landmark': landmark,
            'translation': translation,
            'remapped': remapped,
            'collision_free': set(remapped).isdisjoint(set(player_cards)),
        })
    return rows


def benchmark_mode(mode: str, iterations: int, landmark_count: int = 25):
    rng = __import__('random').Random(12345)
    if mode == 'full':
        dist = sample_full_distribution(iterations)
    elif mode == 'reduced_1911':
        dist = sample_reduced_1911_distribution(iterations)
    elif mode == 'landmark_25':
        dist = sample_landmark_distribution(iterations, landmark_count=landmark_count)
    else:
        raise ValueError(f'Unsupported mode: {mode!r}')

    start = time.perf_counter()
    seen = Counter()
    for _ in range(iterations):
        board = random_from_distribution(dist, rng)
        seen[board] += 1
    elapsed = time.perf_counter() - start

    summary = {
        'mode': dist['mode'],
        'iterations': iterations,
        'elapsed_seconds': round(elapsed, 4),
        'iterations_per_second': round(iterations / elapsed, 3) if elapsed > 0 else 0.0,
        'distinct_boards_seen': len(seen),
        'sample_weight_total': sum(seen.values()),
        'top_boards': [{'board': list(board), 'count': count} for board, count in seen.most_common(5)],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Reduced-flop benchmark harness for preflop sampling experiments.')
    parser.add_argument('--iterations', type=int, default=10000, help='iterations to run in each benchmark mode')
    parser.add_argument('--landmark-count', type=int, default=25, help='landmark subset size to use when mode=landmark')
    parser.add_argument('--modes', nargs='+', default=['full', 'reduced_1911', 'landmark_25'],
                        help='benchmark modes to run; valid values: full, reduced_1911, landmark_25')
    parser.add_argument('--show-remaps', action='store_true', help='print a few collision-safe remap examples')
    args = parser.parse_args()

    aggregated = []
    for mode in args.modes:
        run_summary = benchmark_mode(mode, args.iterations, landmark_count=args.landmark_count)
        aggregated.append(run_summary)

    if args.show_remaps:
        print('remap_examples=' + json.dumps(check_remap_examples(), indent=2, sort_keys=True))

    print('benchmark_summary=' + json.dumps(aggregated, indent=2, sort_keys=True))

    output_path = Path('tmp/reduced_flop_benchmark.json')
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True) + '\n')
    print(f'benchmark_report_written={output_path}')


if __name__ == '__main__':
    main()
