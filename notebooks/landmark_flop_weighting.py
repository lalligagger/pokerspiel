import json
import itertools
from collections import Counter
import numpy as np
from pokerkit import Card

# 1. Paste the raw 25-flop list you found
RAW_25_FLOPS = [
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


def extract_features(cards: tuple) -> np.ndarray:
    """
    Extracts numerical features from a 3-card flop to place it in vector space.
    Features: [HighRank, MidRank, LowRank, IsMonotone, IsTwoTone, IsPaired, TotalGaps]
    """
    pk_cards = [Card.parse(c) for c in cards]
    
    # Ranks (PokerKit index: 0 for '2' up to 12 for 'A')
    ranks = sorted([c.rank.index for c in pk_cards], reverse=True)
    
    # Suit pattern
    suits = [str(c.suit) for c in pk_cards]
    max_suit_count = max(suits.count(s) for s in suits)
    is_monotone = 1.0 if max_suit_count == 3 else 0.0
    is_twotone = 1.0 if max_suit_count == 2 else 0.0
    
    # Pairing status
    is_paired = 1.0 if len(set(ranks)) < 3 else 0.0
    
    # Straight connectivity (gaps between sorted cards)
    gap1 = max(0, ranks[0] - ranks[1] - 1)
    gap2 = max(0, ranks[1] - ranks[2] - 1)
    total_gaps = float(gap1 + gap2)
    
    # Return numerical array for distance math
    return np.array([
        float(ranks[0]), float(ranks[1]), float(ranks[2]),
        is_monotone, is_twotone, is_paired, total_gaps
    ])

def main():
    print("Preparing landmark feature vectors...")
    landmark_vectors = [extract_features(flop) for flop in RAW_25_FLOPS]
    
    # Create an empty tally sheet for our 25 landmarks
    # We use a string key so it's easy to print/save to JSON later
    tally = {str(flop): 0 for flop in RAW_25_FLOPS}
    
    # 2. Build a full deck of 52 cards using PokerKit naming
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    deck = [f"{r}{s}" for r in ranks for s in suits]
    
    # Generate all 22,100 unique physical card combinations
    all_possible_flops = list(itertools.combinations(deck, 3))
    print(f"Processing all {len(all_possible_flops)} physical flops...")
    
    # 3. Nearest Neighbor Loop
    for idx, physical_flop in enumerate(all_possible_flops):
        if idx % 5000 == 0 and idx > 0:
            print(f"  Processed {idx} flops...")
            
        flop_vector = extract_features(physical_flop)
        
        # Calculate Euclidean distance to every landmark
        distances = [np.linalg.norm(flop_vector - lm_vector) for lm_vector in landmark_vectors]
        
        # Find the index of the closest landmark flag
        closest_landmark_idx = np.argmin(distances)
        closest_flop_tuple = RAW_25_FLOPS[closest_landmark_idx]
        
        # Log the hit in our tally sheet
        tally[str(closest_flop_tuple)] += 1

    print("\n--- Processing Complete ---")
    
    # 4. Crucial Invariant Sanity Check
    total_tallied = sum(tally.values())
    print(f"Total Flops Tallied: {total_tallied} (Must be exactly 22100)")
    assert total_tallied == 22100, "Math error! Missing card subsets detected."
    
    # 5. Output formats
    print("\n[Raw Integer Weights Mapping]")
    print(json.dumps(tally, indent=2))
    
    print("\n[Normalized Probability Weights Mapping]")
    normalized_weights = {key: val / 22100.0 for key, val in tally.items()}
    print(json.dumps(normalized_weights, indent=4))

if __name__ == "__main__":
    main()
