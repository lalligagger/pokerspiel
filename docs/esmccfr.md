I’m checking the actual algorithmic structure against external-sampling MCCFR so I can confirm exactly what is and isn’t being followed in this short-deck PokerKit implementation.

Searched for regex `external sampling|chance sampling|cfr\(|mccfr|counterfactual|reach probability|opponent reach|sampling` (`pokerkit_test` · `*.py`), 23 results

## Short answer

Not fully. The current framework is aligned with the CFR family and with the right game-tree structure, but it is not yet a faithful external-sampling MCCFR implementation in the formal sense.

What is in place in `mccfr_launcher.py` and `pokerkit_poc.py` is:

- canonical infoset keys
- legal action generation from PokerKit
- reducer-based filtering rather than custom action logic
- recursive game-tree traversal
- regret table with strategy extraction
- reach-probability style weighting

That is the right architectural direction, but it is not yet the exact formal procedure of external-sampling MCCFR.

---

## Why it is not formal external-sampling MCCFR

Formal external-sampling MCCFR does this:

- each iteration samples chance outcomes and opponent actions
- for the current player, all legal actions are traversed
- regrets are updated using counterfactual values weighted by opponent reach probability and chance reach
- only the traversed player’s regrets are updated for that sampled iteration
- utility is computed as player-specific expected payoff from terminal states

In our current setup:

- the recursive walk is effectively full traversal, not external sampling
- the code does not explicitly sample chance and opponent actions before updating regret
- the update is simplified and not yet separated cleanly by player / iteration / sampling regime
- the utility vector is still a simplified stack-based proxy, not a full game-theoretic utility decomposition tied to player-specific counterfactual values

So the current implementation is best described as:

- a recursive CFR-style prototype
- grounded in real PokerKit legality
- structurally close to MCCFR
- but not yet a complete external-sampling MCCFR algorithm

---

## What is correct about the framework

The following pieces are consistent with MCCFR formalism:

- extensive-form game tree over PokerKit states
- infoset key based on player, street, public board, private cards, action history, legal family
- strategy from regret values
- recursive action expansion over legal actions
- regret accumulation as a function of action utility differences
- using PokerKit legal family as the action source of truth

That is the right abstraction for a real MCCFR implementation in short-deck HUNL.

---

## What still needs to be true for a formal external-sampling MCCFR

To match the formal method, we need:

1. player-indexed update logic
   - update regrets for the acting player only
   - use opponent reach probability, not a symmetric scalar

2. explicit sampling regime
   - external sampling: sample chance and opponent actions
   - do not traverse every branch in an un-sampled full tree

3. correct counterfactual value formula
   - $R(I,a) \leftarrow R(I,a) + \pi_{-i}^{\sigma}(I) \cdot (u_i(\sigma|_{I \to a}) - u_i(\sigma, I))$

4. proper terminal utility
   - from player perspective, not a raw stack vector approximation
   - include payoff scaling for the current iteration

5. deterministic legal-action handling
   - preserve PokerKit-generated legal family
   - filter with our policy rules, but never replace the native legal set

---

## Bottom line

The framework is definitely on the correct formal path, but it is not yet “external sampling MCCFR” in the textbook sense.

It is more accurately:

- a PokerKit-grounded recursive CFR prototype
- valid enough to test infoset structure and legality
- not yet the final external-sampling MCCFR algorithm

If you want, I can convert this from prototype to proper external-sampling MCCFR next, while keeping the same PokerKit legality layer and the same reporting format in `mccfr_launcher.py`.