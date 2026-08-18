# pokerkit-openspiel

## HULH infoset definition

In heads-up limit hold’em, an infoset is the full set of game states that a player cannot distinguish because their private cards are hidden and the player only sees the public information available at that point.

For a real acting player in HULH, the relevant infoset is not just “the current street.” It is the combination of:

- player identity and acting position
- hidden hole cards for that player
- public board cards on the current street
- current pot and bet-to-call context
- action history on the current hand
- the street and betting round we are in
- any legal action family available at the node

Concretely, an infoset for player i at a decision point is:

- street: preflop, flop, turn, or river
- current player: i
- private holding: the exact two-card hole combination for i
- board state: public cards on board so far
- pot context: current pot size, contribution to call, outstanding bet sizes
- action history: all prior actions made on this hand, including checks, calls, bets, raises, and folds
- legal actions: the action set available from PokerKit at that node, after any wrapper filtering or abstraction

So the infoset is effectively:

- I = (player, hole_cards, board, pot_context, history, street, legal_actions)

This is the state that must be treated as one information set during learning: the player cannot tell apart histories that have the same public state and same hidden cards, and the solver must choose a strategy over this set.

A simple example is a preflop infoset:

- player = 1
- hole cards = {Ac, Kd}
- board = {}
- pot context = 3 (big blind + small blind contribution in the standard toy setup)
- history = []
- street = preflop
- legal actions = {fold, call/check, bet/raise}

This is one infoset for player 1. A different private holding such as {As, 7s} or a different action history such as a previous raise would belong to a different infoset, even if the public board is the same.

The same principle applies on later streets. For example, on the flop, the infoset includes the current player, public flop cards, pot size, current bet sizes, the action sequence so far, and whether the player is facing a bet, a raise, or a check.

The key point is that regret minimization operates over infosets, not over exact game-tree nodes. The solver learns a strategy for each infoset, and the strategy is shared across all identical hidden-card/public-history scenarios that the player cannot distinguish.

## From infosets to regret minimization

CFR and MCCFR learn by maintaining regret values for actions at each infoset. At a given infoset, the algorithm compares the utility of each available action against the current strategy mix and accumulates the difference.

If a player would have done better by taking action a instead of following the current policy, that positive regret is added to the regret for a. Over time, the strategy is updated toward actions that have historically been better.

The intuition is straightforward:

- each infoset has a set of legal actions
- each action has an expected value under the current strategy profile
- the regret for taking a different action is the improvement that would have occurred relative to the currently played action
- the average strategy gradually shifts toward actions with positive cumulative regret

Note on terminology: expected value is the standard poker concept for the average value of an action under a distribution over opponent actions and chance; regret is the difference between the value of the action you took and the value of the best action you could have taken at that infoset. So regret is not a separate poker concept in opposition to EV; it is a running measure of how much better an alternative action would have been than the current one. In other words, regret is the “value gap” between the chosen action and the best available action, while expected value is the baseline value we are comparing against.

This is exactly why regret minimization is a natural fit for imperfect-information poker: the game is not a simple single-agent optimization problem, because the player does not know the opponent’s private cards, but the solver can still update policies on the right information sets.

## MCCFR and its variants in HULH

MCCFR is a sampled version of CFR. Instead of traversing the entire game tree at every iteration, MCCFR samples only part of the game tree or a subset of action branches according to a sampling scheme. This makes it much more practical for large poker games.

For HULH, the usual pattern is:

- use the exact game rules from a PokerKit-backed wrapper
- represent each decision as an infoset keyed by the relevant hidden/public information
- keep per-infoset regrets and average strategy tables
- sample trajectories through the game tree
- update regrets only along the sampled path(s)
- accumulate average strategy over time

Different MCCFR variants differ in what is sampled:

- external sampling: sample opponent and chance actions, update the player’s own action set along the sampled trajectory
- outcome sampling: sample a terminal outcome or an action path and update only the sampled branch
- full CFR: traverse the whole tree and update all actions

In practice, for HULH, the wrapper-based OpenSpiel flow uses the real game object, keeps the PokerKit legal action space intact, and then samples states and policies at valid actor nodes. The solver does not need a custom recursive poker solver; it uses the real OpenSpiel MCCFR machinery on the actual game representation, while the reporting and filtering layer compresses the extracted policy into meaningful human-readable infosets and ranges.

The high-level story is:

1. the state is represented as a real HULH game tree with imperfect information
2. the relevant infoset encodes hidden cards, public board, action history, and pot context
3. regret minimization updates that infoset-level policy over time
4. MCCFR samples the tree to make the updates tractable
5. the final average strategy approximates the equilibrium strategy profile for the HULH game

This is the core solver model currently being used in the wrapper-based HULH flow.

---


new "lightweight" run for preflop ranges
```
docker compose run pokerkit-open-spiel \
  python app_solver.py hulh \
  --iterations 400000 \
  --checkpoint-every 2000 \
  --preset hulh-preflop \
  --samples 4000 \
  --stability-threshold 0.01 \
  --stop-patience 3 \
  --solver outcome \
  --report-mode summary \
  --artifact-mode lightweight \
  --checkpoint-history-limit 2 \
  --range-last-n 2000 \
  --output-json /app/overnight_runs/hulh_400k_lightweight/report.json
```
