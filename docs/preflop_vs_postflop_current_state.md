# Preflop vs. Postflop Sampling and Library Strategy

## Current state

The current solver architecture keeps a clear separation between:

- real game-tree training
- selected-node checkpoint reporting for preflop
- diagnostic / query-time postflop probing

This separation is intentional. The goal is to preserve the solver's natural state distribution while still exposing a compact, interpretable set of exact-state views for reporting and API queries.

## Preflop: good fit for a selected-node checkpoint library

Preflop is the right place to build a compact checkpoint library because:

- the action history is short and canonical
- the key spots are structurally repeated across many deal states
- the state space is small enough to meaningfully aggregate by history family and hand bucket
- we can normalize a handful of historical decision points such as:
  - first_to_act
  - response_to_limp
  - response_to_open
  - response_to_limp_raise
  - response_to_open_3bet
  - response_to_open_4bet
  - response_to_open_5bet

This creates a reasonable reporting surface for ranges without changing the underlying learning distribution.

The current design intentionally treats these as selected-node observations rather than as a custom training regime. In other words, they are for:

- checkpointing policy at important exact spots
- aggregating action frequencies over a canonical hand family
- exposing range-style APIs to the dashboard

not for rewriting the game tree or forcing specific deal-state frequencies into training.

## Postflop: not a good candidate for a dense library in the same manner

Postflop is qualitatively different:

- the exact board + hole-card + history combinations explode combinatorially
- many states are highly sparse or ephemeral
- the action histories are longer and much less reusable across deal states
- forcing particular boards or betting sequences would significantly distort the natural training distribution

For that reason, we should not treat postflop the same way as preflop. A dense postflop library of canonical spots is not a natural fit unless we deliberately design a constrained training objective or a separate abstraction layer.

## Why forcing deal states is risky

Forcing deal states and betting histories to make MCCFR "train around" a selected spot can be valid only under explicit experimental design, such as:

- a constrained training distribution
- a warm-start or targeted bootstrapping regime
- a deliberately biased sampling objective that is clearly labeled as such

Without that framing, this is simply a changed data distribution, which means:

- the policy is no longer a standard average-policy estimate for the original game
- the resulting range may seem more stable than it really is
- post-flop or deeper exact-state outputs can become overfit to the forced sample path rather than the true game equilibrium

This is why the current postflop path remains diagnostic and query-time rather than a dense custom library.

## Current practical architecture

The working model is:

1. Preflop selected-node checkpoints are sampled and aggregated for reporting.
2. Postflop exact lookups remain query-time and diagnostics-focused.
3. The API exposes postflop exact/range retrieval behind min-iteration/stability gating.
4. The training loop continues to use the natural game-state distribution rather than a synthetic, forced-state path.

This preserves both:

- solver validity
- meaningful observability

without turning selected-node sampling into a biased training process.

## Recommendation

Keep the preflop selected-node library model. It is a good fit for the structure of the game and the reporting goals.

Keep postflop sparse and exact. It should remain a targeted sampling/inspection tool rather than a dense library unless the project later adds a deliberate, well-specified constrained-training design.
