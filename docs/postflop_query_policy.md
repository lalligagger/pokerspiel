# Post-flop query policy

This note defines the canonical policy for post-flop queries in this repo.

## Scope

We support exactly two query-time post-flop modes:

1. exact infoset lookup
2. range estimate over a chosen hand subset

There is no startup-time post-flop probe setup. We do not pre-sample or precompute post-flop query sets during solver initialization.

## Design principle

We intentionally keep two layers separate:

- raw wrapper semantics from PokerKit/OpenSpiel
- project-facing readability and convenience labels

The raw wrapper is authoritative for exact game state.
The project layer is authoritative for user-facing names and compact range expressions.

## 1) Action IDs and mapping

The raw wrapper exposes legal action IDs as wrapper-specific integers. Those IDs are not a stable semantic layer we should treat as universal poker meaning.

For the project-facing API, we intentionally keep a small HULH-oriented normalization layer:

- fold -> 0
- check_call -> 1
- bet_raise -> 4

This is a convenience layer for HULH policy review and should be treated as a project shorthand, not as a raw PokerKit semantic contract.

## 2) Hand representation

We support two distinct representations:

- hand classes: AA, AKo, QJs
- raw combos: [As, Ad], [As, Ks], [Qd, Js]

These are not the same thing.

- a hand class is a human-readable category
- a raw combo is an exact two-card object or tuple

To avoid ambiguity, raw combos must be treated as actual card tuples or arrays rather than ultra-compact strings.

Example canonical raw combo shape:

```json
{"combo": ["As", "Ad"]}
```

Example hand-class display shape:

```json
{"class": "AKo"}
```

## 3) History

History is fine as a project-facing shorthand, for example:

- bet
- bet, call
- bet, bet, raise

This is valid for readability, but exact state matching must still be based on the canonical underlying game state and the precise action sequence.

## 4) Infoset identity

The infoset ID must be strict and 1:1 once game type and player-count assumptions are fixed.

The canonical infoset key should be built from the exact game context, not from a human-readable display string.

Recommended key fields:

- game = hulh
- player = acting player index
- board = sorted exact board cards
- hole = sorted exact hole cards for the acting player
- history = exact action sequence

Example:

```text
game=hulh|player=0|board=[Ah,Kd,2c]|hole=[As,Qs]|history=[bet,bet]
```

This is a strict ID. It is not a compact display label, and it should never be replaced by a hand class such as AKo or a shorthand like AsQs.

## 5) Supported query modes

### Exact infoset lookup

Inputs:

- board
- history
- hole cards for the acting player
- acting player

Output:

- exact current policy probabilities at that infoset
- strict infoset key

### Range estimate

Inputs:

- board
- history
- chosen hand subset
- acting player

Output:

- average action frequencies across the supplied subset
- no startup precomputation
- no hidden implicit sampling setup

## 6) What we avoid

We intentionally avoid:

- pre-run post-flop probe seeding
- implicit selected-node post-flop setup during solver startup
- mixing hand classes and raw combo labels in one canonical ID field
- treating project shorthand as raw PokerKit semantics

## 7) Summary

The safe rule is:

- raw PokerKit wrapper state is the truth source
- exact infoset keys are strict and deterministic
- project-facing names and compact labels are convenience layers for human readability
- hand classes and raw combos are separate concepts and must not be conflated
