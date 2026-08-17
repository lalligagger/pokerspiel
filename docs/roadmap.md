# PokerKit Short-Deck MCCFR Roadmap

## Goal

Integrate a PokerKit-backed short-deck HUNL MCCFR loop while preserving PokerKit legal action generation and filtering through a policy layer instead of inventing a custom action engine.

## Principles

- PokerKit legal actions are the source of truth.
- Policy filtering is a constraint layer, not a replacement for PokerKit legality.
- Keep compact string representations for MCCFR internal keys, but preserve raw tuple actions at the boundary.
- Use a strict smoke-test loop before enabling training logic.

## Phase 1: State and legality validation

### 1.1 Terminal-state validation
- [x] Confirm a full hand can run from preflop through showdown or forced terminal state.
- [x] Check that state progression through streets is stable and consistent.
- [x] Confirm `status` flips to a terminal state under the actual PokerKit game.
- [x] Ensure no unknown streets appear in the progression loop.

### 1.2 Policy-filter validation
- [x] Validate legal actions are filtered correctly without removing all legal actions unexpectedly.
- [x] Confirm the reducer preserves PokerKit legal families while enforcing policy constraints.
- [x] Confirm the strict mode is safe for training use.

### 1.3 Node/action observation validation
- [x] Validate per-state node keys and actor identity under real PokerKit states.
- [x] Confirm observed histories are grouped by betting history and state context, not just a raw iteration label.
- [x] Confirm action families and sizes remain consistent with PokerKit legal action generation.

## Phase 2: Compact MCCFR action and node encoding

### 2.1 Raw-to-compact translation
- [x] Keep raw PokerKit action tuples as the external truth.
- [x] Provide canonical compact string encodings for trainer internals.
- [x] Round-trip compact strings back to tuple form without ambiguity.

Examples:
- `('check_or_call', 0)` -> `check_or_call:0`
- `('fold', 0)` -> `fold:0`
- `('bet_or_raise', 4)` -> `bet_or_raise:4`
- `('bet_or_raise', 16)` -> `bet_or_raise:16`

### 2.2 Trainer keying
- [x] Use canonical state-aware keys in MCCFR tables.
- [x] Keep readable debug output using compact human-friendly names when needed.
- [x] Ensure key normalization is consistent across streets and actor positions.

This is now represented as a compact key of the form:
- `street:pX:board=...:hole=...:hist=...:legal=...`

## Phase 3: MCCFR integration

### 3.1 Single-hand MCCFR smoke test
- [x] Build a trainer loop that plays one hand end-to-end in a smoke-test style.
- [x] Confirm legal actions are generated from PokerKit and filtered through policy.
- [x] Confirm each chosen action is applied correctly via `apply_action`.
- [x] Confirm regret tables and strategy tables update and display nonzero regret entries.

### 3.2 Uniform policy baseline
- [x] Use uniform policy as the default smoke-test strategy.
- [x] Confirm the runner reaches terminal states over repeated iterations.
- [x] Track node accumulation and legal action counts under the filtered policy.
- [x] Verify the output is saved to a repo-visible JSON artifact with Docker output-path support.

### 3.3 Multi-iteration training
- [x] Run repeated iterations with the uniform baseline and observe node accumulation.
- [x] Check strategy updates are stable and finite across the node table.
- [x] Inspect node accumulation and action frequencies in the live summary output.

## Phase 4: Real MCCFR trainer loop

### 4.1 Counterfactual regret update
- [ ] Replace the smoke-style regret proxy with actual counterfactual regret updates.
- [ ] Use canonical infoset keys consistently during traversal.
- [ ] Weight regret updates by reach probabilities and action probabilities.
- [ ] Confirm the regret table changes under true CFR-style updates, not just sampled terminal outcomes.

### 4.2 Strategy normalization and stability
- [ ] Normalize strategy from positive regrets per infoset.
- [ ] Ensure strategy probabilities remain finite and sum to 1.
- [ ] Ensure repeated updates do not drift into invalid values.

### 4.3 Longer-run validation
- [ ] Run a modest multi-iteration MCCFR smoke test with live summary output.
- [ ] Verify node counts stabilize and values remain finite across refresh cycles.
- [ ] Inspect the final table for repeated keys, illegal actions, and consistent legal families.

## Phase 5: Production checks

- [ ] No synthetic action families beyond PokerKit legal actions.
- [ ] No hardcoded false assumptions about betting sizes or street invariants.
- [ ] No unknown-street states in aggregated diagnostics.
- [ ] Terminal progression and node accumulation remain stable over long runs.
- [ ] The Docker bootstrap remains the canonical environment for OpenSpiel + PokerKit validation.

## Immediate next step

Implement the real counterfactual regret update and strategy normalization using the canonical node keying already in place, then validate the first true MCCFR smoke run over a modest iteration count.

## Files of interest

- `mccfr_launcher.py`
- `pokerkit_poc.py`
- `node_action_probe.py`
- `test_action_space_reducer.py`
- `shortdeck_hunl_action_overrides.json`
- `pokerkit_fork/pokerkit/state.py`
- `docker-compose.yml`
