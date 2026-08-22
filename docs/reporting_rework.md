# Reporting and exact-state rework proposal

## Objective

Build a production-grade solver architecture that is:

- exact-state driven
- flat-array and integer-indexed in the hot path
- player-aware for all preflop lookup logic
- memory-safe and checkpoint-friendly
- inspectable through the API while training is paused or stopped
- free of steady-state range sampling unless a clearly diagnostic, unsupported path requires it

This proposal intentionally allows internal breakage while preserving the end-state API contract as much as practical.

---

## Guiding principles

1. No object-heavy hot path
   - no Python dicts or nested objects for each infoset/state in the training loop
   - no ad hoc per-iteration reconstruction of state from strings

2. Exact-state first
   - all runtime logic should be driven from exact compact state, not sampled approximations
   - reporting and export are downstream consumers of exact state, not upstream drivers

3. Player-partitioned lookup model
   - player 0 and player 1 should not share the same global infoset namespace
   - lookup resolution must clearly separate contexts by player

4. Read-only API serving
   - public range endpoints should answer from compact lookup tables, not by reconstructing ad hoc state on each request

5. Sampling only when necessary
   - sampled ranges are acceptable only as a temporary diagnostic aid for unsupported or exploratory deep post-flop scenarios
   - never as the normal production flow for preflop or core solver state

6. Graceful stop with API liveness
   - stopping the solver should halt training, preserve the object, and keep the app responsive for inspection and read-only queries

---

## End-state architecture

### 1) Solver core

The solver core lives in [app_solver.py](../app_solver.py) and should operate around compact flat arrays:

- infoset_id registry
- strategy array
- avg_strategy array
- regret array
- reach array
- visits array
- legal action metadata
- canonical labels / metadata index

The hot path uses only integer ids and array indexing, not Python objects.

### 2) Player-aware state partitioning

Exact state must be split by player so that strategy and reach are tracked under a clear namespace:

- player 0 state table
- player 1 state table
- shared canonical encoders and legal-action mappings

This avoids collisions between asymmetrically interpreted infosets.

### 3) Exact preflop range lookup

The preflop range service should use the same exact-state encoding model, not ad hoc sampling.

A request resolves as:

- canonicalize request into player/context/history/hand bucket
- resolve to a compact infoset id
- fetch exact action probabilities from the flat table
- materialize the result for the API

The public API can remain very close to the current UX, but the implementation should be backed by exact flat tables.

### 4) Reporting/export layer

The reporting/export layer is downstream from exact state and should derive summaries from the same flat model.

Reporting is for:

- selected-node summaries
- strategic diagnostics
- range export artifacts
- checkpoint summaries
- runtime profiling

It should not be used as a fallback source of truth for the training runtime.

---

## Implementation plan

### Phase 0 — lock the target architecture

Goals:

- agree on exact-state flat-array production model
- agree that only diagnostic sampling remains outside the steady-state path
- agree that the API stays familiar but the internal model changes

Tasks:

1. Freeze the production constraints
2. Confirm public endpoints remain largely unchanged
3. Identify legacy code to retire
4. Record the target architecture in repo docs

Exit criteria:

- there is a clear end-state architecture
- no one is proposing a mixed object-heavy + sample-driven runtime as the default route

---

### Phase 1 — replace the legacy hot-path state with flat arrays

Goals:

- remove the heavy object model from the training path
- use integer infoset ids and arrays for state tracking

Tasks:

1. Introduce flat solver state structures in [app_solver.py](../app_solver.py)
2. Replace string-based state keys with compact integer encodings
3. Add canonical encoding helpers for:
   - player
   - street
   - action history family
   - preflop context
   - hand bucket / combo bucket
4. Track strategy, avg strategy, reach, visits in arrays
5. Keep arrays memmap-backed or fixed-size where appropriate

Cleanup:

- remove legacy per-state Python dicts and nested structures from the hot loop
- delete dead helper code that existed only to support the old object model
- retire legacy debug-only accumulation paths that are no longer needed

Exit criteria:

- solver loop works with integer ids and arrays only
- no heavy Python objects in the hot path

---

### Phase 2 — split exact state by player

Goals:

- make player-aware lookup explicit and robust
- avoid cross-player collisions in range and reach accounting

Tasks:

1. Partition state tables by player
2. Encode player into the canonical infoset id
3. Ensure lookups always resolve against the correct player table
4. Update all range and reporting functions to accept player-specific semantics

Cleanup:

- remove any global or shared namespace assumptions
- delete compatibility layers that smuggled player logic into a single table

Exit criteria:

- player 0 and player 1 are fully isolated at the infoset and table level

---

### Phase 3 — build exact preflop range lookup tables

Goals:

- serve preflop ranges from exact state, not samples
- preserve the external API shape while changing the internal model

Tasks:

1. Build a preflop lookup index keyed by canonical infoset id
2. Resolve each request to the player-specific, context-specific table
3. Fetch exact policy rows from the flat arrays
4. Materialize the same response format the API expects
5. Keep API read-only behavior intact

Cleanup:

- remove any code that synthesizes range outputs from sampled snapshots
- delete synthetic aggregate helpers that only approximate exact lookup data

Exit criteria:

- range queries resolve directly from exact state
- no production route depends on range sampling

---

### Phase 4 — use exact state for reporting and export

Goals:

- make export and summaries originate from exact solver state
- keep reports diagnostic and non-authoritative

Tasks:

1. Rework [range_export.py](../range_export.py) to read exact state tables
2. Retain selected-node summaries only as derived diagnostics
3. Keep range dumps consistent with the canonical flat tables
4. Remove snapshot-driven report generation from the normal runtime path

Cleanup:

- remove aggregate helper code created only for legacy sampling patterns
- eliminate duplicate reporting logic with no single source of truth

Exit criteria:

- reports are reproducible from exact state
- reporting no longer drives policy behavior

---

### Phase 5 — add checkpoint telemetry and graceful stop semantics

Goals:

- protect the long-running process from unbounded growth or memory pressure
- keep the solver inspectable after a graceful stop

Tasks:

1. Add checkpoint payload with:
   - iteration
   - memory usage
   - disk footprint
   - state summary
   - stop threshold status
2. Add cooperative training stop path that:
   - sets a stop flag
   - exits the training loop cleanly
   - preserves the solver object in memory
   - keeps the app alive and queryable
3. Ensure the API can continue serving read-only diagnostics while paused

Cleanup:

- remove stale threshold logic that mixes distinct concepts
- consolidate config parsing so safety and stability thresholds remain separate

Exit criteria:

- solver can stop gracefully without object destruction
- service remains inspectable and queryable in stop state

---

### Phase 6 — restrict sampling to explicit diagnostics only

Goals:

- enforce the rule that the steady-state path is exact-state-only
- leave sampling available only for unsupported exploratory work

Tasks:

1. Locate all range sampling code paths
2. Isolate them behind a diagnostic flag or submodule
3. Ensure normal runtime paths never call them by default
4. Use them only for unsupported deep post-flop exploratory work

Cleanup:

- delete dead fallback sampling code in primary pipelines
- remove fallback logic that pollutes production outputs

Exit criteria:

- production runtime does not rely on sampling
- the only remaining sampling is intentionally diagnostic

---

### Phase 7 — validation: local and Docker

Goals:

- prove the new structure behaves correctly in practice

Tasks:

1. Validate imports and syntax for updated solver and API code
2. Run Docker build and container startup
3. Check health and status endpoints
4. Hit preflop range endpoints
5. Validate graceful-stop behavior while API remains available

Exit criteria:

- build passes
- container starts
- health/status and range queries work
- graceful stop does not destroy the solver object

---

### Phase 8 — remove compatibility shims and finalize cleanup

Goals:

- finish the internal rework and leave a coherent production architecture

Tasks:

1. Remove transitional compatibility wrappers
2. Delete stale code paths and dead helper modules
3. Simplify service and solver logic to rely on the exact-state model only
4. Keep only necessary public API contracts

Cleanup:

- no leftovers from the old object-heavy state model
- no temporary sampling fallback in normal flow

Exit criteria:

- the repo reflects the clean, exact-state, production architecture only

---

## Expected benefit

This rework delivers:

- memory-safe training
- better serviceability during long runs
- exact range servicing for preflop
- clean separation between solver, range service, and reporting
- graceful stop semantics without breaking the API surface
- a production path that is easier to reason about and extend

---

## Practical working rule

At every step, prefer:

- exact state over sample approximations
- flat arrays over Python objects
- player-aware keying over global ambiguity
- read-only API query serving over reconstruction

This keeps the system aligned with the production goals while still allowing internal-breaking changes during the refactor.
