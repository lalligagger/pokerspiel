# DeepCFR integration assessment and implementation path

## Executive summary

Supporting a DeepCFR OpenSpiel model is feasible in this codebase, but it is not a small extension of the current HULH MCCFR runtime. The current solver architecture is optimized for a selected-node, flat-memory, checkpoint-driven external solver loop with compact reporting and read-only API inspection. DeepCFR changes the core training model, the representation of policy parameters, the training loop cadence, and the shape of outputs.

The practical path is to treat DeepCFR as a separate model family, not as a drop-in replacement for the current solver object. The right pattern is:

- keep the current flat checkpoint/report pipeline intact
- add a second solver backend behind the same service interface
- expose DeepCFR outputs through the same range/inspection contracts
- keep training state, network parameters, and checkpoint artifacts separate from the lightweight HULH selected-node path

This keeps the current live API stable while enabling a new training engine when ready.

## Current architecture: what it is built for

The project currently uses a selected-node, production-oriented MCCFR runner with these characteristics:

- compact, array-like state tracking
- selected-node probes and aggregated range summaries
- checkpoint-driven stability and reporting
- read-only API inspection while the solver is running
- emphasis on bounded memory and graceful shutdown without deleting the solver state

The main code paths are centered around:

- `app_solver.py` for solver benchmarking, selected-node summary, checkpoints, report generation
- `api/service.py` for the live solver runtime and API surface
- `api/router.py` for inspection endpoints
- `cfg/*.json` for runtime configuration and thresholds

This is a very good fit for a production external MCCFR or HULH-like sampled solver, but it is not a deep-learning training pipeline by default.

## Why DeepCFR is different

DeepCFR is conceptually and operationally different from the current implementation.

### 1. Training is network-based, not policy-table-based

The current design is mostly policy-centric and aggregate-based:

- average policy is computed from repeated iterations
- selected-node probes sample policy states
- aggregates summarize action frequencies
- checkpoints store compact summaries, not model weights

DeepCFR instead trains neural networks approximating counterfactual values and policies. That means the runtime must manage:

- model parameters and optimizer state
- training batch logic
- value network updates
- policy network updates
- a larger and more expensive memory footprint than compact MCCFR tables

This shifts the main problem from "keep policy objects small" to "train and persist neural model state safely".

### 2. The training loop is not the same shape

The current `run_iteration()`-style loop is short, compact, and incremental. DeepCFR typically has a structure like:

- traverse sampled game tree
- compute counterfactual values and regrets
- accumulate policy/value losses
- perform training steps over optimizer batches
- periodically checkpoint model weights and training state

This is much more batch-oriented and model-oriented than the current code path. A direct swap into the present loop is not realistic.

### 3. OpenSpiel DeepCFR integration is different from the custom wrapper path

OpenSpiel already provides DeepCFR-related abstractions and utilities in the broader OpenSpiel ecosystem, but the current project is using a specific Python poker wrapper plus custom policy-sampling logic. That means the DeepCFR integration likely needs:

- a model adapter layer for the OpenSpiel game state
- a policy/value network wrapper that maps state features into action logits/values
- a deployment path that uses either OpenSpiel’s neural training utilities or a custom trainer built around the same state abstraction

This is not a trivial addition to the current file set; it is a second training backend.

## Difficulty assessment

### Overall difficulty: medium-to-high

This is not a one-file feature. It is a medium-to-high difficulty integration because it affects:

- solver runtime design
- memory model
- checkpoint format
- API semantics
- deployment sizing
- training observability and persistence

### Difficulty by area

#### 1. Core model implementation: high

Implementing DeepCFR from scratch or integrating with OpenSpiel requires a reliable approach for:

- state feature encoding
- value approximation
- policy head output
- regret accumulation or strategy updates
- optimizer step scheduling

This is the highest-complexity part.

#### 2. Inference and query API: medium

The current API is already built around selected-node policy inspection. DeepCFR can satisfy this if we expose policy queries as network-inference results, but the logic must map from game state to action probabilities through the trained model. That is manageable if we keep the API layer thin and delegate to the model.

#### 3. Checkpointing and persistence: medium-high

DeepCFR checkpointing is substantially heavier than the current compact summary. We need:

- model weights
- optimizer state
- training iteration metadata
- schedule/config metadata
- optional evaluation summaries

This needs a dedicated artifact format, not just the current compact report flow.

#### 4. Deployment and infra: medium

The current production pattern is already tuned for compact memory and a long-running process. DeepCFR adds:

- more memory for network activations
- more GPU/CPU pressure depending on training mode
- larger checkpoint files
- more robust health reporting

This is manageable, but it should be treated as a heavier workload than the present HULH run.

## Practical implementation strategy

### Recommended direction: separate backend, shared API

The cleanest way to support DeepCFR in this codebase is to keep the existing selected-node MCCFR runtime and add a separate DeepCFR backend behind the same service layer.

Suggested architecture:

- `SolverService` becomes a backend dispatcher
- `solver_name` can choose among:
  - `external`
  - `outcome`
  - `deepcfr`
- each backend implements a common interface:
  - start()
  - stop()
  - health()
  - status()
  - request_probe()
  - get_preflop_range()
  - checkpoint/serialize model state

This keeps the API surface stable while allowing the training engine to change underneath.

### Backend split

A good structure is:

- current HULH/MCCFR backend: compact, low-overhead, checkpoint-based
- DeepCFR backend: neural, heavier, weight-based, more explicit checkpoint management

Do not try to merge them into one runtime class unless absolutely necessary. A backend split is much safer and easier to reason about.

## Concrete path to a DeepCFR integration

### Phase 1: define the execution contract

Add a backend interface with clearly defined responsibilities:

- how model state is initialized
- how iteration updates happen
- how policy inference is computed for selected nodes
- what checkpoint metadata is saved
- how to preserve inspectability while training is paused

This is the foundation required for testable integration.

### Phase 2: add DeepCFR model scaffolding

Create a new module, for example:

- `deepcfr_model.py`
- or a `backends/deepcfr/` package

This module should own:

- game abstraction and state encoding
- policy network
- value network
- training loop primitives
- checkpoint serialization

This should be isolated from `app_solver.py` to prevent mixing with the compact solver logic.

### Phase 3: hook into the service layer

In `api/service.py`, add a model-selection switch:

- if `solver_name == "deepcfr"`, create a DeepCFR runtime instead of the current solver
- use the same live API contracts
- keep the selected-node summarization logic working as a read-only projection of the DeepCFR policy output

At this point, the API can remain unchanged even as the underlying training backend changes.

### Phase 4: checkpoint format

The current checkpoint format is a summary-based artifact. DeepCFR will require a different path:

- model weights file
- optimizer state file
- training metadata JSON
- range snapshot summary JSON
- optional evaluation metrics

This should be a distinct artifact lineage from the current HULH checkpoint files, otherwise the project will conflate compact policy summaries with neural model weights.

### Phase 5: query-time inference

The API endpoints for range and spot inspection should not depend on a specific backend implementation. They should instead call a common inference method that returns action probabilities for a given state. For DeepCFR this means:

- encode the state to the network input
- run policy head inference
- normalize to action probabilities
- return the same response contract as current selected-node reports

This keeps the public API stable and allows different training backends behind it.

### Phase 6: operational readiness

Once the model is integrated:

- add checkpoint cadence rules
- add memory and disk usage metrics for bigger model checkpoints
- add a graceful pause/restart path for long training runs
- ensure the API remains inspectable while training is paused

This last step is especially important because the current stack has already invested in runtime safety and persistent queryability.

## Key technical risks

### 1. Feature mismatch between current APIs and DeepCFR

The current API is designed around selected-node summaries and compact range exports. DeepCFR is richer but not necessarily aligned to those exact reporting assumptions. This is manageable but requires explicit projection logic from model output to public response shapes.

### 2. Memory pressure and checkpoint size

DeepCFR will likely require a different memory model than the current compact flat arrays. The runtime should assume it will exceed the current resource envelope unless GPU or larger host capacity is explicitly available.

### 3. Training stability and observability

DeepCFR training can be less predictable than the current MCCFR approach. We need richer logs and checkpoints to explain convergence, not just action-frequency passing thresholds.

## Recommendation

The correct path is not to force DeepCFR into the current HULH solver implementation. Instead:

- keep the current solver as the default compact production backend
- add a separate DeepCFR backend with its own runtime and checkpoint policies
- standardize the public API contracts around range queries and selected-node policy inspection
- keep runtime safety/pause semantics consistent across backends

This gives the project the best chance to support DeepCFR without destabilizing the live API or the current production solver.

## What to do next

The next concrete steps are:

1. define a `SolverBackend` protocol and create the backend split in `api/service.py`
2. add a new `deepcfr_model.py` module with state encoding and network skeleton
3. implement a minimal DeepCFR policy inference path for a single selected-node query
4. add a checkpoint artifact format dedicated to model weights and optimizer state
5. add a config profile for DeepCFR runs with heavier memory assumptions

This is the safest, most maintainable path.

## Bottom line

DeepCFR support is feasible, but it should be treated as a new backend family, not an in-place enhancement of the current compact MCCFR solver. The implementation difficulty is medium-to-high, with the biggest challenges being model lifecycle, checkpointing, and deployment sizing.

The project is already structurally ready for this move because it has:

- a clean API layer
- a runtime state model
- checkpoint-based observability
- graceful stop semantics
- a strong emphasis on keeping the service inspectable while live

Those are exactly the foundations needed to support a heavier DeepCFR backend cleanly.
