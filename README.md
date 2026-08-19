# pokerspiel

A lightweight HULH poker solver and API wrapper built around a long-lived OpenSpiel/PokerKit runtime.

The current architecture is intentionally simple:

- one app process owns the live solver thread
- the solver continues iterating in memory
- API endpoints expose health, selected-node status, probing, and preflop lookups
- runner profiles decide how a run is launched for local or GCE execution

## What this repo is for

This repo is meant to support a practical solver workflow rather than only a one-off CLI benchmark.

The main goals are:

- run a long-lived solver process
- inspect the active policy in-flight
- export selected-node range snapshots and stability checkpoints
- expose read-only API access to the solver state
- switch between local and remote deployment with a small config profile

## Architecture

### Runtime model

The runtime is organized around a single service object that owns the live solver and the current in-memory policy state. The API layer reads from that shared state rather than creating a second solver instance.

This avoids the common failure mode where:

- app startup creates one solver
- status endpoints read another object
- the API looks healthy but the solver is not actually the same runtime being tracked

The implementation is centered in:

- `api/app.py`: FastAPI app lifecycle
- `api/service.py`: singleton-backed live solver adapter
- `api/router.py`: read-only endpoints
- `app_solver.py`: core solver, selected-node logic, checkpointing, artifacts
- `runner.sh`: profile-driven launcher for local vs GCE runs

### Solver status semantics

The live `/status` endpoint is a convergence signal, not a full-game exploitability metric.

It is based on selected-node policy drift across checkpoints:

- compare action frequencies and hand policy deltas for the active node set
- aggregate recent movement into max delta and average delta
- expose those values as the live stability signal

This is a useful operational metric for “is the policy still moving materially?” but it is not the same as a full-game exploitability calculation.

## Current default model

The default solver setup is still a full preflop node family for HULH. That means the active selected-node set is broad enough to support queryable preflop range views and policy stability tracking across the main preflop action families.

This is a good default for now because it keeps the public API useful and the live solver state interpretable.

## Config and launch model

The repo uses JSON profiles as the launch layer, and the runner converts those profiles into the concrete solver invocation.

The config files live in:

- `cfg/solve_config_light.json`
- `cfg/solve_config_heavy.json`

The runner reads those profiles and resolves local vs remote execution.

A config file is divided into a few conceptual sections:

- run metadata: name, deploy target, project, zone
- solver settings: model, preset, iterations, stopping rules
- reporting settings: output path, range export behavior, artifact mode
- `solver_env`: values that should be materialized as environment variables when the run is launched

The important rule is:

- JSON config files are for launch presets and run setup
- the live app runtime owns its own in-memory state
- the API does not read config JSON files directly at runtime

## Local quick start

Run a local solver profile:

```bash
bash runner.sh ./cfg/solve_config_light.json local
```

This resolves the config and launches the selected solver command through Docker compose.

## GCE quick start

> [!IMPORTANT]
> The GCE flow requires a Google Cloud account and the relevant GCP services enabled for VM creation and firewall configuration. Local-only runs do not require GCP.

Run the same profile against the configured GCE path:

```bash
bash runner.sh ./cfg/solve_config_light.json gce
```

The deploy scripts under `deploy/` handle the VM and Docker startup flow, and the runner can be used as the main orchestration point.

## API endpoints

The live service exposes read-only endpoints such as:

- `/health`
- `/status`
- `/probe`
- `/bulk-probe`
- `/preflop/{spot}/{hand}`

These are designed to answer operational questions like:

- is the solver still running?
- how far has it moved since the last checkpoint?
- what is the current policy for a given selected node or hand?
- what is the current preflop range summary for a selected spot?

## Running the solver directly

The core solver entry point is still the CLI form in `app_solver.py`.

Example slim run for a lightweight range report:

```bash
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

## Current scope and limits

This project is intentionally focused on a practical, observable solver workflow.

Current limits include:

- selected-node policy drift is used as the live stability signal
- full exploitability is not the live runtime metric
- preflop is the most mature benchmark surface today
- post-flop node selection and deeper benchmark framing are still active design work

That is an intentional boundary: the system is built to be useful and inspectable now, without pretending to be a full equilibrium solver framework overnight.

## Summary

This repo is a practical HULH solver runtime with:

- a long-lived solver process
- selected-node stability tracking
- preflop policy and range reporting
- local/GCE launch orchestration
- a read-only API layer for operational inspection

The goal is to keep the engine stable and observable while still being flexible enough to expand into more sophisticated benchmark and reference strategies later.
