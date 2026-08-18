# Live Probe API contract

This package defines the schema and contract for a freerunning solver that keeps training in the background and serves selected-node range requests on demand.

## Design intent

- The solver remains the existing long-running training engine.
- The API package only exposes a service contract.
- No changes are made to the training loop or solver internals in this package.
- Local or cloud deployment only changes the network endpoint, not the protocol.

## Endpoints

- GET /health
- GET /status
- POST /probe
- POST /bulk-probe

## Scope

The API is intentionally narrow:

- read solver health
- read convergence readiness
- request a selected-node range snapshot
- batch a few selected-node snapshots together

It does not include:

- solver configuration mutation
- training control toggles
- full-tree export endpoints
- exploitability calculation APIs

## Schema files

- openapi.yaml: OpenAPI contract for the live-probe service
- contracts.py: Python dataclass definitions for the same schema

## Local example URL

http://localhost:8000/probe

## Remote example URL

https://example.remote-host/probe

## Notes

This is a contract-first, service-layer wrapper. The actual implementation can later target:

- a local Flask/FastAPI server
- a local Docker service
- a VM or Cloud Run/GKE endpoint

without changing the API shape.
