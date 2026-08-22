# API cheat sheet

This page summarizes the live solver API as implemented in the app and matches the request/response contracts in the code.

## Base URL

Use the deployed app URL, typically:

```text
http://localhost:8080
```

or the public VM endpoint when deployed remotely.

---

## 1) Health and status

### GET /health

Purpose:
- quick liveness check for the long-running solver process

Example:

```bash
curl http://localhost:8080/health
```

Typical response:

```json
{
  "status": "running",
  "iteration": 4000,
  "stable": true,
  "last_probe_at": 4000,
  "ready_for_queries": true,
  "message": "solver is running; read-only probe APIs are enabled"
}
```

### GET /status

Purpose:
- solver readiness and convergence snapshot

Example:

```bash
curl http://localhost:8080/status
```

Typical response:

```json
{
  "solver": "outcome",
  "iteration": 4000,
  "stable": true,
  "stability": {
    "passed": true,
    "max_abs_delta": 0.002,
    "avg_abs_delta": 0.001,
    "threshold": 0.01,
    "matched_nodes": 5
  },
  "ready_for_queries": true,
  "last_probe_at": 4000,
  "min_iteration": 1000,
  "probe_budget_remaining": 1326
}
```

---

## 2) Selected-node probes

### POST /probe

Purpose:
- materialize a selected-node policy snapshot on demand

Request body:

```json
{
  "node": "first_to_act",
  "history": [],
  "samples": 1326,
  "min_iteration": 0,
  "include_stability": true,
  "include_hands": true,
  "action_filter": ["fold", "check_call", "bet_raise"]
}
```

Notes:
- `node` is the selected-node name, not a raw state ID
- `history` is the exact action history for that node
- `samples` is the deal-sample count for that request
- `min_iteration` prevents a query before the solver has reached the requested training depth
- action keys are normalized to:
  - `fold`
  - `check_call`
  - `bet_raise`

Example response:

```json
{
  "iteration": 4000,
  "node": "first_to_act",
  "display_name": "first_to_act",
  "history": [],
  "sample_count": 1326,
  "action_frequencies": {
    "fold": 0.12,
    "check_call": 0.48,
    "bet_raise": 0.40
  },
  "hands": [],
  "ready": true
}
```

### POST /bulk-probe

Purpose:
- request several selected-node probes in one payload

Request body:

```json
{
  "requests": [
    {
      "node": "first_to_act",
      "history": [],
      "samples": 1326
    },
    {
      "node": "response_to_open",
      "history": ["bet"],
      "samples": 1326
    }
  ]
}
```

Example response:

```json
{
  "results": [
    {
      "node": "first_to_act",
      "ready": true,
      "action_frequencies": {
        "fold": 0.15,
        "check_call": 0.45,
        "bet_raise": 0.40
      }
    },
    {
      "node": "response_to_open",
      "ready": true,
      "action_frequencies": {
        "fold": 0.1,
        "check_call": 0.3,
        "bet_raise": 0.6
      }
    }
  ],
  "failed": []
}
```

---

## 3) Preflop lookups

### GET /preflop/{spot}/{hand}

Purpose:
- single-hand lookup for a fixed preflop spot
- canonical hand labels are compact values like `TT`, `AKs`, `AQo`

Examples:

```bash
curl "http://localhost:8080/preflop/open/TT"
curl "http://localhost:8080/preflop/response_to_open/AKs"
```

Aliases supported by the API layer:
- `open` => `response_to_open`
- `first` => `first_to_act`
- `3bet`, `threebet` => `response_to_open_3bet`
- `4bet`, `fourbet` => `response_to_open_4bet`

Example response:

```json
{
  "spot": "response_to_open",
  "hand": "TT",
  "iteration": 4000,
  "frequencies": {
    "fold": 0.1,
    "check_call": 0.2,
    "bet_raise": 0.7
  },
  "ready": true,
  "message": null
}
```

### GET /preflop/open

Purpose:
- convenience alias for the common open-spot query

Usage:

```bash
curl "http://localhost:8080/preflop/open?hand=TT"
```

This is effectively the same as:

```bash
curl "http://localhost:8080/preflop/response_to_open/TT"
```

---

## 4) Postflop exact lookup

### POST /postflop/exact

Purpose:
- exact infoset lookup for a fixed board, action history, and hole cards
- this is the strict exact-match mode

Request body:

```json
{
  "board": ["Ah", "Kd", "2c"],
  "history": ["bet", "bet"],
  "hole_cards": ["As", "Qs"],
  "player": 0,
  "samples": 32,
  "min_iteration": 0
}
```

Notes:
- `board` is the exact current board
- `history` is the exact action sequence
- `hole_cards` are the acting player’s exact cards in that state
- `player` is the acting player index
- `samples` controls the number of states checked for that exact infoset

Example response:

```json
{
  "iteration": 4000,
  "board": ["Ah", "Kd", "2c"],
  "history": ["bet", "bet"],
  "hole_cards": ["As", "Qs"],
  "player": 0,
  "exact_infoset_key": "game=hulh|player=0|board=['2c','Ah','Kd']|hole=['As','Qs']|history=['bet','bet']",
  "action_probabilities": {
    "fold": 0.25,
    "check_call": 0.25,
    "bet_raise": 0.5
  },
  "sample_count": 32,
  "ready": true,
  "message": "live exact postflop infoset policy from current in-memory solver policy"
}
```

This is the exact-match mode; it is not a hand-class estimate.

---

## 5) Postflop range estimate

### POST /postflop/range

Purpose:
- estimate average action frequencies across a chosen hand subset at a postflop infoset

Request body:

```json
{
  "board": ["Ah", "Kd", "2c"],
  "history": ["bet", "bet"],
  "hands": ["AsQs", "AcKc", "QdJd"],
  "player": 0,
  "samples": 32,
  "min_iteration": 0
}
```

Notes:
- `hands` is a selected hand subset to aggregate over
- the service samples matching postflop states for each hand and averages their action distribution
- this is intentionally not the same as exact infoset lookup

Example response:

```json
{
  "iteration": 4000,
  "board": ["Ah", "Kd", "2c"],
  "history": ["bet", "bet"],
  "hands": ["AsQs", "AcKc", "QdJd"],
  "player": 0,
  "hand_count": 3,
  "action_frequencies": {
    "fold": 0.2,
    "check_call": 0.3,
    "bet_raise": 0.5
  },
  "sample_count": 3,
  "ready": true,
  "message": "live postflop range estimate over the chosen hand subset"
}
```

---

## 6) Clean request patterns

These are the request bodies worth using in normal client code:

### Selected node probe

```json
{
  "node": "first_to_act",
  "history": [],
  "samples": 1326
}
```

### Exact postflop infoset lookup

```json
{
  "board": ["Ah", "Kd", "2c"],
  "history": ["bet", "bet"],
  "hole_cards": ["As", "Qs"],
  "player": 0,
  "samples": 32
}
```

### Range-estimate postflop query

```json
{
  "board": ["Ah", "Kd", "2c"],
  "history": ["bet", "bet"],
  "hands": ["AsQs", "AcKc"],
  "player": 0,
  "samples": 32
}
```

---

## 7) Notes on semantics

The project intentionally separates:

- raw PokerKit/OpenSpiel state semantics
- project-facing action names and compact hand labels

The API is built around normalized keys like:

- `fold`, `check_call`, `bet_raise`
- `TT`, `AKs`, `AQo`

Those are convenience labels for client usage, while strict exact infoset identity remains tied to the actual board, hole cards, history, and acting player.

This is the policy described in [docs/postflop_query_policy.md](docs/postflop_query_policy.md).
