#!/usr/bin/env python3
"""Live preflop range-grid viewer for the poker solver API.

This script serves a browser page that renders a 13x13 starting-hand grid where
cell area is vertically split by fold/call/raise frequencies, matching the
`range_grid.py` style.

Features:
- Polls the configured solver API endpoint.
- Dropdown of available preflop spots.
- Selecting a spot triggers a fresh probe request and re-renders the figure.
- Auto-refresh on a configurable interval (default 5 minutes).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

RANKS = list("AKQJT98765432")
RANK_IDX: Dict[str, int] = {rank: idx for idx, rank in enumerate(RANKS)}


def build_grid_labels(ranks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return the full set of 169 canonical preflop matrix labels for a 13x13 grid."""
    ranks = list(ranks or RANKS)
    labels: List[Dict[str, Any]] = []
    for i in range(len(ranks)):
        for j in range(len(ranks)):
            if i == j:
                label = f"{ranks[i]}{ranks[j]}"
            else:
                hi_idx = i if i < j else j
                lo_idx = j if i < j else i
                suffix = "s" if i < j else "o"
                label = f"{ranks[hi_idx]}{ranks[lo_idx]}{suffix}"
            labels.append({"i": i, "j": j, "label": label})
    return labels


DEFAULT_SPOTS = [
    "first_to_act",
    "response_to_open",
    "response_to_limp_raise",
    "response_to_open_3bet",
    "response_to_open_4bet",
    "response_to_open_5bet",
]

DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_SAMPLES = 1326
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30


def http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def hand_to_category(hand: str) -> Optional[str]:
    text = str(hand or "").strip()
    if not text:
        return None

    upper = text.upper()
    if len(upper) == 2 and upper[0] in RANK_IDX and upper[1] in RANK_IDX:
        if upper[0] == upper[1]:
            return upper
        hi, lo = sorted((upper[0], upper[1]), key=lambda r: RANK_IDX[r])
        return f"{hi}{lo}"

    if len(upper) == 3 and upper[0] in RANK_IDX and upper[1] in RANK_IDX and upper[2] in {"S", "O"}:
        if upper[0] == upper[1]:
            return upper[:2]
        hi, lo = sorted((upper[0], upper[1]), key=lambda r: RANK_IDX[r])
        return f"{hi}{lo}{upper[2]}"

    return None


def category_to_cell(category: str) -> Optional[Tuple[int, int]]:
    cat = hand_to_category(category)
    if cat is None:
        return None
    normalized = cat.upper()
    if len(normalized) == 2 and normalized[0] == normalized[1] and normalized[0] in RANK_IDX:
        idx = RANK_IDX[normalized[0]]
        return idx, idx
    if len(normalized) == 3 and normalized[0] in RANK_IDX and normalized[1] in RANK_IDX and normalized[2] in {"S", "O"}:
        if normalized[0] == normalized[1]:
            return RANK_IDX[normalized[0]], RANK_IDX[normalized[1]]
        i = RANK_IDX[normalized[0]]
        j = RANK_IDX[normalized[1]]
        if normalized[2] == "S":
            return i, j
        return j, i
    return None


def range_hands_to_matrices(range_payload: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    """Return action matrices keyed by the 13x13 grid.

    Unseen cells remain at zero mass so the plotting layer can render them as
    "not in range" rather than implicitly defaulting to a full fold bucket.
    The API may also attach explicit prior_fold_mass metadata for the selected spot.
    """
    fold = [[0.0 for _ in range(13)] for _ in range(13)]
    call = [[0.0 for _ in range(13)] for _ in range(13)]
    raise_ = [[0.0 for _ in range(13)] for _ in range(13)]

    totals: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for hand_record in (range_payload.get("hands") or []):
        hand_name = str(hand_record.get("hand") or "")
        category = hand_to_category(hand_name)
        if category is None:
            continue
        policy = hand_record.get("policy") or {}
        f_val = float(policy.get("fold", 0.0) or 0.0)
        c_val = float(policy.get("check_call", policy.get("call", 0.0)) or 0.0)
        r_val = float(policy.get("bet_raise", policy.get("raise", policy.get("bet", 0.0))) or 0.0)

        bucket = totals.setdefault(category, {"f": 0.0, "c": 0.0, "r": 0.0})
        bucket["f"] += f_val
        bucket["c"] += c_val
        bucket["r"] += r_val
        counts[category] = counts.get(category, 0) + 1

    for category, sums in totals.items():
        cell = category_to_cell(category)
        if cell is None:
            continue
        i, j = cell
        n = max(counts.get(category, 1), 1)
        f_val = sums["f"] / n
        c_val = sums["c"] / n
        r_val = sums["r"] / n
        total = f_val + c_val + r_val
        if total <= 0:
            f_val, c_val, r_val = 0.0, 0.0, 0.0
        else:
            f_val, c_val, r_val = f_val / total, c_val / total, r_val / total
        fold[i][j] = max(0.0, min(1.0, f_val))
        call[i][j] = max(0.0, min(1.0, c_val))
        raise_[i][j] = max(0.0, min(1.0, r_val))

    return {"F": fold, "C": call, "R": raise_}


def fetch_spot_payload(
    api_base_url: str,
    spot: str,
    samples: int,
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    range_url = api_base_url.rstrip("/") + f"/preflop/{spot}/range"

    try:
        range_payload = http_json(range_url, timeout=max(1, int(request_timeout_seconds)))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": str(exc)}
        return {
            "ok": False,
            "error": detail,
            "status": {"iteration": None, "ready_for_queries": False},
            "spot": spot,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"preflop range fetch timed out or failed: {exc}",
            "status": {"iteration": None, "ready_for_queries": False},
            "spot": spot,
        }

    if not range_payload.get("ready", False):
        return {
            "ok": False,
            "error": range_payload.get("message") or f"preflop range for {spot} is not ready",
            "status": {
                "iteration": range_payload.get("iteration"),
                "ready_for_queries": False,
            },
            "spot": spot,
        }

    matrices = range_hands_to_matrices(range_payload)
    metadata = range_payload.get("metadata") or {}
    prior_fold_mass = float(metadata.get("prior_fold_mass", 0.0) or 0.0)
    return {
        "ok": True,
        "spot": spot,
        "status": {
            "iteration": range_payload.get("iteration"),
            "ready_for_queries": bool(range_payload.get("ready")),
        },
        "range": range_payload,
        "matrices": matrices,
        "metadata": {
            "prior_fold_mass": prior_fold_mass,
            "branch_valid": bool(metadata.get("branch_valid", True)),
            "zero_means_not_in_range": bool(metadata.get("zero_means_not_in_range", True)),
            "branch_model": metadata.get("branch_model", "conditional_after_prior_folds"),
        },
        "ranks": RANKS,
        "grid_labels": build_grid_labels(),
        "fetched_at": int(time.time()),
    }


def render_html(
    spots: List[str],
    default_spot: str,
    interval_seconds: int,
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> str:
    spots_json = json.dumps(spots)
    default_spot_json = json.dumps(default_spot)
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Preflop Range Grid</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      body {{
        margin: 0;
        background: #ffffff;
        color: #222;
        font-family: Menlo, Monaco, Consolas, monospace;
      }}
      .topbar {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-bottom: 1px solid #d9dde3;
        background: #f7f9fc;
      }}
      .label {{ font-size: 12px; color: #4a5563; }}
      select {{
        font-family: inherit;
        font-size: 13px;
        padding: 6px 8px;
        border: 1px solid #c8d0da;
        border-radius: 6px;
        background: white;
      }}
      .status {{ font-size: 12px; color: #5c6c80; }}
      #plot {{ width: 900px; height: 900px; margin: 12px auto; }}
      #node-table-wrap {{ width: 900px; margin: 0 auto 24px auto; }}
      #node-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
      #node-table th, #node-table td {{ border: 1px solid #d9dde3; padding: 6px 8px; text-align: left; }}
      #node-table th {{ background: #f3f6fa; }}
      #error {{ width: 900px; margin: 0 auto 16px auto; font-size: 12px; color: #9f1239; }}
      .collapsible-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
        user-select: none;
      }}
      .collapsible-body.collapsed {{
        display: none;
      }}
    </style>
  </head>
  <body>
    <div class="topbar">
      <span class="label">Spot</span>
      <select id="spot-select"></select>
      <span class="label" style="margin-left: 8px;">Checkpoint</span>
      <select id="checkpoint-select">
        <option value="latest">latest</option>
      </select>
      <span id="status" class="status"></span>
    </div>

    <div id="summary-panel" style="width: 900px; margin: 16px auto 0 auto;">
      <div class="summary-grid" style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px;">
        <div class="card" style="border: 1px solid #d9dde3; border-radius: 8px; background: #f7f9fc; padding: 12px;">
          <h3 style="margin: 0 0 8px 0; font-size: 13px;">Current stability</h3>
          <div id="stability-summary" class="metric-block" style="font-size: 12px; line-height: 1.6; color: #2d3748;"></div>
        </div>
        <div class="card" style="border: 1px solid #d9dde3; border-radius: 8px; background: #f7f9fc; padding: 12px;">
          <h3 style="margin: 0 0 8px 0; font-size: 13px;">Memory utilization</h3>
          <div id="memory-summary" class="metric-block" style="font-size: 12px; line-height: 1.6; color: #2d3748;"></div>
        </div>
        <div class="card" style="border: 1px solid #d9dde3; border-radius: 8px; background: #f7f9fc; padding: 12px;">
          <h3 style="margin: 0 0 8px 0; font-size: 13px;">Root starting hands</h3>
          <div id="root-summary" class="metric-block" style="font-size: 12px; line-height: 1.6; color: #2d3748;"></div>
        </div>
      </div>

      <div class="card" style="border: 1px solid #d9dde3; border-radius: 8px; background: white; padding: 12px; margin-bottom: 16px;">
        <h3 style="margin: 0 0 8px 0; font-size: 14px;">Historical stability</h3>
        <div id="stability-history" style="width: 100%; height: 220px;"></div>
      </div>
    </div>

    <div id="plot"></div>

    <div id="node-summary-panel" style="width: 900px; margin: 0 auto 24px auto;">
      <div class="card" style="border: 1px solid #d9dde3; border-radius: 8px; background: white; padding: 12px;">
        <div id="node-summary-toggle" class="collapsible-header" aria-expanded="false">
          <h3 style="margin: 0; font-size: 14px;">Node summary</h3>
          <span id="node-summary-toggle-label">Show</span>
        </div>
        <div id="node-summary-body" class="collapsible-body collapsed" style="margin-top: 12px;">
          <table id="node-table" style="border-collapse: collapse; width: 100%; font-size: 12px;">
            <thead><tr><th style="text-align: left; border: 1px solid #d9dde3; padding: 6px 8px;">node</th><th style="text-align: left; border: 1px solid #d9dde3; padding: 6px 8px;">count</th><th style="text-align: left; border: 1px solid #d9dde3; padding: 6px 8px;">fold</th><th style="text-align: left; border: 1px solid #d9dde3; padding: 6px 8px;">check/call</th><th style="text-align: left; border: 1px solid #d9dde3; padding: 6px 8px;">bet/raise</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <div id="error"></div>

    <script>
      const spots = {spots_json};
      const defaultSpot = {default_spot_json};
      const refreshMs = {max(1, int(interval_seconds))} * 1000;
      const requestTimeoutMs = {max(1, int(request_timeout_seconds))} * 1000;
      const STORAGE_KEY = 'pokerspiel_dashboard_status_history';
      const SPOT_KEY = 'pokerspiel_dashboard_selected_spot';

      function loadStatusHistory() {{
        try {{
          const raw = localStorage.getItem(STORAGE_KEY);
          if (!raw) return [];
          const parsed = JSON.parse(raw);
          return Array.isArray(parsed) ? parsed : [];
        }} catch (err) {{
          return [];
        }}
      }}

      function persistStatusHistory(history) {{
        try {{
          localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
        }} catch (err) {{
          // ignore storage failures in privacy-restricted browsers
        }}
      }}

      function loadSelectedSpot(defaultValue) {{
        try {{
          const saved = localStorage.getItem(SPOT_KEY);
          if (saved && spots.includes(saved)) return saved;
        }} catch (err) {{
          // ignore storage failures in privacy-restricted browsers
        }}
        return defaultValue;
      }}

      function persistSelectedSpot(spot) {{
        try {{
          localStorage.setItem(SPOT_KEY, spot);
        }} catch (err) {{
          // ignore storage failures in privacy-restricted browsers
        }}
      }}

      const statusHistory = loadStatusHistory();
      let rootStartingHands = 0;

      const spotSelect = document.getElementById('spot-select');
      const checkpointSelect = document.getElementById('checkpoint-select');
      const statusEl = document.getElementById('status');
      const errorEl = document.getElementById('error');
      const stabilitySummaryEl = document.getElementById('stability-summary');
      const memorySummaryEl = document.getElementById('memory-summary');
      const rootSummaryEl = document.getElementById('root-summary');
      const nodeSummaryToggleEl = document.getElementById('node-summary-toggle');
      const nodeSummaryBodyEl = document.getElementById('node-summary-body');
      const nodeSummaryToggleLabelEl = document.getElementById('node-summary-toggle-label');
      const checkpointHistoryCache = {{}};
      let currentSpotRangePayload = null;
      let selectedCheckpointValue = 'latest';

      for (const spot of spots) {{
        const opt = document.createElement('option');
        opt.value = spot;
        opt.textContent = spot;
        if (spot === loadSelectedSpot(defaultSpot)) opt.selected = true;
        spotSelect.appendChild(opt);
      }}

      function category(i, j, ranks) {{
        if (i === j) return ranks[i] + ranks[j];
        const hi = i < j ? i : j;
        const lo = i < j ? j : i;
        const suffix = i < j ? 's' : 'o';
        return ranks[hi] + ranks[lo] + suffix;
      }}

      function formatFreq(value) {{
        const number = Number(value ?? 0);
        if (!Number.isFinite(number)) return '0.000';
        return number.toFixed(3);
      }}

      function formatPct(value) {{
        const number = Number(value ?? 0);
        if (!Number.isFinite(number)) return 'n/a';
        return `${{(number * 100).toFixed(1)}}%`;
      }}

      function renderStabilityStatus(statusPayload) {{
        const stability = statusPayload && statusPayload.stability ? statusPayload.stability : null;
        const passed = stability && stability.passed !== undefined ? stability.passed : 'n/a';
        const avg = stability && stability.avg_abs_delta !== undefined ? Number(stability.avg_abs_delta) : null;
        const max = stability && stability.max_abs_delta !== undefined ? Number(stability.max_abs_delta) : null;
        const threshold = stability && stability.threshold !== undefined ? Number(stability.threshold) : null;
        const matched = stability && stability.matched_nodes !== undefined ? Number(stability.matched_nodes) : null;

        stabilitySummaryEl.innerHTML = (
          '<div><strong>passed:</strong> ' + String(passed) + '</div>' +
          '<div><strong>avg abs delta:</strong> ' + (avg === null ? 'n/a' : avg.toFixed(4)) + '</div>' +
          '<div><strong>max abs delta:</strong> ' + (max === null ? 'n/a' : max.toFixed(4)) + '</div>' +
          '<div><strong>threshold:</strong> ' + (threshold === null ? 'n/a' : threshold.toFixed(4)) + '</div>' +
          '<div><strong>matched nodes:</strong> ' + (matched === null ? 'n/a' : matched) + '</div>'
        );
      }}

      function renderMemoryStatus(statusPayload) {{
        const telemetry = statusPayload && statusPayload.telemetry ? statusPayload.telemetry : null;
        if (!telemetry) {{
          memorySummaryEl.innerHTML = '<div>no telemetry yet</div>';
          return;
        }}

        const total = Number(telemetry.total_memory_mb || 0);
        const used = Number(telemetry.used_memory_mb || 0);
        const availableRatio = Number(telemetry.memory_available_ratio || 0);
        const usedPct = total > 0 ? (used / total) * 100 : 0;

        memorySummaryEl.innerHTML = (
          '<div><strong>used:</strong> ' + used.toFixed(1) + ' / ' + total.toFixed(1) + ' MB</div>' +
          '<div><strong>utilization:</strong> ' + usedPct.toFixed(1) + '%</div>' +
          '<div><strong>available ratio:</strong> ' + formatPct(availableRatio) + '</div>'
        );
      }}

      function renderRootSummary(count) {{
        const value = Number(count || 0);
        rootSummaryEl.innerHTML = (
          '<div><strong>starting hands:</strong> ' + (Number.isFinite(value) && value > 0 ? value : 'n/a') + '</div>' +
          '<div><strong>spot:</strong> ' + defaultSpot + '</div>'
        );
      }}

      function clearStoredDashboardState() {{
        try {{
          localStorage.removeItem(STORAGE_KEY);
        }} catch (err) {{
          // ignore storage failures in privacy-restricted browsers
        }}
        try {{
          localStorage.removeItem(SPOT_KEY);
        }} catch (err) {{
          // ignore storage failures in privacy-restricted browsers
        }}
        statusHistory.length = 0;
      }}

      function clearStabilityHistoryPlot() {{
        try {{
          Plotly.purge('stability-history');
        }} catch (err) {{
          // ignore if Plotly is not ready yet
        }}
        document.getElementById('stability-history').innerHTML = '<div style="font-size:12px; color:#5c6c80;">no stability history yet</div>';
      }}

      function resetRuntimeCards() {{
        clearStoredDashboardState();
        renderStabilityStatus(null);
        renderMemoryStatus(null);
        renderRootSummary(0);
        renderNodeSummaryTable([]);
        clearStabilityHistoryPlot();
      }}

      function renderHistoricalStability(statusHistoryArray) {{
        if (!Array.isArray(statusHistoryArray) || statusHistoryArray.length === 0) {{
          clearStabilityHistoryPlot();
          return;
        }}

        const xs = statusHistoryArray.map(entry => Number(entry.iteration || 0));
        const avg = statusHistoryArray.map(entry => Number(entry.avg ?? 0));
        const max = statusHistoryArray.map(entry => Number(entry.max ?? 0));

        const traceAvg = {{
          type: 'scatter',
          mode: 'lines+markers',
          x: xs,
          y: avg,
          name: 'avg abs delta',
          line: {{ color: '#3b82f6', width: 2 }},
          marker: {{ size: 5 }},
        }};

        const traceMax = {{
          type: 'scatter',
          mode: 'lines+markers',
          x: xs,
          y: max,
          name: 'max abs delta',
          line: {{ color: '#ef4444', width: 2, dash: 'dot' }},
          marker: {{ size: 5 }},
        }};

        const layout = {{
          margin: {{ l: 40, r: 20, t: 10, b: 35 }},
          paper_bgcolor: 'white',
          plot_bgcolor: '#fafbfc',
          legend: {{ orientation: 'h', y: 1.15 }},
          xaxis: {{ title: 'iteration' }},
          yaxis: {{ title: 'delta' }},
        }};

        Plotly.newPlot('stability-history', [traceAvg, traceMax], layout, {{ responsive: true, displayModeBar: false }});
      }}

      function renderNodeSummaryTable(summaryRows) {{
        const tbody = document.querySelector('#node-table tbody');
        tbody.innerHTML = '';

        if (!Array.isArray(summaryRows) || summaryRows.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="5" style="padding: 8px; color: #4a5563;">no node summaries available yet</td></tr>';
          return;
        }}

        for (const row of summaryRows) {{
          const freqs = row && row.action_frequencies ? row.action_frequencies : (row && row.policy ? row.policy : {{}});
          const nodeName = row && row.node_name ? row.node_name : (row && row.hand ? row.hand : 'n/a');
          const sampleCount = Number(row && row.sample_count !== undefined ? row.sample_count : (row && row.count !== undefined ? row.count : 0));
          const tr = document.createElement('tr');
          tr.innerHTML = (
            '<td style="border: 1px solid #d9dde3; padding: 6px 8px;">' + nodeName + '</td>' +
            '<td style="border: 1px solid #d9dde3; padding: 6px 8px;">' + sampleCount + '</td>' +
            '<td style="border: 1px solid #d9dde3; padding: 6px 8px;">' + formatFreq(freqs.fold) + '</td>' +
            '<td style="border: 1px solid #d9dde3; padding: 6px 8px;">' + formatFreq(freqs['check_call']) + '</td>' +
            '<td style="border: 1px solid #d9dde3; padding: 6px 8px;">' + formatFreq(freqs['bet_raise']) + '</td>'
          );
          tbody.appendChild(tr);
        }}
      }}

      function updateCheckpointOptions(spot) {{
        const entries = checkpointHistoryCache[spot] || {{}};
        const iterations = Object.keys(entries).map(Number).sort((a, b) => b - a);
        const currentValue = selectedCheckpointValue === 'latest' || iterations.includes(Number(selectedCheckpointValue)) ? selectedCheckpointValue : 'latest';
        checkpointSelect.innerHTML = '<option value="latest">latest</option>';
        for (const iteration of iterations) {{
          const option = document.createElement('option');
          option.value = String(iteration);
          option.textContent = 'iter ' + String(iteration);
          if (String(iteration) === String(currentValue)) {{
            option.selected = true;
          }}
          checkpointSelect.appendChild(option);
        }}
        checkpointSelect.value = currentValue;
        selectedCheckpointValue = currentValue;
      }}

      function renderEmptyPlot(message) {{
        const layout = {{
          title: message || 'waiting for first checkpoint',
          width: 900,
          height: 900,
          margin: {{ l: 40, r: 20, t: 60, b: 40 }},
          paper_bgcolor: 'white',
          plot_bgcolor: '#fafbfc',
          xaxis: {{ visible: false }},
          yaxis: {{ visible: false }},
          showlegend: false,
        }};
        Plotly.newPlot('plot', [{{
          type: 'scatter',
          mode: 'markers',
          x: [],
          y: [],
          marker: {{ opacity: 0 }},
          hoverinfo: 'skip',
          showlegend: false,
        }}], layout, {{ responsive: true, displayModeBar: false }});
      }}

      function renderSelectedSnapshot(payload) {{
        if (!payload || !payload.ok) {{
          renderEmptyPlot('waiting for first checkpoint');
          renderRootSummary(0);
          renderNodeSummaryTable([]);
          statusEl.textContent = 'waiting for first checkpoint';
          return;
        }}
        buildFigure(payload);
        const summaryRows = Array.isArray(payload.range && payload.range.hands)
          ? payload.range.hands
          : [];
        if (Number(payload.range && payload.range.hand_count) > 0) {{
          rootStartingHands = Number(payload.range.hand_count);
        }} else if (Array.isArray(payload.range && payload.range.hands)) {{
          rootStartingHands = payload.range.hands.length;
        }}
        renderRootSummary(rootStartingHands);
        renderNodeSummaryTable(summaryRows);
        const iter = payload.status && payload.status.iteration !== undefined ? payload.status.iteration : 'n/a';
        const ready = payload.status && payload.status.ready_for_queries !== undefined ? payload.status.ready_for_queries : false;
        const label = selectedCheckpointValue === 'latest' ? 'latest' : 'checkpoint ' + String(selectedCheckpointValue);
        statusEl.textContent = 'iteration=' + String(iter) + ' ready=' + String(ready) + ' snapshot=' + String(label) + ' fetched=' + new Date(payload.fetched_at * 1000).toLocaleTimeString();
      }}

      function categoryMapFromGridLabels(labels) {{
        const map = new Map();
        for (const entry of labels || []) {{
          const key = String(entry.i) + ':' + String(entry.j);
          map.set(key, entry.label);
        }}
        return map;
      }}

      function buildFigure(payload) {{
        const ranks = payload.ranks;
        const F = payload.matrices && payload.matrices.F ? payload.matrices.F : Array.from({{ length: 13 }}, () => Array(13).fill(0));
        const C = payload.matrices && payload.matrices.C ? payload.matrices.C : Array.from({{ length: 13 }}, () => Array(13).fill(0));
        const R = payload.matrices && payload.matrices.R ? payload.matrices.R : Array.from({{ length: 13 }}, () => Array(13).fill(0));
        const metadata = payload.metadata || {{}};
        const zeroMeansNotInRange = metadata.zero_means_not_in_range !== false;
        const shapeLabels = Array.isArray(payload.grid_labels) ? payload.grid_labels : [];
        const labelMap = categoryMapFromGridLabels(shapeLabels);
        const shapes = [];
        const annotations = [];

        const xPad = 0.10;
        const yPad = 0.10;
        const innerPad = 0.03;

        for (let i = 0; i < 13; i++) {{
          for (let j = 0; j < 13; j++) {{
            let f = Number(F[i][j] || 0);
            let c = Number(C[i][j] || 0);
            let r = Number(R[i][j] || 0);
            const total = f + c + r;
            const x0 = j + xPad;
            const x1 = j + 1.0 - xPad;
            const y0 = i + yPad;
            const y1 = i + 1.0 - yPad;

            const isOutOfRange = total <= 0 && zeroMeansNotInRange;
            shapes.push({{
              type: 'rect',
              xref: 'x', yref: 'y',
              x0: x0, x1: x1, y0: y0, y1: y1,
              fillcolor: isOutOfRange ? '#ededed' : 'white',
              line: {{ color: 'rgba(0,0,0,0.22)', width: 0.6 }},
              layer: 'below',
            }});

            if (isOutOfRange) {{
              annotations.push({{
                x: j + 0.5,
                y: i + 0.5,
                xref: 'x', yref: 'y',
                text: labelMap.get(String(i) + ':' + String(j)) || category(i, j, ranks),
                showarrow: false,
                font: {{ size: 9, color: '#7a7a7a', family: 'monospace' }},
              }});
              continue;
            }}

            const norm = f + c + r;
            if (norm <= 0) {{
              continue;
            }}
            f /= norm;
            c /= norm;
            r /= norm;

            const foldEnd = y0 + (y1 - y0) * f;
            const callEnd = y0 + (y1 - y0) * (f + c);

            if (f > 0) {{
              shapes.push({{
                type: 'rect',
                xref: 'x', yref: 'y',
                x0: x0 + innerPad, x1: x1 - innerPad,
                y0: y0 + innerPad, y1: Math.min(y1 - innerPad, foldEnd),
                fillcolor: '#dfeefe',
                line: {{ color: 'rgba(0,0,0,0.12)', width: 0.3 }},
                layer: 'below',
              }});
            }}

            if (c > 0) {{
              shapes.push({{
                type: 'rect',
                xref: 'x', yref: 'y',
                x0: x0 + innerPad, x1: x1 - innerPad,
                y0: Math.max(y0 + innerPad, foldEnd), y1: Math.min(y1 - innerPad, callEnd),
                fillcolor: '#b9e4b9',
                line: {{ color: 'rgba(0,0,0,0.12)', width: 0.3 }},
                layer: 'below',
              }});
            }}

            if (r > 0) {{
              shapes.push({{
                type: 'rect',
                xref: 'x', yref: 'y',
                x0: x0 + innerPad, x1: x1 - innerPad,
                y0: Math.max(y0 + innerPad, callEnd), y1: y1 - innerPad,
                fillcolor: '#ff6b6b',
                line: {{ color: 'rgba(0,0,0,0.12)', width: 0.3 }},
                layer: 'below',
              }});
            }}

            annotations.push({{
              x: j + 0.5,
              y: i + 0.5,
              xref: 'x', yref: 'y',
              text: labelMap.get(String(i) + ':' + String(j)) || category(i, j, ranks),
              showarrow: false,
              font: {{ size: 9, color: '#333333', family: 'monospace' }},
            }});
          }}
        }}

        const trace = {{
          type: 'scatter',
          x: [0],
          y: [0],
          mode: 'markers',
          marker: {{ opacity: 0 }},
          hoverinfo: 'skip',
          showlegend: false,
        }};

        const layout = {{
          title: payload.spot + ' action split',
          width: 900,
          height: 900,
          margin: {{ l: 40, r: 20, t: 60, b: 40 }},
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          showlegend: false,
          shapes: shapes,
          annotations: annotations,
          xaxis: {{
            tickmode: 'array',
            tickvals: [...Array(13).keys()],
            ticktext: ranks,
            side: 'top',
            range: [-0.5, 13.5],
            showgrid: false,
            zeroline: false,
            tickfont: {{ size: 11 }},
          }},
          yaxis: {{
            tickmode: 'array',
            tickvals: [...Array(13).keys()],
            ticktext: ranks,
            range: [13.5, -0.5],
            showgrid: false,
            zeroline: false,
            tickfont: {{ size: 11 }},
          }},
        }};

        Plotly.newPlot('plot', [trace], layout, {{ responsive: true, displayModeBar: false }});
      }}

      async function refreshStatus() {{
        try {{
          const resp = await fetch('/status');
          const status = await resp.json();
          const hasTelemetry = status && status.telemetry && typeof status.telemetry === 'object';
          const hasIteration = status && status.iteration !== null && status.iteration !== undefined && Number(status.iteration) >= 0;

          if (!hasTelemetry || !hasIteration) {{
            renderStabilityStatus(null);
            renderMemoryStatus(null);
            renderHistoricalStability([]);
            return;
          }}

          if (!status || !status.stability) {{
            renderStabilityStatus(null);
            renderMemoryStatus(null);
            renderHistoricalStability([]);
            return;
          }}

          renderStabilityStatus(status);
          renderMemoryStatus(status);

          if (status && status.stability) {{
            const avg = Number(status.stability.avg_abs_delta ?? 0);
            const max = Number(status.stability.max_abs_delta ?? 0);
            const iteration = Number(status.iteration || 0);
            const last = statusHistory[statusHistory.length - 1];
            if (!last || last.iteration !== iteration || last.avg !== avg || last.max !== max) {{
              statusHistory.push({{ iteration, avg, max }});
            }}
            if (statusHistory.length > 25) statusHistory.shift();
            persistStatusHistory(statusHistory);
            renderHistoricalStability(statusHistory);
          }}

          if (Array.isArray(status.selected_node_summary) && status.selected_node_summary.length) {{
            renderNodeSummaryTable(status.selected_node_summary);
          }}
        }} catch (err) {{
          console.warn('status refresh failed', err);
          renderStabilityStatus(null);
          renderMemoryStatus(null);
          renderHistoricalStability([]);
        }}
      }}

      async function refreshSpot() {{
        const spot = spotSelect.value;
        statusEl.textContent = 'loading ' + String(spot) + '...';
        errorEl.textContent = '';
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), requestTimeoutMs);
        try {{
          const resp = await fetch('/grid-data?spot=' + encodeURIComponent(spot), {{ signal: controller.signal }});
          const payload = await resp.json();
          if (!payload.ok) {{
            throw new Error(typeof payload.error === 'string' ? payload.error : JSON.stringify(payload.error));
          }}
          currentSpotRangePayload = payload;
          const iter = payload.status && payload.status.iteration !== undefined ? Number(payload.status.iteration) : null;
          if (Number.isFinite(iter) && iter > 0) {{
            const perSpot = checkpointHistoryCache[spot] || {{}};
            perSpot[iter] = payload;
            checkpointHistoryCache[spot] = perSpot;
            updateCheckpointOptions(spot);
          }}
          if (selectedCheckpointValue === 'latest' || !selectedCheckpointValue) {{
            renderSelectedSnapshot(payload);
          }} else if (checkpointHistoryCache[spot] && checkpointHistoryCache[spot][Number(selectedCheckpointValue)]) {{
            renderSelectedSnapshot(checkpointHistoryCache[spot][Number(selectedCheckpointValue)]);
          }} else {{
            renderSelectedSnapshot(payload);
            selectedCheckpointValue = 'latest';
            updateCheckpointOptions(spot);
          }}
        }} catch (err) {{
          const timeoutLabel = Number(requestTimeoutMs / 1000).toFixed(0);
          const detail = err && err.name === 'AbortError' ? 'request timed out after ' + String(timeoutLabel) + 's' : (err.message || String(err));
          errorEl.textContent = 'Error: ' + String(detail);
          statusEl.textContent = 'request failed';
          renderEmptyPlot('waiting for first checkpoint');
          renderNodeSummaryTable([]);
        }} finally {{
          clearTimeout(timer);
        }}
      }}

      spotSelect.addEventListener('change', () => {{
        persistSelectedSpot(spotSelect.value);
        selectedCheckpointValue = 'latest';
        updateCheckpointOptions(spotSelect.value);
        refreshSpot();
      }});

      checkpointSelect.addEventListener('change', () => {{
        selectedCheckpointValue = checkpointSelect.value;
        const spot = spotSelect.value;
        const snapshot = selectedCheckpointValue === 'latest'
          ? currentSpotRangePayload
          : (checkpointHistoryCache[spot] || {{}})[Number(selectedCheckpointValue)];
        if (snapshot) {{
          renderSelectedSnapshot(snapshot);
        }}
      }});

      const restoredSelection = loadSelectedSpot(defaultSpot);
      if (restoredSelection && spots.includes(restoredSelection)) {{
        spotSelect.value = restoredSelection;
      }}

      function toggleNodeSummary(forceOpen) {{
        const willOpen = typeof forceOpen === 'boolean' ? forceOpen : nodeSummaryBodyEl.classList.contains('collapsed');
        nodeSummaryBodyEl.classList.toggle('collapsed', !willOpen);
        nodeSummaryToggleEl.setAttribute('aria-expanded', String(willOpen));
        nodeSummaryToggleLabelEl.textContent = willOpen ? 'Hide' : 'Show';
      }}

      nodeSummaryToggleEl.addEventListener('click', () => toggleNodeSummary());

      resetRuntimeCards();
      renderEmptyPlot('waiting for first checkpoint');
      renderRootSummary(rootStartingHands);
      renderHistoricalStability(statusHistory);
      updateCheckpointOptions(defaultSpot);
      toggleNodeSummary(false);

      refreshStatus();
      refreshSpot();
      setInterval(async () => {{
        await refreshStatus();
        await refreshSpot();
      }}, refreshMs);
    </script>
  </body>
</html>
"""


class DashboardHTTPServer:
    def __init__(
        self,
        api_base_url: str,
        spots: List[str],
        default_spot: str,
        samples: int,
        interval_seconds: int,
        host: str,
        port: int,
        request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.api_base_url = api_base_url
        self.spots = list(spots)
        self.default_spot = default_spot
        self.samples = int(samples)
        self.interval_seconds = int(interval_seconds)
        self.host = host
        self.port = port
        self.request_timeout_seconds = max(1, int(request_timeout_seconds))
        self._server: Optional[ThreadingHTTPServer] = None

    def serve(self) -> None:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/index.html"}:
                    html = render_html(
                        parent.spots,
                        parent.default_spot,
                        parent.interval_seconds,
                        parent.request_timeout_seconds,
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.end_headers()
                    self.wfile.write(html.encode("utf-8"))
                    return

                if parsed.path == "/status":
                    try:
                        payload = http_json(
                            parent.api_base_url.rstrip("/") + "/status",
                            timeout=max(1, int(parent.request_timeout_seconds)),
                        )
                    except Exception as exc:
                        payload = {"error": str(exc), "solver": None, "iteration": None, "stable": False}
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(200 if "error" not in payload else 502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if parsed.path == "/grid-data":
                    query = parse_qs(parsed.query)
                    requested = (query.get("spot") or [parent.default_spot])[0]
                    spot = requested if requested in parent.spots else parent.default_spot
                    payload = fetch_spot_payload(
                        parent.api_base_url,
                        spot,
                        parent.samples,
                        parent.request_timeout_seconds,
                    )
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live 13x13 preflop range-grid viewer.")
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("POKERSPIEL_API_URL", "http://localhost:8000"),
        help="Base URL for the running solver API (e.g. http://35.199.156.62:8080).",
    )
    parser.add_argument(
        "--spots",
        default=",".join(DEFAULT_SPOTS),
        help="Comma-separated preflop spot names for dropdown.",
    )
    parser.add_argument("--default-spot", default=DEFAULT_SPOTS[0], help="Initial dropdown selection.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Probe sample count.")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Auto-refresh interval in seconds (default 300).",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="API request timeout in seconds for dashboard range fetches (default 30).",
    )
    parser.add_argument("--serve-host", default="127.0.0.1", help="Host for local viewer server.")
    parser.add_argument("--serve-port", type=int, default=8765, help="Port for local viewer server.")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spots = [item.strip() for item in str(args.spots or "").split(",") if item.strip()]
    if not spots:
        spots = list(DEFAULT_SPOTS)

    default_spot = args.default_spot if args.default_spot in spots else spots[0]

    if args.open_browser:
        webbrowser.open(f"http://{args.serve_host}:{args.serve_port}/")

    server = DashboardHTTPServer(
        api_base_url=args.api_base_url,
        spots=spots,
        default_spot=default_spot,
        samples=max(1, int(args.samples)),
        interval_seconds=max(1, int(args.interval)),
        host=args.serve_host,
        port=args.serve_port,
        request_timeout_seconds=max(1, int(args.request_timeout)),
    )

    try:
        server.serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
