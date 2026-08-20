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
    text = str(hand or "").strip().upper()
    if not text:
        return None

    if len(text) == 2 and text[0] in RANK_IDX and text[1] in RANK_IDX:
        if text[0] == text[1]:
            return text
        hi, lo = sorted((text[0], text[1]), key=lambda r: RANK_IDX[r], reverse=True)
        return f"{hi}{lo}"

    if len(text) == 3 and text[0] in RANK_IDX and text[1] in RANK_IDX and text[2] in {"S", "O"}:
        if text[0] == text[1]:
            return text[:2]
        hi, lo = sorted((text[0], text[1]), key=lambda r: RANK_IDX[r], reverse=True)
        return f"{hi}{lo}{text[2]}"

    return None


def category_to_cell(category: str) -> Optional[Tuple[int, int]]:
    cat = hand_to_category(category)
    if cat is None:
        return None
    if len(cat) == 2 and cat[0] == cat[1] and cat[0] in RANK_IDX:
        idx = RANK_IDX[cat[0]]
        return idx, idx
    if len(cat) == 3 and cat[0] in RANK_IDX and cat[1] in RANK_IDX and cat[2] in {"S", "O"}:
        if cat[0] == cat[1]:
            return RANK_IDX[cat[0]], RANK_IDX[cat[1]]
        i = RANK_IDX[cat[0]]
        j = RANK_IDX[cat[1]]
        if cat[2] == "S":
            return i, j
        return j, i
    return None


def range_hands_to_matrices(range_payload: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    fold = [[1.0 for _ in range(13)] for _ in range(13)]
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
            f_val, c_val, r_val = 1.0, 0.0, 0.0
        else:
            f_val, c_val, r_val = f_val / total, c_val / total, r_val / total
        fold[i][j] = max(0.0, min(1.0, f_val))
        call[i][j] = max(0.0, min(1.0, c_val))
        raise_[i][j] = max(0.0, min(1.0, r_val))

    return {"F": fold, "C": call, "R": raise_}


def fetch_spot_payload(api_base_url: str, spot: str, samples: int) -> Dict[str, Any]:
    range_url = api_base_url.rstrip("/") + f"/preflop/{spot}/range"

    try:
        range_payload = http_json(range_url, timeout=12)
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
    return {
        "ok": True,
        "spot": spot,
        "status": {
            "iteration": range_payload.get("iteration"),
            "ready_for_queries": bool(range_payload.get("ready")),
        },
        "range": range_payload,
        "matrices": matrices,
        "ranks": RANKS,
        "grid_labels": build_grid_labels(),
        "fetched_at": int(time.time()),
    }


def render_html(spots: List[str], default_spot: str, interval_seconds: int) -> str:
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
    </style>
  </head>
  <body>
    <div class="topbar">
      <span class="label">Spot</span>
      <select id="spot-select"></select>
      <span id="status" class="status"></span>
    </div>
    <div id="plot"></div>
    <div id="node-table-wrap">
      <h3 style="margin: 0 0 8px 0; font-size: 14px;">Observed preflop nodes</h3>
      <table id="node-table">
        <thead><tr><th>node</th><th>count</th><th>fold</th><th>check/call</th><th>bet/raise</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div id="error"></div>

    <script>
      const spots = {spots_json};
      const defaultSpot = {default_spot_json};
      const refreshMs = {max(1, int(interval_seconds))} * 1000;

      const spotSelect = document.getElementById('spot-select');
      const statusEl = document.getElementById('status');
      const errorEl = document.getElementById('error');

      for (const spot of spots) {{
        const opt = document.createElement('option');
        opt.value = spot;
        opt.textContent = spot;
        if (spot === defaultSpot) opt.selected = true;
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

      function renderNodeTable(summaryRows) {{
        const tbody = document.querySelector('#node-table tbody');
        tbody.innerHTML = '';
        if (!Array.isArray(summaryRows) || summaryRows.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="5">range data is sourced from /preflop/&lt;spot&gt;/range only</td></tr>';
          return;
        }}

        for (const row of summaryRows) {{
          const freqs = row && row.policy ? row.policy : {{}};
          const tr = document.createElement('tr');
          tr.innerHTML = (
            '<td>' + (row.hand || 'hand') + '</td>' +
            '<td>' + Number(row.hand ? 1 : 0) + '</td>' +
            '<td>' + formatFreq(freqs.fold) + '</td>' +
            '<td>' + formatFreq(freqs['check_call']) + '</td>' +
            '<td>' + formatFreq(freqs['bet_raise']) + '</td>'
          );
          tbody.appendChild(tr);
        }}
      }}

      function buildFigure(payload) {{
        const ranks = payload.ranks;
        const F = payload.matrices.F;
        const C = payload.matrices.C;
        const R = payload.matrices.R;
        const shapeLabels = Array.isArray(payload.grid_labels) ? payload.grid_labels : [];
        const labelMap = new Map(shapeLabels.map(entry => [String(entry.i) + ':' + String(entry.j), entry.label]));
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
            if (total <= 0) {{
              f = 1.0;
              c = 0.0;
              r = 0.0;
            }}
            const norm = f + c + r;
            f /= norm;
            c /= norm;
            r /= norm;

            const x0 = j + xPad;
            const x1 = j + 1.0 - xPad;
            const y0 = i + yPad;
            const y1 = i + 1.0 - yPad;

            shapes.push({{
              type: 'rect',
              xref: 'x', yref: 'y',
              x0: x0, x1: x1, y0: y0, y1: y1,
              fillcolor: 'white',
              line: {{ color: 'rgba(0,0,0,0.22)', width: 0.6 }},
              layer: 'below',
            }});

            const foldEnd = y0 + (y1 - y0) * f;
            const callEnd = y0 + (y1 - y0) * (f + c);

            if (f > 0) {{
              const fy0 = y0 + innerPad;
              const fy1 = (c === 0 && r === 0) ? (y1 - innerPad) : foldEnd;
              shapes.push({{
                type: 'rect',
                xref: 'x', yref: 'y',
                x0: x0 + innerPad, x1: x1 - innerPad,
                y0: fy0, y1: fy1,
                fillcolor: '#dfeefe',
                line: {{ color: 'rgba(0,0,0,0.12)', width: 0.3 }},
                layer: 'below',
              }});
            }}

            if (c > 0) {{
              const cy0 = Math.max(y0 + innerPad, foldEnd);
              const cy1 = r === 0 ? (y1 - innerPad) : callEnd;
              shapes.push({{
                type: 'rect',
                xref: 'x', yref: 'y',
                x0: x0 + innerPad, x1: x1 - innerPad,
                y0: cy0, y1: cy1,
                fillcolor: '#b9e4b9',
                line: {{ color: 'rgba(0,0,0,0.12)', width: 0.3 }},
                layer: 'below',
              }});
            }}

            if (r > 0) {{
              const ry0 = Math.max(y0 + innerPad, callEnd);
              shapes.push({{
                type: 'rect',
                xref: 'x', yref: 'y',
                x0: x0 + innerPad, x1: x1 - innerPad,
                y0: ry0, y1: y1 - innerPad,
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
          title: `${{payload.spot}} action split`,
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

      async function refreshSpot() {{
        const spot = spotSelect.value;
        statusEl.textContent = `loading ${{spot}}...`;
        errorEl.textContent = '';
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 15000);
        try {{
          const resp = await fetch(`/grid-data?spot=${{encodeURIComponent(spot)}}`, {{ signal: controller.signal }});
          const payload = await resp.json();
          if (!payload.ok) {{
            throw new Error(typeof payload.error === 'string' ? payload.error : JSON.stringify(payload.error));
          }}
          buildFigure(payload);
          const summaryRows = Array.isArray(payload.range && payload.range.hands)
            ? payload.range.hands
            : [];
          renderNodeTable(summaryRows);
          const iter = payload.status && payload.status.iteration !== undefined ? payload.status.iteration : 'n/a';
          const ready = payload.status && payload.status.ready_for_queries !== undefined ? payload.status.ready_for_queries : false;
          statusEl.textContent = `iteration=${{iter}} ready=${{ready}} source=/preflop/${{encodeURIComponent(spot)}}/range fetched=${{new Date(payload.fetched_at * 1000).toLocaleTimeString()}}`;
        }} catch (err) {{
          const detail = err && err.name === 'AbortError' ? 'request timed out after 15s' : (err.message || String(err));
          errorEl.textContent = `Error: ${{detail}}`;
          statusEl.textContent = 'request failed';
          renderNodeTable([]);
        }} finally {{
          clearTimeout(timer);
        }}
      }}

      spotSelect.addEventListener('change', () => {{
        refreshSpot();
      }});

      refreshSpot();
      setInterval(refreshSpot, refreshMs);
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
    ) -> None:
        self.api_base_url = api_base_url
        self.spots = list(spots)
        self.default_spot = default_spot
        self.samples = int(samples)
        self.interval_seconds = int(interval_seconds)
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None

    def serve(self) -> None:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/index.html"}:
                    html = render_html(parent.spots, parent.default_spot, parent.interval_seconds)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html.encode("utf-8"))
                    return

                if parsed.path == "/grid-data":
                    query = parse_qs(parsed.query)
                    requested = (query.get("spot") or [parent.default_spot])[0]
                    spot = requested if requested in parent.spots else parent.default_spot
                    payload = fetch_spot_payload(parent.api_base_url, spot, parent.samples)
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
    )

    try:
        server.serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
