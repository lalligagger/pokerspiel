#!/usr/bin/env python3
"""One-off dashboarding script for older preflop APIs.

This script targets the legacy endpoint:
  GET /preflop/{spot}/{hand}

It loops through every canonical hand class for a target spot, fetches the
single-hand policy frequencies, and assembles a full 13x13 grid representation
that can be consumed by a simple Plotly dashboard or exported as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

RANKS = list("AKQJT98765432")
RANK_TO_IDX = {rank: idx for idx, rank in enumerate(RANKS)}


def normalize_api_base_url(raw: str) -> str:
    """Strip duplicated scheme prefixes such as http://http://... that can creep into copy-pasted URLs."""
    value = (raw or "").strip().rstrip("/")
    if not value:
        return value
    for prefix in ("http://http://", "https://https://", "http://https://", "https://http://"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def canonical_hand_classes() -> List[str]:
    """Return the full 169 hand classes in the same matrix convention as the live dashboard.

    Suited hands stay in the upper-right triangle, while offsuit hands mirror into the
    lower-left triangle using the same rank ordering. Pairs stay on the diagonal.
    """
    classes: List[str] = []
    seen: set[str] = set()

    for i in range(len(RANKS)):
        for j in range(len(RANKS)):
            if i == j:
                label = f"{RANKS[i]}{RANKS[j]}"
            elif i < j:
                hi = RANKS[i]
                lo = RANKS[j]
                label = f"{hi}{lo}s"
            else:
                hi = RANKS[j]
                lo = RANKS[i]
                label = f"{hi}{lo}o"
            if label not in seen:
                classes.append(label)
                seen.add(label)
    return classes


def api_hand_lookup_key(hand: str) -> str:
    """Flip low-to-high labels like 23s -> 32s and lowercase the suffix for the legacy backend."""
    text = str(hand or "").strip()
    if len(text) == 3 and text[0] in RANK_TO_IDX and text[1] in RANK_TO_IDX and text[2].upper() in {"S", "O"}:
        suffix = "s" if text[2].upper() == "S" else "o"
        if text[0] == text[1]:
            return f"{text[0]}{text[1]}{suffix}"
        if RANK_TO_IDX[text[0]] > RANK_TO_IDX[text[1]]:
            return f"{text[1]}{text[0]}{suffix}"
        return f"{text[0]}{text[1]}{suffix}"
    return text.upper()


def category_to_cell(category: str) -> Optional[Tuple[int, int]]:
    cat = str(category or "").strip().upper()
    if len(cat) == 2 and cat[0] == cat[1] and cat[0] in RANK_TO_IDX:
        idx = RANK_TO_IDX[cat[0]]
        return idx, idx
    if len(cat) == 3 and cat[0] in RANK_TO_IDX and cat[1] in RANK_TO_IDX and cat[2] in {"S", "O"}:
        if cat[0] == cat[1]:
            return RANK_TO_IDX[cat[0]], RANK_TO_IDX[cat[1]]
        if RANK_TO_IDX[cat[0]] > RANK_TO_IDX[cat[1]]:
            return None
        i = RANK_TO_IDX[cat[0]]
        j = RANK_TO_IDX[cat[1]]
        if cat[2] == "S":
            return i, j
        return j, i
    return None


def http_json(url: str, timeout: int = 90) -> Dict[str, Any]:
    request = Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def fetch_hand_data(api_base_url: str, spot: str, hand: str) -> Dict[str, Any]:
    normalized_base = normalize_api_base_url(api_base_url)
    lookup_hand = api_hand_lookup_key(hand)
    url = f"{normalized_base}/preflop/{spot}/{lookup_hand}"
    try:
        response = http_json(url, timeout=90)
        if isinstance(response, dict) and response.get("hand"):
            response["hand"] = str(response["hand"]).strip()
        return response
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": str(exc)}
        return {"spot": spot, "hand": hand, "ready": False, "error": detail, "frequencies": {}}
    except Exception as exc:
        return {"spot": spot, "hand": hand, "ready": False, "error": str(exc), "frequencies": {}}


def fetch_probe_data(api_base_url: str, node: str, samples: int = 1326) -> Dict[str, Any]:
    """Query the selected-node probe API and return the live hand-level policies for a node."""
    normalized_base = normalize_api_base_url(api_base_url)
    url = f"{normalized_base}/probe"
    payload = {
        "node": node,
        "history": [],
        "samples": max(int(samples), 1),
        "include_hands": True,
        "include_stability": False,
    }
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {"node": node, "hands": [], "ready": False, "message": "empty probe response"}
            return json.loads(body)
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": str(exc)}
        return {"node": node, "hands": [], "ready": False, "message": detail}
    except Exception as exc:
        return {"node": node, "hands": [], "ready": False, "message": str(exc)}


def probe_rows_to_legacy_rows(node_response: Dict[str, Any], spot: str) -> List[Dict[str, Any]]:
    """Convert /probe hand policies to the legacy row shape used by the HTML export."""
    rows: List[Dict[str, Any]] = []
    for hand_entry in node_response.get("hands") or []:
        hand = str(hand_entry.get("hand") or "").strip()
        if not hand:
            continue
        policy = hand_entry.get("policy") or {}
        rows.append({
            "spot": spot,
            "hand": hand,
            "iteration": node_response.get("iteration"),
            "ready": bool(node_response.get("ready", True)),
            "frequencies": {
                "fold": float(policy.get("fold", 0.0) or 0.0),
                "check_call": float(policy.get("check_call", policy.get("call", 0.0)) or 0.0),
                "bet_raise": float(policy.get("bet_raise", policy.get("raise", policy.get("bet", 0.0))) or 0.0),
            },
            "message": node_response.get("message"),
            "error": None,
        })
    return rows


def build_grid_data(results: Iterable[Dict[str, Any]]) -> Dict[str, List[List[float]]]:
    fold = [[0.0 for _ in range(13)] for _ in range(13)]
    call = [[0.0 for _ in range(13)] for _ in range(13)]
    raise_ = [[0.0 for _ in range(13)] for _ in range(13)]

    for item in results:
        hand = str(item.get("hand") or "").strip()
        if not hand:
            continue
        cell = category_to_cell(hand)
        if cell is None:
            continue
        freqs = item.get("frequencies") or {}
        if not freqs and not item.get("ready", False):
            continue

        f_val = float(freqs.get("fold", 0.0) or 0.0)
        c_val = float(freqs.get("check_call", freqs.get("call", 0.0)) or 0.0)
        r_val = float(freqs.get("bet_raise", freqs.get("raise", 0.0)) or 0.0)

        i, j = cell
        fold[i][j] = f_val
        call[i][j] = c_val
        raise_[i][j] = r_val

    return {"F": fold, "C": call, "R": raise_}


def render_html(title: str, data: Dict[str, Any]) -> str:
    """Render the simplest plotly-based dashboard page from assembled grid data."""
    spot = data.get("spot") or "unknown"
    matrices = data.get("matrices") or {"F": [[0.0] * 13 for _ in range(13)], "C": [[0.0] * 13 for _ in range(13)], "R": [[0.0] * 13 for _ in range(13)]}
    rows = data.get("rows") or []
    rows_json = json.dumps(rows)
    title_html = escape(title)
    spot_html = escape(spot)
    hand_count = len(rows)

    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>__TITLE__</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      body { font-family: Menlo, Monaco, Consolas, monospace; margin: 0; background: #fff; color: #111; }
      #plot { width: 920px; height: 920px; margin: 20px auto; }
      .meta { padding: 12px 20px 0 20px; font-size: 12px; color: #445; }
      table { width: 900px; margin: 0 auto 32px auto; border-collapse: collapse; font-size: 12px; }
      th, td { border: 1px solid #d0d7e2; padding: 6px 8px; text-align: left; }
      th { background: #f5f7fb; }
    </style>
  </head>
  <body>
    <div class="meta">Spot: __SPOT__ | Hand classes: __COUNT__</div>
    <div id="plot"></div>
    <table>
      <thead><tr><th>hand</th><th>fold</th><th>check/call</th><th>bet/raise</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
    <script>
      const rows = __ROWS__;
      const F = __F__;
      const C = __C__;
      const R = __R__;
      const ranks = __RANKS__;

      const tbody = document.getElementById('tbody');
      for (const row of rows) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + row.hand + '</td><td>' + Number(row.frequencies.fold || 0).toFixed(3) + '</td><td>' + Number(row.frequencies.check_call || 0).toFixed(3) + '</td><td>' + Number(row.frequencies.bet_raise || 0).toFixed(3) + '</td>';
        tbody.appendChild(tr);
      }

      const shapes = [];
      const annotations = [];
      const hoverPoints = [];
      const xPad = 0.08;
      const yPad = 0.08;
      const innerPad = 0.02;
      for (let i = 0; i < 13; i++) {
        for (let j = 0; j < 13; j++) {
          let f = Number(F[i][j] || 0);
          let c = Number(C[i][j] || 0);
          let r = Number(R[i][j] || 0);
          const total = f + c + r;
          if (total <= 0) {
            f = 1; c = 0; r = 0;
          }
          const hand = (i === j) ? (ranks[i] + ranks[j]) : (i < j ? (ranks[i] + ranks[j] + 's') : (ranks[j] + ranks[i] + 'o'));
          const x0 = j + xPad;
          const x1 = j + 1.0 - xPad;
          const y0 = i + yPad;
          const y1 = i + 1.0 - yPad;
          shapes.push({ type: 'rect', xref: 'x', yref: 'y', x0: x0, x1: x1, y0: y0, y1: y1, fillcolor: 'white', line: { color: 'rgba(0,0,0,0.2)', width: 0.4 }, layer: 'below' });
          const foldEnd = y0 + (y1 - y0) * (f / (f + c + r));
          const callEnd = y0 + (y1 - y0) * ((f + c) / (f + c + r));
          if (f > 0) {
            shapes.push({ type: 'rect', xref: 'x', yref: 'y', x0: x0 + innerPad, x1: x1 - innerPad, y0: y0 + innerPad, y1: foldEnd, fillcolor: '#dfeefe', line: { color: 'rgba(0,0,0,0.05)', width: 0.2 }, layer: 'below' });
          }
          if (c > 0) {
            shapes.push({ type: 'rect', xref: 'x', yref: 'y', x0: x0 + innerPad, x1: x1 - innerPad, y0: Math.max(y0 + innerPad, foldEnd), y1: Math.min(callEnd, y1 - innerPad), fillcolor: '#b9e4b9', line: { color: 'rgba(0,0,0,0.05)', width: 0.2 }, layer: 'below' });
          }
          if (r > 0) {
            shapes.push({ type: 'rect', xref: 'x', yref: 'y', x0: x0 + innerPad, x1: x1 - innerPad, y0: Math.max(y0 + innerPad, callEnd), y1: y1 - innerPad, fillcolor: '#ff6b6b', line: { color: 'rgba(0,0,0,0.05)', width: 0.2 }, layer: 'below' });
          }
          hoverPoints.push({ x: j + 0.5, y: i + 0.5, hand, f, c, r });
          annotations.push({ x: j + 0.5, y: i + 0.5, xref: 'x', yref: 'y', text: hand, showarrow: false, font: { size: 8, color: '#222' } });
        }
      }

      const hoverTrace = {
        type: 'scatter',
        mode: 'markers',
        x: hoverPoints.map((p) => p.x),
        y: hoverPoints.map((p) => p.y),
        customdata: hoverPoints.map((p) => [p.hand, p.f, p.c, p.r]),
        marker: { size: 12, opacity: 0, color: 'rgba(0,0,0,0)' },
        hovertemplate: '<b>%{customdata[0]}</b><br>fold: %{customdata[1]:.3f}<br>check/call: %{customdata[2]:.3f}<br>bet/raise: %{customdata[3]:.3f}<extra></extra>',
        showlegend: false,
      };
      const layout = {
        title: 'Legacy preflop action split: ' + (rows[0] ? rows[0].spot : 'n/a'),
        width: 920,
        height: 920,
        margin: { l: 40, r: 20, t: 60, b: 40 },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        hovermode: 'closest',
        hoverlabel: { font: { size: 14 }, bgcolor: 'rgba(17,17,17,0.92)' },
        shapes: shapes,
        annotations: annotations,
        xaxis: { tickmode: 'array', tickvals: Array.from({ length: 13 }, (_, i) => i), ticktext: ranks, side: 'top', range: [-0.5, 13.5], showgrid: false, zeroline: false },
        yaxis: { tickmode: 'array', tickvals: Array.from({ length: 13 }, (_, i) => i), ticktext: ranks, range: [13.5, -0.5], showgrid: false, zeroline: false }
      };
      Plotly.newPlot('plot', [hoverTrace], layout, { responsive: true, displayModeBar: false });
    </script>
  </body>
</html>
""".replace("__TITLE__", title_html).replace("__SPOT__", spot_html).replace("__COUNT__", str(hand_count)).replace("__ROWS__", rows_json).replace("__F__", json.dumps(matrices.get('F') or [])).replace("__C__", json.dumps(matrices.get('C') or [])).replace("__R__", json.dumps(matrices.get('R') or [])).replace("__RANKS__", json.dumps(RANKS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Legacy preflop dashboard builder for /preflop/{spot}/{hand}.')
    parser.add_argument('--api-base-url', default=os.getenv('POKERSPIEL_API_URL', 'http://localhost:8080'), help='Base URL for the legacy solver API.')
    parser.add_argument('--spot', default='response_to_open', help='Preflop spot name to query, e.g. first_to_act or response_to_open.')
    parser.add_argument('--output', default='legacy_preflop_grid.json', help='Where to write the assembled JSON data.')
    parser.add_argument('--html-output', default='legacy_preflop_grid.html', help='Optional HTML dashboard export path.')
    parser.add_argument('--serve-port', type=int, default=None, help='Optional local port to serve the HTML dashboard after assembly.')
    parser.add_argument('--open-browser', action='store_true', help='Open the local dashboard automatically after assembly.')
    parser.add_argument('--probe-node', default=None, help='Optional explicit selected-node probe name to use instead of the per-hand fetch path.')
    parser.add_argument('--probe-samples', type=int, default=1326, help='Number of sampled hands to request from the /probe endpoint when fallback is used.')
    return parser.parse_args()


def legacy_rows_for_spot(api_base_url: str, spot: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hand in canonical_hand_classes():
        data = fetch_hand_data(api_base_url, spot, hand)
        rows.append({
            'spot': spot,
            'hand': hand,
            'iteration': data.get('iteration'),
            'ready': bool(data.get('ready', False)),
            'frequencies': data.get('frequencies') or {},
            'message': data.get('message'),
            'error': data.get('error'),
        })
    return rows


def main() -> None:
    args = parse_args()
    api_base_url = normalize_api_base_url(args.api_base_url)

    spot = args.spot
    probe_node = args.probe_node or spot
    classes = canonical_hand_classes()
    rows: List[Dict[str, Any]] = []
    data_source = 'legacy_per_hand'
    row_by_hand: Dict[str, Dict[str, Any]] = {}

    if args.probe_node is not None:
        probe_response = fetch_probe_data(api_base_url, probe_node, samples=args.probe_samples)
        rows = probe_rows_to_legacy_rows(probe_response, spot)
        data_source = 'probe'
    else:
        rows = legacy_rows_for_spot(api_base_url, spot)
        if not rows or not any(row.get('ready') for row in rows):
            probe_response = fetch_probe_data(api_base_url, probe_node, samples=args.probe_samples)
            rows = probe_rows_to_legacy_rows(probe_response, spot)
            data_source = 'probe_fallback'

    for row in rows:
        row_by_hand[row['hand']] = row

    matrices = build_grid_data(rows)
    payload = {
        'spot': spot,
        'hand_count': len(rows),
        'rows': rows,
        'matrices': matrices,
        'hand_classes': classes,
        'ready_count': sum(1 for row in rows if row.get('ready')),
        'data_source': data_source,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')

    html_payload = {**payload, 'rows': [row for row in rows if row.get('ready')]}
    html = render_html(f'Legacy preflop dashboard: {spot}', html_payload)
    with open(args.html_output, 'w', encoding='utf-8') as f:
        f.write(html)

    print(json.dumps({
        'spot': spot,
        'hand_count': len(rows),
        'ready_count': payload['ready_count'],
        'data_source': payload.get('data_source'),
        'json_output': args.output,
        'html_output': args.html_output,
    }, indent=2))

    if args.serve_port:
        import http.server
        import socketserver
        import webbrowser
        import threading

        class LegacyHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))

            def log_message(self, format: str, *args: Any) -> None:
                return

        with socketserver.TCPServer(('127.0.0.1', args.serve_port), LegacyHandler) as httpd:
            if args.open_browser:
                webbrowser.open(f'http://127.0.0.1:{args.serve_port}/')
            print(f'Legacy dashboard serving at http://127.0.0.1:{args.serve_port}/')
            httpd.serve_forever()


if __name__ == '__main__':
    main()
