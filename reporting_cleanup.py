#!/usr/bin/env python3
"""Compact checkpoint summary exporter for large poker training runs.

This tool reads a directory of checkpoint JSON artifacts produced by
profile_wrapper_solver.py, extracts the selected-node policy summaries, and writes a
small CSV summary suitable for trend analysis or plotting without loading the full
multi-GB run payloads into the editor.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

DEFAULT_WANTED_NODES = [
    "first_to_act",
    "response_to_limp",
    "response_to_limp_raise",
    "response_to_open",
    "response_to_open_3bet",
]

DEFAULT_WANTED_HANDS = [
    "22",
    "66",
    "AA",
    "A2o",
    "A2s",
    "A5s",
    "AKo",
    "AKs",
    "KQo",
    "KTs",
    "QTs",
    "JTs",
    "T9s",
]

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
RANK_TO_VALUE = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
VALUE_TO_RANK = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}


def iter_checkpoint_files(input_dir: str | Path) -> List[Path]:
    base = Path(input_dir)
    files = []
    for path in sorted(base.glob("report_checkpoint_*.json")):
        if path.name.endswith("_stability.json"):
            continue
        files.append(path)
    return files


def get_node_name(node: Dict[str, Any]) -> str | None:
    if not isinstance(node, dict):
        return None
    return node.get("display_name") or node.get("history_label") or node.get("name")


def get_hand_policy(node: Dict[str, Any], hand: str) -> Dict[str, float]:
    if not isinstance(node, dict):
        return {}
    for item in node.get("hands", []):
        if isinstance(item, dict) and item.get("hand") == hand:
            policy = item.get("policy") or {}
            return {str(k): float(v) for k, v in policy.items()}
    return {}


def parse_hand_label_to_ranks(label: str) -> List[int]:
    if not isinstance(label, str):
        return []
    clean = label.strip()
    if not clean:
        return []

    if clean.endswith("s") or clean.endswith("o"):
        clean = clean[:-1]
    if len(clean) == 2 and clean[0].isdigit() and clean[1].isdigit():
        return [int(clean[0]), int(clean[1])]
    if len(clean) == 2 and clean[0].isdigit() and clean[1] in RANK_TO_VALUE:
        return [int(clean[0]), RANK_TO_VALUE[clean[1]]]
    if len(clean) == 2 and clean[0] in RANK_TO_VALUE and clean[1].isdigit():
        return [RANK_TO_VALUE[clean[0]], int(clean[1])]
    if len(clean) == 2 and clean[0] in RANK_TO_VALUE and clean[1] in RANK_TO_VALUE:
        return [RANK_TO_VALUE[clean[0]], RANK_TO_VALUE[clean[1]]]

    match = re.match(r"([2-9TJQKA])([2-9TJQKA])", clean)
    if match:
        return [RANK_TO_VALUE[match.group(1)], RANK_TO_VALUE[match.group(2)]]
    return []


def build_rank_matrix(node: Dict[str, Any]) -> Dict[tuple[int, int], Dict[str, Any]]:
    matrix: Dict[tuple[int, int], Dict[str, Any]] = {}
    if not isinstance(node, dict):
        return matrix

    for entry in node.get("hands", []):
        if not isinstance(entry, dict):
            continue
        hand = str(entry.get("hand", ""))
        ranks = parse_hand_label_to_ranks(hand)
        if len(ranks) != 2:
            continue

        low, high = sorted(ranks)
        low_idx = RANKS.index(VALUE_TO_RANK[low]) if low in VALUE_TO_RANK else 0
        high_idx = RANKS.index(VALUE_TO_RANK[high]) if high in VALUE_TO_RANK else 0

        is_pair = low == high
        is_suited = hand.endswith("s")
        is_offsuit = hand.endswith("o")

        if is_pair:
            row = col = high_idx
        elif is_suited:
            row = high_idx
            col = low_idx
        elif is_offsuit:
            row = low_idx
            col = high_idx
        else:
            row = high_idx
            col = low_idx

        key = (row, col)
        if key not in matrix:
            matrix[key] = {"sample_count": 0, "fold": 0.0, "check_call": 0.0, "bet_raise": 0.0}

        policy = entry.get("policy") or {}
        sample_count = int(entry.get("sample_count") or 0)
        if sample_count <= 0:
            sample_count = 1
        matrix[key]["sample_count"] += sample_count
        matrix[key]["fold"] += float(policy.get("fold", 0.0)) * sample_count
        matrix[key]["check_call"] += float(policy.get("check_call", 0.0)) * sample_count
        matrix[key]["bet_raise"] += float(policy.get("bet_raise", 0.0)) * sample_count

    for values in matrix.values():
        count = max(values["sample_count"], 1)
        values["fold"] = values["fold"] / count
        values["check_call"] = values["check_call"] / count
        values["bet_raise"] = values["bet_raise"] / count
    return matrix


def top_cells_for_node(node: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    matrix = build_rank_matrix(node)
    ranked = []
    for (row, col), values in matrix.items():
        count = values.get("sample_count", 0)
        dominant_action = max(
            float(values.get("fold", 0.0)),
            float(values.get("check_call", 0.0)),
            float(values.get("bet_raise", 0.0)),
        )
        ranked.append({
            "cell": (row, col),
            "row_label": RANKS[row],
            "col_label": RANKS[col],
            "sample_count": count,
            "fold": values.get("fold", 0.0),
            "check_call": values.get("check_call", 0.0),
            "bet_raise": values.get("bet_raise", 0.0),
            "dominant_action": dominant_action,
        })
    ranked.sort(key=lambda item: (item["dominant_action"], item["sample_count"]), reverse=True)
    return ranked[:top_n]


def format_cell_value(values: Dict[str, Any]) -> str:
    if not values:
        return "0.000|0.000|0.000"
    return f"F={float(values.get('fold', 0.0)):.3f}|C={float(values.get('check_call', 0.0)):.3f}|R={float(values.get('bet_raise', 0.0)):.3f}"


def latest_checkpoint_file(input_dir: str | Path) -> Path | None:
    files = iter_checkpoint_files(input_dir)
    if not files:
        return None
    return max(files, key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1)


def export_node_grid_csv(input_dir: str | Path, output_dir: str | Path, node_name: str):
    checkpoint = latest_checkpoint_file(input_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No report_checkpoint_*.json files found in {input_dir}")

    with open(checkpoint, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    rp = obj.get("range_policies", {}) if isinstance(obj, dict) else {}
    nodes = rp.get("nodes", []) if isinstance(rp, dict) else []
    target = None
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if (node.get("display_name") or node.get("history_label") or node.get("name")) == node_name:
            target = node
            break
    if target is None:
        raise ValueError(f"Node {node_name!r} not found in {checkpoint}")

    matrix = build_rank_matrix(target)
    top_cells = top_cells_for_node(target, top_n=3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{node_name}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spot", node_name])
        writer.writerow(["iteration", obj.get("iteration")])
        writer.writerow([])
        writer.writerow(["top_3_cells", "row_rank", "col_rank", "sample_count", "fold", "check_call", "bet_raise"])
        for cell in top_cells:
            writer.writerow([
                "top_cell",
                cell["row_label"],
                cell["col_label"],
                cell["sample_count"],
                f"{cell['fold']:.3f}",
                f"{cell['check_call']:.3f}",
                f"{cell['bet_raise']:.3f}",
            ])
        writer.writerow([])

        for action_name in ["fold", "check_call", "bet_raise"]:
            writer.writerow([])
            writer.writerow([f"{action_name}_13x13", *RANKS])
            for row_idx, row_rank in enumerate(RANKS):
                vals = [row_rank]
                for col_idx, col_rank in enumerate(RANKS):
                    cell = matrix.get((row_idx, col_idx), {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0})
                    vals.append(f"{float(cell.get(action_name, 0.0)):.3f}")
                writer.writerow(vals)

    return output_path


def build_summary_row(obj: Dict[str, Any], wanted_nodes: Iterable[str], wanted_hands: Iterable[str]) -> Dict[str, Any]:
    rp = obj.get("range_policies", {}) if isinstance(obj, dict) else {}
    nodes = rp.get("nodes", []) if isinstance(rp, dict) else []
    node_map: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = get_node_name(node)
        if name in wanted_nodes:
            node_map[name] = node

    row: Dict[str, Any] = {"iteration": obj.get("iteration")}
    for node_name in wanted_nodes:
        node = node_map.get(node_name)
        if not isinstance(node, dict):
            row[node_name] = {}
            continue

        af = node.get("action_frequencies", {})
        row[node_name] = {
            "fold": af.get("fold"),
            "check_call": af.get("check_call"),
            "bet_raise": af.get("bet_raise"),
        }
        for hand in wanted_hands:
            row[f"{node_name}:{hand}"] = get_hand_policy(node, hand)
    return row


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def summarize_checkpoint_directory(input_dir: str | Path, wanted_nodes: Iterable[str] | None = None, wanted_hands: Iterable[str] | None = None, step: int = 5):
    wanted_nodes = list(wanted_nodes or DEFAULT_WANTED_NODES)
    wanted_hands = list(wanted_hands or DEFAULT_WANTED_HANDS)

    rows: List[Dict[str, Any]] = []
    for path in iter_checkpoint_files(input_dir):
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if not isinstance(obj, dict):
            continue
        rows.append(build_summary_row(obj, wanted_nodes, wanted_hands))

    header = ["iteration"]
    for node_name in wanted_nodes:
        header.extend([f"{node_name}_F", f"{node_name}_C", f"{node_name}_R"])
    for hand in wanted_hands:
        header.extend([f"{hand}_F", f"{hand}_C", f"{hand}_R"])

    csv_rows = []
    for row in rows[::step]:
        vals = [str(row.get("iteration"))]
        for node_name in wanted_nodes:
            policy = row.get(node_name, {}) or {}
            vals.extend([
                fmt_value(policy.get("fold")),
                fmt_value(policy.get("check_call")),
                fmt_value(policy.get("bet_raise")),
            ])
        for hand in wanted_hands:
            policy = row.get(f"{wanted_nodes[0]}:{hand}", {}) or {}
            vals.extend([
                fmt_value(policy.get("fold")),
                fmt_value(policy.get("check_call")),
                fmt_value(policy.get("bet_raise")),
            ])
        csv_rows.append(vals)

    return header, csv_rows


def main():
    parser = argparse.ArgumentParser(description="Summarize a checkpoint directory into a compact CSV and 13x13 per-spot grids.")
    parser.add_argument("input_dir", help="directory containing report_checkpoint_*.json files")
    parser.add_argument("--output-csv", default=None, help="output CSV path; defaults to <input_dir>/summary.csv")
    parser.add_argument("--grid-dir", default=None, help="directory for per-spot 13x13 grid CSVs; defaults to <input_dir>/spot_grids")
    parser.add_argument("--step", type=int, default=5, help="sample every Nth checkpoint when writing summary rows")
    parser.add_argument("--nodes", nargs="*", default=DEFAULT_WANTED_NODES, help="selected nodes to summarize")
    parser.add_argument("--hands", nargs="*", default=DEFAULT_WANTED_HANDS, help="selected hands to summarize")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "summary.csv"
    grid_dir = Path(args.grid_dir) if args.grid_dir else input_dir / "spot_grids"

    header, rows = summarize_checkpoint_directory(
        input_dir,
        wanted_nodes=args.nodes,
        wanted_hands=args.hands,
        step=args.step,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    generated_files = []
    for node_name in args.nodes:
        grid_path = export_node_grid_csv(input_dir, grid_dir, node_name)
        generated_files.append(str(grid_path))

    print(f"Wrote {output_csv} with {len(rows)} rows")
    print(f"Wrote {len(generated_files)} per-spot grid CSVs in {grid_dir}")


if __name__ == "__main__":
    main()
