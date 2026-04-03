#!/usr/bin/env python3
"""Show best sweep configuration from nnm_eval_compare results."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print best sweep setting from nnm_eval_compare results.json."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Explicit path to results.json. If omitted, newest match from --results-glob is used.",
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="data/results/nnm_eval_compare_*/results.json",
        help="Glob for auto-discovery when --results-file is omitted.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg",
        help=(
            "Metric to optimize. Short names (ndcg/recall/mrr/precision/hit_rate) "
            "are expanded to @top_k."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override k for short metric names. Default: config.top_k from results.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["transformed", "delta", "baseline"],
        default="transformed",
        help="Which score family to sort by.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="How many top sweep rows to print.",
    )
    return parser.parse_args()


def _resolve_results_path(args: argparse.Namespace) -> Path:
    if args.results_file:
        path = Path(args.results_file)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        return path

    matches = sorted(glob.glob(args.results_glob))
    if not matches:
        raise FileNotFoundError(f"No results found for glob: {args.results_glob}")
    return Path(matches[-1])


def _metric_key(raw_metric: str, k: int) -> str:
    m = (raw_metric or "").strip()
    if not m:
        return f"ndcg@{k}"
    if "@" in m:
        return m
    if m in {"ndcg", "recall", "mrr", "precision", "hit_rate"}:
        return f"{m}@{k}"
    return m


def _as_float(value: Any, default: float = float("-inf")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fallback_sweep(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = data.get("config", {})
    return [
        {
            "entry_agg": cfg.get("entry_agg", "mean_top3"),
            "min_entry_token_hits": int(cfg.get("min_entry_token_hits", 1)),
            "baseline": data.get("baseline", {}),
            "transformed": data.get("transformed", {}),
            "delta_transformed_minus_baseline": data.get("delta_transformed_minus_baseline", {}),
        }
    ]


def main() -> int:
    args = parse_args()
    path = _resolve_results_path(args)
    data = json.loads(path.read_text(encoding="utf-8"))

    cfg = data.get("config", {})
    k = int(args.top_k) if args.top_k is not None else int(cfg.get("top_k", 10))
    metric_key = _metric_key(args.metric, k)

    sweep = data.get("sweep_results")
    if not isinstance(sweep, list) or not sweep:
        sweep = _fallback_sweep(data)

    def score_of(row: Dict[str, Any]) -> float:
        if args.sort_by == "transformed":
            return _as_float(row.get("transformed", {}).get(metric_key))
        if args.sort_by == "delta":
            return _as_float(row.get("delta_transformed_minus_baseline", {}).get(metric_key))
        return _as_float(row.get("baseline", {}).get(metric_key))

    rows = sorted(sweep, key=score_of, reverse=True)
    best = rows[0]

    b = _as_float(best.get("baseline", {}).get(metric_key), default=float("nan"))
    t = _as_float(best.get("transformed", {}).get(metric_key), default=float("nan"))
    d = _as_float(
        best.get("delta_transformed_minus_baseline", {}).get(metric_key),
        default=float("nan"),
    )

    print(f"results: {path}")
    print(f"metric: {metric_key} (sort_by={args.sort_by})")
    print(
        "best: "
        f"entry_agg={best.get('entry_agg')} "
        f"min_entry_token_hits={best.get('min_entry_token_hits')}"
    )
    print(f"score: transformed={t:.4f} delta={d:+.4f} baseline={b:.4f}")

    n = max(1, int(args.show_top))
    print("")
    print(f"top {min(n, len(rows))}:")
    for idx, row in enumerate(rows[:n], start=1):
        rb = _as_float(row.get("baseline", {}).get(metric_key), default=float("nan"))
        rt = _as_float(row.get("transformed", {}).get(metric_key), default=float("nan"))
        rd = _as_float(
            row.get("delta_transformed_minus_baseline", {}).get(metric_key),
            default=float("nan"),
        )
        print(
            f"{idx:02d} "
            f"agg={row.get('entry_agg')} "
            f"min_hits={row.get('min_entry_token_hits')} "
            f"transformed={rt:.4f} "
            f"delta={rd:+.4f} "
            f"baseline={rb:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
