"""Cross-framework timing aggregator.

Walks ``outputs/train/{dataset}/{model}/{framework}/seed_*/timing.json``,
groups runs by ``(dataset, model)``, computes mean ± std across seeds
within each framework, and prints a side-by-side comparison table.

Wall-clock only by design. GPU util / power / energy / model latency
were prototyped earlier but removed to keep the surface small; this
script only reads what ``RunTiming`` currently dumps.

Usage:
    uv run python -m src.benchmarks.compare_timing
    uv run python -m src.benchmarks.compare_timing --dataset MIND-small
    uv run python -m src.benchmarks.compare_timing --models GLORY,NRMS
    uv run python -m src.benchmarks.compare_timing --json report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.core.io.timing import RunTiming, dict_to_run_timing

console = Console()

DEFAULT_OUTPUT_ROOT = Path("outputs/train")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def discover_timing_files(
    root: Path,
    *,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    frameworks: list[str] | None = None,
    encoders: list[str] | None = None,
    split_strategies: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Find ``timing.json`` files matching the given filters.

    Expects the post-encoder-slug path layout:
        ``<root>/<dataset>/<model>/<framework>/<encoder>/<split>/seed_*/timing.json``

    Returns a list of ``(timing_json_path, encoder_slug)`` pairs since
    the encoder slug is not stored inside ``RunTiming`` and must come
    from the path.
    """
    paths: list[tuple[Path, str]] = []
    for p in root.glob("*/*/*/*/*/seed_*/timing.json"):
        try:
            dataset = p.parents[5].name
            model = p.parents[4].name
            framework = p.parents[3].name
            encoder = p.parents[2].name
            split = p.parents[1].name
        except IndexError:
            continue
        if datasets and dataset not in datasets:
            continue
        if models and model not in models:
            continue
        if frameworks and framework not in frameworks:
            continue
        if encoders and encoder not in encoders:
            continue
        if split_strategies and split not in split_strategies:
            continue
        paths.append((p, encoder))
    return sorted(paths)


def load_runs(paths: list[tuple[Path, str]]) -> list[RunTiming]:
    """Load ``RunTiming`` from each path; attach encoder slug as an
    ad-hoc attribute (not persisted in the dump, inferred from path).
    """
    runs: list[RunTiming] = []
    for path, encoder in paths:
        try:
            with open(path) as f:
                d = json.load(f)
            rt = dict_to_run_timing(d)
            rt.encoder = encoder  # type: ignore[attr-defined]
            runs.append(rt)
        except Exception as e:
            console.log(f"[yellow]Skipping {path}: {e}[/yellow]")
    return runs


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _mean_steady_state(values: list[float]) -> float | None:
    """Mean across all but the first entry (warmup-excluded)."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return sum(values[1:]) / (len(values) - 1)


def aggregate_runs(
    runs: list[RunTiming],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Group by (dataset, model, framework, encoder); reduce across seeds."""
    by_key: dict[tuple[str, str, str, str], list[RunTiming]] = defaultdict(list)
    for r in runs:
        encoder = getattr(r, "encoder", "")
        by_key[(r.dataset_name, r.model_name, r.framework, encoder)].append(r)

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, group in by_key.items():
        n_seeds = len(group)
        params = group[0].n_params_total

        ttfs: list[float] = []
        train_mean_epoch: list[float] = []
        train_throughput: list[float] = []
        val_mean_epoch: list[float] = []
        val_throughput: list[float] = []
        test_seconds: list[float] = []
        total_seconds: list[float] = []

        for r in group:
            if r.time_to_first_step_seconds is not None:
                ttfs.append(float(r.time_to_first_step_seconds))
            if r.total_seconds is not None:
                total_seconds.append(float(r.total_seconds))

            # Train epochs — steady-state mean (drop epoch 1, JIT/cuDNN warmup).
            train_wall = [p.wall_seconds for p in r.train_epochs]
            t_thr = [
                p.throughput_samples_per_s
                for p in r.train_epochs
                if p.throughput_samples_per_s is not None
            ]
            v = _mean_steady_state(train_wall)
            if v is not None:
                train_mean_epoch.append(v)
            v = _mean_steady_state(t_thr)
            if v is not None:
                train_throughput.append(v)

            val_wall = [p.wall_seconds for p in r.val_epochs]
            v_thr = [
                p.throughput_samples_per_s
                for p in r.val_epochs
                if p.throughput_samples_per_s is not None
            ]
            v = _mean_steady_state(val_wall)
            if v is not None:
                val_mean_epoch.append(v)
            v = _mean_steady_state(v_thr)
            if v is not None:
                val_throughput.append(v)

            if r.test is not None:
                test_seconds.append(r.test.wall_seconds)

        result[key] = {
            "n_seeds": n_seeds,
            "n_params": params,
            "ttfs_s": _mean_std(ttfs),
            "train_mean_epoch_s": _mean_std(train_mean_epoch),
            "train_samples_per_s": _mean_std(train_throughput),
            "val_mean_epoch_s": _mean_std(val_mean_epoch),
            "val_samples_per_s": _mean_std(val_throughput),
            "test_seconds": _mean_std(test_seconds),
            "total_seconds": _mean_std(total_seconds),
        }
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(mean_std: tuple[float, float]) -> str:
    m, s = mean_std
    if m != m:  # NaN
        return "—"
    return f"{m:.2f} ± {s:.2f}" if s > 0 else f"{m:.2f}"


def _fmt_int(mean_std: tuple[float, float]) -> str:
    m, s = mean_std
    if m != m:
        return "—"
    return f"{int(m):,} ± {int(s):,}" if s > 0 else f"{int(m):,}"


def render_per_run_breakdown(runs: list[RunTiming]) -> None:
    """For each run, print every-epoch wall times + both means.

    ``all-epochs mean`` is the plain arithmetic mean over every epoch
    that actually ran (so e.g. 8/10 if early-stop fired).
    ``steady mean`` drops epoch 1 (JIT/cuDNN warmup) for a comparison-
    fair number; that's what the aggregate table shows.
    """
    by_key: dict[tuple[str, str, str, str], list[RunTiming]] = defaultdict(list)
    for r in runs:
        encoder = getattr(r, "encoder", "")
        by_key[(r.dataset_name, r.model_name, r.framework, encoder)].append(r)

    for (dataset, model, fw, enc), group in sorted(by_key.items()):
        for r in sorted(group, key=lambda x: x.seed):
            t = Table(
                title=(
                    f"[bold]{dataset} / {model} / {fw} / {enc} / seed {r.seed}[/bold] "
                    f"— per-epoch breakdown"
                ),
                show_lines=False,
            )
            t.add_column("epoch", justify="right")
            t.add_column("train (s)", justify="right")
            t.add_column("val (s)", justify="right")
            t.add_column("train samples/s", justify="right")

            te = [p.wall_seconds for p in r.train_epochs]
            ve = [p.wall_seconds for p in r.val_epochs]
            for i, (tp, vp) in enumerate(zip(r.train_epochs, r.val_epochs), 1):
                t.add_row(
                    str(i),
                    f"{tp.wall_seconds:.2f}",
                    f"{vp.wall_seconds:.2f}",
                    f"{int(tp.throughput_samples_per_s):,}"
                    if tp.throughput_samples_per_s is not None
                    else "—",
                )
            # Means as footer rows.
            if te:
                t.add_row(
                    "[dim]all mean[/dim]",
                    f"[dim]{sum(te) / len(te):.2f}[/dim]",
                    f"[dim]{sum(ve) / len(ve):.2f}[/dim]" if ve else "—",
                    "",
                )
            if len(te) > 1:
                t.add_row(
                    "[dim]steady mean[/dim]",
                    f"[dim]{sum(te[1:]) / (len(te) - 1):.2f}[/dim]",
                    f"[dim]{sum(ve[1:]) / (len(ve) - 1):.2f}[/dim]"
                    if len(ve) > 1
                    else "—",
                    "[dim](epoch 1 dropped — warmup)[/dim]",
                )
            test_s = r.test.wall_seconds if r.test else float("nan")
            ttfs = r.time_to_first_step_seconds or 0
            console.print(t)
            console.print(
                f"  TTFS: {ttfs:.2f}s  |  test: {test_s:.2f}s  |  "
                f"total: {r.total_seconds:.1f}s  |  "
                f"params: {r.n_params_total:,}"
            )
            console.print()


def render_table(agg: dict[tuple[str, str, str, str], dict[str, Any]]) -> None:
    """Print one Rich table per (dataset, model); rows = (framework, encoder)."""
    by_dm: dict[tuple[str, str], list[tuple[str, str, dict[str, Any]]]] = defaultdict(
        list
    )
    for (dataset, model, fw, enc), stats in agg.items():
        by_dm[(dataset, model)].append((fw, enc, stats))

    for (dataset, model), rows in sorted(by_dm.items()):
        table = Table(
            title=f"[bold]{dataset} / {model}[/bold]",
            show_lines=True,
        )
        table.add_column("framework", style="bold")
        table.add_column("encoder", style="bold")
        table.add_column("seeds", justify="right")
        table.add_column("params (M)", justify="right")
        table.add_column("time-to-first-step (s)", justify="right")
        table.add_column("train epoch (s)", justify="right")
        table.add_column("train samples/s", justify="right")
        table.add_column("val epoch (s)", justify="right")
        table.add_column("test (s)", justify="right")
        table.add_column("total (s)", justify="right")

        for fw, enc, st in sorted(rows):
            params_m = st["n_params"] / 1e6
            table.add_row(
                fw,
                enc,
                str(st["n_seeds"]),
                f"{params_m:.2f}",
                _fmt(st["ttfs_s"]),
                _fmt(st["train_mean_epoch_s"]),
                _fmt_int(st["train_samples_per_s"]),
                _fmt(st["val_mean_epoch_s"]),
                _fmt(st["test_seconds"]),
                _fmt(st["total_seconds"]),
            )
        console.print(table)


def to_json_payload(
    agg: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (dataset, model, fw, enc), st in sorted(agg.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "framework": fw,
            "encoder": enc,
        }
        for k, v in st.items():
            if isinstance(v, tuple):
                row[f"{k}_mean"] = v[0]
                row[f"{k}_std"] = v[1]
            else:
                row[k] = v
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Path containing dataset/model/framework/seed_*/timing.json.",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        help="Filter by dataset (repeatable).",
    )
    ap.add_argument(
        "--models",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="Comma-separated model names.",
    )
    ap.add_argument(
        "--frameworks",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="Comma-separated framework names (jax, pytorch).",
    )
    ap.add_argument(
        "--encoders",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="Comma-separated encoder slugs (e.g. glove,bert,bert_token,bert_dual).",
    )
    ap.add_argument(
        "--splits",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="Comma-separated split strategies (random, chronological, dev_as_val).",
    )
    ap.add_argument(
        "--json",
        type=Path,
        help="Write aggregated rows as JSON to this path (in addition to the table).",
    )
    ap.add_argument(
        "--per-epoch",
        "-v",
        action="store_true",
        help="Print the per-epoch breakdown for each run before the aggregate table.",
    )
    args = ap.parse_args()

    paths = discover_timing_files(
        args.root,
        datasets=args.dataset,
        models=args.models,
        frameworks=args.frameworks,
        encoders=args.encoders,
        split_strategies=args.splits,
    )
    if not paths:
        console.log("[yellow]No timing.json files found.[/yellow]")
        return
    console.log(f"Found {len(paths)} timing.json file(s).")

    runs = load_runs(paths)
    if args.per_epoch:
        render_per_run_breakdown(runs)
    agg = aggregate_runs(runs)
    render_table(agg)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(to_json_payload(agg), f, indent=2)
        console.log(f"Wrote aggregated JSON to {args.json}")


if __name__ == "__main__":
    main()
