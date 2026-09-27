"""Replay coverage across sample-floor shapes, per PREREG.md (issue #37).

No new data: reuses the ten `ArFit`s committed in #31
(`docs/analysis/2026-09-27-block-length/results/ar_fits.csv`). For each
shape in PREREG.md's fixed grid -- contiguous runs of n in {60, 90, 120,
180, 270, 365, 540, 730} issuance dates, then season-shaped grids of Y in
{1..6} JJA segments -- and for each of the ten models, this simulates
`--sims` (default 1000) slices with `weather_forecast_audit.block_length.
replay_coverage_segments` and records the share of nominal 95% bias
intervals that cover the model's true mean.

A shape/model whose coverage falls below 0.90 while some shape with fewer
blocks already passed (PREREG.md's "noise protocol", rule 2) is re-run once
at `--rerun-sims` (default 2000) on a fresh seed; both numbers are reported
and the re-run's coverage is what the decision rule (rule 3/4, via
`weather_forecast_audit.block_length.choose_min_sample_blocks`) sees.

Every (shape, model) replay is an independent task with its own
pre-assigned `numpy.random.SeedSequence` child, so results are exactly
reproducible regardless of `--workers` -- they are dispatched to a
`concurrent.futures.ProcessPoolExecutor` and collected by (shape, model)
key, never by completion order.

Run as `uv run python docs/analysis/2026-09-27-sample-floor/replay.py`
from the repo root. `--sims`/`--rerun-sims` below their pre-registered
defaults are for smoke-testing only; the generated RESULTS.md carries a
loud banner whenever either is non-default.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from weather_forecast_audit import scoring
from weather_forecast_audit.block_length import (
    ArFit,
    ShapeResult,
    choose_min_sample_blocks,
    contiguous_dates,
    n_blocks,
    needs_rerun,
    replay_coverage_segments,
    season_segments,
)

_HERE = Path(__file__).resolve().parent
_AR_FITS_PATH = _HERE.parent / "2026-09-27-block-length" / "results" / "ar_fits.csv"
_DEFAULT_OUT = _HERE

_SEED_INITIAL = 20260929
_SEED_RERUN = 20260930
_DEFAULT_SIMS = 1000
_DEFAULT_RERUN_SIMS = 2000
_N_BOOT = 500
_COVERAGE_THRESHOLD = 0.90
_N_MODELS = 10

# PREREG.md "Shapes": contiguous ascending n, then season ascending Y. This
# order is a constant, not read off any data -- it fixes the seed indexing.
_CONTIGUOUS_NS: tuple[int, ...] = (60, 90, 120, 180, 270, 365, 540, 730)
_SEASON_YEARS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_HERE,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


class Model:
    """One of the ten committed `ArFit`s plus the labels PREREG reports it by."""

    __slots__ = ("fit", "lead_day", "series", "variable")

    def __init__(self, series: str, variable: str, lead_day: int, fit: ArFit) -> None:
        self.series = series
        self.variable = variable
        self.lead_day = lead_day
        self.fit = fit


class Shape:
    """One replay shape: its segments and the block/date counts they imply."""

    __slots__ = ("kind", "label", "n_blocks", "n_dates", "segments")

    def __init__(
        self,
        label: str,
        kind: str,
        segments: list[list[date]],
        block_days: int,
    ) -> None:
        all_dates = [d for segment in segments for d in segment]
        self.label = label
        self.kind = kind
        self.segments = segments
        self.n_dates = len(all_dates)
        self.n_blocks = n_blocks(all_dates, block_days)


def _load_models(path: Path) -> list[Model]:
    if not path.exists():
        msg = f"no ar_fits.csv at {path}; run #31's analyze.py first"
        raise SystemExit(msg)
    frame = pl.read_csv(path)
    if frame.height != 10:
        msg = (
            f"expected exactly 10 rows in {path}, found {frame.height}; "
            "PREREG.md has no rule for a different model count"
        )
        raise SystemExit(msg)

    models: list[Model] = []
    for row in frame.iter_rows(named=True):
        phi_text = row["phi"] or ""
        phi = np.array([float(v) for v in str(phi_text).split(";") if v != ""])
        fit = ArFit(
            p=int(row["p"]),
            phi=phi,
            sigma2=float(row["sigma2"]),
            mean=float(row["mean"]),
        )
        models.append(Model(row["series"], row["variable"], int(row["lead_day"]), fit))
    return models


def _build_shapes(block_days: int) -> list[Shape]:
    shapes: list[Shape] = []
    for n in _CONTIGUOUS_NS:
        shapes.append(
            Shape(f"contiguous:{n}", "contiguous", contiguous_dates(n), block_days)
        )
    for year_count in _SEASON_YEARS:
        shapes.append(
            Shape(
                f"season:Y{year_count}",
                "season",
                season_segments(year_count),
                block_days,
            )
        )
    return shapes


# -- replay execution ---------------------------------------------------------


# Module-level so it is picklable across `ProcessPoolExecutor` worker processes.
def _replay_task(
    fit: ArFit,
    segments: list[list[date]],
    block_days: int,
    n_sims: int,
    n_boot: int,
    seed_seq: np.random.SeedSequence,
) -> float:
    rng = np.random.default_rng(seed_seq)
    return replay_coverage_segments(
        fit, segments, block_days, n_sims=n_sims, rng=rng, n_boot=n_boot
    )


_Task = tuple[int, int, ArFit, list[list[date]], int, int, int, np.random.SeedSequence]


def _run_batch(tasks: list[_Task], workers: int) -> dict[tuple[int, int], float]:
    if not tasks:
        return {}
    results: dict[tuple[int, int], float] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _replay_task, fit, segments, block_days, n_sims, n_boot, seed
            ): (shape_idx, model_idx)
            for (
                shape_idx,
                model_idx,
                fit,
                segments,
                block_days,
                n_sims,
                n_boot,
                seed,
            ) in tasks
        }
        for future in futures:
            results[futures[future]] = future.result()
    return results


# -- RESULTS.md -----------------------------------------------------------------


def _write_results_md(
    *,
    out_dir: Path,
    shapes: list[Shape],
    models: list[Model],
    coverage_matrix: list[list[float]],
    rerun_results: dict[tuple[int, int], float],
    flagged: list[int],
    final_shape_results: list[ShapeResult],
    min_sample_blocks: int | None,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    lines: list[str] = []
    is_smoke = args.sims != _DEFAULT_SIMS or args.rerun_sims != _DEFAULT_RERUN_SIMS
    if is_smoke:
        lines.append(
            "**SMOKE RUN -- not the pre-registered replay** "
            f"(sims={args.sims}, rerun_sims={args.rerun_sims}; PREREG.md "
            f"specifies {_DEFAULT_SIMS} / {_DEFAULT_RERUN_SIMS})."
        )
        lines.append("")

    lines.append("# Sample-floor coverage replay results (#37)")
    lines.append("")
    lines.append(
        "Generated by `docs/analysis/2026-09-27-sample-floor/replay.py`. "
        "Do not hand-edit; re-run the script to regenerate."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- `replay.py` SHA-256: `{_sha256(Path(__file__).resolve())}`")
    lines.append(f"- `ar_fits.csv` SHA-256: `{_sha256(_AR_FITS_PATH)}`")
    lines.append(f"- Git HEAD: `{_git_head()}`")
    lines.append(
        f"- sims: {args.sims}, rerun_sims: {args.rerun_sims}, n_boot: {_N_BOOT}, "
        f"block_days: {scoring.BLOCK_DAYS}"
    )
    lines.append(f"- elapsed: {elapsed:.1f}s")
    lines.append("")

    lines.append("## Shapes")
    lines.append("")
    lines.append("| shape | kind | n_dates | n_blocks |")
    lines.append("| --- | --- | --- | --- |")
    for shape in shapes:
        lines.append(
            f"| {shape.label} | {shape.kind} | {shape.n_dates} | {shape.n_blocks} |"
        )
    lines.append("")

    lines.append("## Coverage by shape and model")
    lines.append("")
    lines.append(
        "| shape | n_blocks | model | variable | lead_day | coverage | rerun_coverage |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for shape_idx, shape in enumerate(shapes):
        for model_idx, model in enumerate(models):
            coverage = coverage_matrix[shape_idx][model_idx]
            rerun = rerun_results.get((shape_idx, model_idx))
            rerun_text = f"{rerun:.4f}" if rerun is not None else ""
            lines.append(
                f"| {shape.label} | {shape.n_blocks} | {model.series} | "
                f"{model.variable} | {model.lead_day} | {coverage:.4f} | {rerun_text} |"
            )
    lines.append("")

    lines.append("## Rule 2: re-runs")
    lines.append("")
    if flagged:
        for shape_idx in flagged:
            shape = shapes[shape_idx]
            lines.append(
                f"- `{shape.label}` (n_blocks={shape.n_blocks}) flagged: a shape "
                "with fewer blocks passed."
            )
            for model_idx, model in enumerate(models):
                if coverage_matrix[shape_idx][model_idx] < _COVERAGE_THRESHOLD:
                    rerun = rerun_results.get((shape_idx, model_idx))
                    rerun_text = f"{rerun:.4f}" if rerun is not None else "(not run)"
                    lines.append(
                        f"  - {model.series} {model.variable} lead{model.lead_day}: "
                        f"{coverage_matrix[shape_idx][model_idx]:.4f} -> {rerun_text} "
                        f"(n_sims={args.rerun_sims})"
                    )
    else:
        lines.append(
            "No shape triggered rule 2 (no failure sat above an already-passing, "
            "smaller-block shape)."
        )
    lines.append("")

    lines.append("## Decision rule walkthrough")
    lines.append("")
    for shape_idx, shape in enumerate(shapes):
        result = final_shape_results[shape_idx]
        shape_passes = all(c >= _COVERAGE_THRESHOLD for c in result.coverages)
        lines.append(
            f"- `{shape.label}` (n_blocks={shape.n_blocks}): "
            f"{'PASS' if shape_passes else 'FAIL'} "
            f"(min coverage after rule 2 = {min(result.coverages):.4f})"
        )
    lines.append("")

    if min_sample_blocks is not None:
        lines.append(
            f"## MIN_SAMPLE_BLOCKS\n\n`MIN_SAMPLE_BLOCKS = {min_sample_blocks}`."
        )
    else:
        lines.append(
            "## MIN_SAMPLE_BLOCKS\n\n"
            "No `k` satisfies the rule: the largest-block shape still fails "
            "after rule 2. Per PREREG.md rule 4, `scoring.MIN_SAMPLE_DATES` / "
            "`min_sample_flag` is not changed, and #16 stays blocked."
        )
    lines.append("")

    interpretation = _HERE / "INTERPRETATION.md"
    lines.append("## Interpretation")
    lines.append("")
    if interpretation.exists():
        lines.append(interpretation.read_text().rstrip())
    else:
        lines.append("(not yet written: add INTERPRETATION.md and re-run)")
    lines.append("")

    (out_dir / "RESULTS.md").write_text("\n".join(lines))


# -- CLI ------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="directory to write results/ and RESULTS.md into",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=_DEFAULT_SIMS,
        help=f"simulations per (shape, model) (PREREG.md default {_DEFAULT_SIMS})",
    )
    parser.add_argument(
        "--rerun-sims",
        type=int,
        default=_DEFAULT_RERUN_SIMS,
        help=(
            f"simulations for rule-2 re-runs (PREREG.md default {_DEFAULT_RERUN_SIMS})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="ProcessPoolExecutor worker count (default: os.cpu_count())",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    start = time.perf_counter()

    if scoring.BLOCK_DAYS != 14:
        msg = (
            f"expected scoring.BLOCK_DAYS == 14, found {scoring.BLOCK_DAYS}; "
            "PREREG.md has no rule for a different block length"
        )
        raise SystemExit(msg)
    block_days = scoring.BLOCK_DAYS

    models = _load_models(_AR_FITS_PATH)
    shapes = _build_shapes(block_days)
    n_shapes = len(shapes)

    initial_seeds = np.random.SeedSequence(_SEED_INITIAL).spawn(n_shapes * _N_MODELS)
    rerun_seeds = np.random.SeedSequence(_SEED_RERUN).spawn(n_shapes * _N_MODELS)

    initial_tasks: list[_Task] = [
        (
            shape_idx,
            model_idx,
            model.fit,
            shape.segments,
            block_days,
            args.sims,
            _N_BOOT,
            initial_seeds[shape_idx * _N_MODELS + model_idx],
        )
        for shape_idx, shape in enumerate(shapes)
        for model_idx, model in enumerate(models)
    ]
    initial_results = _run_batch(initial_tasks, args.workers)

    coverage_matrix = [
        [initial_results[(shape_idx, model_idx)] for model_idx in range(len(models))]
        for shape_idx in range(n_shapes)
    ]

    shape_results = [
        ShapeResult(shape.label, shape.n_blocks, tuple(coverage_matrix[i]))
        for i, shape in enumerate(shapes)
    ]
    flagged = needs_rerun(shape_results, threshold=_COVERAGE_THRESHOLD)

    rerun_tasks: list[_Task] = [
        (
            shape_idx,
            model_idx,
            models[model_idx].fit,
            shapes[shape_idx].segments,
            block_days,
            args.rerun_sims,
            _N_BOOT,
            rerun_seeds[shape_idx * _N_MODELS + model_idx],
        )
        for shape_idx in flagged
        for model_idx in range(len(models))
        if coverage_matrix[shape_idx][model_idx] < _COVERAGE_THRESHOLD
    ]
    rerun_results = _run_batch(rerun_tasks, args.workers)

    final_matrix = [
        [
            rerun_results.get(
                (shape_idx, model_idx), coverage_matrix[shape_idx][model_idx]
            )
            for model_idx in range(len(models))
        ]
        for shape_idx in range(n_shapes)
    ]
    final_shape_results = [
        ShapeResult(shape.label, shape.n_blocks, tuple(final_matrix[i]))
        for i, shape in enumerate(shapes)
    ]
    min_sample_blocks = choose_min_sample_blocks(
        final_shape_results, threshold=_COVERAGE_THRESHOLD
    )

    elapsed = time.perf_counter() - start

    out_dir = args.out
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = [
        {
            "shape": shape.label,
            "kind": shape.kind,
            "n_dates": shape.n_dates,
            "n_blocks": shape.n_blocks,
            "model": model.series,
            "variable": model.variable,
            "lead_day": model.lead_day,
            "coverage": coverage_matrix[shape_idx][model_idx],
            "rerun_coverage": rerun_results.get((shape_idx, model_idx)),
        }
        for shape_idx, shape in enumerate(shapes)
        for model_idx, model in enumerate(models)
    ]
    pl.DataFrame(coverage_rows).write_csv(results_dir / "coverage.csv")

    _write_results_md(
        out_dir=out_dir,
        shapes=shapes,
        models=models,
        coverage_matrix=coverage_matrix,
        rerun_results=rerun_results,
        flagged=flagged,
        final_shape_results=final_shape_results,
        min_sample_blocks=min_sample_blocks,
        args=args,
        elapsed=elapsed,
    )

    print(f"MIN_SAMPLE_BLOCKS: {min_sample_blocks}")
    print(f"wrote {results_dir} and {out_dir / 'RESULTS.md'}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
