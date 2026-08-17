#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.11.1",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "plots"

FIGURE_SIZE = (9.2, 4.5)
CONSTRUCTION_FIGURE_SIZE = (9.2, 8.0)
LINE_WIDTH = 1.8
MARKER_SIZE = 5.5

EF_SEARCH_VALUES = (16, 32, 64, 128, 256, 512)
GRAPH_VALUES = (4, 8, 16, 32, 64)
CONSTRUCTION_VALUES = (32, 64, 100, 128, 200, 400)
SIZE_VALUES = (100_000, 250_000, 500_000, 1_000_000)
MODES = ("sequential", "dynamic", "batched")
MODE_LABELS = {
    "sequential": "Sequential",
    "dynamic": "Dynamic",
    "batched": "Batched",
}
MODE_COLORS = {
    "sequential": "#222222",
    "dynamic": "#D55E00",
    "batched": "#009E73",
}
MODE_MARKERS = {
    "sequential": "o",
    "dynamic": "^",
    "batched": "s",
}

REPORT_PATHS = {
    "search_sift": "measure/m-sweep/sift-1m-normal.json",
    "search_mnist": "measure/m-sweep/mnist-60k-normal.json",
    "construction": "build/ef-construction-sweep/sift-1m.json",
    "size_build_100k": "build/size-sweep/sift-100k.json",
    "size_build_250k": "build/size-sweep/sift-250k.json",
    "size_build_500k": "build/size-sweep/sift-500k.json",
    "size_build_1m": "build/size-sweep/sift-1m.json",
    "size_measure_100k": "measure/size-sweep/sift-100k-normal.json",
    "size_measure_250k": "measure/size-sweep/sift-250k-normal.json",
    "size_measure_500k": "measure/size-sweep/sift-500k-normal.json",
    "size_measure_1m": "measure/size-sweep/sift-1m-normal.json",
    "size_parallel_100k": "build/parallel-construction/size-sift-100k.json",
    "size_parallel_250k": "build/parallel-construction/size-sift-250k.json",
    "size_parallel_500k": "build/parallel-construction/size-sift-500k.json",
    "size_parallel_1m": "build/parallel-construction/size-sift-1m.json",
    "parallel_construction": "build/parallel-construction/sift-1m.json",
    "pq_normal": "measure/m-sweep/sift-1m-normal.json",
    "pq32": "measure/pq/m-sweep/sift-1m-pq32.json",
    "pq64": "measure/pq/m-sweep/sift-1m-pq64.json",
    "pq128": "measure/pq/m-sweep/sift-1m-pq128.json",
}
BUILD_REPORTS = {
    "construction",
    "size_parallel_100k",
    "size_parallel_250k",
    "size_parallel_500k",
    "size_parallel_1m",
    "parallel_construction",
}

GRAPH_COLORS = {
    4: "#0072B2",
    8: "#E69F00",
    16: "#009E73",
    32: "#D55E00",
    64: "#CC79A7",
}
CONSTRUCTION_COLORS = {
    32: "#0072B2",
    64: "#D55E00",
    100: "#009E73",
    128: "#CC79A7",
    200: "#E69F00",
    400: "#56B4E9",
}
MARKERS = ("o", "s", "^", "D", "v", "P")


def load_reports() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    required_run_fields = {
        "m",
        "m0",
        "ef_construction",
        "ef_search",
        "recall",
        "qps",
        "memory_mib",
    }
    parallel_run_fields = {
        "build_mode",
        "build_repetition",
        "build_threads",
        "effective_build_threads",
        "build_time_s",
        "insert_qps",
    }

    for name, relative_path in REPORT_PATHS.items():
        path = RESULTS / relative_path
        if not path.is_file():
            raise SystemExit(f"missing benchmark report: {path}")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid JSON report: {path}: {error}") from error
        if not isinstance(report, dict) or not isinstance(report.get("runs"), list):
            raise SystemExit(f"invalid report structure: {path}")
        if report.get("effective_k") != 10:
            raise SystemExit(f"expected recall@10 report: {path}")
        for run in report["runs"]:
            if not required_run_fields.issubset(run):
                missing = sorted(required_run_fields - run.keys())
                raise SystemExit(f"{path} is missing run fields: {', '.join(missing)}")
            if name in BUILD_REPORTS:
                if not parallel_run_fields.issubset(run):
                    missing = sorted(parallel_run_fields - run.keys())
                    raise SystemExit(
                        f"{path} is missing parallel run fields: {', '.join(missing)}"
                    )
                if run["build_mode"] in ("dynamic", "batched") and (
                    run["build_threads"] is None
                    or run["effective_build_threads"] is None
                ):
                    raise SystemExit(f"{path} has a parallel run without worker counts")
                if run["build_time_s"] is None or run["insert_qps"] is None:
                    raise SystemExit(
                        f"{path} has a build run without construction metrics"
                    )
        reports[name] = report

    return reports


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "legend.frameon": False,
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.5,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "figure.titlesize": 12,
            "figure.titleweight": "bold",
            "svg.fonttype": "none",
        }
    )


def style_axis(axis: Any) -> None:
    axis.grid(True, which="major", color="#D9D9D9", alpha=0.65, linewidth=0.5)
    axis.grid(False, which="minor")
    axis.tick_params(direction="out", length=3, width=0.8)


def figure_header(figure: Any, title: str, subtitle: str) -> None:
    layout = figure.get_layout_engine()
    if layout is not None:
        layout.set(rect=(0.0, 0.0, 1.0, 0.92))
    figure.text(
        0.5,
        0.995,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.95,
        subtitle,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )


def require_search_sweep(rows: list[dict[str, Any]], context: str) -> None:
    values = tuple(sorted(int(row["ef_search"]) for row in rows))
    if values != EF_SEARCH_VALUES:
        expected = ", ".join(str(value) for value in EF_SEARCH_VALUES)
        actual = ", ".join(str(value) for value in values) or "none"
        raise SystemExit(
            f"{context} expected ef_search values [{expected}], found [{actual}]"
        )


def require_one(rows: list[dict[str, Any]], context: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise SystemExit(f"{context} expected one run, found {len(rows)}")
    return rows[0]


def save_figure(figure: Any, path: Path) -> None:
    figure.savefig(path, format="svg", bbox_inches="tight")
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def format_qps_tick(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def configure_qps_axis(axis: Any) -> None:
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(format_qps_tick))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="y", which="minor", length=0)


def configure_fixed_log_ticks(
    axis: Any, values: tuple[int, ...], labels: tuple[str, ...]
) -> None:
    axis.xaxis.set_major_locator(FixedLocator(values))
    axis.xaxis.set_major_formatter(FixedFormatter(labels))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())


def search_curve(
    axis: Any, rows: list[dict[str, Any]], color: str, marker: str, label: str
) -> None:
    rows = sorted(rows, key=lambda row: int(row["ef_search"]))
    axis.plot(
        [float(row["recall"]) for row in rows],
        [float(row["qps"]) for row in rows],
        color=color,
        marker=marker,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=label,
    )


def selected_graph_rows(report: dict[str, Any], m: int) -> list[dict[str, Any]]:
    return [
        row
        for row in report["runs"]
        if row["m"] == m and row["m0"] == 2 * m and row["ef_construction"] == 200
    ]


def plot_search_tradeoff(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, sharey=True, layout="constrained"
    )
    panels = (
        (axes[0], "search_sift", "SIFT-1M"),
        (axes[1], "search_mnist", "MNIST-60k"),
    )

    for axis, report_name, title in panels:
        report = reports[report_name]
        for index, m in enumerate(GRAPH_VALUES):
            rows = selected_graph_rows(report, m)
            require_search_sweep(rows, f"{report_name} M={m}")
            search_curve(axis, rows, GRAPH_COLORS[m], MARKERS[index], str(m))
        axis.set_title(title)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("QPS")
        configure_qps_axis(axis)
        style_axis(axis)

    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="M (M0=2M)",
        loc="outside lower center",
        ncol=5,
    )
    figure_header(
        figure,
        "Graph degree search trade-off",
        "Sequential | ef_construction=200 | ef_search points: 16, 32, 64, 128, 256, 512",
    )
    save_figure(figure, output_dir / "search_tradeoff.svg")
    plt.close(figure)


def plot_construction_tradeoff(
    reports: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=CONSTRUCTION_FIGURE_SIZE, layout="constrained"
    )
    report = reports["construction"]
    quality_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    build_axis = axes[1, 1]
    quality_axes[1].sharex(quality_axes[0])
    quality_axes[1].sharey(quality_axes[0])
    quality_axes[2].sharex(quality_axes[0])
    quality_axes[2].sharey(quality_axes[0])

    for axis, mode in zip(quality_axes, MODES, strict=True):
        for index, value in enumerate(CONSTRUCTION_VALUES):
            rows = [
                row
                for row in report["runs"]
                if row["m"] == 16
                and row["m0"] == 32
                and row["ef_construction"] == value
                and row["build_mode"] == mode
            ]
            require_search_sweep(rows, f"construction {mode} ef_construction={value}")
            search_curve(
                axis,
                rows,
                CONSTRUCTION_COLORS[value],
                MARKERS[index],
                str(value),
            )
        title = MODE_LABELS[mode]
        if mode != "sequential":
            title += " (11 workers)"
        axis.set_title(title)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("QPS")
        configure_qps_axis(axis)
        style_axis(axis)

    quality_axes[1].set_ylabel("")
    for mode in MODES:
        build_rows = []
        for value in CONSTRUCTION_VALUES:
            row = require_one(
                [
                    row
                    for row in report["runs"]
                    if row["m"] == 16
                    and row["m0"] == 32
                    and row["ef_construction"] == value
                    and row["build_mode"] == mode
                    and row["ef_search"] == EF_SEARCH_VALUES[0]
                ],
                f"construction build {mode} ef_construction={value}",
            )
            if mode != "sequential" and row["effective_build_threads"] != 11:
                raise SystemExit(f"construction {mode} row did not use 11 workers")
            build_rows.append(row)
        build_axis.plot(
            CONSTRUCTION_VALUES,
            [row["build_time_s"] for row in build_rows],
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            label=MODE_LABELS[mode],
        )

    build_axis.set_title("Construction time")
    build_axis.set_xlabel("ef_construction")
    build_axis.set_ylabel("Build time (s)")
    build_axis.set_xscale("log")
    construction_labels = tuple(str(value) for value in CONSTRUCTION_VALUES)
    configure_fixed_log_ticks(build_axis, CONSTRUCTION_VALUES, construction_labels)
    build_axis.tick_params(axis="x", labelrotation=25)
    for label in build_axis.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axis(build_axis)
    build_axis.legend(title="Build mode", loc="best")
    handles, labels = quality_axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, title="ef_construction", loc="outside lower center", ncol=3
    )
    figure_header(
        figure,
        "Construction effort trade-off on SIFT-1M",
        "M=16, M0=32 | ef_search points: 16, 32, 64, 128, 256, 512",
    )
    save_figure(figure, output_dir / "construction_tradeoff.svg")
    plt.close(figure)


def plot_size_scaling(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    figure, (build_axis, memory_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, layout="constrained"
    )
    build_names = (
        "size_build_100k",
        "size_build_250k",
        "size_build_500k",
        "size_build_1m",
    )
    measure_names = (
        "size_measure_100k",
        "size_measure_250k",
        "size_measure_500k",
        "size_measure_1m",
    )
    parallel_names = (
        "size_parallel_100k",
        "size_parallel_250k",
        "size_parallel_500k",
        "size_parallel_1m",
    )

    mode_times = {
        "sequential": [
            require_one(reports[name]["runs"], f"{name} sequential build")[
                "build_time_s"
            ]
            for name in build_names
        ],
        "dynamic": [],
        "batched": [],
    }
    for mode in ("dynamic", "batched"):
        for name in parallel_names:
            row = require_one(
                [row for row in reports[name]["runs"] if row["build_mode"] == mode],
                f"{name} {mode} build",
            )
            if row["effective_build_threads"] != 11:
                raise SystemExit(f"{name} {mode} row did not use 11 workers")
            mode_times[mode].append(row["build_time_s"])

    memories = [reports[name]["runs"][0]["memory_mib"] for name in measure_names]

    for mode in MODES:
        build_axis.plot(
            SIZE_VALUES,
            mode_times[mode],
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            label=MODE_LABELS[mode],
        )
    memory_axis.plot(SIZE_VALUES, memories, color="#0072B2", marker="o")
    build_axis.set_title("Construction time")
    memory_axis.set_title("Loaded index memory")
    for axis, ylabel in (
        (build_axis, "Build time (s)"),
        (memory_axis, "Logical index memory (MiB)"),
    ):
        axis.set_xscale("log")
        configure_fixed_log_ticks(axis, SIZE_VALUES, ("100k", "250k", "500k", "1M"))
        axis.set_xlabel("Dataset size")
        axis.set_ylabel(ylabel)
        style_axis(axis)

    build_axis.legend(title="Build mode", loc="best")
    figure_header(
        figure,
        "SIFT dataset-size scaling",
        "M=16, M0=32 | ef_construction=200 | parallel: 11 workers",
    )
    save_figure(figure, output_dir / "size_scaling.svg")
    plt.close(figure)


def pq_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report["runs"]
        if row["m"] == 16 and row["m0"] == 32 and row["ef_construction"] == 200
    ]


def plot_pq_tradeoff(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    figure, (search_axis, memory_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, layout="constrained"
    )
    representations = (
        ("Normal", "pq_normal", "#222222", "o"),
        ("PQ-32", "pq32", "#0072B2", "s"),
        ("PQ-64", "pq64", "#D55E00", "^"),
        ("PQ-128", "pq128", "#009E73", "D"),
    )

    memory_labels = []
    memory_values = []
    memory_colors = []
    for label, report_name, color, marker in representations:
        rows = pq_rows(reports[report_name])
        require_search_sweep(rows, f"product quantization {label}")
        search_curve(search_axis, rows, color, marker, label)
        memory_labels.append(label)
        memory_values.append(float(rows[0]["memory_mib"]))
        memory_colors.append(color)

    memory_axis.bar(memory_labels, memory_values, color=memory_colors, width=0.7)
    memory_axis.set_ylabel("Logical index memory (MiB)")
    memory_axis.tick_params(axis="x", rotation=0)
    search_axis.set_xlabel("Recall@10")
    search_axis.set_ylabel("QPS")
    configure_qps_axis(search_axis)
    style_axis(search_axis)
    style_axis(memory_axis)
    handles, labels = search_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Representation",
        loc="outside lower center",
        ncol=4,
    )
    figure_header(
        figure,
        "Normal HNSW versus product quantization on SIFT-1M",
        "M=16, M0=32 | ef_construction=200 | ef_search points: 16, 32, 64, 128, 256, 512",
    )
    save_figure(figure, output_dir / "pq_tradeoff.svg")
    plt.close(figure)


def summary(rows: list[dict[str, Any]], field: str) -> tuple[float, float, float]:
    values = sorted(float(row[field]) for row in rows)
    return median(values), values[0], values[-1]


def error_bars(summaries: list[tuple[float, float, float]]) -> list[list[float]]:
    return [
        [median_value - minimum for median_value, minimum, _ in summaries],
        [maximum - median_value for median_value, _, maximum in summaries],
    ]


def parallel_rows(
    report: dict[str, Any], mode: str, build_threads: int | None = None
) -> list[dict[str, Any]]:
    return [
        row
        for row in report["runs"]
        if row.get("build_mode") == mode
        and (build_threads is None or row.get("build_threads") == build_threads)
    ]


def plot_parallel_construction(
    reports: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    report = reports["parallel_construction"]
    requested_threads = [
        int(row["build_threads"])
        for row in report["runs"]
        if row.get("build_mode") in ("dynamic", "batched")
    ]
    if not requested_threads:
        raise SystemExit("parallel construction report has no parallel rows")
    worker_counts = tuple(sorted(set(requested_threads)))
    expected_worker_counts = (1, 2, 4, 8, 11)
    if worker_counts != expected_worker_counts:
        raise SystemExit(
            "parallel construction expected workers "
            f"{expected_worker_counts}, found {worker_counts}"
        )

    full_machine_threads = max(requested_threads)
    rows_by_mode = {
        "sequential": parallel_rows(report, "sequential"),
        "dynamic": parallel_rows(report, "dynamic", full_machine_threads),
        "batched": parallel_rows(report, "batched", full_machine_threads),
    }
    if any(not rows for rows in rows_by_mode.values()):
        raise SystemExit("parallel construction report is missing full-machine rows")
    if len(rows_by_mode["sequential"]) != 3:
        raise SystemExit("parallel construction expected 3 sequential repetitions")
    for mode in ("dynamic", "batched"):
        for workers in expected_worker_counts:
            count = len(parallel_rows(report, mode, workers))
            if count != 3:
                raise SystemExit(
                    f"parallel construction expected 3 {mode}-{workers} repetitions, "
                    f"found {count}"
                )

    build_summaries = [summary(rows_by_mode[mode], "build_time_s") for mode in MODES]
    build_medians = [item[0] for item in build_summaries]
    baseline = build_medians[0]
    speedup_summaries = [(1.0, 1.0, 1.0)]
    speedup_summaries.extend(
        (baseline / median_time, baseline / maximum, baseline / minimum)
        for median_time, minimum, maximum in build_summaries[1:]
    )
    speedups = [item[0] for item in speedup_summaries]

    figure, (overview_axis, scaling_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, layout="constrained"
    )
    labels = [MODE_LABELS[mode] for mode in MODES]
    colors = [MODE_COLORS[mode] for mode in MODES]
    overview_axis.bar(
        labels,
        speedups,
        yerr=error_bars(speedup_summaries),
        color=colors,
        capsize=4,
        error_kw={"elinewidth": 1, "capthick": 1},
    )
    for index, value in enumerate(speedups):
        overview_axis.text(
            index,
            speedup_summaries[index][2] * 1.04,
            f"{value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    overview_axis.axhline(1.0, color="#222222", linewidth=0.9, linestyle="--")
    overview_axis.set_ylim(0.0, max(item[2] for item in speedup_summaries) * 1.18)
    overview_axis.set_title(
        f"Mode comparison (parallel: {full_machine_threads} workers)"
    )
    overview_axis.set_ylabel("Speedup vs sequential")
    overview_axis.tick_params(axis="x", rotation=20)
    style_axis(overview_axis)

    for mode in ("dynamic", "batched"):
        rows = parallel_rows(report, mode)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["build_threads"]), []).append(row)
        thread_counts = sorted(grouped)
        values = [summary(grouped[threads], "insert_qps") for threads in thread_counts]
        scaling_axis.errorbar(
            thread_counts,
            [item[0] for item in values],
            yerr=error_bars(values),
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            capsize=4,
            label=MODE_LABELS[mode],
        )

    sequential_median, _, _ = summary(rows_by_mode["sequential"], "insert_qps")
    scaling_axis.axhline(
        sequential_median,
        color=MODE_COLORS["sequential"],
        linestyle="--",
        linewidth=1.2,
        label="Sequential baseline",
    )
    scaling_axis.set_title("Worker scaling")
    scaling_axis.set_xlabel("Requested worker count")
    scaling_axis.set_ylabel("Construction throughput (inserts/s)")
    scaling_axis.set_xticks(
        sorted(
            {
                int(row["build_threads"])
                for row in report["runs"]
                if row.get("build_mode") in ("dynamic", "batched")
            }
        )
    )
    scaling_axis.legend(title="Build mode", loc="best")
    style_axis(scaling_axis)

    effective_threads = {
        int(row["effective_build_threads"])
        for mode in ("dynamic", "batched")
        for row in rows_by_mode[mode]
    }
    if len(effective_threads) != 1:
        raise SystemExit(
            "parallel construction rows disagree on effective worker count"
        )
    if effective_threads.pop() != full_machine_threads:
        raise SystemExit(
            "parallel construction requested and effective workers disagree"
        )
    figure_header(
        figure,
        "Parallel construction performance on SIFT-1M",
        "M=16, M0=32 | ef_construction=128 | median and min-max over 3 builds",
    )
    save_figure(figure, output_dir / "parallel_construction.svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for the SVG files",
    )
    args = parser.parse_args()

    configure_style()
    reports = load_reports()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_search_tradeoff(reports, args.output_dir)
    plot_construction_tradeoff(reports, args.output_dir)
    plot_size_scaling(reports, args.output_dir)
    plot_pq_tradeoff(reports, args.output_dir)
    plot_parallel_construction(reports, args.output_dir)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
