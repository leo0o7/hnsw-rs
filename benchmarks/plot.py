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
LINE_WIDTH = 1.8
MARKER_SIZE = 5.5

REPORT_PATHS = {
    "search_sift": "measure/m-sweep/sift-1m-normal.json",
    "search_mnist": "measure/m-sweep/mnist-60k-normal.json",
    "construction_build": "build/ef-construction-sweep/sift-1m.json",
    "construction_measure": "measure/ef-construction-sweep/sift-1m-normal.json",
    "size_build_100k": "build/size-sweep/sift-100k.json",
    "size_build_250k": "build/size-sweep/sift-250k.json",
    "size_build_500k": "build/size-sweep/sift-500k.json",
    "size_build_1m": "build/size-sweep/sift-1m.json",
    "size_measure_100k": "measure/size-sweep/sift-100k-normal.json",
    "size_measure_250k": "measure/size-sweep/sift-250k-normal.json",
    "size_measure_500k": "measure/size-sweep/sift-500k-normal.json",
    "size_measure_1m": "measure/size-sweep/sift-1m-normal.json",
    "pq_normal": "measure/m-sweep/sift-1m-normal.json",
    "pq32": "measure/pq/m-sweep/sift-1m-pq32.json",
    "pq64": "measure/pq/m-sweep/sift-1m-pq64.json",
    "pq128": "measure/pq/m-sweep/sift-1m-pq128.json",
}
OPTIONAL_REPORT_PATHS = {
    "parallel_construction": "build/parallel-construction/sift-1m.json",
}

GRAPH_COLORS = {
    6: "#0072B2",
    16: "#D55E00",
    32: "#009E73",
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

    report_paths = [
        (name, relative_path, True) for name, relative_path in REPORT_PATHS.items()
    ]
    report_paths.extend(
        (name, relative_path, False)
        for name, relative_path in OPTIONAL_REPORT_PATHS.items()
    )
    for name, relative_path, required in report_paths:
        path = RESULTS / relative_path
        if not path.is_file():
            if required:
                raise SystemExit(f"missing benchmark report: {path}")
            continue
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
            if name == "parallel_construction":
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
        for index, m in enumerate((6, 16, 32)):
            rows = selected_graph_rows(report, m)
            search_curve(
                axis, rows, GRAPH_COLORS[m], MARKERS[index], f"M={m}, M0={2 * m}"
            )
        axis.set_title(title)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("QPS")
        configure_qps_axis(axis)
        style_axis(axis)

    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("Graph degree search trade-off")
    figure.savefig(
        output_dir / "search_tradeoff.svg", format="svg", bbox_inches="tight"
    )
    plt.close(figure)


def plot_construction_tradeoff(
    reports: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    figure, (search_axis, build_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, layout="constrained"
    )
    measure_rows = reports["construction_measure"]["runs"]
    build_rows = reports["construction_build"]["runs"]

    construction_values = (32, 64, 100, 128, 200, 400)
    for index, value in enumerate(construction_values):
        rows = [
            row
            for row in measure_rows
            if row["m"] == 16 and row["m0"] == 32 and row["ef_construction"] == value
        ]
        search_curve(
            search_axis, rows, CONSTRUCTION_COLORS[value], MARKERS[index], str(value)
        )

    build_rows = sorted(build_rows, key=lambda row: int(row["ef_construction"]))
    build_axis.plot(
        [row["ef_construction"] for row in build_rows],
        [row["build_time_s"] for row in build_rows],
        color="#0072B2",
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
    )
    search_axis.set_xlabel("Recall@10")
    search_axis.set_ylabel("QPS")
    configure_qps_axis(search_axis)
    build_axis.set_xlabel("ef_construction")
    build_axis.set_ylabel("Build time (s)")
    build_axis.set_xscale("log")
    construction_labels = tuple(str(value) for value in construction_values)
    configure_fixed_log_ticks(build_axis, construction_values, construction_labels)
    build_axis.tick_params(axis="x", labelrotation=25)
    for label in build_axis.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axis(search_axis)
    style_axis(build_axis)
    handles, labels = search_axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, title="ef_construction", loc="outside lower center", ncol=3
    )
    figure.suptitle("Construction effort trade-off on SIFT-1M")
    figure.savefig(
        output_dir / "construction_tradeoff.svg", format="svg", bbox_inches="tight"
    )
    plt.close(figure)


def plot_size_scaling(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    figure, (build_axis, memory_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, layout="constrained"
    )
    sizes = (100_000, 250_000, 500_000, 1_000_000)
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

    build_times = [reports[name]["runs"][0]["build_time_s"] for name in build_names]
    memories = [reports[name]["runs"][0]["memory_mib"] for name in measure_names]

    build_axis.plot(sizes, build_times, color="#0072B2", marker="o")
    memory_axis.plot(sizes, memories, color="#D55E00", marker="s")
    for axis, ylabel in (
        (build_axis, "Build time (s)"),
        (memory_axis, "Logical index memory (MiB)"),
    ):
        axis.set_xscale("log")
        configure_fixed_log_ticks(axis, sizes, ("100k", "250k", "500k", "1M"))
        axis.set_xlabel("Dataset size")
        axis.set_ylabel(ylabel)
        style_axis(axis)

    figure.suptitle("SIFT dataset-size scaling")
    figure.savefig(output_dir / "size_scaling.svg", format="svg", bbox_inches="tight")
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
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.suptitle("Normal HNSW versus product quantization on SIFT-1M")
    figure.savefig(output_dir / "pq_tradeoff.svg", format="svg", bbox_inches="tight")
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
        and (
            build_threads is None or row.get("build_threads") == build_threads
        )
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

    full_machine_threads = max(requested_threads)
    rows_by_mode = {
        "sequential": parallel_rows(report, "sequential"),
        "dynamic": parallel_rows(report, "dynamic", full_machine_threads),
        "batched": parallel_rows(report, "batched", full_machine_threads),
    }
    if any(not rows for rows in rows_by_mode.values()):
        raise SystemExit("parallel construction report is missing full-machine rows")

    modes = ("sequential", "dynamic", "batched")
    labels = ("Sequential", "Dynamic", "Batched")
    colors = ("#222222", "#D55E00", "#009E73")
    build_summaries = [summary(rows_by_mode[mode], "build_time_s") for mode in modes]
    recall_summaries = [summary(rows_by_mode[mode], "recall") for mode in modes]
    build_medians = [item[0] for item in build_summaries]
    recall_medians = [item[0] for item in recall_summaries]
    baseline = build_medians[0]

    figure, (build_axis, recall_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, layout="constrained"
    )
    build_axis.bar(
        labels,
        build_medians,
        yerr=error_bars(build_summaries),
        color=colors,
        capsize=4,
        error_kw={"elinewidth": 1, "capthick": 1},
    )
    for index, value in enumerate(build_medians):
        build_axis.text(
            index,
            build_summaries[index][2] * 1.04,
            f"{baseline / value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    recall_axis.bar(
        labels,
        recall_medians,
        yerr=error_bars(recall_summaries),
        color=colors,
        capsize=4,
        error_kw={"elinewidth": 1, "capthick": 1},
    )
    recall_axis.axhline(
        recall_medians[0], color="#222222", linewidth=0.9, linestyle="--"
    )
    build_axis.set_ylabel("Build time (s)")
    recall_axis.set_ylabel("Recall@10")
    recall_axis.set_ylim(
        max(0.0, min(item[1] for item in recall_summaries) - 0.02),
        min(1.0, max(item[2] for item in recall_summaries) + 0.01),
    )
    for axis in (build_axis, recall_axis):
        axis.tick_params(axis="x", rotation=20)
        style_axis(axis)
    effective_threads = {
        int(row["effective_build_threads"])
        for mode in ("dynamic", "batched")
        for row in rows_by_mode[mode]
    }
    if len(effective_threads) != 1:
        raise SystemExit("parallel construction rows disagree on effective worker count")
    figure.suptitle(
        f"Parallel construction on SIFT-1M ({effective_threads.pop()} workers)"
    )
    figure.savefig(
        output_dir / "parallel_construction.svg", format="svg", bbox_inches="tight"
    )
    plt.close(figure)


def plot_parallel_scaling(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    report = reports["parallel_construction"]
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    colors = {"dynamic": "#D55E00", "batched": "#009E73"}
    labels = {"dynamic": "Dynamic", "batched": "Batched"}
    for mode in ("dynamic", "batched"):
        rows = parallel_rows(report, mode)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["build_threads"]), []).append(row)
        thread_counts = sorted(grouped)
        values = [summary(grouped[threads], "insert_qps") for threads in thread_counts]
        medians = [item[0] for item in values]
        axis.errorbar(
            thread_counts,
            medians,
            yerr=error_bars(values),
            color=colors[mode],
            marker="o" if mode == "dynamic" else "s",
            capsize=4,
            label=labels[mode],
        )

    sequential = parallel_rows(report, "sequential")
    sequential_median, _, _ = summary(sequential, "insert_qps")
    axis.axhline(
        sequential_median,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="Sequential baseline",
    )
    axis.set_xlabel("Requested worker count")
    axis.set_ylabel("Construction throughput (inserts/s)")
    axis.set_xticks(
        sorted(
            {
                int(row["build_threads"])
                for row in report["runs"]
                if row.get("build_mode") in ("dynamic", "batched")
            }
        )
    )
    axis.legend(loc="best")
    style_axis(axis)
    figure.suptitle("Parallel construction scaling on SIFT-1M")
    figure.savefig(output_dir / "parallel_scaling.svg", format="svg", bbox_inches="tight")
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
    if "parallel_construction" in reports:
        plot_parallel_construction(reports, args.output_dir)
        plot_parallel_scaling(reports, args.output_dir)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
