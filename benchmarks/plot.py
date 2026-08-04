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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for the four SVG files",
    )
    args = parser.parse_args()

    configure_style()
    reports = load_reports()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_search_tradeoff(reports, args.output_dir)
    plot_construction_tradeoff(reports, args.output_dir)
    plot_size_scaling(reports, args.output_dir)
    plot_pq_tradeoff(reports, args.output_dir)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
