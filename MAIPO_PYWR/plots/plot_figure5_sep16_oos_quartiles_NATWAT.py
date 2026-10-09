#!/usr/bin/env python3
"""Recreate manuscript Figure 5 with Sep16 policies and 108 OOS climates.

For every period, drought indicator, scenario, and performance objective, the
best value across all archived Sep16 policies is calculated first. These
scenario-level optima are then averaged within the lowest and highest climate
quartiles (27 scenarios each) for final-20-year temperature, precipitation,
streamflow, and glacier melt. The same end-of-century climate groups are used
in both period columns. No Pywr or Borg run is performed.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_diff_climate_oos_pareto_frontier_NATWAT as climate
import plot_diff_climate_oos_pareto_frontier_four_quartiles_NATWAT as quartiles


POLICY_RUN = "sep16_decost"
RESULTS_FOLDER = "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
PERIODS = ("A", "B")
PERIOD_LABELS = {"A": "2020–2040", "B": "2080–2100"}
INDICATORS = ("SRI3", "SRI6", "SRI12", "SPI3", "SPI6", "SPI12")
OBJECTIVES = (
    ("cost", "Total Cost (M$)", "min"),
    ("urban", "Urban Reliability", "max"),
    ("agriculture", "Agricultural Reliability", "max"),
)
VARIABLES = (
    ("T", "Temperature"),
    ("P", "Precipitation"),
    ("S", "Streamflow"),
    ("G", "Glacier melt"),
)
VARIABLE_COLORS = {
    "T": "#994455",
    "P": "#6699cc",
    "S": "#004488",
    "G": "#997700",
}
N_SCENARIOS = 108


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=script_dir / RESULTS_FOLDER
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=script_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--temperature-folder", default="knn_weap_transient_no_drought_pert"
    )
    parser.add_argument(
        "--precipitation-folder", default="knn_weap_transient_no_drought_pert"
    )
    parser.add_argument("--temperature-prefix", default="temp_transient_")
    parser.add_argument("--precipitation-prefix", default="precip_transient_")
    parser.add_argument("--climate-file-suffix", default="_1979_2099")
    parser.add_argument(
        "--glacier-file",
        default="CMIP6_Glacier_Melt_108scenarios_no_drought_pert.csv",
    )
    parser.add_argument(
        "--period-b-oos-folder", default="transient_108_no_drought_pert_periodB"
    )
    parser.add_argument("--temperature-column", default="LBR_2375")
    parser.add_argument("--precipitation-column", default="LBR_2375")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def scalar_text(value):
    return str(np.asarray(value).reshape(-1)[0])


def load_cache(cache_dir, indicator, period):
    path = Path(cache_dir) / f"{indicator}_{period}_oos_performance.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed OOS cache: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "policies",
            "scenario_names",
            "urban_reliability",
            "agricultural_reliability",
            "total_cost",
            "completed",
            "indicator",
            "period",
            "policy_run",
            "num_scenarios",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Cache {path} is missing: {', '.join(sorted(missing))}")
        item = {
            "path": path,
            "policies": saved["policies"].copy(),
            "scenario_names": saved["scenario_names"].astype(str),
            "urban": saved["urban_reliability"].copy(),
            "agriculture": saved["agricultural_reliability"].copy(),
            "cost": saved["total_cost"].copy(),
            "completed": int(np.asarray(saved["completed"]).reshape(-1)[0]),
            "indicator": scalar_text(saved["indicator"]),
            "period": scalar_text(saved["period"]),
            "policy_run": scalar_text(saved["policy_run"]).lower(),
            "num_scenarios": int(
                np.asarray(saved["num_scenarios"]).reshape(-1)[0]
            ),
        }

    if (item["indicator"], item["period"]) != (indicator, period):
        raise ValueError(f"Cache identity mismatch: {path}")
    if item["policy_run"] != POLICY_RUN:
        raise ValueError(
            f"Cache {path} has policy_run={item['policy_run']!r}; "
            f"expected {POLICY_RUN!r}"
        )
    if item["num_scenarios"] != N_SCENARIOS:
        raise ValueError(f"Expected 108 scenarios in {path}")
    if item["scenario_names"].shape != (N_SCENARIOS,):
        raise ValueError(f"Expected 108 scenario names in {path}")
    if len(set(item["scenario_names"].tolist())) != N_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {path}")
    policies = item["policies"]
    if policies.ndim != 2 or policies.shape[1] != 4:
        raise ValueError(f"Expected n-by-4 policies in {path}; got {policies.shape}")
    if item["completed"] != len(policies):
        raise ValueError(f"Incomplete cache {path}")
    expected_shape = (len(policies), N_SCENARIOS)
    for key in ("cost", "urban", "agriculture"):
        values = item[key]
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {key} array in {path}")
    return item


def align_cache(cache_item, reference_names):
    lookup = {
        climate.normalized_scenario_name(name): index
        for index, name in enumerate(cache_item["scenario_names"])
    }
    normalized_reference = [
        climate.normalized_scenario_name(name) for name in reference_names
    ]
    if len(lookup) != N_SCENARIOS or len(set(normalized_reference)) != N_SCENARIOS:
        raise ValueError("Normalized scenario names must remain unique")
    missing = [name for name in normalized_reference if name not in lookup]
    extra = sorted(set(lookup).difference(normalized_reference))
    if missing or extra:
        raise ValueError(
            f"Scenario mismatch in {cache_item['path']}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    indices = [lookup[name] for name in normalized_reference]
    for key in ("cost", "urban", "agriculture"):
        cache_item[key] = cache_item[key][:, indices]
    cache_item["scenario_names"] = np.asarray(reference_names, dtype=str)


def calculate_climate_metrics(args, scenario_names):
    data_root = args.data_root.expanduser().resolve()
    temperature_dir = climate.resolved_under(data_root, args.temperature_folder)
    precipitation_dir = climate.resolved_under(data_root, args.precipitation_folder)
    glacier_file = climate.resolved_under(data_root, args.glacier_file)
    streamflow_dir = climate.resolved_under(data_root, args.period_b_oos_folder)
    temperature, temperature_sources, missing_temperature = climate.calculate_raw_metric(
        temperature_dir,
        args.temperature_prefix,
        scenario_names,
        args.temperature_column,
        allow_missing=False,
        file_suffix=args.climate_file_suffix,
    )
    precipitation, precipitation_sources, missing_precipitation = (
        climate.calculate_raw_metric(
            precipitation_dir,
            args.precipitation_prefix,
            scenario_names,
            args.precipitation_column,
            allow_missing=False,
            file_suffix=args.climate_file_suffix,
        )
    )
    if missing_temperature or missing_precipitation:
        raise RuntimeError("All 108 temperature and precipitation files are required")
    metrics = {
        "temperature": temperature,
        "precipitation": precipitation,
        "streamflow": climate.calculate_streamflow(streamflow_dir, scenario_names),
        "glacier": climate.calculate_glacier(glacier_file, scenario_names),
    }
    return metrics, temperature_sources, precipitation_sources


def best_scenario_performance(caches):
    best = {}
    for period in PERIODS:
        for indicator in INDICATORS:
            item = caches[(period, indicator)]
            best[(period, indicator, "cost")] = item["cost"].min(axis=0) / 1e6
            best[(period, indicator, "urban")] = item["urban"].max(axis=0)
            best[(period, indicator, "agriculture")] = item["agriculture"].max(axis=0)
    return best


def summarize(best, scenario_names, membership):
    name_to_index = {str(name): index for index, name in enumerate(scenario_names)}
    rows = []
    values = {}
    for objective, _, direction in OBJECTIVES:
        for period in PERIODS:
            for indicator in INDICATORS:
                scenario_values = best[(period, indicator, objective)]
                for code, label in VARIABLES:
                    for tail, quartile in (("low", "Q1"), ("high", "Q4")):
                        group = f"{code}_{quartile}"
                        names = membership[group]
                        indices = [name_to_index[str(name)] for name in names]
                        mean_value = float(np.mean(scenario_values[indices]))
                        values[(objective, period, indicator, code, tail)] = mean_value
                        rows.append(
                            {
                                "objective": objective,
                                "objective_direction": direction,
                                "period": period,
                                "period_label": PERIOD_LABELS[period],
                                "indicator": indicator,
                                "climate_code": code,
                                "climate_variable": label,
                                "climate_tail": tail,
                                "quartile": quartile,
                                "scenario_count": len(indices),
                                "mean_scenario_best_performance": mean_value,
                            }
                        )
    return values, rows


def padded_bounds(values, objective):
    array = np.asarray(values, dtype=float)
    lower = float(array.min())
    upper = float(array.max())
    span = upper - lower
    if span <= 0.0:
        span = max(abs(lower), 1.0) * 0.1
    pad = 0.08 * span
    lower -= pad
    upper += pad
    if objective in ("urban", "agriculture"):
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    return lower, upper


def add_panel_label(axis, label):
    axis.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-27, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        annotation_clip=False,
    )


def require_matplotlib_panel_alignment(fig, axes, output_base, tolerance_pt=1.5):
    fig.canvas.draw()
    axes = np.asarray(axes)
    if axes.shape != (3, 2):
        raise RuntimeError(f"Expected a 3-by-2 axes grid; got {axes.shape}")
    width_pt, height_pt = fig.get_size_inches() * 72.0
    boxes = []
    for row in range(3):
        for column in range(2):
            position = axes[row, column].get_position()
            boxes.append(
                {
                    "id": chr(ord("a") + row * 2 + column),
                    "row": row,
                    "column": column,
                    "left": float(position.x0 * width_pt),
                    "bottom": float(position.y0 * height_pt),
                    "right": float(position.x1 * width_pt),
                    "top": float(position.y1 * height_pt),
                    "width": float(position.width * width_pt),
                    "height": float(position.height * height_pt),
                }
            )
    deviations = {
        "max_width_difference": max(box["width"] for box in boxes)
        - min(box["width"] for box in boxes),
        "max_height_difference": max(box["height"] for box in boxes)
        - min(box["height"] for box in boxes),
        "max_row_top_difference": max(
            abs(boxes[row * 2]["top"] - boxes[row * 2 + 1]["top"])
            for row in range(3)
        ),
        "max_column_left_difference": max(
            max(boxes[row * 2 + column]["left"] for row in range(3))
            - min(boxes[row * 2 + column]["left"] for row in range(3))
            for column in range(2)
        ),
    }
    passed = all(value <= tolerance_pt for value in deviations.values())
    report = {
        "schema_version": 1,
        "figure_size_pt": [float(width_pt), float(height_pt)],
        "tolerance_pt": tolerance_pt,
        "panels": boxes,
        "deviations_pt": deviations,
        "verdict": "PASS" if passed else "FIX BEFORE DELIVERY",
    }
    Path(f"{output_base}.alignment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"3-by-2 panel alignment failed: {deviations}")


def plot_figure(values, output_base, dpi):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), sharex=True)
    x_positions = np.arange(len(INDICATORS), dtype=float)
    offsets = {"T": -0.24, "P": -0.08, "S": 0.08, "G": 0.24}

    for row, (objective, ylabel, _) in enumerate(OBJECTIVES):
        row_values = [
            value
            for key, value in values.items()
            if key[0] == objective
        ]
        y_bounds = padded_bounds(row_values, objective)
        for column, period in enumerate(PERIODS):
            axis = axes[row, column]
            for indicator_index, indicator in enumerate(INDICATORS):
                for code, _ in VARIABLES:
                    x_value = x_positions[indicator_index] + offsets[code]
                    low = values[(objective, period, indicator, code, "low")]
                    high = values[(objective, period, indicator, code, "high")]
                    axis.vlines(
                        x_value,
                        min(low, high),
                        max(low, high),
                        color="0.25",
                        alpha=0.72,
                        linewidth=0.65,
                        zorder=1,
                    )
                    axis.scatter(
                        x_value,
                        high,
                        s=18,
                        color=VARIABLE_COLORS[code],
                        alpha=1.0,
                        edgecolors="none",
                        zorder=3,
                    )
                    axis.scatter(
                        x_value,
                        low,
                        s=18,
                        color=VARIABLE_COLORS[code],
                        alpha=0.42,
                        edgecolors="none",
                        zorder=2,
                    )
            axis.set_ylim(*y_bounds)
            axis.set_xlim(-0.55, len(INDICATORS) - 0.45)
            axis.grid(axis="y", color="0.88", linewidth=0.55)
            axis.set_axisbelow(True)
            if row == 0:
                axis.set_title(PERIOD_LABELS[period], fontsize=8)
            if column == 0:
                axis.set_ylabel(ylabel)
            else:
                axis.tick_params(axis="y", labelleft=False)
            if row == 2:
                axis.set_xticks(x_positions, INDICATORS)
            else:
                axis.tick_params(axis="x", labelbottom=False)
            add_panel_label(axis, chr(ord("a") + row * 2 + column))

    legend_handles = []
    for code, label in VARIABLES:
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=4.5,
                    color=VARIABLE_COLORS[code],
                    alpha=1.0,
                    label=f"High {label} (Q4)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=4.5,
                    color=VARIABLE_COLORS[code],
                    alpha=0.42,
                    label=f"Low {label} (Q1)",
                ),
            ]
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        columnspacing=1.1,
        handletextpad=0.35,
        fontsize=6.5,
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.12, top=0.955, hspace=0.23, wspace=0.12
    )
    require_matplotlib_panel_alignment(fig, axes, output_base)
    fig.savefig(f"{output_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    fig.savefig(f"{output_base}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_climate_outputs(
    output_dir,
    scenario_names,
    metrics,
    membership,
    rank_rows,
    quartile_summary_rows,
    temperature_sources,
    precipitation_sources,
):
    classification_dir = Path(output_dir) / "climate_classification"
    metric_rows = []
    for index, scenario in enumerate(scenario_names):
        metric_rows.append(
            {
                "scenario": str(scenario),
                "temperature_source": str(temperature_sources[index]),
                "precipitation_source": str(precipitation_sources[index]),
                "final_20yr_temperature_LBR_2375": metrics["temperature"][index],
                "final_20yr_precipitation_LBR_2375": metrics["precipitation"][index],
                "final_20yr_streamflow": metrics["streamflow"][index],
                "final_20yr_glacier_times_0pt017": metrics["glacier"][index],
            }
        )
    membership_rows = [
        {"climate_group": group, "scenario": str(scenario)}
        for group in quartiles.GROUP_ORDER
        for scenario in membership[group]
    ]
    paths = (
        classification_dir / "oos_climate_scenario_metrics.csv",
        classification_dir / "oos_climate_four_quartile_assignments.csv",
        classification_dir / "oos_climate_four_quartile_summary.csv",
        classification_dir / "oos_climate_four_quartile_membership.csv",
    )
    for path, rows in zip(
        paths,
        (metric_rows, rank_rows, quartile_summary_rows, membership_rows),
    ):
        write_csv(path, rows)
    return paths


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else results_dir / "out_of_sample" / "simulation_cache"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "out_of_sample" / "figure5_quartile_performance"
    )
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory missing: {results_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"OOS cache directory missing: {cache_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    caches = {
        (period, indicator): load_cache(cache_dir, indicator, period)
        for period in PERIODS
        for indicator in INDICATORS
    }
    reference_names = caches[("B", INDICATORS[0])]["scenario_names"].astype(str)
    for item in caches.values():
        align_cache(item, reference_names)

    print("Calculating final-20-year OOS climate metrics...", flush=True)
    metrics, temperature_sources, precipitation_sources = calculate_climate_metrics(
        args, reference_names
    )
    membership, rank_rows, quartile_summary_rows = quartiles.stable_four_quartiles(
        reference_names, metrics
    )
    for code, _ in VARIABLES:
        if len(membership[f"{code}_Q1"]) != 27 or len(membership[f"{code}_Q4"]) != 27:
            raise RuntimeError(f"Expected exactly 27 scenarios in {code} Q1 and Q4")

    classification_paths = write_climate_outputs(
        output_dir,
        reference_names,
        metrics,
        membership,
        rank_rows,
        quartile_summary_rows,
        temperature_sources,
        precipitation_sources,
    )

    best = best_scenario_performance(caches)
    values, summary_rows = summarize(best, reference_names, membership)
    summary_path = output_dir / "figure5_sep16_oos_quartile_performance.csv"
    write_csv(summary_path, summary_rows)

    figure_base = output_dir / "figure5_sep16_oos_high_low_quartile_performance"
    plot_figure(values, figure_base, args.dpi)

    method_path = output_dir / "figure5_sep16_oos_method.txt"
    method_path.write_text(
        "Sep16 OOS Figure 5 method\n\n"
        "All unique policies contained in each completed Sep16 OOS cache are "
        "used. For every period, indicator, and scenario, the minimum total "
        "cost, maximum urban reliability, and maximum agricultural reliability "
        "are calculated independently across policies. Thus, the policy "
        "attaining the best value may differ among objectives and scenarios. "
        "The resulting scenario-level best values are then averaged within "
        "each climate group.\n\n"
        "Climate groups are based on the shifted final-20-year temperature, "
        "precipitation, streamflow, and glacier-melt metrics. Scenarios are "
        "sorted ascending by metric value and then scenario name. Ranks 1-27 "
        "form Q1 (low) and ranks 82-108 form Q4 (high). The same end-of-century "
        "groups are applied to both the 2020-2040 and 2080-2100 performance "
        "columns. Scenario matching is by name, not column position. No model "
        "simulation or optimization is rerun.\n",
        encoding="utf-8",
    )

    print(f"Policy run : {POLICY_RUN}")
    print(f"Cache      : {cache_dir}")
    print(f"Output     : {output_dir}")
    for code, label in VARIABLES:
        print(
            f"  {label}: Q1={len(membership[f'{code}_Q1'])}, "
            f"Q4={len(membership[f'{code}_Q4'])}"
        )
    print("Saved:")
    for path in classification_paths:
        print(f"  {path}")
    for path in (summary_path, method_path):
        print(f"  {path}")
    for suffix in (".png", ".pdf", ".svg", ".tiff", ".alignment.json"):
        print(f"  {figure_base}{suffix}")


if __name__ == "__main__":
    main()
