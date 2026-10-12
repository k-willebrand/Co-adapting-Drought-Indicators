#!/usr/bin/env python3
"""Plot Sep16 Period-B OOS Pareto sets for all four climate quartiles.

The script does not rerun Pywr. It reads the completed 108-scenario OOS
performance caches, calculates the same final-20-year precipitation,
temperature, streamflow, and glacier-melt metrics as the existing different-
climate workflow, and assigns every scenario to one of four equal-sized
quartiles. Ties are resolved reproducibly by scenario name, so Q1--Q4 each
contain exactly 27 scenarios. Objectives and three-objective non-dominance are
recomputed independently within every quartile.
"""

import csv
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

import plot_diff_climate_oos_pareto_frontier_NATWAT as fixed


POLICY_RUN = "sep16_decost"
RESULTS_FOLDER = "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
VARIABLES = (
    ("P", "Precipitation", "precipitation"),
    ("T", "Temperature", "temperature"),
    ("S", "Streamflow", "streamflow"),
    ("G", "Glacier melt", "glacier"),
)
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
GROUP_ORDER = tuple(
    f"{code}_{quartile}"
    for code, _, _ in VARIABLES
    for quartile in QUARTILES
)


mpl.rcParams.update(
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


def stable_four_quartiles(scenario_names, metrics):
    """Return 27-scenario Q1--Q4 groups and auditable rank records."""
    names = np.asarray(scenario_names, dtype=str)
    if names.shape != (fixed.N_SCENARIOS,):
        raise ValueError(
            f"Expected {fixed.N_SCENARIOS} scenario names; got {names.shape}"
        )
    if len(set(names.tolist())) != fixed.N_SCENARIOS:
        raise ValueError("Scenario names must be unique before quartile ranking")

    group_size = fixed.N_SCENARIOS // 4
    if group_size * 4 != fixed.N_SCENARIOS:
        raise ValueError("Scenario count is not divisible into four equal quartiles")

    membership = {}
    rank_rows = []
    summary_rows = []
    for code, label, metric_name in VARIABLES:
        values = np.asarray(metrics[metric_name], dtype=float)
        if values.shape != (fixed.N_SCENARIOS,):
            raise ValueError(
                f"Expected {fixed.N_SCENARIOS} {metric_name} values; got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            missing = names[~np.isfinite(values)]
            raise ValueError(
                f"Cannot rank {metric_name}: non-finite values for "
                + ", ".join(missing.tolist())
            )

        # np.lexsort uses the last key as primary: metric ascending, then name.
        order = np.lexsort((names, values))
        assigned = []
        for quartile_index, quartile in enumerate(QUARTILES):
            start = quartile_index * group_size
            stop = start + group_size
            selected_indices = order[start:stop]
            group = f"{code}_{quartile}"
            selected_names = tuple(names[selected_indices].tolist())
            membership[group] = selected_names
            assigned.extend(selected_names)

            selected_values = values[selected_indices]
            summary_rows.append(
                {
                    "climate_code": code,
                    "climate_variable": metric_name,
                    "climate_label": label,
                    "quartile": quartile,
                    "group": group,
                    "scenario_count": len(selected_indices),
                    "rank_start": start + 1,
                    "rank_end": stop,
                    "metric_min": float(selected_values.min()),
                    "metric_max": float(selected_values.max()),
                }
            )
            for within_index, scenario_index in enumerate(selected_indices, start=1):
                rank_rows.append(
                    {
                        "climate_code": code,
                        "climate_variable": metric_name,
                        "quartile": quartile,
                        "group": group,
                        "overall_rank": start + within_index,
                        "rank_within_quartile": within_index,
                        "scenario": str(names[scenario_index]),
                        "metric_value": float(values[scenario_index]),
                    }
                )

        if len(assigned) != fixed.N_SCENARIOS or len(set(assigned)) != fixed.N_SCENARIOS:
            raise RuntimeError(f"{metric_name} quartiles do not partition all scenarios")
        if any(len(membership[f"{code}_{q}"]) != group_size for q in QUARTILES):
            raise RuntimeError(f"{metric_name} quartiles are not all size {group_size}")

    return membership, rank_rows, summary_rows


def write_classification_tables(
    output_dir,
    scenario_names,
    metrics,
    membership,
    rank_rows,
    summary_rows,
    temperature_sources,
    precipitation_sources,
):
    """Write standard metrics plus explicit four-quartile audit tables."""
    classification_dir = Path(output_dir) / "climate_classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    # Keep the existing wide metrics/membership files for compatibility.
    metrics_path, membership_path = fixed.write_climate_tables(
        output_dir,
        scenario_names,
        metrics,
        membership,
        temperature_sources,
        precipitation_sources,
    )

    rank_path = classification_dir / "oos_climate_four_quartile_assignments.csv"
    with rank_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)

    summary_path = classification_dir / "oos_climate_four_quartile_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    method_path = classification_dir / "oos_climate_four_quartile_method.txt"
    method_path.write_text(
        "Four-quartile climate grouping method\n\n"
        "For precipitation, temperature, streamflow, and glacier melt, the "
        "scenario-level metric is the same shifted final-20-year mean used by "
        "the existing different-climate workflow. All 108 OOS scenarios are "
        "sorted ascending by (metric value, scenario name). Sorted ranks 1-27, "
        "28-54, 55-81, and 82-108 define Q1, Q2, Q3, and Q4, respectively. "
        "Scenario name is used only as a deterministic tie-breaker. Therefore, "
        "the four groups are mutually exclusive, exhaustive, and each contains "
        "27 scenarios. Within each group and drought indicator, objectives are "
        "averaged across the 27 scenarios and three-objective non-dominance is "
        "recomputed using total cost (minimize), urban reliability (maximize), "
        "and agricultural reliability (maximize).\n",
        encoding="utf-8",
    )
    return metrics_path, membership_path, rank_path, summary_path, method_path


def panel_title(quartile, count):
    descriptions = {
        "Q1": "lowest 25%",
        "Q2": "25–50%",
        "Q3": "50–75%",
        "Q4": "highest 25%",
    }
    return f"{quartile}: {descriptions[quartile]} (n={count})"


def require_four_panel_alignment(fig, json_out, tolerance_pt=1.5):
    """Require equal plot-area geometry within rows and columns."""
    fig.canvas.draw()
    axes = [axis for axis in fig.axes if axis.get_visible()]
    if len(axes) != 4:
        raise RuntimeError(f"Expected four comparable axes; found {len(axes)}")

    width_pt, height_pt = fig.get_size_inches() * 72.0
    boxes = []
    for label, axis in zip(QUARTILES, axes):
        position = axis.get_position()
        boxes.append(
            {
                "id": label,
                "left": float(position.x0 * width_pt),
                "bottom": float(position.y0 * height_pt),
                "right": float(position.x1 * width_pt),
                "top": float(position.y1 * height_pt),
                "width": float(position.width * width_pt),
                "height": float(position.height * height_pt),
            }
        )
    deviations = {
        "max_width_difference": max(b["width"] for b in boxes)
        - min(b["width"] for b in boxes),
        "max_height_difference": max(b["height"] for b in boxes)
        - min(b["height"] for b in boxes),
        "top_row_top_difference": abs(boxes[0]["top"] - boxes[1]["top"]),
        "bottom_row_bottom_difference": abs(boxes[2]["bottom"] - boxes[3]["bottom"]),
        "left_column_left_difference": abs(boxes[0]["left"] - boxes[2]["left"]),
        "right_column_right_difference": abs(boxes[1]["right"] - boxes[3]["right"]),
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
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Four-panel alignment failed: {deviations}")


def save_figure(fig, output_base, dpi):
    output_base = Path(output_base)
    require_four_panel_alignment(
        fig, Path(str(output_base) + ".alignment.json"), tolerance_pt=1.5
    )
    fig.savefig(Path(str(output_base) + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(Path(str(output_base) + ".pdf"), bbox_inches="tight")
    fig.savefig(Path(str(output_base) + ".svg"), bbox_inches="tight")
    fixed.plt.close(fig)


def plot_urban_cost_grid(output_base, code, variable_label, membership, results, dpi):
    fig, axes = fixed.plt.subplots(
        2, 2, figsize=(7.2, 6.35), sharex=True, sharey=True
    )
    for panel_index, (axis, quartile) in enumerate(zip(axes.flat, QUARTILES)):
        group = f"{code}_{quartile}"
        for indicator in fixed.INDICATORS:
            item = results[(group, indicator)]
            data = item["objectives"][item["pareto_mask"]]
            axis.scatter(
                data[:, 0] / 1e6,
                data[:, 1],
                color=fixed.COLORS[indicator],
                marker=fixed.MARKERS[indicator],
                edgecolors=(
                    "none"
                    if fixed.MARKERS[indicator] == "o"
                    else fixed.COLORS[indicator]
                ),
                s=19,
                alpha=0.76,
            )
        axis.set_title(panel_title(quartile, len(membership[group])), fontsize=8)
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.text(
            -0.12,
            1.07,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    for axis in axes[1, :]:
        axis.set_xlabel("Mean total cost (M$)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean urban reliability")
    fig.suptitle(
        f"2080–2100 OOS Pareto sets by {variable_label} quartile",
        fontsize=10,
        y=0.985,
    )
    fig.legend(
        handles=fixed.indicator_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.085, top=0.84, hspace=0.25, wspace=0.12
    )
    save_figure(fig, output_base, dpi)


def plot_parallel_grid(
    output_base, code, variable_label, membership, results, bounds, dpi
):
    fig, axes = fixed.plt.subplots(2, 2, figsize=(8.0, 6.6), sharey=True)
    for panel_index, (axis, quartile) in enumerate(zip(axes.flat, QUARTILES)):
        group = f"{code}_{quartile}"
        fixed.draw_parallel_axes(axis, bounds)
        line_count = 0
        for indicator in fixed.INDICATORS:
            normalized = fixed.normalize_parallel(
                fixed.pareto_performance(results[(group, indicator)]), bounds
            )
            line_count += len(normalized)
            for row in normalized:
                axis.plot(
                    range(len(fixed.AXIS_NAMES)),
                    row,
                    color=fixed.COLORS[indicator],
                    linewidth=0.65,
                    alpha=0.24,
                    zorder=2,
                )
        axis.set_title(
            f"{panel_title(quartile, len(membership[group]))}; {line_count} policies",
            fontsize=8,
        )
        axis.text(
            -0.07,
            1.07,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    fig.suptitle(
        f"2080–2100 OOS three-objective Pareto policies by {variable_label} quartile",
        fontsize=10,
        y=0.988,
    )
    fig.legend(
        handles=fixed.indicator_handles(markers=False),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.928),
        ncol=6,
        frameon=False,
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.085, top=0.83, hspace=0.31, wspace=0.12
    )
    save_figure(fig, output_base, dpi)


def main():
    fixed.POLICY_RUN_FOLDERS[POLICY_RUN] = RESULTS_FOLDER
    fixed.GROUP_ORDER = GROUP_ORDER
    args = fixed.parse_args()
    if args.policy_run != POLICY_RUN:
        raise ValueError(
            f"This script is for --policy-run {POLICY_RUN!r}; got {args.policy_run!r}"
        )

    results_dir = Path(
        args.results_dir or fixed.SCRIPT_DIR / RESULTS_FOLDER
    ).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    cache_dir = Path(
        args.cache_dir or results_dir / "out_of_sample" / "simulation_cache"
    ).expanduser().resolve()
    output_dir = Path(
        args.output_dir
        or results_dir / "out_of_sample" / "diff_climates_four_quartiles"
    ).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory missing: {results_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"OOS simulation cache directory missing: {cache_dir}")

    temperature_dir = fixed.resolved_under(data_root, args.temperature_folder)
    precipitation_dir = fixed.resolved_under(data_root, args.precipitation_folder)
    glacier_file = fixed.resolved_under(data_root, args.glacier_file)
    streamflow_dir = fixed.resolved_under(data_root, args.period_b_oos_folder)
    if not glacier_file.is_file():
        raise FileNotFoundError(f"Glacier file missing: {glacier_file}")

    pareto_dir = output_dir / "pareto_data"
    urban_cost_dir = output_dir / "urban_reliability_vs_cost"
    parallel_dir = output_dir / "parallel_axes"
    for directory in (output_dir, pareto_dir, urban_cost_dir, parallel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    caches = {}
    reference_scenarios = None
    for indicator in fixed.INDICATORS:
        cache = fixed.load_oos_cache(cache_dir, indicator, POLICY_RUN)
        if reference_scenarios is None:
            reference_scenarios = cache["scenario_names"]
        elif not np.array_equal(reference_scenarios, cache["scenario_names"]):
            raise ValueError(f"OOS scenario names/order differ for B/{indicator}")
        caches[indicator] = cache

    print("Calculating final-20-year OOS climate metrics...", flush=True)
    temperature, temperature_sources, missing_temperature = fixed.calculate_raw_metric(
        temperature_dir,
        args.temperature_prefix,
        reference_scenarios,
        args.temperature_column,
        allow_missing=False,
        shuffled_fallback=args.temperature_prefix == "temp_raw_",
        file_suffix=args.climate_file_suffix,
    )
    precipitation, precipitation_sources, missing_precipitation = (
        fixed.calculate_raw_metric(
            precipitation_dir,
            args.precipitation_prefix,
            reference_scenarios,
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
        "streamflow": fixed.calculate_streamflow(streamflow_dir, reference_scenarios),
        "glacier": fixed.calculate_glacier(glacier_file, reference_scenarios),
    }
    membership, rank_rows, summary_rows = stable_four_quartiles(
        reference_scenarios, metrics
    )
    classification_paths = write_classification_tables(
        output_dir,
        reference_scenarios,
        metrics,
        membership,
        rank_rows,
        summary_rows,
        temperature_sources,
        precipitation_sources,
    )

    print(f"Policy run        : {POLICY_RUN}")
    print(f"Policy results    : {results_dir}")
    print(f"OOS cache         : {cache_dir}")
    print(f"P/T data          : {temperature_dir}")
    print(f"Streamflow data   : {streamflow_dir}")
    print(f"Glacier data      : {glacier_file}")
    print(f"Output            : {output_dir}")
    for group in GROUP_ORDER:
        print(f"  {group}: {len(membership[group])} OOS scenarios")

    results = fixed.build_group_results(caches, membership)
    saved_tables = []
    for group in GROUP_ORDER:
        path = pareto_dir / f"oos_periodB_{group}_pareto.csv"
        fixed.save_group_pareto(path, group, len(membership[group]), results)
        saved_tables.append(path)

    bounds = fixed.global_parallel_bounds(results)
    saved_figures = []
    for code, variable_label, _ in VARIABLES:
        urban_base = (
            urban_cost_dir / f"oos_periodB_four_quartiles_urban_cost_pareto_{code}"
        )
        parallel_base = (
            parallel_dir / f"oos_periodB_four_quartiles_parallel_pareto_{code}"
        )
        plot_urban_cost_grid(
            urban_base, code, variable_label, membership, results, args.dpi
        )
        plot_parallel_grid(
            parallel_base,
            code,
            variable_label,
            membership,
            results,
            bounds,
            args.dpi,
        )
        for base in (urban_base, parallel_base):
            saved_figures.extend(
                Path(str(base) + suffix)
                for suffix in (".png", ".pdf", ".svg", ".alignment.json")
            )

    print("Saved climate classification:")
    for path in classification_paths:
        print(f"  {path}")
    print(f"Saved {len(saved_tables)} quartile-specific Pareto tables in {pareto_dir}")
    print("Saved 2x2 quartile figures:")
    for path in saved_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
