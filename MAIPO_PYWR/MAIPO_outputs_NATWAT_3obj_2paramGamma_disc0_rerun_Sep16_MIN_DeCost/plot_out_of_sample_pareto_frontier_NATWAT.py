"""Plot Pareto frontiers after evaluating policies on 108 OOS scenarios.

For every indicator-period cache, the three objectives are averaged over all
108 out-of-sample scenarios. Three-objective non-dominance is then recomputed
with total cost minimized and both reliabilities maximized. The script reports
both per-indicator Pareto sets and a joint set where all six indicators compete
within each period.
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


INDICATORS = ("SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12")
PERIODS = ("A", "B")
PERIOD_LABELS = {"A": "2020–2040", "B": "2080–2100"}
N_SCENARIOS = 108

COLORS = {
    "SPI3": "#cfe8ff",
    "SPI6": "#3d85c6",
    "SPI12": "#0b2e59",
    "SRI3": "#ffd9a0",
    "SRI6": "#e0842a",
    "SRI12": "#7a3d00",
}
MARKERS = {
    "SPI3": "o",
    "SPI6": "o",
    "SPI12": "o",
    "SRI3": "^",
    "SRI6": "^",
    "SRI12": "^",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_RUN_FOLDERS = {
    "aug24": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug24_MIN",
    "aug28": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug28_MIN",
    "sep03_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep3_MIN_DeCost",
    "sep09_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep9_MIN_DeCost",
    "sep16_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost",
}
NO_DROUGHT_OOS_DATA_FOLDERS = {
    "A": "transient_108_no_drought_pert_periodA",
    "B": "transient_108_no_drought_pert_periodB",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-run",
        choices=tuple(POLICY_RUN_FOLDERS),
        default="aug28",
        help=(
            "optimization/OOS policy run to plot (default: aug28); "
            "*_decost runs include demand restriction cost at $1/m3"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="optional explicit override for the folder selected by --policy-run",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: <selected-results-dir>/out_of_sample",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_cache(cache_dir, indicator, period, expected_policy_run):
    path = os.path.join(cache_dir, f"{indicator}_{period}_oos_performance.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing completed cache: {path}\n"
            "Run submit_out_of_sample_simulation_NATWAT.sh first."
        )
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
            "data_folder",
            "num_scenarios",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Cache {path} is missing: {', '.join(sorted(missing))}")
        policies = saved["policies"].copy()
        scenario_names = saved["scenario_names"].astype(str)
        urban = saved["urban_reliability"].copy()
        agriculture = saved["agricultural_reliability"].copy()
        cost = saved["total_cost"].copy()
        demand_cost = (
            saved["demand_restriction_cost"].copy()
            if "demand_restriction_cost" in saved.files
            else None
        )
        completed = int(saved["completed"])
        saved_indicator = str(saved["indicator"])
        saved_period = str(saved["period"])
        saved_policy_run = str(saved["policy_run"]).lower()
        saved_data_folder = str(saved["data_folder"])
        saved_num_scenarios = int(saved["num_scenarios"])

    if saved_indicator != indicator or saved_period != period:
        raise ValueError(
            f"Cache identity mismatch in {path}: metadata is "
            f"{saved_indicator}/{saved_period}, expected {indicator}/{period}"
        )
    if saved_policy_run != expected_policy_run:
        raise ValueError(
            f"Cache {path} contains policy_run={saved_policy_run!r}; "
            f"expected {expected_policy_run!r}"
        )
    if expected_policy_run in {"sep09_decost", "sep16_decost"}:
        expected_data_folder = NO_DROUGHT_OOS_DATA_FOLDERS[period]
        if saved_data_folder != expected_data_folder:
            raise ValueError(
                f"Cache {path} was generated from data_folder={saved_data_folder!r}; "
                f"expected {expected_data_folder!r}"
            )
    if saved_num_scenarios != N_SCENARIOS:
        raise ValueError(
            f"Cache metadata in {path} says {saved_num_scenarios} scenarios; expected 108"
        )
    if scenario_names.size != N_SCENARIOS:
        raise ValueError(f"{path} contains {scenario_names.size} scenarios; expected 108")
    if len(set(scenario_names)) != N_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {path}")
    expected = (len(policies), N_SCENARIOS)
    if policies.ndim != 2 or policies.shape[1] != 4:
        raise ValueError(f"Expected n-by-4 policies in {path}; got {policies.shape}")
    if completed != len(policies):
        raise ValueError(f"Incomplete cache {path}: {completed}/{len(policies)} policies")
    for name, values in (("urban", urban), ("agriculture", agriculture), ("cost", cost)):
        if values.shape != expected:
            raise ValueError(f"Unexpected {name} shape {values.shape} in {path}; expected {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} values in {path}")
    if demand_cost is not None:
        if demand_cost.shape != expected:
            raise ValueError(
                f"Unexpected demand-cost shape {demand_cost.shape} in {path}; "
                f"expected {expected}"
            )
        if not np.all(np.isfinite(demand_cost)):
            raise ValueError(f"Non-finite demand-restriction cost values in {path}")

    # Objective order used everywhere below: cost, urban reliability, agricultural reliability.
    objectives = np.column_stack(
        (cost.mean(axis=1), urban.mean(axis=1), agriculture.mean(axis=1))
    )
    return {
        "path": path,
        "policies": policies,
        "scenario_names": scenario_names,
        "objectives": objectives,
        "mean_demand_restriction_cost": (
            demand_cost.mean(axis=1) if demand_cost is not None else None
        ),
    }


def nondominated_mask(objectives):
    """Return mask for cost=min, urban=max, agriculture=max."""
    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError("Expected an n-by-3 objective array")
    if not np.all(np.isfinite(objectives)):
        raise ValueError("Objectives contain non-finite values")
    minimized = np.column_stack((objectives[:, 0], -objectives[:, 1], -objectives[:, 2]))
    keep = np.ones(len(minimized), dtype=bool)
    for index, point in enumerate(minimized):
        dominates = np.all(minimized <= point, axis=1) & np.any(minimized < point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return keep


def assign_joint_masks(results):
    for period in PERIODS:
        sizes = [len(results[(period, indicator)]["objectives"]) for indicator in INDICATORS]
        combined = np.vstack(
            [results[(period, indicator)]["objectives"] for indicator in INDICATORS]
        )
        joint = nondominated_mask(combined)
        start = 0
        for indicator, size in zip(INDICATORS, sizes):
            results[(period, indicator)]["joint_mask"] = joint[start:start + size]
            start += size


def write_summary_csv(path, results):
    header = [
        "period",
        "indicator",
        "policy_index",
        "contract_threshold",
        "contract_action",
        "demand_threshold",
        "demand_action",
        "mean_total_cost",
        "mean_total_cost_million",
        "mean_demand_restriction_cost",
        "mean_demand_restriction_cost_million",
        "mean_urban_reliability",
        "mean_agricultural_reliability",
        "indicator_pareto",
        "joint_pareto",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for period in PERIODS:
            for indicator in INDICATORS:
                item = results[(period, indicator)]
                for index, (policy, objective) in enumerate(
                    zip(item["policies"], item["objectives"])
                ):
                    demand_cost = item["mean_demand_restriction_cost"]
                    mean_demand_cost = (
                        float(demand_cost[index]) if demand_cost is not None else float("nan")
                    )
                    writer.writerow(
                        [period, indicator, index]
                        + policy.tolist()
                        + [
                            objective[0],
                            objective[0] / 1e6,
                            mean_demand_cost,
                            mean_demand_cost / 1e6,
                            objective[1],
                            objective[2],
                            int(item["indicator_mask"][index]),
                            int(item["joint_mask"][index]),
                        ]
                    )


def legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker=MARKERS[indicator],
            linestyle="none",
            color=COLORS[indicator],
            markerfacecolor=COLORS[indicator],
            markersize=7,
            label=indicator,
        )
        for indicator in INDICATORS
    ]


def require_matplotlib_panel_alignment(fig, output_base, tolerance_pt=1.5):
    """Require aligned plot areas within each subplot row and column."""
    fig.canvas.draw()
    axes = [axis for axis in fig.axes if axis.get_visible()]
    if len(axes) < 2:
        raise RuntimeError("Expected at least two visible Pareto panels")
    width_pt, height_pt = fig.get_size_inches() * 72.0
    panels = []
    row_groups = {}
    column_groups = {}
    for index, axis in enumerate(axes):
        position = axis.get_position()
        spec = axis.get_subplotspec()
        row = int(spec.rowspan.start)
        column = int(spec.colspan.start)
        panel = {
            "id": f"panel_{index + 1}",
            "row": row,
            "column": column,
            "left": float(position.x0 * width_pt),
            "bottom": float(position.y0 * height_pt),
            "right": float(position.x1 * width_pt),
            "top": float(position.y1 * height_pt),
            "width": float(position.width * width_pt),
            "height": float(position.height * height_pt),
        }
        panels.append(panel)
        row_groups.setdefault(row, []).append(panel)
        column_groups.setdefault(column, []).append(panel)

    deviations = {}
    for row, group in row_groups.items():
        if len(group) > 1:
            for field in ("top", "bottom", "width", "height"):
                values = [panel[field] for panel in group]
                deviations[f"row_{row}_{field}"] = max(values) - min(values)
    for column, group in column_groups.items():
        if len(group) > 1:
            for field in ("left", "right", "width", "height"):
                values = [panel[field] for panel in group]
                deviations[f"column_{column}_{field}"] = max(values) - min(values)
    passed = all(value <= tolerance_pt for value in deviations.values())
    report = {
        "schema_version": 1,
        "figure_size_pt": [float(width_pt), float(height_pt)],
        "tolerance_pt": tolerance_pt,
        "panels": panels,
        "deviations_pt": deviations,
        "verdict": "PASS" if passed else "FIX BEFORE DELIVERY",
    }
    with open(output_base + ".alignment.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not passed:
        raise RuntimeError(
            f"Pareto panel alignment exceeds {tolerance_pt} pt: {deviations}"
        )


def save_figure_bundle(fig, path, dpi):
    output_base, _ = os.path.splitext(path)
    require_matplotlib_panel_alignment(fig, output_base)
    fig.savefig(output_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base + ".pdf", bbox_inches="tight")
    fig.savefig(output_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_pairwise(path, results, mask_name, title, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex="col", sharey="col")
    projections = (
        (0, 1, "Total Cost (M$)", "Urban Reliability"),
        (0, 2, "Total Cost (M$)", "Agricultural Reliability"),
        (1, 2, "Urban Reliability", "Agricultural Reliability"),
    )
    for row, period in enumerate(PERIODS):
        for indicator in INDICATORS:
            item = results[(period, indicator)]
            data = item["objectives"][item[mask_name]].copy()
            data[:, 0] /= 1e6
            for column, (x_index, y_index, x_label, y_label) in enumerate(projections):
                axes[row, column].scatter(
                    data[:, x_index],
                    data[:, y_index],
                    s=28,
                    alpha=0.8,
                    color=COLORS[indicator],
                    marker=MARKERS[indicator],
                    edgecolors="none",
                )
                axes[row, column].set_xlabel(x_label)
                axes[row, column].set_ylabel(y_label)
                axes[row, column].grid(alpha=0.18, linewidth=0.6)
        axes[row, 0].annotate(
            PERIOD_LABELS[period],
            xy=(-0.22, 0.5),
            xycoords="axes fraction",
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.88, wspace=0.27, hspace=0.22)
    save_figure_bundle(fig, path, dpi)


def plot_cost_urban(path, results, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    for axis, period in zip(axes, PERIODS):
        for indicator in INDICATORS:
            item = results[(period, indicator)]
            data = item["objectives"][item["indicator_mask"]]
            axis.scatter(
                data[:, 0] / 1e6,
                data[:, 1],
                s=30,
                alpha=0.8,
                color=COLORS[indicator],
                marker=MARKERS[indicator],
                edgecolors="none",
            )
        axis.set_title(PERIOD_LABELS[period])
        axis.set_xlabel("Mean Total Cost (M$)")
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("Mean Urban Reliability")
    fig.suptitle("OOS 3-objective Pareto sets: cost–urban projection", fontsize=13)
    axes[1].legend(
        handles=legend_handles(),
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.86, wspace=0.16)
    save_figure_bundle(fig, path, dpi)


def main():
    args = parse_args()
    selected_results_dir = os.path.join(SCRIPT_DIR, POLICY_RUN_FOLDERS[args.policy_run])
    results_dir = os.path.abspath(args.results_dir or selected_results_dir)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(results_dir, "out_of_sample")
    )
    cache_dir = os.path.join(output_dir, "simulation_cache")
    pareto_dir = os.path.join(output_dir, "pareto_data")
    figures_dir = os.path.join(output_dir, "figures")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    for directory in (output_dir, pareto_dir, figures_dir):
        os.makedirs(directory, exist_ok=True)

    print(f"Policy run: {args.policy_run}")
    print(f"Results   : {results_dir}")
    print(f"OOS output: {output_dir}")

    results = {}
    for period in PERIODS:
        reference_scenarios = None
        for indicator in INDICATORS:
            item = load_cache(cache_dir, indicator, period, args.policy_run)
            if args.policy_run.endswith("_decost") and item[
                "mean_demand_restriction_cost"
            ] is None:
                raise ValueError(
                    f"DeCost cache lacks DemandRestrictionCost: {item['path']}"
                )
            if reference_scenarios is None:
                reference_scenarios = item["scenario_names"]
            elif not np.array_equal(reference_scenarios, item["scenario_names"]):
                raise ValueError(
                    f"Scenario names/order differ among Period {period} caches; "
                    f"first mismatch is {indicator}"
                )
            item["indicator_mask"] = nondominated_mask(item["objectives"])
            results[(period, indicator)] = item
            print(
                f"{period}/{indicator}: {int(item['indicator_mask'].sum())} "
                f"indicator-Pareto policies of {len(item['policies'])}"
            )

    assign_joint_masks(results)
    for period in PERIODS:
        counts = ", ".join(
            f"{indicator}={int(results[(period, indicator)]['joint_mask'].sum())}"
            for indicator in INDICATORS
        )
        print(f"Period {period} joint Pareto composition: {counts}")

    summary_path = os.path.join(pareto_dir, "out_of_sample_policy_means_and_pareto.csv")
    per_indicator_path = os.path.join(
        figures_dir, "oos_3obj_pareto_pairwise_by_period.png"
    )
    joint_path = os.path.join(
        figures_dir, "oos_3obj_joint_pareto_pairwise_by_period.png"
    )
    cost_urban_path = os.path.join(
        figures_dir, "oos_3obj_pareto_cost_urban.png"
    )
    write_summary_csv(summary_path, results)
    plot_pairwise(
        per_indicator_path,
        results,
        "indicator_mask",
        "Out-of-sample performance: per-indicator 3-objective Pareto sets (108 scenarios)",
        args.dpi,
    )
    plot_pairwise(
        joint_path,
        results,
        "joint_mask",
        "Out-of-sample performance: joint 3-objective Pareto set (108 scenarios)",
        args.dpi,
    )
    plot_cost_urban(cost_urban_path, results, args.dpi)

    print("Saved:")
    for path in (summary_path, per_indicator_path, joint_path, cost_urban_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
