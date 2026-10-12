"""Plot climate-conditioned Pareto sets for the NATWAT three-objective run.

This is stage two of the analysis. It classifies the 36 CMIP6 scenarios using
the same end-of-century 20-year climate-state definitions as manuscript Figure
5, loads scenario-level policy performance produced by
simulate_diff_climate_policies_NATWAT.py, and recomputes three-objective
non-dominance within each climate group.

Each climate group gets one six-panel figure: periods A/B by the three pairwise
projections of the true three-objective Pareto set. The six drought indicators
are overlaid in every panel.
"""

import argparse
import csv
import itertools
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INDICATORS = ("SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12")
PERIODS = ("A", "B")
PERIOD_LABELS = {"A": "2020–2040", "B": "2080–2100"}
MODELS = (
    "ACCESS-CM2",
    "CNRM-CM6-1",
    "INM-CM4-8",
    "INM-CM5-0",
    "MIROC6",
    "MIROC-ES2L",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-MM",
)
EXPERIMENTS = ("ssp126", "ssp245", "ssp370", "ssp585")
SCENARIOS = tuple(f"{model}_{experiment}" for model, experiment in itertools.product(MODELS, EXPERIMENTS))
STREAMFLOW_FILES = ("COLORADO", "MAIPO", "YESO", "LAGUNANEGRA", "VOLCAN", "MAIPOEXTRA")
ROLLING_WEEKS = 20 * 52

GROUPS = {
    "high_T": ("temperature", ">=", 10.0),
    "low_T": ("temperature", "<=", 8.0),
    "high_P": ("precipitation", ">=", 11.0),
    "low_P": ("precipitation", "<=", 9.0),
    "high_S": ("streamflow", ">=", 55.0),
    "low_S": ("streamflow", "<=", 40.0),
}
GROUP_ORDER = ("high_S", "low_S", "high_T", "low_T", "high_P", "low_P")

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
DEFAULT_RESULTS_DIR = os.path.join(
    SCRIPT_DIR, "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug24_MIN"
)
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "diff_cliamtes")
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "data")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--streamflow-folder",
        default="TDP_scenarios_DS_2paramGamma_periodB",
        help="end-of-century streamflow folder under data-root",
    )
    parser.add_argument("--temperature-precipitation-folder", default="UQM_weap_TempC_new")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def numeric_column(path, zero_based_column=3):
    values = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) <= zero_based_column:
            raise ValueError(
                f"Expected at least {zero_based_column + 1} columns in {path}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) <= zero_based_column:
                raise ValueError(f"Short row in {path}:{line_number}")
            try:
                values.append(float(row[zero_based_column]))
            except ValueError as exc:
                raise ValueError(f"Non-numeric climate value in {path}:{line_number}") from exc
    result = np.asarray(values, dtype=float)
    if result.size < ROLLING_WEEKS + 1:
        raise ValueError(
            f"Need at least {ROLLING_WEEKS + 1} weekly rows in {path}; got {result.size}"
        )
    return result


def shifted_final_rolling_mean(values):
    # Equivalent to series.shift().rolling(20*52).mean().iloc[-1].
    window = np.asarray(values[-(ROLLING_WEEKS + 1):-1], dtype=float)
    if window.size != ROLLING_WEEKS or not np.all(np.isfinite(window)):
        raise ValueError("The final 20-year rolling window is incomplete or non-finite")
    return float(np.mean(window))


def multi_scenario_matrix(path, scenario_names):
    columns = {name: [] for name in scenario_names}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        missing = [name for name in scenario_names if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} scenario columns in {path}: {', '.join(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            for name in scenario_names:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid value for {name} in {path}:{line_number}"
                    ) from exc
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"Scenario columns have inconsistent lengths in {path}")
    return np.column_stack([columns[name] for name in scenario_names])


def calculate_climate_metrics(data_root, tp_folder_name, streamflow_folder_name):
    tp_folder = os.path.join(data_root, tp_folder_name)
    streamflow_folder = os.path.join(data_root, streamflow_folder_name)
    if not os.path.isdir(tp_folder):
        raise FileNotFoundError(f"Temperature/precipitation folder missing: {tp_folder}")
    if not os.path.isdir(streamflow_folder):
        raise FileNotFoundError(f"Streamflow folder missing: {streamflow_folder}")

    temperature = np.empty(len(SCENARIOS), dtype=float)
    precipitation = np.empty(len(SCENARIOS), dtype=float)
    for index, scenario in enumerate(SCENARIOS):
        p_path = os.path.join(tp_folder, f"PP_semanal_{scenario}.csv")
        t_path = os.path.join(tp_folder, f"Tprom_semanal_{scenario}.csv")
        if not os.path.isfile(p_path):
            raise FileNotFoundError(f"Precipitation file missing: {p_path}")
        if not os.path.isfile(t_path):
            raise FileNotFoundError(f"Temperature file missing: {t_path}")
        precipitation[index] = shifted_final_rolling_mean(numeric_column(p_path))
        temperature[index] = shifted_final_rolling_mean(numeric_column(t_path))

    total_streamflow = None
    row_count = None
    for basename in STREAMFLOW_FILES:
        path = os.path.join(streamflow_folder, f"{basename}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Streamflow file missing: {path}")
        values = multi_scenario_matrix(path, SCENARIOS)
        if row_count is None:
            row_count = values.shape[0]
            total_streamflow = np.zeros_like(values, dtype=float)
        elif values.shape[0] != row_count:
            raise ValueError(f"Streamflow row count differs in {path}")
        total_streamflow += values

    streamflow = np.asarray(
        [shifted_final_rolling_mean(total_streamflow[:, index]) for index in range(len(SCENARIOS))]
    )
    return {
        "temperature": temperature,
        "precipitation": precipitation,
        "streamflow": streamflow,
    }


def classify_scenarios(metrics):
    membership = {}
    for group, (variable, operator, threshold) in GROUPS.items():
        values = metrics[variable]
        if operator == ">=":
            mask = values >= threshold
        else:
            mask = values <= threshold
        names = tuple(np.asarray(SCENARIOS)[mask].tolist())
        if not names:
            raise RuntimeError(f"Climate group {group} contains no scenarios")
        membership[group] = names
    return membership


def write_climate_tables(output_dir, metrics, membership):
    classification_dir = os.path.join(output_dir, "climate_classification")
    os.makedirs(classification_dir, exist_ok=True)
    metrics_path = os.path.join(classification_dir, "climate_scenario_metrics.csv")
    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scenario", "final_20yr_temperature", "final_20yr_precipitation", "final_20yr_streamflow"]
            + list(GROUP_ORDER)
        )
        for index, scenario in enumerate(SCENARIOS):
            writer.writerow(
                [scenario, metrics["temperature"][index], metrics["precipitation"][index], metrics["streamflow"][index]]
                + [int(scenario in membership[group]) for group in GROUP_ORDER]
            )

    membership_path = os.path.join(classification_dir, "climate_group_membership.csv")
    with open(membership_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["climate_group", "scenario"])
        for group in GROUP_ORDER:
            for scenario in membership[group]:
                writer.writerow([group, scenario])
    return metrics_path, membership_path


def load_policy_cache(cache_dir, indicator, period):
    path = os.path.join(cache_dir, f"{indicator}_{period}_policy_performance.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Simulation cache missing: {path}\nRun submit_diff_climate_simulation_NATWAT.sh first."
        )
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "policies",
            "scenario_names",
            "urban_reliability",
            "agricultural_reliability",
            "total_cost",
            "completed",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Cache {path} lacks: {', '.join(sorted(missing))}")
        policies = saved["policies"].copy()
        scenario_names = saved["scenario_names"].astype(str)
        urban = saved["urban_reliability"].copy()
        agriculture = saved["agricultural_reliability"].copy()
        cost = saved["total_cost"].copy()
        completed = int(saved["completed"])

    expected_shape = (len(policies), len(scenario_names))
    if completed != len(policies):
        raise ValueError(f"Incomplete cache {path}: {completed}/{len(policies)} policies")
    for name, values in (("urban", urban), ("agriculture", agriculture), ("cost", cost)):
        if values.shape != expected_shape:
            raise ValueError(f"Unexpected {name} shape {values.shape} in {path}; expected {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} values in {path}")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError(f"Duplicate scenario names in {path}")
    return {
        "policies": policies,
        "scenario_names": scenario_names,
        "urban": urban,
        "agriculture": agriculture,
        "cost": cost,
    }


def group_means(cache, scenario_group):
    name_to_index = {name: index for index, name in enumerate(cache["scenario_names"])}
    missing = [name for name in scenario_group if name not in name_to_index]
    if missing:
        raise ValueError("Cache is missing classified scenarios: " + ", ".join(missing))
    indices = [name_to_index[name] for name in scenario_group]
    return np.column_stack(
        (
            cache["cost"][:, indices].mean(axis=1),
            cache["urban"][:, indices].mean(axis=1),
            cache["agriculture"][:, indices].mean(axis=1),
        )
    )


def nondominated_mask(objectives):
    """Return 3-D Pareto mask for [cost (min), urban (max), agriculture (max)]."""
    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError("Expected an n-by-3 objective array")
    if not np.all(np.isfinite(objectives)):
        raise ValueError("Objective array contains non-finite values")
    minimization = np.column_stack((objectives[:, 0], -objectives[:, 1], -objectives[:, 2]))
    keep = np.ones(len(minimization), dtype=bool)
    for index, point in enumerate(minimization):
        dominates = np.all(minimization <= point, axis=1) & np.any(minimization < point, axis=1)
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return keep


def save_group_pareto(path, group, scenario_count, results):
    header = [
        "climate_group",
        "scenario_count",
        "period",
        "indicator",
        "policy_index",
        "contract_threshold",
        "contract_action",
        "demand_threshold",
        "demand_action",
        "mean_total_cost",
        "mean_total_cost_million",
        "mean_urban_reliability",
        "mean_agricultural_reliability",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for period in PERIODS:
            for indicator in INDICATORS:
                item = results[(period, indicator)]
                for policy_index in np.where(item["pareto_mask"])[0]:
                    policy = item["policies"][policy_index]
                    objective = item["objectives"][policy_index]
                    writer.writerow(
                        [group, scenario_count, period, indicator, int(policy_index)]
                        + policy.tolist()
                        + [objective[0], objective[0] / 1e6, objective[1], objective[2]]
                    )


def plot_group(path, group, scenario_count, results, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex="col", sharey="col")
    projections = (
        (0, 1, "Total Cost (M$)", "Urban Reliability"),
        (0, 2, "Total Cost (M$)", "Agricultural Reliability"),
        (1, 2, "Urban Reliability", "Agricultural Reliability"),
    )
    for row, period in enumerate(PERIODS):
        for indicator in INDICATORS:
            item = results[(period, indicator)]
            data = item["objectives"][item["pareto_mask"]].copy()
            data[:, 0] /= 1e6
            for column, (x_index, y_index, x_label, y_label) in enumerate(projections):
                axes[row, column].scatter(
                    data[:, x_index],
                    data[:, y_index],
                    color=COLORS[indicator],
                    marker=MARKERS[indicator],
                    edgecolors="none" if MARKERS[indicator] == "o" else COLORS[indicator],
                    s=26,
                    alpha=0.78,
                    label=indicator,
                )
                axes[row, column].set_xlabel(x_label)
                axes[row, column].set_ylabel(y_label)
                axes[row, column].grid(alpha=0.18, linewidth=0.6)
        axes[row, 0].annotate(
            PERIOD_LABELS[period],
            xy=(-0.22, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    variable, operator, threshold = GROUPS[group]
    pretty_variable = variable.capitalize()
    fig.suptitle(
        f"Three-objective Pareto sets under {group.replace('_', ' ')} scenarios "
        f"({pretty_variable} {operator} {threshold:g}; n={scenario_count})",
        fontsize=14,
        y=0.995,
    )
    handles = [
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
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=6, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.88, wspace=0.27, hspace=0.22)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)
    cache_dir = os.path.join(output_dir, "simulation_cache")
    pareto_dir = os.path.join(output_dir, "pareto_data")
    figures_dir = os.path.join(output_dir, "figures")
    for directory in (output_dir, pareto_dir, figures_dir):
        os.makedirs(directory, exist_ok=True)
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    print(f"Results directory : {results_dir}")
    print(f"Output directory  : {output_dir}")
    print("Calculating end-of-century climate states...")
    metrics = calculate_climate_metrics(
        data_root,
        args.temperature_precipitation_folder,
        args.streamflow_folder,
    )
    membership = classify_scenarios(metrics)
    metrics_path, membership_path = write_climate_tables(output_dir, metrics, membership)
    for group in GROUP_ORDER:
        print(f"  {group}: {len(membership[group])} scenarios")

    caches = {}
    for period in PERIODS:
        for indicator in INDICATORS:
            caches[(period, indicator)] = load_policy_cache(cache_dir, indicator, period)

    saved_figures = []
    saved_tables = []
    for group in GROUP_ORDER:
        results = {}
        for period in PERIODS:
            for indicator in INDICATORS:
                cache = caches[(period, indicator)]
                objectives = group_means(cache, membership[group])
                mask = nondominated_mask(objectives)
                results[(period, indicator)] = {
                    "policies": cache["policies"],
                    "objectives": objectives,
                    "pareto_mask": mask,
                }
                print(
                    f"{group:6s} {period}/{indicator}: {int(mask.sum())} Pareto policies "
                    f"of {len(mask)}"
                )

        table_path = os.path.join(pareto_dir, f"{group}_pareto.csv")
        figure_path = os.path.join(figures_dir, f"pareto_{group}.png")
        save_group_pareto(table_path, group, len(membership[group]), results)
        plot_group(figure_path, group, len(membership[group]), results, args.dpi)
        saved_tables.append(table_path)
        saved_figures.append(figure_path)

    print("Saved climate classification:")
    print(f"  {metrics_path}")
    print(f"  {membership_path}")
    print("Saved Pareto tables:")
    for path in saved_tables:
        print(f"  {path}")
    print("Saved figures:")
    for path in saved_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
