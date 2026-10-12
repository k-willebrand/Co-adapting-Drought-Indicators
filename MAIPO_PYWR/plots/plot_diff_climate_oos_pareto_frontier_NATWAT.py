#!/usr/bin/env python3
"""Plot Period-B OOS Pareto sets conditioned on test-set P, T, S, and G.

This script does not rerun Pywr. It reads completed OOS performance caches,
classifies the same 108 test scenarios from their own climate data, and
recomputes per-indicator 3-D non-dominance within every high/low climate group.
For each climate variable it writes paired high/low urban-cost and parallel-axis
figures. Scenario matching is always name-based. In particular, the transient
P/T files omit the ``_no_drought_pert`` suffix used by the OOS caches and wide
streamflow/glacier tables; that known suffix difference is normalized explicitly.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_diff_climate_pareto_frontier_NATWAT as training


INDICATORS = tuple(training.INDICATORS)
PERIOD = "B"
N_SCENARIOS = 108
ROLLING_WEEKS = 20 * 52
STREAMFLOW_FILES = tuple(training.STREAMFLOW_FILES)
COLORS = training.COLORS
MARKERS = training.MARKERS

GROUPS = {
    "high_P": ("precipitation", ">=", 11.0),
    "low_P": ("precipitation", "<=", 9.0),
    "high_T": ("temperature", ">=", 10.0),
    "low_T": ("temperature", "<=", 8.0),
    "high_S": ("streamflow", ">=", 55.0),
    "low_S": ("streamflow", "<=", 40.0),
    "high_G": ("glacier", ">", 5.0),
    "low_G": ("glacier", "<", 0.1),
}
GROUP_PAIRS = (
    ("P", "Precipitation", "high_P", "low_P"),
    ("T", "Temperature", "high_T", "low_T"),
    ("S", "Streamflow", "high_S", "low_S"),
    ("G", "Glacier melt", "high_G", "low_G"),
)
GROUP_ORDER = tuple(group for _, _, high, low in GROUP_PAIRS for group in (high, low))

AXIS_NAMES = ("Total Cost", "Urban Reliability", "Agricultural Reliability")
AXIS_LABELS = (
    "Total Cost (M$)\n(minimize)",
    "Urban Reliability\n(maximize)",
    "Agricultural Reliability\n(maximize)",
)
FLIP_AXIS = np.asarray((True, False, False), dtype=bool)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = (
    SCRIPT_DIR / "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug28_MIN"
)
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
POLICY_RUN_FOLDERS = {
    "aug28": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug28_MIN",
    "sep09_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep9_MIN_DeCost",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-run", choices=tuple(POLICY_RUN_FOLDERS), default="aug28"
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--temperature-folder", default="syn_temp_raw")
    parser.add_argument("--precipitation-folder", default="syn_prep_raw")
    parser.add_argument("--temperature-prefix", default="temp_raw_")
    parser.add_argument("--precipitation-prefix", default="precip_raw_")
    parser.add_argument(
        "--climate-file-suffix",
        default="",
        help="suffix between the scenario name and .csv, e.g. _1979_2099",
    )
    parser.add_argument(
        "--glacier-file", default="CMIP6_Glacier_Melt_108scenarios.csv"
    )
    parser.add_argument("--period-b-oos-folder", default="TDP_108_periodB")
    parser.add_argument("--temperature-column", default="LBR_2375")
    parser.add_argument("--precipitation-column", default="LBR_2375")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="default: <results-dir>/out_of_sample/simulation_cache",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: <results-dir>/out_of_sample/diff_climates",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def resolved_under(root, path_text):
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def tail_data_lines(path, number):
    """Return final non-empty CSV data lines without scanning the full file."""
    path = Path(path)
    chunks = []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        newline_count = 0
        while position > 0 and newline_count <= number:
            read_size = min(1024 * 1024, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    lines = [line for line in b"".join(reversed(chunks)).splitlines() if line.strip()]
    if len(lines) < number:
        raise ValueError(f"Need at least {number} data rows in {path}")
    return lines[-number:]


def read_header(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"Missing CSV header: {path}")
    return header


def shifted_final_column_mean(path, column_name):
    """Equivalent to df[column].shift().rolling(1040).mean().iloc[-1]."""
    header = read_header(path)
    if column_name not in header:
        raise ValueError(f"Column {column_name!r} is missing from {path}")
    column = header.index(column_name)
    lines = tail_data_lines(path, ROLLING_WEEKS + 1)
    values = np.empty(ROLLING_WEEKS, dtype=float)
    for index, raw in enumerate(lines[:-1]):
        fields = raw.decode("utf-8-sig").split(",")
        try:
            values[index] = float(fields[column])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid {column_name!r} near end of {path}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite final 20-year values in {path}")
    return float(values.mean())


def normalized_scenario_name(name):
    """Return the common scenario identity used across OOS data products."""
    name = str(name)
    if name.endswith("_no_drought_pert"):
        name = name[: -len("_no_drought_pert")]
    if name.endswith("_v2"):
        name = name[:-3]
    return name


def raw_file_map(folder, prefix, file_suffix=""):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Raw climate folder missing: {folder}")
    mapping = {}
    for path in sorted(folder.glob(f"{prefix}*.csv")):
        scenario = path.name[len(prefix) : -4]
        if file_suffix:
            if not scenario.endswith(file_suffix):
                continue
            scenario = scenario[: -len(file_suffix)]
        key = normalized_scenario_name(scenario)
        if key in mapping:
            raise ValueError(f"Duplicate raw climate file for {key} in {folder}")
        mapping[key] = path
    return mapping


def shuffled_temperature_name(scenario):
    scenario = str(scenario)
    if scenario.endswith("_v2"):
        scenario = scenario[:-3]
    return f"{scenario}_1979_2099"


def calculate_raw_metric(
    folder, prefix, scenario_names, column_name, allow_missing,
    shuffled_fallback=False, file_suffix="",
):
    paths = raw_file_map(folder, prefix, file_suffix=file_suffix)
    shuffled_paths = raw_file_map(folder, "temp_shuffled_") if shuffled_fallback else {}
    metrics = np.full(len(scenario_names), np.nan, dtype=float)
    sources = np.full(len(scenario_names), "", dtype=object)
    missing = []
    for index, scenario in enumerate(scenario_names):
        path = paths.get(normalized_scenario_name(scenario))
        if path is None and shuffled_fallback:
            path = shuffled_paths.get(shuffled_temperature_name(scenario))
        if path is None:
            missing.append(str(scenario))
            continue
        metrics[index] = shifted_final_column_mean(path, column_name)
        sources[index] = str(path)
    if missing and not allow_missing:
        detail = "\n".join(f"  {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} required {prefix.rstrip('_')} files:\n{detail}"
        )
    return metrics, sources, tuple(missing)


def shifted_final_wide_means(path, scenario_names):
    """Read final shifted 20-year means for named columns in one wide CSV."""
    header = read_header(path)
    missing = [str(name) for name in scenario_names if str(name) not in header]
    if missing:
        detail = "\n".join(f"  {name}" for name in missing)
        raise ValueError(f"{path} is missing {len(missing)} scenario columns:\n{detail}")
    columns = [header.index(str(name)) for name in scenario_names]
    lines = tail_data_lines(path, ROLLING_WEEKS + 1)
    values = np.empty((ROLLING_WEEKS, len(columns)), dtype=float)
    row_keys = []
    key_columns = min(columns)
    for row_index, raw in enumerate(lines[:-1]):
        fields = raw.decode("utf-8-sig").split(",")
        row_keys.append(tuple(fields[:key_columns]))
        try:
            values[row_index] = [float(fields[column]) for column in columns]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid final 20-year scenario value in {path}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite final 20-year scenario values in {path}")
    return values.mean(axis=0), tuple(row_keys)


def calculate_streamflow(folder, scenario_names):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"OOS streamflow folder missing: {folder}")
    total = np.zeros(len(scenario_names), dtype=float)
    reference_keys = None
    for basename in STREAMFLOW_FILES:
        path = folder / f"{basename}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"OOS streamflow file missing: {path}")
        means, row_keys = shifted_final_wide_means(path, scenario_names)
        if reference_keys is None:
            reference_keys = row_keys
        elif row_keys != reference_keys:
            raise ValueError(f"Final 20-year row labels differ in {path}")
        total += means
    return total


def calculate_glacier(path, scenario_names):
    means, _ = shifted_final_wide_means(path, scenario_names)
    return means * 0.017


def apply_operator(values, operator, threshold):
    finite = np.isfinite(values)
    if operator == ">=":
        return finite & (values >= threshold)
    if operator == "<=":
        return finite & (values <= threshold)
    if operator == ">":
        return finite & (values > threshold)
    if operator == "<":
        return finite & (values < threshold)
    raise ValueError(f"Unsupported operator: {operator}")


def classify_scenarios(scenario_names, metrics):
    names = np.asarray(scenario_names, dtype=str)
    membership = {}
    for group in GROUP_ORDER:
        variable, operator, threshold = GROUPS[group]
        selected = tuple(names[apply_operator(metrics[variable], operator, threshold)].tolist())
        if not selected:
            raise RuntimeError(
                f"OOS climate group {group} is empty under "
                f"{variable} {operator} {threshold:g}"
            )
        membership[group] = selected
    return membership


def display_value(value):
    return "" if not np.isfinite(value) else f"{value:.12g}"


def write_climate_tables(
    output_dir,
    scenario_names,
    metrics,
    membership,
    temperature_sources,
    precipitation_sources,
):
    classification_dir = Path(output_dir) / "climate_classification"
    classification_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = classification_dir / "oos_climate_scenario_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "temperature_source",
                "precipitation_source",
                "final_20yr_temperature_LBR_2375",
                "final_20yr_precipitation_LBR_2375",
                "final_20yr_streamflow",
                "final_20yr_glacier_times_0pt017",
                "temperature_available",
            ]
            + list(GROUP_ORDER)
        )
        for index, scenario in enumerate(scenario_names):
            writer.writerow(
                [
                    scenario,
                    temperature_sources[index],
                    precipitation_sources[index],
                    display_value(metrics["temperature"][index]),
                    display_value(metrics["precipitation"][index]),
                    display_value(metrics["streamflow"][index]),
                    display_value(metrics["glacier"][index]),
                    int(np.isfinite(metrics["temperature"][index])),
                ]
                + [int(str(scenario) in membership[group]) for group in GROUP_ORDER]
            )

    membership_path = classification_dir / "oos_climate_group_membership.csv"
    with membership_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["climate_group", "scenario"])
        for group in GROUP_ORDER:
            for scenario in membership[group]:
                writer.writerow([group, scenario])
    return metrics_path, membership_path


def load_oos_cache(cache_dir, indicator, expected_policy_run):
    path = Path(cache_dir) / f"{indicator}_{PERIOD}_oos_performance.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"OOS cache missing: {path}\nRun the selected policy set's OOS simulation first."
        )
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "policies", "scenario_names", "urban_reliability",
            "agricultural_reliability", "total_cost", "completed", "num_scenarios",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Cache {path} is missing: {', '.join(sorted(missing))}")
        item = {
            "policies": saved["policies"].copy(),
            "scenario_names": saved["scenario_names"].astype(str),
            "urban": saved["urban_reliability"].copy(),
            "agriculture": saved["agricultural_reliability"].copy(),
            "cost": saved["total_cost"].copy(),
            "completed": int(saved["completed"]),
            "num_scenarios": int(saved["num_scenarios"]),
            "path": path,
        }
        if "policy_run" in saved.files:
            saved_policy_run = str(np.asarray(saved["policy_run"]).reshape(-1)[0]).lower()
            if saved_policy_run != expected_policy_run:
                raise ValueError(
                    f"Cache {path} has policy_run={saved_policy_run!r}; "
                    f"expected {expected_policy_run!r}"
                )
        elif expected_policy_run != "aug28":
            raise ValueError(f"Cache {path} lacks required policy_run metadata")
    policies = item["policies"]
    scenarios = item["scenario_names"]
    if item["num_scenarios"] != N_SCENARIOS or len(scenarios) != N_SCENARIOS:
        raise ValueError(f"Expected 108 scenarios in {path}; got {len(scenarios)}")
    if len(set(scenarios)) != N_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {path}")
    if policies.ndim != 2 or policies.shape[1] != 4:
        raise ValueError(f"Expected n-by-4 policies in {path}; got {policies.shape}")
    if item["completed"] != len(policies):
        raise ValueError(f"Incomplete cache {path}: {item['completed']}/{len(policies)}")
    expected = (len(policies), N_SCENARIOS)
    for name in ("urban", "agriculture", "cost"):
        if item[name].shape != expected or not np.all(np.isfinite(item[name])):
            raise ValueError(f"Invalid {name} array in {path}")
    return item


def group_objectives(cache, scenario_group):
    lookup = {str(name): index for index, name in enumerate(cache["scenario_names"])}
    missing = [name for name in scenario_group if name not in lookup]
    if missing:
        raise ValueError(f"Cache {cache['path']} lacks: {', '.join(missing)}")
    indices = [lookup[name] for name in scenario_group]
    return np.column_stack(
        (
            cache["cost"][:, indices].mean(axis=1),
            cache["urban"][:, indices].mean(axis=1),
            cache["agriculture"][:, indices].mean(axis=1),
        )
    )


def build_group_results(caches, membership):
    results = {}
    for group in GROUP_ORDER:
        for indicator in INDICATORS:
            cache = caches[indicator]
            objectives = group_objectives(cache, membership[group])
            mask = training.nondominated_mask(objectives)
            results[(group, indicator)] = {
                "policies": cache["policies"],
                "objectives": objectives,
                "pareto_mask": mask,
            }
            print(
                f"{group:6s} B/{indicator}: {int(mask.sum())} Pareto policies "
                f"of {len(mask)}", flush=True,
            )
    return results


def save_group_pareto(path, group, scenario_count, results):
    header = [
        "climate_group", "scenario_count", "period", "indicator", "policy_index",
        "contract_threshold", "contract_action", "demand_threshold", "demand_action",
        "mean_total_cost", "mean_total_cost_million", "mean_urban_reliability",
        "mean_agricultural_reliability",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for indicator in INDICATORS:
            item = results[(group, indicator)]
            for policy_index in np.flatnonzero(item["pareto_mask"]):
                policy = item["policies"][policy_index]
                objective = item["objectives"][policy_index]
                writer.writerow(
                    [group, scenario_count, PERIOD, indicator, int(policy_index)]
                    + policy.tolist()
                    + [objective[0], objective[0] / 1e6, objective[1], objective[2]]
                )


def indicator_handles(markers=True):
    return [
        Line2D(
            [0], [0], marker=MARKERS[indicator] if markers else None,
            linestyle="none" if markers else "-", color=COLORS[indicator],
            markerfacecolor=COLORS[indicator], linewidth=2.5, markersize=7,
            label=indicator,
        )
        for indicator in INDICATORS
    ]


def group_title(group, count):
    side = "High" if group.startswith("high_") else "Low"
    variable, operator, threshold = GROUPS[group]
    return f"{side} {variable.replace('_', ' ')} ({operator} {threshold:g}; n={count})"


def save_figure_pair(fig, output_base, dpi):
    fig.savefig(Path(str(output_base) + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(Path(str(output_base) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_urban_cost_pair(output_base, variable_label, high, low, membership, results, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True, sharey=True)
    for axis, group in zip(axes, (high, low)):
        for indicator in INDICATORS:
            item = results[(group, indicator)]
            data = item["objectives"][item["pareto_mask"]]
            axis.scatter(
                data[:, 0] / 1e6, data[:, 1], color=COLORS[indicator],
                marker=MARKERS[indicator],
                edgecolors="none" if MARKERS[indicator] == "o" else COLORS[indicator],
                s=31, alpha=0.8,
            )
        axis.set_title(group_title(group, len(membership[group])))
        axis.set_xlabel("Mean Total Cost (M$)")
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("Mean Urban Reliability")
    fig.suptitle(
        f"Period B OOS urban reliability–cost Pareto sets by {variable_label}",
        fontsize=13, y=0.985,
    )
    fig.legend(
        handles=indicator_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=6, frameon=False,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.77, wspace=0.12)
    save_figure_pair(fig, output_base, dpi)


def pareto_performance(item):
    values = item["objectives"][item["pareto_mask"]].copy()
    values[:, 0] /= 1e6
    return values


def global_parallel_bounds(results):
    combined = np.vstack([pareto_performance(item) for item in results.values()])
    if combined.size == 0 or not np.all(np.isfinite(combined)):
        raise ValueError("Cannot calculate parallel-axis bounds")
    return combined.min(axis=0), combined.max(axis=0)


def normalize_parallel(values, bounds):
    lower, upper = bounds
    span = np.where(upper > lower, upper - lower, 1.0)
    normalized = (values - lower) / span
    normalized[:, FLIP_AXIS] = 1.0 - normalized[:, FLIP_AXIS]
    return normalized


def draw_parallel_axes(axis, bounds, n_ticks=5):
    lower, upper = bounds
    for axis_index, name in enumerate(AXIS_NAMES):
        axis.axvline(axis_index, color="0.25", linewidth=1.0, zorder=1)
        for fraction in np.linspace(0.0, 1.0, n_ticks):
            if FLIP_AXIS[axis_index]:
                value = upper[axis_index] - fraction * (upper[axis_index] - lower[axis_index])
            else:
                value = lower[axis_index] + fraction * (upper[axis_index] - lower[axis_index])
            label = f"{value:,.1f}" if name == "Total Cost" else f"{value:.3f}"
            axis.text(
                axis_index, fraction, label, ha="center", va="center", fontsize=7,
                bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
                zorder=4,
            )
    axis.set_xticks(range(len(AXIS_NAMES)))
    axis.set_xticklabels(AXIS_LABELS, fontsize=9)
    axis.set_xlim(-0.25, len(AXIS_NAMES) - 0.75)
    axis.set_ylim(-0.05, 1.05)
    axis.set_yticks([])
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)


def plot_parallel_pair(
    output_base, variable_label, high, low, membership, results, bounds, dpi
):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    for axis, group in zip(axes, (high, low)):
        draw_parallel_axes(axis, bounds)
        line_count = 0
        for indicator in INDICATORS:
            normalized = normalize_parallel(
                pareto_performance(results[(group, indicator)]), bounds
            )
            line_count += len(normalized)
            for row in normalized:
                axis.plot(
                    range(len(AXIS_NAMES)), row, color=COLORS[indicator],
                    linewidth=0.85, alpha=0.28, zorder=2,
                )
        axis.set_title(
            f"{group_title(group, len(membership[group]))}; {line_count} policies"
        )
    fig.suptitle(
        f"Period B OOS three-objective Pareto policies by {variable_label}",
        fontsize=13, y=0.985,
    )
    fig.legend(
        handles=indicator_handles(markers=False), loc="upper center",
        bbox_to_anchor=(0.5, 0.91), ncol=6, frameon=False,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.12, top=0.76, wspace=0.12)
    save_figure_pair(fig, output_base, dpi)


def main():
    args = parse_args()
    results_dir = Path(
        args.results_dir
        or SCRIPT_DIR / POLICY_RUN_FOLDERS[args.policy_run]
    ).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    cache_dir = Path(
        args.cache_dir or results_dir / "out_of_sample" / "simulation_cache"
    ).expanduser().resolve()
    output_dir = Path(
        args.output_dir or results_dir / "out_of_sample" / "diff_climates"
    ).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory missing: {results_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"OOS simulation cache directory missing: {cache_dir}")

    temperature_dir = resolved_under(data_root, args.temperature_folder)
    precipitation_dir = resolved_under(data_root, args.precipitation_folder)
    glacier_file = resolved_under(data_root, args.glacier_file)
    streamflow_dir = resolved_under(data_root, args.period_b_oos_folder)
    if not glacier_file.is_file():
        raise FileNotFoundError(f"Glacier file missing: {glacier_file}")

    pareto_dir = output_dir / "pareto_data"
    urban_cost_dir = output_dir / "urban_reliability_vs_cost"
    parallel_dir = output_dir / "parallel_axes"
    for directory in (output_dir, pareto_dir, urban_cost_dir, parallel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    caches = {}
    reference_scenarios = None
    for indicator in INDICATORS:
        cache = load_oos_cache(cache_dir, indicator, args.policy_run)
        if reference_scenarios is None:
            reference_scenarios = cache["scenario_names"]
        elif not np.array_equal(reference_scenarios, cache["scenario_names"]):
            raise ValueError(f"OOS scenario names/order differ in cache for B/{indicator}")
        caches[indicator] = cache

    print("Calculating OOS climate metrics from test-set data...", flush=True)
    temperature, temperature_sources, missing_temperature = calculate_raw_metric(
        temperature_dir, args.temperature_prefix, reference_scenarios,
        args.temperature_column, allow_missing=False,
        shuffled_fallback=args.temperature_prefix == "temp_raw_",
        file_suffix=args.climate_file_suffix,
    )
    precipitation, precipitation_sources, _ = calculate_raw_metric(
        precipitation_dir, args.precipitation_prefix, reference_scenarios,
        args.precipitation_column, allow_missing=False,
        file_suffix=args.climate_file_suffix,
    )
    streamflow = calculate_streamflow(streamflow_dir, reference_scenarios)
    glacier = calculate_glacier(glacier_file, reference_scenarios)
    metrics = {
        "temperature": temperature,
        "precipitation": precipitation,
        "streamflow": streamflow,
        "glacier": glacier,
    }
    membership = classify_scenarios(reference_scenarios, metrics)
    metrics_path, membership_path = write_climate_tables(
        output_dir, reference_scenarios, metrics, membership,
        temperature_sources, precipitation_sources,
    )

    print(f"Policy run          : {args.policy_run}")
    print(f"Policy results      : {results_dir}")
    print(f"OOS cache           : {cache_dir}")
    print(f"Temperature data    : {temperature_dir}")
    print(f"Precipitation data  : {precipitation_dir}")
    print(f"Streamflow data     : {streamflow_dir}")
    print(f"Glacier data        : {glacier_file}")
    print(f"Output              : {output_dir}")
    print(f"Temperature coverage: {N_SCENARIOS - len(missing_temperature)}/{N_SCENARIOS}")
    if missing_temperature:
        print(
            f"WARNING: {len(missing_temperature)} temperature files are missing; "
            "high_T/low_T use only available scenarios.", flush=True,
        )
    for group in GROUP_ORDER:
        print(f"  {group}: {len(membership[group])} OOS scenarios")

    results = build_group_results(caches, membership)
    saved_tables = []
    for group in GROUP_ORDER:
        path = pareto_dir / f"oos_periodB_{group}_pareto.csv"
        save_group_pareto(path, group, len(membership[group]), results)
        saved_tables.append(path)

    bounds = global_parallel_bounds(results)
    saved_figures = []
    for code, variable_label, high, low in GROUP_PAIRS:
        urban_base = urban_cost_dir / f"oos_periodB_urban_cost_pareto_{code}"
        parallel_base = parallel_dir / f"oos_periodB_parallel_pareto_{code}"
        plot_urban_cost_pair(
            urban_base, variable_label, high, low, membership, results, args.dpi
        )
        plot_parallel_pair(
            parallel_base, variable_label, high, low, membership, results,
            bounds, args.dpi,
        )
        saved_figures.extend(
            [Path(str(urban_base) + suffix) for suffix in (".png", ".pdf")]
            + [Path(str(parallel_base) + suffix) for suffix in (".png", ".pdf")]
        )

    print("Saved climate classification:")
    print(f"  {metrics_path}")
    print(f"  {membership_path}")
    print("Saved Period-B Pareto tables:")
    for path in saved_tables:
        print(f"  {path}")
    print("Saved paired figures:")
    for path in saved_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
