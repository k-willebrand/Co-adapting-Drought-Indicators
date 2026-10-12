#!/usr/bin/env python3
"""Plot Sep16 optimal-policy mechanisms for the ten lowest-melt OOS climates.

The script does not rerun Pywr. It uses the completed Sep16 OOS performance
caches and CDF-action caches, selects fixed SPI12 and SRI12 policies in each
period under two rules (highest urban reliability and closest below a target
mean cost), ranks all 108 OOS climates by mean glacier melt over 2080--2099,
and creates matching six-panel figures for each of the ten lowest-ranked
scenarios.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_cdf_actions_oos_NATWAT as cdf_actions


POLICY_RUN = "sep16_decost"
RESULTS_FOLDER = "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
PERIOD_CONFIG = {
    "A": {
        "label": "2020–2040",
        "start": "2020-01-01",
        "end": "2040-01-01",
        "data_folder": "transient_108_no_drought_pert_periodA",
    },
    "B": {
        "label": "2080–2100",
        "start": "2080-01-01",
        "end": "2100-01-01",
        "data_folder": "transient_108_no_drought_pert_periodB",
    },
}
PLOTTED_INDICATORS = ("SPI12", "SRI12")
N_SCENARIOS = 108
N_SELECTED_SCENARIOS = 10
MELT_RANK_START = "2080-01-01"
MELT_RANK_END = "2100-01-01"
PRECIP_FOLDER = "knn_weap_transient_no_drought_pert"
GLACIER_FILENAME = "CMIP6_Glacier_Melt_108scenarios_no_drought_pert.csv"
GLACIER_YLIM = (0.0, 600.0)

COLORS = {
    "precipitation": "#6699cc",
    "glacier": "#997700",
    "SPI12": "#6699cc",
    "SRI12": "black",
    "SPI12_threshold": "orange",
    "SRI12_threshold": "chocolate",
    "A": "steelblue",
    "B": "firebrick",
}
LINESTYLES = {"SPI12": "--", "SRI12": ":"}

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
    parser.add_argument("--data-root", type=Path, default=script_dir / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <results-dir>/out_of_sample/"
            "low_glacier_optimal_policy_timeseries"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--target-cost-million",
        type=float,
        nargs="+",
        default=[650.0, 5000.0],
        help=(
            "One or more upper bounds on mean OOS total cost in M$. For each "
            "bound, the Pareto policy closest to, but strictly below, that "
            "value is selected (default: 650 5000)."
        ),
    )
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_policy(
    performance_cache_dir,
    action_dir,
    period,
    indicator,
    selection_mode="highest_urban_reliability",
    target_cost_million=None,
):
    performance = cdf_actions.load_performance_cache(
        performance_cache_dir, indicator, period, POLICY_RUN
    )
    if selection_mode == "highest_urban_reliability":
        source_index = cdf_actions.selected_policy_index(
            performance, indicator, period
        )
        selection_rule = (
            "max mean urban reliability among per-indicator OOS Pareto "
            "policies; then min cost, max agricultural reliability, "
            "lexicographically smallest decision vector"
        )
    elif selection_mode == "cost_below_target":
        if target_cost_million is None:
            raise ValueError("target_cost_million is required for cost selection")
        source_index = cdf_actions.target_cost_policy_index(
            performance, indicator, period, target_cost_million
        )
        selection_rule = (
            f"largest mean total cost strictly below {target_cost_million:g} "
            "M$ among per-indicator OOS Pareto policies; then max urban "
            "reliability, max agricultural reliability, lexicographically "
            "smallest decision vector"
        )
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    action_item = cdf_actions.load_action_cache(
        action_dir, indicator, period, POLICY_RUN
    )
    local_index = cdf_actions.matching_selected_row(action_item, source_index)
    if not np.array_equal(
        action_item["scenario_names"].astype(str),
        performance["scenario_names"].astype(str),
    ):
        raise RuntimeError(f"Scenario order differs between {indicator}_{period} caches")
    policy = performance["policies"][source_index].copy()
    objectives = performance["objectives"][source_index].copy()
    return {
        "period": period,
        "indicator": indicator,
        "performance": performance,
        "source_policy_index": int(source_index),
        "action_cache_row": int(local_index),
        "policy": policy,
        "objectives": objectives,
        "selection_mode": selection_mode,
        "selection_rule": selection_rule,
        "total_contracts": action_item["total_contracts"][local_index].copy(),
        "total_demand_restricted_mm3": action_item[
            "total_urban_demand_restricted_mm3"
        ][local_index].copy(),
    }


def select_policy_set(
    performance_cache_dir,
    action_dir,
    selection_mode,
    target_cost_million=None,
):
    return {
        (period, indicator): selected_policy(
            performance_cache_dir,
            action_dir,
            period,
            indicator,
            selection_mode=selection_mode,
            target_cost_million=target_cost_million,
        )
        for period in ("A", "B")
        for indicator in PLOTTED_INDICATORS
    }


def load_glacier_and_rank(glacier_path, expected_names):
    glacier = pd.read_csv(glacier_path, index_col=0)
    glacier.index = pd.to_datetime(glacier.index, errors="raise")
    glacier.columns = glacier.columns.astype(str)
    if glacier.columns.duplicated().any():
        raise ValueError(f"Duplicate glacier scenario names in {glacier_path}")
    expected = [str(name) for name in expected_names]
    if set(glacier.columns) != set(expected):
        missing = sorted(set(expected) - set(glacier.columns))
        extra = sorted(set(glacier.columns) - set(expected))
        raise ValueError(
            f"Glacier/OOS scenario mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    glacier = glacier.loc[:, expected]
    window = glacier.loc[
        (glacier.index >= pd.Timestamp(MELT_RANK_START))
        & (glacier.index < pd.Timestamp(MELT_RANK_END))
    ]
    if window.empty:
        raise ValueError("No glacier-melt rows in the 2080--2099 ranking window")
    metrics = window.mean(axis=0, skipna=True)
    if metrics.isna().any():
        missing = metrics.index[metrics.isna()].tolist()
        raise ValueError(f"No final-20-year glacier values for: {missing}")
    ranked = sorted(
        ((float(metrics[name]), str(name)) for name in expected),
        key=lambda item: (item[0], item[1]),
    )
    selected = ranked[:N_SELECTED_SCENARIOS]
    boundary_value = selected[-1][0]
    boundary_tie_count = sum(
        np.isclose(value, boundary_value, rtol=0.0, atol=1e-12)
        for value, _ in ranked
    )
    rows = []
    selected_names = {name for _, name in selected}
    for rank, (value, name) in enumerate(ranked, start=1):
        rows.append(
            {
                "deterministic_rank": rank,
                "scenario_name": name,
                "mean_glacier_melt_2080_2099_mm3": value,
                "selected_lowest_10": int(name in selected_names),
                "selection_boundary_value_mm3": boundary_value,
                "boundary_tie_count": boundary_tie_count,
                "tie_break_rule": "ascending glacier metric, then scenario name",
            }
        )
    return glacier, selected, rows


def load_indicator(path, expected_names):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, dayfirst=True, errors="raise")
    frame.columns = frame.columns.astype(str)
    expected = [str(name) for name in expected_names]
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate indicator scenario names in {path}")
    if set(frame.columns) != set(expected):
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        raise ValueError(
            f"Indicator/OOS scenario mismatch in {path}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    # Match by scenario name, never by the source CSV column position.
    return frame.loc[:, expected]


def precipitation_path(data_root, scenario_name):
    suffix = "_no_drought_pert"
    if not scenario_name.endswith(suffix):
        raise ValueError(f"Unexpected OOS scenario name: {scenario_name}")
    base = scenario_name[: -len(suffix)]
    return (
        data_root
        / PRECIP_FOLDER
        / f"precip_transient_{base}_1979_2099.csv"
    )


def load_spatial_mean_precipitation(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOS precipitation file: {path}")
    frame = pd.read_csv(path)
    if "year" not in frame or "week" not in frame or frame.shape[1] < 3:
        raise ValueError(f"Unexpected precipitation file structure: {path}")
    dates = pd.to_datetime(
        frame["year"].astype(int).astype(str)
        + frame["week"].astype(int).astype(str).str.zfill(2)
        + "-1",
        format="%Y%W-%w",
        errors="raise",
    )
    values = frame.iloc[:, 2:].apply(pd.to_numeric, errors="raise").mean(axis=1)
    return pd.Series(values.to_numpy(dtype=float), index=dates, name="precipitation")


def period_slice(series, period):
    config = PERIOD_CONFIG[period]
    sliced = series.loc[
        (series.index >= pd.Timestamp(config["start"]))
        & (series.index < pd.Timestamp(config["end"]))
    ]
    if sliced.empty:
        raise ValueError(f"No {series.name} data for Period {period}")
    return sliced


def add_panel_label(axis, label):
    axis.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-27, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        annotation_clip=False,
    )


def format_time_axis(axis):
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(False)


def plot_climate_panel(axis, precipitation, glacier, period, panel_label):
    config = PERIOD_CONFIG[period]
    precipitation = period_slice(precipitation, period)
    glacier = period_slice(glacier.rename("glacier_melt"), period)
    twin = axis.twinx()
    line_p = axis.plot(
        precipitation.index,
        precipitation.values,
        color=COLORS["precipitation"],
        label="Precipitation",
    )[0]
    line_g = twin.plot(
        glacier.index,
        glacier.values,
        color=COLORS["glacier"],
        label="Glacier Melt",
    )[0]
    axis.set_ylabel("Precipitation (mm)", color=COLORS["precipitation"])
    twin.set_ylabel("Glacier Melt (Mm³)", color=COLORS["glacier"])
    axis.tick_params(axis="y", colors=COLORS["precipitation"])
    twin.tick_params(axis="y", colors=COLORS["glacier"])
    axis.set_title(config["label"])
    axis.set_xlim(pd.Timestamp(config["start"]), pd.Timestamp(config["end"]))
    twin.set_xlim(pd.Timestamp(config["start"]), pd.Timestamp(config["end"]))
    axis.set_ylim(0.0, 400.0)
    twin.set_ylim(*GLACIER_YLIM)
    format_time_axis(axis)
    axis.legend(
        [line_p, line_g],
        [line_p.get_label(), line_g.get_label()],
        loc="upper right",
        ncol=1,
    )
    add_panel_label(axis, panel_label)
    return twin


def plot_indicator_panel(axis, indicator_series, selections, period, panel_label):
    config = PERIOD_CONFIG[period]
    indicator_handles = {}
    threshold_handles = {}
    for indicator in PLOTTED_INDICATORS:
        values = period_slice(
            indicator_series[indicator].rename(indicator), period
        )
        indicator_line = axis.plot(
            values.index,
            values.values,
            color=COLORS[indicator],
            label=indicator,
        )[0]
        policy = selections[(period, indicator)]["policy"]
        threshold_lines = []
        for threshold in (policy[0], policy[2]):
            threshold_lines.append(
                axis.axhline(
                    threshold,
                    color=COLORS[f"{indicator}_threshold"],
                    linestyle=LINESTYLES[indicator],
                    linewidth=3.0,
                    label=f"{indicator} Thresholds",
                )
            )
        indicator_handles[indicator] = indicator_line
        threshold_handles[indicator] = threshold_lines[0]

    axis.set_ylabel("SPI or SRI (-)")
    axis.set_title(config["label"])
    axis.set_xlim(pd.Timestamp(config["start"]), pd.Timestamp(config["end"]))
    axis.set_ylim(-6.5, 5.0)
    format_time_axis(axis)
    legend_handles = [
        indicator_handles["SPI12"],
        indicator_handles["SRI12"],
        threshold_handles["SPI12"],
        threshold_handles["SRI12"],
    ]
    legend_labels = [
        "SPI12",
        "SRI12",
        "SPI12 Thresholds",
        "SRI12 Thresholds",
    ]
    axis.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        ncol=2,
    )
    add_panel_label(axis, panel_label)


def require_matplotlib_panel_alignment(
    fig, full_axes, bottom_axes, output_base, tolerance_pt=1.5
):
    """Audit the four full-width panels and two comparable bottom panels."""
    fig.canvas.draw()
    width_pt, height_pt = fig.get_size_inches() * 72.0

    def geometry(axis, panel_id):
        position = axis.get_position()
        return {
            "id": panel_id,
            "left": float(position.x0 * width_pt),
            "bottom": float(position.y0 * height_pt),
            "right": float(position.x1 * width_pt),
            "top": float(position.y1 * height_pt),
            "width": float(position.width * width_pt),
            "height": float(position.height * height_pt),
        }

    panels = [
        geometry(axis, label)
        for axis, label in zip(full_axes + bottom_axes, tuple("abcdef"))
    ]
    deviations = {}
    for field in ("left", "right", "width"):
        values = [panel[field] for panel in panels[:4]]
        deviations[f"full_width_{field}"] = max(values) - min(values)
    for field in ("top", "bottom", "width", "height"):
        values = [panel[field] for panel in panels[4:]]
        deviations[f"bottom_row_{field}"] = max(values) - min(values)
    deviations["outer_left"] = abs(panels[0]["left"] - panels[4]["left"])
    deviations["outer_right"] = abs(panels[0]["right"] - panels[5]["right"])
    passed = all(value <= tolerance_pt for value in deviations.values())
    report = {
        "schema_version": 1,
        "figure_size_pt": [float(width_pt), float(height_pt)],
        "tolerance_pt": tolerance_pt,
        "panels": panels,
        "deviations_pt": deviations,
        "verdict": "PASS" if passed else "FIX BEFORE DELIVERY",
    }
    Path(f"{output_base}.alignment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            f"Six-panel alignment exceeds {tolerance_pt} pt: {deviations}"
        )


def save_figure(fig, full_axes, bottom_axes, output_base, dpi):
    require_matplotlib_panel_alignment(fig, full_axes, bottom_axes, output_base)
    fig.savefig(f"{output_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    plt.close(fig)


def make_figure(
    output_base,
    scenario_name,
    rank,
    melt_metric,
    precipitation,
    glacier,
    indicators,
    selections,
    scenario_index,
    dpi,
):
    fig = plt.figure(figsize=(10.0, 12.0), constrained_layout=True)
    grid = fig.add_gridspec(
        ncols=3,
        nrows=9,
        width_ratios=(1.0, 0.2, 1.0),
        height_ratios=(1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0),
        wspace=0.0,
        hspace=0.0,
    )
    ax_a = fig.add_subplot(grid[0, 0:3])
    ax_b = fig.add_subplot(grid[2, 0:3])
    ax_c = fig.add_subplot(grid[4, 0:3])
    ax_d = fig.add_subplot(grid[6, 0:3])
    ax_e = fig.add_subplot(grid[8, 0])
    ax_f = fig.add_subplot(grid[8, 2])

    plot_climate_panel(ax_a, precipitation, glacier, "A", "a")
    plot_indicator_panel(ax_b, indicators["A"], selections, "A", "b")
    plot_climate_panel(ax_c, precipitation, glacier, "B", "c")
    plot_indicator_panel(ax_d, indicators["B"], selections, "B", "d")

    group_centers = np.asarray((0.0, 0.65))
    bar_width = 0.25
    period_offsets = {"A": -0.15, "B": 0.15}
    contracts = {}
    restrictions = {}
    for period in ("A", "B"):
        period_contracts = []
        period_restrictions = []
        for indicator in PLOTTED_INDICATORS:
            selection = selections[(period, indicator)]
            period_contracts.append(
                selection["total_contracts"][scenario_index] / 1_000.0
            )
            period_restrictions.append(
                selection["total_demand_restricted_mm3"][scenario_index]
            )
        contracts[period] = period_contracts
        restrictions[period] = period_restrictions
        ax_e.bar(
            group_centers + period_offsets[period],
            period_contracts,
            color=COLORS[period],
            width=bar_width,
            label=PERIOD_CONFIG[period]["label"],
        )
        ax_f.bar(
            group_centers + period_offsets[period],
            period_restrictions,
            color=COLORS[period],
            width=bar_width,
        )

    ax_e.set_xticks(group_centers, PLOTTED_INDICATORS)
    ax_e.set_xlim(-0.35, 1.0)
    ax_e.set_ylim(0.0, 1900.0)
    ax_e.set_ylabel("Contracts (thousands)")
    ax_e.legend(frameon=False, loc="upper left")
    ax_e.grid(False)
    add_panel_label(ax_e, "e")

    ax_f.set_xticks(group_centers, PLOTTED_INDICATORS)
    ax_f.set_xlim(-0.35, 1.0)
    ax_f.set_ylabel("Demand Restricted (Mm³)")
    ax_f.grid(False)
    add_panel_label(ax_f, "f")

    save_figure(fig, [ax_a, ax_b, ax_c, ax_d], [ax_e, ax_f], output_base, dpi)
    return contracts, restrictions


def policy_table_rows(selections, policy_set, target_cost_million=None):
    rows = []
    for period in ("A", "B"):
        for indicator in PLOTTED_INDICATORS:
            selection = selections[(period, indicator)]
            policy = selection["policy"]
            objective = selection["objectives"]
            mean_cost_million = objective[0] / 1_000_000.0
            rows.append(
                {
                    "policy_set": policy_set,
                    "period": period,
                    "date_range": PERIOD_CONFIG[period]["label"],
                    "indicator": selection["indicator"],
                    "source_policy_index": selection["source_policy_index"],
                    "action_cache_row": selection["action_cache_row"],
                    "contract_threshold": policy[0],
                    "contract_action": policy[1],
                    "demand_threshold": policy[2],
                    "demand_action": policy[3],
                    "target_cost_million": target_cost_million,
                    "distance_below_target_million": (
                        None
                        if target_cost_million is None
                        else target_cost_million - mean_cost_million
                    ),
                    "oos_mean_total_cost": objective[0],
                    "oos_mean_total_cost_million": mean_cost_million,
                    "oos_mean_urban_reliability": objective[1],
                    "oos_mean_agricultural_reliability": objective[2],
                    "selection_rule": selection["selection_rule"],
                }
            )
    return rows


def scenario_action_row(
    rank,
    scenario_name,
    scenario_index,
    melt_metric,
    selections,
    contracts,
    restrictions,
    policy_set,
):
    row = {
        "policy_set": policy_set,
        "rank": rank,
        "scenario_name": scenario_name,
        "scenario_index": scenario_index,
        "mean_glacier_melt_2080_2099_mm3": melt_metric,
    }
    for period in ("A", "B"):
        for indicator_index, indicator in enumerate(PLOTTED_INDICATORS):
            prefix = f"period_{period}_{indicator}"
            row[f"{prefix}_source_policy_index"] = selections[
                (period, indicator)
            ]["source_policy_index"]
            row[f"{prefix}_total_contracts_thousand_shares"] = contracts[
                period
            ][indicator_index]
            row[f"{prefix}_total_demand_restricted_mm3"] = restrictions[
                period
            ][indicator_index]
    return row


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    target_costs = []
    for target_cost in args.target_cost_million:
        if not np.isfinite(target_cost) or target_cost <= 0.0:
            raise ValueError(
                "Every --target-cost-million value must be finite and positive"
            )
        if target_cost not in target_costs:
            target_costs.append(target_cost)
    results_dir = args.results_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir
        / "out_of_sample"
        / "low_glacier_optimal_policy_timeseries"
    )
    performance_cache_dir = results_dir / "out_of_sample" / "simulation_cache"
    cdf_dir = results_dir / "out_of_sample" / "cdf_actions"
    action_dir = cdf_dir / "simulation_cache"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        results_dir,
        data_root,
        performance_cache_dir,
        action_dir,
        data_root / GLACIER_FILENAME,
        data_root / PRECIP_FOLDER,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path is missing: {path}")

    highest_selections = select_policy_set(
        performance_cache_dir,
        action_dir,
        "highest_urban_reliability",
    )
    selection_sets = {
        "highest_urban_reliability": highest_selections,
    }
    cost_runs = {}
    for target_cost in target_costs:
        target_tag = cdf_actions.target_cost_tag(target_cost)
        policy_set = f"cost_below_{target_tag}"
        if policy_set in cost_runs:
            raise ValueError(f"Duplicate target-cost output tag: {target_tag}")
        selections = select_policy_set(
            performance_cache_dir,
            action_dir,
            "cost_below_target",
            target_cost_million=target_cost,
        )
        cost_figures_dir = figures_dir / policy_set
        cost_figures_dir.mkdir(parents=True, exist_ok=True)
        cost_runs[policy_set] = {
            "target_cost_million": target_cost,
            "target_tag": target_tag,
            "selections": selections,
            "figures_dir": cost_figures_dir,
            "action_rows": [],
        }
        selection_sets[policy_set] = selections
    reference_selection = highest_selections[("A", "SPI12")]
    scenario_names = reference_selection["performance"]["scenario_names"].astype(str)
    for policy_set, selections in selection_sets.items():
        for (period, indicator), selection in selections.items():
            if not np.array_equal(
                scenario_names,
                selection["performance"]["scenario_names"].astype(str),
            ):
                raise RuntimeError(
                    f"OOS scenario order differs for {policy_set} "
                    f"{indicator}_{period}"
                )
    scenario_lookup = {name: index for index, name in enumerate(scenario_names)}

    glacier, selected_scenarios, climate_rows = load_glacier_and_rank(
        data_root / GLACIER_FILENAME, scenario_names
    )
    write_csv(
        output_dir / "glacier_melt_scenario_ranking.csv",
        list(climate_rows[0]),
        climate_rows,
    )

    indicators = {}
    for period in ("A", "B"):
        config = PERIOD_CONFIG[period]
        indicators[period] = {
            indicator: load_indicator(
                data_root / config["data_folder"] / f"{indicator}.csv",
                scenario_names,
            )
            for indicator in PLOTTED_INDICATORS
        }

    highest_policy_rows = policy_table_rows(
        highest_selections, "highest_urban_reliability"
    )
    write_csv(
        output_dir / "selected_policies.csv",
        list(highest_policy_rows[0]),
        highest_policy_rows,
    )
    for policy_set, cost_run in cost_runs.items():
        cost_policy_rows = policy_table_rows(
            cost_run["selections"],
            policy_set,
            target_cost_million=cost_run["target_cost_million"],
        )
        write_csv(
            output_dir
            / f"selected_policies_cost_below_{cost_run['target_tag']}.csv",
            list(cost_policy_rows[0]),
            cost_policy_rows,
        )

    highest_action_rows = []
    for rank, (melt_metric, scenario_name) in enumerate(selected_scenarios, start=1):
        scenario_index = scenario_lookup[scenario_name]
        precipitation = load_spatial_mean_precipitation(
            precipitation_path(data_root, scenario_name)
        )
        scenario_glacier = glacier[scenario_name].rename("glacier_melt")
        scenario_indicators = {
            period: {
                indicator: indicators[period][indicator][scenario_name]
                for indicator in PLOTTED_INDICATORS
            }
            for period in ("A", "B")
        }
        safe_name = scenario_name.replace("_no_drought_pert", "")
        highest_contracts, highest_restrictions = make_figure(
            figures_dir / f"rank{rank:02d}_{safe_name}",
            scenario_name,
            rank,
            melt_metric,
            precipitation,
            scenario_glacier,
            scenario_indicators,
            highest_selections,
            scenario_index,
            args.dpi,
        )
        highest_action_rows.append(
            scenario_action_row(
                rank,
                scenario_name,
                scenario_index,
                melt_metric,
                highest_selections,
                highest_contracts,
                highest_restrictions,
                "highest_urban_reliability",
            )
        )
        for policy_set, cost_run in cost_runs.items():
            cost_contracts, cost_restrictions = make_figure(
                cost_run["figures_dir"] / f"rank{rank:02d}_{safe_name}",
                scenario_name,
                rank,
                melt_metric,
                precipitation,
                scenario_glacier,
                scenario_indicators,
                cost_run["selections"],
                scenario_index,
                args.dpi,
            )
            cost_run["action_rows"].append(
                scenario_action_row(
                    rank,
                    scenario_name,
                    scenario_index,
                    melt_metric,
                    cost_run["selections"],
                    cost_contracts,
                    cost_restrictions,
                    policy_set,
                )
            )
    write_csv(
        output_dir / "scenario_action_summary.csv",
        list(highest_action_rows[0]),
        highest_action_rows,
    )
    for cost_run in cost_runs.values():
        write_csv(
            output_dir
            / (
                "scenario_action_summary_cost_below_"
                f"{cost_run['target_tag']}.csv"
            ),
            list(cost_run["action_rows"][0]),
            cost_run["action_rows"],
        )

    target_list_text = ", ".join(f"{value:g}" for value in target_costs)
    target_output_text = "\n".join(
        f"- {cost_run['target_cost_million']:g} M$: "
        f"figures/cost_below_{cost_run['target_tag']}/"
        for cost_run in cost_runs.values()
    )

    method = f"""Sep16 low-glacier optimal-policy time-series method

All scenarios are from the 108-scenario out-of-sample ensemble. Scenarios are
ranked by ascending mean glacier melt from 2080-01-01 through 2099-12-31.
Exactly ten scenarios are retained; ties are resolved by scenario name.

Both periods show SPI12 and SRI12. The original figure set selects one fixed
policy from each of the four indicator-period OOS Pareto sets by maximum mean
urban reliability across all 108 OOS scenarios. Ties are resolved by lower
mean total cost, higher mean agricultural reliability, and then the
lexicographically smallest decision vector.

The additional figure sets use target costs of {target_list_text} M$. For each
target and each of the four OOS Pareto sets, the policy with the largest mean
total cost strictly below that target is selected (the closest policy from
below). Policies at or above the applicable target are excluded. Exact-cost
ties are resolved by higher mean urban reliability, higher mean agricultural
reliability, and then the lexicographically smallest decision vector. Each
selection rule uses the same four fixed policies for all ten climates, allowing
the figures to be compared without changing the selected scenarios.

Panels a and c show spatial-mean weekly precipitation and glacier melt. Panels
b and d show both drought indicators. For each indicator, both its contract
and demand-restriction thresholds are drawn with the same indicator-specific
style, matching the manuscript figure; coincident thresholds therefore appear
as one line. Panels e and f each contain four bars (two indicators by two
periods). Panel e shows total triggered contract purchases in thousand shares;
it is not a physical water volume. Panel f shows cumulative urban demand
restricted in Mm3. Action totals are read from the completed Sep16 CDF-action
caches; this plotting script does not rerun Pywr. The original figures remain
in figures/. Cost-constrained figure folders are:
{target_output_text}
"""
    (output_dir / "method.txt").write_text(method, encoding="utf-8")
    print(f"Selected scenarios and six-panel figures written to: {output_dir}")


if __name__ == "__main__":
    main()
