#!/usr/bin/env python3
"""Plot Sep16 low-glacier mechanisms with counterfactual reliability.

The ten scenarios and the four fixed SPI12/SRI12 policies are selected exactly
as in the existing Sep16 low-glacier highest-urban-reliability workflow. The
script reads completed all-108 weekly reliability caches and creates two visual
variants for each scenario: action-active weeks as points or as filled areas.
No Pywr model or optimization is run here.
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

import plot_low_glacier_optimal_policy_timeseries_NATWAT as original


POLICY_RUN = original.POLICY_RUN
PERIODS = ("A", "B")
INDICATORS = original.PLOTTED_INDICATORS
N_SCENARIOS = original.N_SCENARIOS
INDICATOR_COLORS = {"SPI12": "#3d85c6", "SRI12": "#d47a1f"}
PERIOD_LINESTYLES = {"A": "-", "B": "--"}
RELIABILITY_STYLES = {
    "no_policy": {"linestyle": "-", "linewidth": 1.8, "marker": None},
    "SPI12": {"linestyle": (0, (1, 1.5)), "linewidth": 1.45, "marker": "o"},
    "SRI12": {"linestyle": (0, (5, 2)), "linewidth": 1.45, "marker": "^"},
}
FORMAL_PERIOD_LABELS = {"A": "2020–2040", "B": "2080–2100"}
SECONDS_PER_WEEK = 7 * 24 * 60 * 60


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
        "--results-dir",
        type=Path,
        default=(
            script_dir
            / "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=script_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reliability-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--trigger-style",
        nargs="+",
        choices=("points", "fill"),
        default=("points", "fill"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def scalar_text(value):
    return str(np.asarray(value).reshape(-1)[0])


def load_weekly_cache(cache_dir, period, policy_type, expected_names, expected_policy):
    path = Path(cache_dir) / f"{period}_{policy_type}_weekly_reliability.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing weekly reliability cache: {path}\n"
            "Run the six-task reliability simulation first."
        )
    required = {
        "completed",
        "policy_run",
        "period",
        "policy_type",
        "source_policy_index",
        "policy",
        "scenario_names",
        "dates",
        "contract_action",
        "demand_restriction_factor",
        "unrestricted_demand",
        "restricted_demand",
        "actual_pt1_delivery",
        "weekly_failure",
        "cumulative_reliability",
        "trigger_union",
        "official_reliability",
        "demand_restriction_cost_per_m3",
        "min_demand_retention_factor",
    }
    with np.load(path, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Cache {path} is missing: {', '.join(sorted(missing))}")
        item = {name: saved[name].copy() for name in required}
    identity = (
        scalar_text(item["policy_run"]).lower(),
        scalar_text(item["period"]),
        scalar_text(item["policy_type"]),
    )
    if identity != (POLICY_RUN, period, policy_type):
        raise ValueError(f"Weekly cache identity mismatch: {path}")
    if int(np.asarray(item["completed"]).reshape(-1)[0]) != 1:
        raise ValueError(f"Weekly cache is incomplete: {path}")
    names = item["scenario_names"].astype(str)
    if not np.array_equal(names, np.asarray(expected_names, dtype=str)):
        raise ValueError(f"Weekly cache scenario order mismatch: {path}")
    if not np.allclose(item["policy"], expected_policy, rtol=0.0, atol=1e-12):
        raise ValueError(f"Weekly cache selected policy mismatch: {path}")
    if not np.isclose(
        float(np.asarray(item["demand_restriction_cost_per_m3"]).reshape(-1)[0]),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Weekly cache does not use Sep16 $1/m3 demand cost: {path}")
    if not np.isclose(
        float(np.asarray(item["min_demand_retention_factor"]).reshape(-1)[0]),
        0.7,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Weekly cache does not use Sep16 30% curtailment cap: {path}")
    dates = pd.DatetimeIndex(item["dates"].astype("datetime64[ns]"))
    n_weeks = len(dates)
    for name in (
        "contract_action",
        "demand_restriction_factor",
        "unrestricted_demand",
        "restricted_demand",
        "actual_pt1_delivery",
        "weekly_failure",
        "cumulative_reliability",
        "trigger_union",
    ):
        values = np.asarray(item[name])
        if values.shape != (n_weeks, N_SCENARIOS):
            raise ValueError(f"Invalid {name} shape in {path}: {values.shape}")
    official = np.asarray(item["official_reliability"], dtype=float).reshape(-1)
    if official.shape != (N_SCENARIOS,):
        raise ValueError(f"Invalid official reliability in {path}")
    if not np.allclose(
        np.asarray(item["cumulative_reliability"], dtype=float)[-1],
        official,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(f"Cached cumulative reliability validation failed: {path}")
    item["dates"] = dates
    item["official_reliability"] = official
    item["path"] = path
    return item


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


def monthly_period(series, dates, aggregation):
    selected = series.loc[(series.index >= dates[0]) & (series.index <= dates[-1])]
    if selected.empty:
        raise ValueError(f"No {series.name} values for {dates[0]} to {dates[-1]}")
    if aggregation == "sum":
        monthly = selected.resample("MS").sum(min_count=1)
    elif aggregation == "mean":
        monthly = selected.resample("MS").mean()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if monthly.isna().any():
        raise ValueError(f"Missing monthly {series.name} values")
    return monthly


def plot_combined_climate(axis, precipitation, glacier, weekly_caches):
    twin = axis.twinx()
    handles = []
    max_precipitation = 0.0
    for period in PERIODS:
        dates = weekly_caches[(period, "no_policy")]["dates"]
        monthly_p = monthly_period(precipitation, dates, "sum")
        monthly_g = monthly_period(glacier, dates, "mean")
        elapsed_p = np.arange(len(monthly_p), dtype=float) / 12.0
        elapsed_g = np.arange(len(monthly_g), dtype=float) / 12.0
        max_precipitation = max(max_precipitation, float(monthly_p.max()))
        handles.append(
            axis.plot(
                elapsed_p,
                monthly_p.values,
                color="#4f8fbd",
                linestyle=PERIOD_LINESTYLES[period],
                linewidth=1.9,
                label=f"Precipitation, {FORMAL_PERIOD_LABELS[period]}",
                zorder=3,
            )[0]
        )
        handles.append(
            twin.plot(
                elapsed_g,
                monthly_g.values,
                color="#997700",
                linestyle=PERIOD_LINESTYLES[period],
                linewidth=2.25 if period == "B" else 1.9,
                label=f"Glacier melt, {FORMAL_PERIOD_LABELS[period]}",
                zorder=4,
            )[0]
        )
    axis.set_xlim(0.0, 20.0)
    axis.set_xticks((0, 5, 10, 15, 20))
    axis.set_xlabel("Elapsed years within period")
    axis.set_ylim(0.0, max(1.0, max_precipitation * 1.08))
    twin.set_ylim(-15.0, 600.0)
    axis.set_ylabel("Monthly precipitation (mm)", color="#4f8fbd")
    twin.set_ylabel("Glacier melt (Mm³)", color="#997700")
    axis.tick_params(axis="y", colors="#4f8fbd")
    twin.tick_params(axis="y", colors="#997700")
    axis.grid(False)
    axis.legend(handles=handles, loc="upper right", ncol=2, fontsize=6.5)
    add_panel_label(axis, "a")
    return twin


def aligned_indicator(series, dates, indicator):
    aligned = series.reindex(dates)
    if aligned.isna().any():
        missing = dates[aligned.isna()]
        raise ValueError(
            f"{indicator} is missing {len(missing)} formal weekly dates; "
            f"first={missing[0] if len(missing) else 'none'}"
        )
    return aligned.to_numpy(dtype=float)


def plot_indicator_and_reliability(
    indicator_axis,
    reliability_axis,
    period,
    scenario_index,
    scenario_indicators,
    weekly_caches,
    trigger_style,
    panel_label,
):
    dates = weekly_caches[(period, "no_policy")]["dates"]
    for indicator in INDICATORS:
        cache = weekly_caches[(period, indicator)]
        values = aligned_indicator(
            scenario_indicators[indicator], dates, indicator
        )
        trigger = np.asarray(cache["trigger_union"][:, scenario_index], dtype=bool)
        color = INDICATOR_COLORS[indicator]
        indicator_axis.plot(
            dates,
            values,
            color=color,
            linewidth=1.35,
            label=indicator,
            zorder=3,
        )
        if trigger_style == "points":
            indicator_axis.scatter(
                dates[trigger],
                values[trigger],
                s=8,
                color=color,
                edgecolors="none",
                alpha=0.9,
                zorder=4,
            )
        elif trigger_style == "fill":
            indicator_axis.fill_between(
                dates,
                -6.5,
                values,
                where=trigger,
                color=color,
                alpha=0.15,
                linewidth=0.0,
                interpolate=False,
                zorder=1,
            )
        else:
            raise ValueError(f"Unknown trigger style: {trigger_style}")

    indicator_axis.set_xlim(dates[0], dates[-1])
    indicator_axis.set_ylim(-6.5, 5.0)
    indicator_axis.set_ylabel("SPI or SRI (-)")
    indicator_axis.set_title(FORMAL_PERIOD_LABELS[period], fontsize=8)
    indicator_axis.tick_params(axis="x", labelbottom=False)
    indicator_axis.grid(False)
    indicator_axis.legend(loc="lower right", ncol=2, fontsize=6.5)
    add_panel_label(indicator_axis, panel_label)

    labels = {
        "no_policy": "No policy",
        "SPI12": "SPI12 policy",
        "SRI12": "SRI12 policy",
    }
    for policy_type in ("no_policy", "SPI12", "SRI12"):
        cache = weekly_caches[(period, policy_type)]
        reliability = np.asarray(
            cache["cumulative_reliability"][:, scenario_index], dtype=float
        )
        style = RELIABILITY_STYLES[policy_type]
        final = reliability[-1]
        reliability_axis.plot(
            dates,
            reliability,
            color="black",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=52 if style["marker"] is not None else None,
            markersize=2.6,
            markerfacecolor="white",
            markeredgewidth=0.6,
            label=f"{labels[policy_type]} (final R={final:.3f})",
            zorder=3,
        )
    reliability_axis.set_xlim(dates[0], dates[-1])
    reliability_axis.set_ylim(0.0, 1.02)
    reliability_axis.set_ylabel("Cumulative\nurban reliability")
    reliability_axis.xaxis.set_major_locator(mdates.YearLocator(2))
    reliability_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    reliability_axis.grid(axis="y", color="0.9", linewidth=0.5)
    reliability_axis.legend(loc="lower right", ncol=3, fontsize=6.0)


def plot_action_bars(ax_contracts, ax_restrictions, selections, scenario_index):
    group_centers = np.asarray((0.0, 0.65))
    bar_width = 0.25
    period_offsets = {"A": -0.15, "B": 0.15}
    period_colors = {"A": "steelblue", "B": "firebrick"}
    contracts = {}
    restrictions = {}
    for period in PERIODS:
        period_contracts = []
        period_restrictions = []
        for indicator in INDICATORS:
            selection = selections[(period, indicator)]
            period_contracts.append(
                selection["total_contracts"][scenario_index] / 1_000.0
            )
            period_restrictions.append(
                selection["total_demand_restricted_mm3"][scenario_index]
            )
        contracts[period] = period_contracts
        restrictions[period] = period_restrictions
        ax_contracts.bar(
            group_centers + period_offsets[period],
            period_contracts,
            color=period_colors[period],
            width=bar_width,
            label=FORMAL_PERIOD_LABELS[period],
        )
        ax_restrictions.bar(
            group_centers + period_offsets[period],
            period_restrictions,
            color=period_colors[period],
            width=bar_width,
        )
    for axis in (ax_contracts, ax_restrictions):
        axis.set_xticks(group_centers, INDICATORS)
        axis.set_xlim(-0.35, 1.0)
        axis.grid(False)
    ax_contracts.set_ylim(0.0, 1900.0)
    ax_contracts.set_ylabel("Contracts (thousands)")
    ax_contracts.legend(loc="upper left", fontsize=6.5)
    ax_restrictions.set_ylabel("Demand Restricted (Mm³)")
    add_panel_label(ax_contracts, "d")
    add_panel_label(ax_restrictions, "e")
    return contracts, restrictions


def axis_geometry(fig, axis, panel_id):
    width_pt, height_pt = fig.get_size_inches() * 72.0
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


def require_matplotlib_panel_alignment(
    fig, full_axes, bottom_axes, output_base, tolerance_pt=1.5
):
    fig.canvas.draw()
    full_boxes = [
        axis_geometry(fig, axis, panel_id)
        for axis, panel_id in zip(full_axes, ("a", "b1", "b2", "c1", "c2"))
    ]
    bottom_boxes = [
        axis_geometry(fig, axis, panel_id)
        for axis, panel_id in zip(bottom_axes, ("d", "e"))
    ]
    deviations = {}
    for field in ("left", "right", "width"):
        values = [box[field] for box in full_boxes]
        deviations[f"full_width_{field}"] = max(values) - min(values)
    for field in ("top", "bottom", "width", "height"):
        values = [box[field] for box in bottom_boxes]
        deviations[f"bottom_row_{field}"] = max(values) - min(values)
    passed = all(value <= tolerance_pt for value in deviations.values())
    report = {
        "schema_version": 1,
        "figure_size_pt": [
            float(value) for value in fig.get_size_inches() * 72.0
        ],
        "tolerance_pt": tolerance_pt,
        "panels": full_boxes + bottom_boxes,
        "deviations_pt": deviations,
        "verdict": "PASS" if passed else "FIX BEFORE DELIVERY",
    }
    Path(f"{output_base}.alignment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Panel alignment failed: {deviations}")


def save_figure(fig, full_axes, bottom_axes, output_base, dpi):
    require_matplotlib_panel_alignment(
        fig, full_axes, bottom_axes, output_base
    )
    fig.savefig(f"{output_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    fig.savefig(f"{output_base}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_figure(
    output_base,
    scenario_name,
    rank,
    melt_metric,
    precipitation,
    glacier,
    scenario_indicators,
    selections,
    weekly_caches,
    scenario_index,
    trigger_style,
    dpi,
):
    fig = plt.figure(figsize=(7.2, 9.8), constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=(1.05, 1.55, 1.55, 0.95),
        width_ratios=(1.0, 0.16, 1.0),
    )
    ax_a = fig.add_subplot(outer[0, :])
    grid_b = outer[1, :].subgridspec(2, 1, height_ratios=(2.0, 1.0), hspace=0.06)
    ax_b_indicator = fig.add_subplot(grid_b[0, 0])
    ax_b_reliability = fig.add_subplot(grid_b[1, 0], sharex=ax_b_indicator)
    grid_c = outer[2, :].subgridspec(2, 1, height_ratios=(2.0, 1.0), hspace=0.06)
    ax_c_indicator = fig.add_subplot(grid_c[0, 0])
    ax_c_reliability = fig.add_subplot(grid_c[1, 0], sharex=ax_c_indicator)
    ax_d = fig.add_subplot(outer[3, 0])
    ax_e = fig.add_subplot(outer[3, 2])
    fig.subplots_adjust(
        left=0.11, right=0.90, bottom=0.065, top=0.94, hspace=0.28, wspace=0.0
    )

    plot_combined_climate(ax_a, precipitation, glacier, weekly_caches)
    plot_indicator_and_reliability(
        ax_b_indicator,
        ax_b_reliability,
        "A",
        scenario_index,
        scenario_indicators["A"],
        weekly_caches,
        trigger_style,
        "b",
    )
    plot_indicator_and_reliability(
        ax_c_indicator,
        ax_c_reliability,
        "B",
        scenario_index,
        scenario_indicators["B"],
        weekly_caches,
        trigger_style,
        "c",
    )
    contracts, restrictions = plot_action_bars(
        ax_d, ax_e, selections, scenario_index
    )
    fig.suptitle(
        f"Low-glacier scenario rank {rank}: {scenario_name}\n"
        f"2080–2100 mean glacier melt = {melt_metric:.3g} Mm³; "
        f"action triggers shown as {trigger_style}",
        fontsize=8,
    )
    save_figure(
        fig,
        [ax_a, ax_b_indicator, ax_b_reliability, ax_c_indicator, ax_c_reliability],
        [ax_d, ax_e],
        output_base,
        dpi,
    )
    return contracts, restrictions


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_weekly_action_totals(selections, weekly_caches):
    """Require new weekly simulations to reproduce the existing action caches."""
    for period in PERIODS:
        for indicator in INDICATORS:
            selection = selections[(period, indicator)]
            cache = weekly_caches[(period, indicator)]
            weekly_contracts = np.asarray(cache["contract_action"], dtype=float).sum(
                axis=0
            )
            weekly_restriction = (
                np.maximum(
                    np.asarray(cache["unrestricted_demand"], dtype=float)
                    - np.asarray(cache["restricted_demand"], dtype=float),
                    0.0,
                ).sum(axis=0)
                * SECONDS_PER_WEEK
                / 1_000_000.0
            )
            if not np.allclose(
                weekly_contracts,
                selection["total_contracts"],
                rtol=0.0,
                atol=1e-6,
            ):
                difference = np.max(
                    np.abs(weekly_contracts - selection["total_contracts"])
                )
                raise RuntimeError(
                    f"Weekly contract totals differ from the existing action cache "
                    f"for {period}/{indicator}; max difference={difference:.12g}"
                )
            if not np.allclose(
                weekly_restriction,
                selection["total_demand_restricted_mm3"],
                rtol=0.0,
                atol=1e-6,
            ):
                difference = np.max(
                    np.abs(
                        weekly_restriction
                        - selection["total_demand_restricted_mm3"]
                    )
                )
                raise RuntimeError(
                    f"Weekly restriction totals differ from the existing action cache "
                    f"for {period}/{indicator}; max difference={difference:.12g} Mm3"
                )


def main():
    args = parse_args()
    trigger_styles = tuple(dict.fromkeys(args.trigger_style))
    results_dir = args.results_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir
        / "out_of_sample"
        / "low_glacier_optimal_policy_timeseries"
        / "reviewer_reliability"
    )
    reliability_cache_dir = (
        args.reliability_cache_dir.expanduser().resolve()
        if args.reliability_cache_dir is not None
        else results_dir
        / "out_of_sample"
        / "low_glacier_optimal_policy_timeseries"
        / "reviewer_reliability_cache"
    )
    performance_cache_dir = results_dir / "out_of_sample" / "simulation_cache"
    action_dir = results_dir / "out_of_sample" / "cdf_actions" / "simulation_cache"
    for path in (
        results_dir,
        data_root,
        performance_cache_dir,
        action_dir,
        reliability_cache_dir,
        data_root / original.GLACIER_FILENAME,
        data_root / original.PRECIP_FOLDER,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = original.select_policy_set(
        performance_cache_dir,
        action_dir,
        "highest_urban_reliability",
    )
    scenario_names = selections[("A", "SPI12")]["performance"][
        "scenario_names"
    ].astype(str)
    scenario_lookup = {name: index for index, name in enumerate(scenario_names)}

    no_policy_vector = np.asarray((-4.0, 0.0, -4.0, 1.0), dtype=float)
    weekly_caches = {}
    for period in PERIODS:
        weekly_caches[(period, "no_policy")] = load_weekly_cache(
            reliability_cache_dir,
            period,
            "no_policy",
            scenario_names,
            no_policy_vector,
        )
        for indicator in INDICATORS:
            selection = selections[(period, indicator)]
            cache = load_weekly_cache(
                reliability_cache_dir,
                period,
                indicator,
                scenario_names,
                selection["policy"],
            )
            cached_index = int(
                np.asarray(cache["source_policy_index"]).reshape(-1)[0]
            )
            if cached_index != selection["source_policy_index"]:
                raise ValueError(
                    f"Weekly cache source policy index mismatch for {period}/{indicator}"
                )
            weekly_caches[(period, indicator)] = cache

    validate_weekly_action_totals(selections, weekly_caches)

    glacier, selected_scenarios, climate_rows = original.load_glacier_and_rank(
        data_root / original.GLACIER_FILENAME, scenario_names
    )
    write_csv(output_dir / "glacier_melt_scenario_ranking.csv", climate_rows)
    indicators = {}
    for period in PERIODS:
        config = original.PERIOD_CONFIG[period]
        indicators[period] = {
            indicator: original.load_indicator(
                data_root / config["data_folder"] / f"{indicator}.csv",
                scenario_names,
            )
            for indicator in INDICATORS
        }

    summary_rows = []
    for rank, (melt_metric, scenario_name) in enumerate(selected_scenarios, start=1):
        scenario_index = scenario_lookup[scenario_name]
        precipitation = original.load_spatial_mean_precipitation(
            original.precipitation_path(data_root, scenario_name)
        )
        scenario_glacier = glacier[scenario_name].rename("glacier_melt")
        scenario_indicators = {
            period: {
                indicator: indicators[period][indicator][scenario_name]
                for indicator in INDICATORS
            }
            for period in PERIODS
        }
        safe_name = scenario_name.replace("_no_drought_pert", "")
        for trigger_style in trigger_styles:
            figure_dir = output_dir / f"trigger_{trigger_style}"
            figure_dir.mkdir(parents=True, exist_ok=True)
            make_figure(
                figure_dir / f"rank{rank:02d}_{safe_name}",
                scenario_name,
                rank,
                melt_metric,
                precipitation,
                scenario_glacier,
                scenario_indicators,
                selections,
                weekly_caches,
                scenario_index,
                trigger_style,
                args.dpi,
            )
        for period in PERIODS:
            row = {
                "rank": rank,
                "scenario_name": scenario_name,
                "scenario_index": scenario_index,
                "mean_glacier_melt_2080_2099_mm3": melt_metric,
                "period": period,
                "period_label": FORMAL_PERIOD_LABELS[period],
            }
            for policy_type in ("no_policy", "SPI12", "SRI12"):
                cache = weekly_caches[(period, policy_type)]
                prefix = policy_type.lower()
                row[f"{prefix}_final_reliability"] = cache[
                    "official_reliability"
                ][scenario_index]
                row[f"{prefix}_failure_weeks"] = int(
                    np.asarray(cache["weekly_failure"][:, scenario_index]).sum()
                )
                row[f"{prefix}_trigger_weeks"] = int(
                    np.asarray(cache["trigger_union"][:, scenario_index]).sum()
                )
            summary_rows.append(row)
    write_csv(output_dir / "selected_scenario_reliability_summary.csv", summary_rows)

    method_path = output_dir / "method.txt"
    method_path.write_text(
        "Sep16 low-glacier reliability time-series method\n\n"
        "The ten scenarios, SPI12/SRI12 policies, and deterministic tie rules "
        "are identical to the existing highest-urban-reliability low-glacier "
        "workflow. Six fixed-policy simulations evaluate all 108 OOS scenarios: "
        "no policy, SPI12, and SRI12 in each period. No optimization is rerun. "
        "The no-policy counterfactual has zero contract purchases and retains "
        "100% of unrestricted urban demand. Sep16 simulations explicitly use "
        "$1/m3 curtailed-demand cost and a minimum demand-retention factor of "
        "0.7.\n\n"
        "Panel a overlays both periods on elapsed time. Weekly precipitation is "
        "summed by calendar month; glacier melt is averaged by calendar month. "
        "Panels b and c show SPI12 and SRI12 plus cumulative urban reliability "
        "for no policy, the SPI12 policy, and the SRI12 policy. An action-active "
        "week is the union of positive contract action and a demand-restriction "
        "factor below one. Trigger-points and trigger-fill figures use the same "
        "underlying weeks. Panels d and e retain the existing cumulative contract "
        "and demand-restriction bars. Weekly failure and cumulative reliability "
        "use the formal restricted-demand definition, and every final cumulative "
        "value is required to match reliability_PT1.\n",
        encoding="utf-8",
    )
    print(f"Saved 10 scenarios for trigger styles {trigger_styles} to: {output_dir}")


if __name__ == "__main__":
    main()
