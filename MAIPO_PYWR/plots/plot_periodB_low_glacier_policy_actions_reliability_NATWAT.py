#!/usr/bin/env python3
"""Plot SPI12/SRI12 actions, unmet demand, and reliability for Period A or B.

The script reuses the completed Sep16 all-108 weekly fixed-policy caches. It
does not run Pywr or Borg. It can plot either the deterministic ten
lowest-glacier-melt OOS scenarios or all 108 OOS scenarios in ascending
glacier-melt rank, using the same highest-urban-reliability SPI12/SRI12
policies as the existing low-glacier workflow.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import plot_low_glacier_optimal_policy_timeseries_NATWAT as original
import plot_low_glacier_policy_reliability_timeseries_NATWAT as reliability_plot


# Set explicitly from the required --period argument in main().
PERIOD = None
INDICATORS = ("SRI12", "SPI12")
POLICY_COLORS = {"SPI12": "#3d85c6", "SRI12": "#d47a1f"}
THRESHOLD_COLORS = {"SPI12": "#8ab6d6", "SRI12": "#e2a164"}
THRESHOLD_STYLES = {
    "SPI12": (0, (6, 3)),
    "SRI12": (0, (2, 2)),
}
PRECIPITATION_COLOR = "#4f8fbd"
GLACIER_COLOR = "#997700"
SECONDS_PER_WEEK = 7 * 24 * 60 * 60
ACTION_EPSILON = 1e-6
COMPACT_SCENARIO = "ACCESS-CM2_ssp585_scen03"


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
    parser.add_argument(
        "--period",
        choices=("A", "B"),
        required=True,
        help="Planning period: A is 2020-2040 and B is 2080-2100.",
    )
    parser.add_argument("--reliability-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alignment-dir", type=Path, default=None)
    parser.add_argument(
        "--plot-style",
        nargs="+",
        choices=("action_bands", "binary_actions"),
        default=("action_bands", "binary_actions"),
        help="Generate one or both alternative action visualizations.",
    )
    parser.add_argument(
        "--scenario-scope",
        choices=("lowest10", "all"),
        default="lowest10",
        help="Plot the lowest-melt 10 scenarios or all 108 in melt-rank order.",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help=(
            "Optionally plot only one exact OOS scenario name; the "
            "_no_drought_pert suffix may be omitted."
        ),
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        choices=("png", "pdf", "svg", "tiff"),
        default=("png", "pdf", "svg", "tiff"),
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write figures directly in --output-dir; requires one plot style.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip CSV and method outputs; useful for a PNG-only figure folder.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.flat_output and len(set(args.plot_style)) != 1:
        parser.error("--flat-output requires exactly one --plot-style")
    return args


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


def scenario_display_name(scenario_name):
    match = re.match(
        r"^(?P<model>.+?)_ssp(?P<ssp>\d{3})_scen(?P<scenario>\d+)",
        scenario_name,
    )
    if match is None:
        return scenario_name.replace("_no_drought_pert", "").replace("_", " ")
    digits = match.group("ssp")
    forcing = f"{digits[0]}-{digits[1]}.{digits[2]}"
    return (
        f"{match.group('model')} SSP{forcing}, "
        f"scenario {int(match.group('scenario'))}"
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def monthly_series(values, dates, name, aggregation="sum"):
    series = pd.Series(np.asarray(values, dtype=float), index=dates, name=name)
    if not np.all(np.isfinite(series.to_numpy())):
        raise ValueError(f"Non-finite weekly values in {name}")
    if aggregation == "sum":
        result = series.resample("MS").sum(min_count=1)
    elif aggregation == "mean":
        result = series.resample("MS").mean()
    else:
        raise ValueError(f"Unsupported monthly aggregation: {aggregation}")
    if result.isna().any():
        raise ValueError(f"Missing monthly values in {name}")
    return result


def monthly_climate(series, dates, name, aggregation):
    selected = series.loc[(series.index >= dates[0]) & (series.index <= dates[-1])]
    if selected.empty:
        raise ValueError(f"No {name} data within the formal Period-{PERIOD} dates")
    selected = selected.rename(name)
    if aggregation == "sum":
        monthly = selected.resample("MS").sum(min_count=1)
    elif aggregation == "mean":
        monthly = selected.resample("MS").mean()
    else:
        raise ValueError(f"Unsupported climate aggregation: {aggregation}")
    if monthly.isna().any():
        raise ValueError(f"Missing monthly {name} values")
    return monthly


def policy_thresholds(selection):
    policy = np.asarray(selection["policy"], dtype=float)
    contract_threshold = float(policy[0])
    demand_threshold = float(policy[2])
    if not np.all(np.isfinite((contract_threshold, demand_threshold))):
        raise ValueError("Policy contains a non-finite threshold")
    return contract_threshold, demand_threshold


def aligned_indicator(series, dates, indicator):
    aligned = series.reindex(dates)
    if aligned.isna().any():
        missing = dates[aligned.isna()]
        raise ValueError(
            f"{indicator} is missing {len(missing)} formal weekly dates; "
            f"first={missing[0] if len(missing) else 'none'}"
        )
    return aligned


def derive_monthly_actions(cache, scenario_index, indicator):
    dates = cache["dates"]
    contracts = monthly_series(
        cache["contract_action"][:, scenario_index],
        dates,
        f"{indicator}_water_contracts",
    )
    restricted_flow = np.maximum(
        cache["unrestricted_demand"][:, scenario_index]
        - cache["restricted_demand"][:, scenario_index],
        0.0,
    )
    restriction_volume = monthly_series(
        restricted_flow * SECONDS_PER_WEEK / 1_000_000.0,
        dates,
        f"{indicator}_demand_restriction_mm3",
    )
    unmet_flow = np.maximum(
        cache["restricted_demand"][:, scenario_index]
        - cache["actual_pt1_delivery"][:, scenario_index],
        0.0,
    )
    unmet_volume = monthly_series(
        unmet_flow * SECONDS_PER_WEEK / 1_000_000.0,
        dates,
        f"{indicator}_unmet_urban_demand_mm3",
    )
    return contracts, restriction_volume, unmet_volume


def validate_action_totals(selections, weekly_caches):
    """Require weekly cache totals to match the existing CDF-action cache."""
    for indicator in ("SPI12", "SRI12"):
        cache = weekly_caches[(PERIOD, indicator)]
        selection = selections[(PERIOD, indicator)]
        contract_totals = np.asarray(cache["contract_action"], dtype=float).sum(
            axis=0
        )
        restriction_totals = (
            np.maximum(
                np.asarray(cache["unrestricted_demand"], dtype=float)
                - np.asarray(cache["restricted_demand"], dtype=float),
                0.0,
            ).sum(axis=0)
            * SECONDS_PER_WEEK
            / 1_000_000.0
        )
        if not np.allclose(
            contract_totals,
            selection["total_contracts"],
            rtol=0.0,
            atol=1e-6,
        ):
            difference = np.max(
                np.abs(contract_totals - selection["total_contracts"])
            )
            raise RuntimeError(
                f"{indicator} contract totals disagree with the existing "
                f"CDF-action cache; max difference={difference:.12g}"
            )
        if not np.allclose(
            restriction_totals,
            selection["total_demand_restricted_mm3"],
            rtol=0.0,
            atol=1e-6,
        ):
            difference = np.max(
                np.abs(
                    restriction_totals
                    - selection["total_demand_restricted_mm3"]
                )
            )
            raise RuntimeError(
                f"{indicator} restriction totals disagree with the existing "
                f"CDF-action cache; max difference={difference:.12g} Mm3"
            )


def format_date_axis(axis, show_labels=True):
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(axis="x", labelbottom=show_labels)
    axis.grid(False)


def plot_climate(axis, precipitation, glacier, dates):
    monthly_p = monthly_climate(
        precipitation, dates, "precipitation", aggregation="sum"
    )
    monthly_g = monthly_climate(
        glacier, dates, "glacier_melt", aggregation="mean"
    )
    line_p = axis.plot(
        monthly_p.index,
        monthly_p.values,
        color=PRECIPITATION_COLOR,
        linewidth=1.8,
        label="Precipitation",
        zorder=3,
    )[0]
    glacier_is_zero = np.allclose(
        monthly_g.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-12
    )
    twin = None
    if glacier_is_zero:
        # A small negative precipitation margin keeps the exact zero line
        # visible without adding an empty secondary axis.
        precipitation_top = max(1.0, float(monthly_p.max()) * 1.08)
        axis.set_ylim(-0.025 * precipitation_top, precipitation_top)
        line_g = axis.plot(
            monthly_g.index,
            np.zeros(len(monthly_g), dtype=float),
            color=GLACIER_COLOR,
            linewidth=1.2,
            label="Glacier melt = 0 Mm³",
            zorder=4,
        )[0]
    else:
        twin = axis.twinx()
        line_g = twin.plot(
            monthly_g.index,
            monthly_g.values,
            color=GLACIER_COLOR,
            linewidth=1.5,
            label="Glacier melt",
            zorder=4,
        )[0]
        glacier_max = max(float(monthly_g.max()), 1.0)
        twin.set_ylim(-0.025 * glacier_max, 1.12 * glacier_max)
        twin.set_xlim(dates[0], dates[-1])
        twin.set_ylabel("Glacier melt (Mm³)", color=GLACIER_COLOR, labelpad=7)
        twin.tick_params(axis="y", colors=GLACIER_COLOR, labelsize=6.5)
        twin.spines["right"].set_visible(True)
        twin.spines["right"].set_color("0.7")
    axis.set_xlim(dates[0], dates[-1])
    if not glacier_is_zero:
        axis.set_ylim(0.0, max(1.0, float(monthly_p.max()) * 1.08))
    axis.set_ylabel("Monthly precipitation (mm)", color=PRECIPITATION_COLOR)
    axis.tick_params(axis="y", colors=PRECIPITATION_COLOR)
    format_date_axis(axis, show_labels=False)
    axis.legend(
        [line_p, line_g],
        [line_p.get_label(), line_g.get_label()],
        loc="upper right",
        ncol=2,
        fontsize=6.5,
    )
    axis.set_title("Hydroclimatic forcing", loc="left", fontsize=7.5, pad=7)
    add_panel_label(axis, "a")
    return twin, monthly_p, monthly_g


def plot_indicators(
    axis,
    scenario_indicators,
    selections,
    dates,
    weekly_caches=None,
    scenario_index=None,
    show_action_bands=False,
):
    if show_action_bands:
        if weekly_caches is None or scenario_index is None:
            raise ValueError("Action-band plotting requires weekly caches and scenario index")
        for indicator in ("SPI12", "SRI12"):
            trigger = np.asarray(
                weekly_caches[(PERIOD, indicator)]["trigger_union"][:, scenario_index],
                dtype=bool,
            )
            axis.fill_between(
                dates,
                0.0,
                1.0,
                where=trigger,
                step="post",
                transform=axis.get_xaxis_transform(),
                color=POLICY_COLORS[indicator],
                alpha=0.10,
                linewidth=0.0,
                zorder=0,
            )
    indicator_handles = []
    for indicator in ("SPI12", "SRI12"):
        values = aligned_indicator(
            scenario_indicators[indicator], dates, indicator
        )
        indicator_handles.append(
            axis.plot(
                dates,
                values.values,
                color=POLICY_COLORS[indicator],
                linewidth=1.35,
                label=indicator,
                zorder=3,
            )[0]
        )

    threshold_handles = []
    for indicator in ("SPI12", "SRI12"):
        thresholds = policy_thresholds(selections[(PERIOD, indicator)])
        first_handle = None
        for threshold in thresholds:
            handle = axis.axhline(
                threshold,
                color=THRESHOLD_COLORS[indicator],
                linestyle=THRESHOLD_STYLES[indicator],
                linewidth=1.1,
                zorder=1,
            )
            if first_handle is None:
                first_handle = handle
        threshold_handles.append(first_handle)
    axis.set_xlim(dates[0], dates[-1])
    axis.set_ylim(-6.5, 5.0)
    axis.set_ylabel("SPI or SRI (-)")
    format_date_axis(axis, show_labels=False)
    axis.legend(
        indicator_handles + threshold_handles,
        ["SPI12", "SRI12", "SPI12 thresholds", "SRI12 thresholds"],
        loc="lower right",
        ncol=2,
        fontsize=6.2,
    )
    axis.set_title("Drought indicators", loc="left", fontsize=7.5, pad=7)
    add_panel_label(axis, "b")


def plot_binary_actions(axis, weekly_caches, scenario_index, dates):
    twin = axis.twinx()
    handles = []
    for indicator in INDICATORS:
        cache = weekly_caches[(PERIOD, indicator)]
        contract_on = (
            np.asarray(cache["contract_action"][:, scenario_index], dtype=float)
            > ACTION_EPSILON
        ).astype(float)
        demand_on = (
            np.asarray(
                cache["demand_restriction_factor"][:, scenario_index], dtype=float
            )
            < 1.0 - ACTION_EPSILON
        ).astype(float)
        handles.append(
            axis.plot(
                dates,
                contract_on,
                color=POLICY_COLORS[indicator],
                linestyle="-",
                linewidth=1.25,
                drawstyle="steps-post",
                label=f"{indicator} contracts",
                zorder=3,
            )[0]
        )
        handles.append(
            twin.plot(
                dates,
                demand_on,
                color=POLICY_COLORS[indicator],
                linestyle="--",
                linewidth=1.25,
                drawstyle="steps-post",
                label=f"{indicator} demand restriction",
                zorder=2,
            )[0]
        )
    axis.set_xlim(dates[0], dates[-1])
    twin.set_xlim(dates[0], dates[-1])
    axis.set_ylim(-0.08, 1.08)
    twin.set_ylim(-0.08, 1.08)
    axis.set_yticks((0.0, 1.0), ("Off", "On"))
    twin.set_yticks((0.0, 1.0), ("Off", "On"))
    axis.set_ylabel("Contracts")
    twin.set_ylabel("Demand restriction")
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color("0.7")
    format_date_axis(axis, show_labels=False)
    legend_handles = [handles[0], handles[2], handles[1], handles[3]]
    axis.legend(
        legend_handles,
        [
            "SRI12 contracts",
            "SPI12 contracts",
            "SRI12 demand restriction",
            "SPI12 demand restriction",
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        fontsize=6.1,
    )
    axis.set_title("Weekly action status", loc="left", fontsize=7.5, pad=9)
    add_panel_label(axis, "c")
    return twin


def plot_monthly_actions(axis, monthly, dates):
    twin = axis.twinx()
    width_days = 11.0
    offsets = {"SRI12": -4.0, "SPI12": 4.0}
    handles = []
    for indicator in INDICATORS:
        contracts, restrictions, _ = monthly[indicator]
        shifted_dates = contracts.index + pd.to_timedelta(
            offsets[indicator], unit="D"
        )
        handles.append(
            axis.bar(
                shifted_dates,
                contracts.values,
                width=width_days,
                color=POLICY_COLORS[indicator],
                alpha=0.27,
                linewidth=0.0,
                label=f"{indicator} contracts",
                zorder=2,
            )
        )
        handles.append(
            twin.plot(
                restrictions.index,
                restrictions.values,
                color=POLICY_COLORS[indicator],
                linewidth=1.25,
                linestyle="-" if indicator == "SRI12" else "--",
                label=f"{indicator} demand restriction",
                zorder=3,
            )[0]
        )
    axis.set_xlim(dates[0], dates[-1])
    twin.set_xlim(dates[0], dates[-1])
    axis.set_ylim(bottom=0.0)
    twin.set_ylim(bottom=0.0)
    axis.set_ylabel("Contracts (shares)", labelpad=7)
    twin.set_ylabel("Demand restriction (Mm³)", labelpad=8)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color("0.7")
    twin.tick_params(axis="y", labelsize=6.5)
    format_date_axis(axis, show_labels=False)
    legend_handles = [handles[0], handles[2], handles[1], handles[3]]
    legend_labels = [
        "SRI12 contracts",
        "SPI12 contracts",
        "SRI12 demand restriction",
        "SPI12 demand restriction",
    ]
    axis.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        fontsize=6.1,
    )
    axis.set_title("Monthly management actions", loc="left", fontsize=7.5, pad=9)
    add_panel_label(axis, "c")
    return twin


def plot_monthly_unmet(axis, monthly, dates):
    width_days = 12.0
    offsets = {"SRI12": -4.0, "SPI12": 4.0}
    for indicator in INDICATORS:
        unmet = monthly[indicator][2]
        axis.bar(
            unmet.index + pd.to_timedelta(offsets[indicator], unit="D"),
            unmet.values,
            width=width_days,
            color=POLICY_COLORS[indicator],
            alpha=0.72,
            linewidth=0.0,
            label=indicator,
        )
    axis.set_xlim(dates[0], dates[-1])
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Unmet urban demand (Mm³/month)")
    format_date_axis(axis, show_labels=True)
    axis.legend(loc="upper right", ncol=2, fontsize=6.5)
    axis.set_title("Monthly shortage events", loc="left", fontsize=7.5, pad=7)
    add_panel_label(axis, "d")


def plot_total_actions(axis, monthly):
    twin = axis.twinx()
    group_centers = np.asarray((0.0, 1.0), dtype=float)
    width = 0.26
    policy_offsets = {"SRI12": -width / 2, "SPI12": width / 2}
    contract_totals = np.asarray(
        [monthly[indicator][0].sum() for indicator in INDICATORS], dtype=float
    )
    restriction_totals = np.asarray(
        [monthly[indicator][1].sum() for indicator in INDICATORS], dtype=float
    )
    for index, indicator in enumerate(INDICATORS):
        color = POLICY_COLORS[indicator]
        axis.bar(
            group_centers[0] + policy_offsets[indicator],
            contract_totals[index],
            width=width,
            color=color,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.78,
        )
        twin.bar(
            group_centers[1] + policy_offsets[indicator],
            restriction_totals[index],
            width=width,
            color=color,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.78,
        )
    axis.set_xticks(group_centers, ("Water contracts", "Demand restriction"))
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(bottom=0.0)
    twin.set_ylim(bottom=0.0)
    axis.set_ylabel("Total contracts (shares)")
    twin.set_ylabel("Total 20-year demand restriction (Mm³)")
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color("0.65")
    policy_handles = [
        Patch(
            facecolor=POLICY_COLORS["SRI12"],
            edgecolor=POLICY_COLORS["SRI12"],
            alpha=0.78,
            label="SRI12",
        ),
        Patch(
            facecolor=POLICY_COLORS["SPI12"],
            edgecolor=POLICY_COLORS["SPI12"],
            alpha=0.78,
            label="SPI12",
        ),
    ]
    axis.legend(
        policy_handles,
        ["SRI12", "SPI12"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=6.0,
    )
    add_panel_label(axis, "e")
    return twin, contract_totals, restriction_totals


def plot_final_reliability(
    axis, weekly_caches, scenario_index, compact_format=False
):
    values = np.asarray(
        [
            weekly_caches[(PERIOD, indicator)]["official_reliability"][
                scenario_index
            ]
            for indicator in INDICATORS
        ],
        dtype=float,
    )
    x = np.arange(len(INDICATORS), dtype=float)
    bars = axis.bar(
        x,
        values,
        width=0.55,
        color=[POLICY_COLORS[indicator] for indicator in INDICATORS],
    )
    axis.set_xticks(x, INDICATORS)
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel(
        "Average urban reliability"
        if compact_format
        else "Final cumulative urban reliability"
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2)
    if not compact_format:
        axis.set_title("Final outcome", loc="left", fontsize=7.5, pad=10)
    add_panel_label(axis, "f")
    return values


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
    fig,
    full_axes,
    bottom_axes,
    output_base,
    full_labels,
    bottom_labels,
):
    tolerance_pt = 1.5
    fig.canvas.draw()
    full_panels = [
        axis_geometry(fig, axis, label)
        for axis, label in zip(full_axes, full_labels)
    ]
    bottom_panels = [
        axis_geometry(fig, axis, label)
        for axis, label in zip(bottom_axes, bottom_labels)
    ]
    panels = full_panels + bottom_panels
    deviations = {}
    for field in ("left", "right", "width"):
        values = [panel[field] for panel in full_panels]
        deviations[f"full_width_{field}"] = max(values) - min(values)
    for field in ("top", "bottom", "width", "height"):
        values = [panel[field] for panel in bottom_panels]
        deviations[f"bottom_row_{field}"] = max(values) - min(values)
    deviations["outer_left"] = abs(
        full_panels[0]["left"] - bottom_panels[0]["left"]
    )
    deviations["outer_right"] = abs(
        full_panels[0]["right"] - bottom_panels[-1]["right"]
    )
    passed = all(value <= tolerance_pt for value in deviations.values())
    report = {
        "schema_version": 1,
        "figure_size_pt": [
            float(value) for value in fig.get_size_inches() * 72.0
        ],
        "tolerance_pt": tolerance_pt,
        "panels": panels,
        "deviations_pt": deviations,
        "verdict": "PASS" if passed else "FIX BEFORE DELIVERY",
    }
    Path(f"{output_base}.alignment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Six-panel alignment failed: {deviations}")


def save_figure(
    fig,
    full_axes,
    bottom_axes,
    output_base,
    dpi,
    full_labels,
    bottom_labels,
    output_formats,
    alignment_output_base=None,
):
    require_matplotlib_panel_alignment(
        fig,
        full_axes,
        bottom_axes,
        alignment_output_base or output_base,
        full_labels,
        bottom_labels,
    )
    if "png" in output_formats:
        fig.savefig(f"{output_base}.png", dpi=dpi, bbox_inches="tight")
    if "pdf" in output_formats:
        fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    if "svg" in output_formats:
        fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    if "tiff" in output_formats:
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
    plot_style,
    dpi,
    output_formats,
    alignment_output_base=None,
):
    compact_format = (
        scenario_name.replace("_no_drought_pert", "") == COMPACT_SCENARIO
    )
    dates = weekly_caches[(PERIOD, "SPI12")]["dates"]
    monthly = {
        indicator: derive_monthly_actions(
            weekly_caches[(PERIOD, indicator)], scenario_index, indicator
        )
        for indicator in INDICATORS
    }

    if plot_style == "action_bands":
        fig = plt.figure(
            figsize=(7.2, 9.3 if compact_format else 10.5),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            4,
            3,
            height_ratios=(1.0, 1.28, 1.08, 1.0),
            width_ratios=(1.0, 0.58, 1.0),
        )
        ax_a = fig.add_subplot(grid[0, :])
        ax_b = fig.add_subplot(grid[1, :])
        ax_c = None
        ax_d = fig.add_subplot(grid[2, :])
        ax_e = fig.add_subplot(grid[3, 0])
        ax_f = fig.add_subplot(grid[3, 2])
    elif plot_style == "binary_actions":
        fig = plt.figure(
            figsize=(7.2, 11.0 if compact_format else 12.6),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            5,
            3,
            height_ratios=(1.0, 1.18, 1.28, 1.08, 1.0),
            width_ratios=(1.0, 0.58, 1.0),
        )
        ax_a = fig.add_subplot(grid[0, :])
        ax_b = fig.add_subplot(grid[1, :])
        ax_c = fig.add_subplot(grid[2, :])
        ax_d = fig.add_subplot(grid[3, :])
        ax_e = fig.add_subplot(grid[4, 0])
        ax_f = fig.add_subplot(grid[4, 2])
    else:
        raise ValueError(f"Unknown plot style: {plot_style}")
    fig.subplots_adjust(
        left=0.12,
        right=0.88,
        bottom=0.055,
        top=0.98 if compact_format else 0.925,
        hspace=0.36 if compact_format else 0.58,
        wspace=0.0,
    )

    _, monthly_p, monthly_g = plot_climate(
        ax_a, precipitation, glacier, dates
    )
    plot_indicators(
        ax_b,
        scenario_indicators,
        selections,
        dates,
        weekly_caches=weekly_caches,
        scenario_index=scenario_index,
        show_action_bands=(plot_style == "action_bands"),
    )
    if plot_style == "binary_actions":
        plot_binary_actions(ax_c, weekly_caches, scenario_index, dates)
    plot_monthly_unmet(ax_d, monthly, dates)
    _, contract_totals, restriction_totals = plot_total_actions(ax_e, monthly)
    final_reliability = plot_final_reliability(
        ax_f,
        weekly_caches,
        scenario_index,
        compact_format=compact_format,
    )
    if not compact_format:
        period_label = original.PERIOD_CONFIG[PERIOD]["label"]
        fig.suptitle(
            f"{period_label} OOS scenario — {scenario_display_name(scenario_name)}",
            fontsize=9,
            fontweight="bold",
            y=0.972,
        )
        fig.text(
            0.5,
            0.952,
            f"Mean glacier melt = {melt_metric:.3g} Mm³",
            ha="center",
            va="center",
            fontsize=7,
            color="0.35",
        )
    save_figure(
        fig,
        [ax_a, ax_b, ax_d]
        if plot_style == "action_bands"
        else [ax_a, ax_b, ax_c, ax_d],
        [ax_e, ax_f],
        output_base,
        dpi,
        ("a", "b", "d")
        if plot_style == "action_bands"
        else ("a", "b", "c", "d"),
        ("e", "f"),
        output_formats,
        alignment_output_base=alignment_output_base,
    )
    return monthly, monthly_p, monthly_g, contract_totals, restriction_totals, final_reliability


def main():
    global PERIOD
    args = parse_args()
    PERIOD = args.period
    plot_styles = tuple(dict.fromkeys(args.plot_style))
    output_formats = tuple(dict.fromkeys(args.output_formats))
    results_dir = args.results_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    performance_cache_dir = results_dir / "out_of_sample" / "simulation_cache"
    action_cache_dir = (
        results_dir / "out_of_sample" / "cdf_actions" / "simulation_cache"
    )
    reliability_cache_dir = (
        args.reliability_cache_dir.expanduser().resolve()
        if args.reliability_cache_dir is not None
        else results_dir
        / "out_of_sample"
        / "low_glacier_optimal_policy_timeseries"
        / "reviewer_reliability_cache"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir
        / "out_of_sample"
        / "low_glacier_optimal_policy_timeseries"
        / f"period{PERIOD}_actions_reliability"
    )
    alignment_dir = (
        args.alignment_dir.expanduser().resolve()
        if args.alignment_dir is not None
        else None
    )
    required_paths = (
        results_dir,
        data_root,
        performance_cache_dir,
        action_cache_dir,
        reliability_cache_dir,
        data_root / original.GLACIER_FILENAME,
        data_root / original.PRECIP_FOLDER,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if alignment_dir is not None:
        alignment_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir = output_dir / "source_data"
    if not args.figures_only:
        source_data_dir.mkdir(parents=True, exist_ok=True)

    selections = {
        (PERIOD, indicator): original.selected_policy(
            performance_cache_dir,
            action_cache_dir,
            PERIOD,
            indicator,
            selection_mode="highest_urban_reliability",
        )
        for indicator in ("SPI12", "SRI12")
    }
    scenario_names = selections[(PERIOD, "SPI12")]["performance"][
        "scenario_names"
    ].astype(str)
    if not np.array_equal(
        scenario_names,
        selections[(PERIOD, "SRI12")]["performance"]["scenario_names"].astype(str),
    ):
        raise RuntimeError("SPI12 and SRI12 scenario order differs")
    scenario_lookup = {name: index for index, name in enumerate(scenario_names)}

    weekly_caches = {}
    for indicator in ("SPI12", "SRI12"):
        selection = selections[(PERIOD, indicator)]
        cache = reliability_plot.load_weekly_cache(
            reliability_cache_dir,
            PERIOD,
            indicator,
            scenario_names,
            selection["policy"],
        )
        cached_index = int(
            np.asarray(cache["source_policy_index"]).reshape(-1)[0]
        )
        if cached_index != selection["source_policy_index"]:
            raise ValueError(f"Weekly cache policy index mismatch for {indicator}")
        weekly_caches[(PERIOD, indicator)] = cache

    validate_action_totals(selections, weekly_caches)
    for indicator in ("SRI12", "SPI12"):
        contract_threshold, demand_threshold = policy_thresholds(
            selections[(PERIOD, indicator)]
        )
        print(
            f"{indicator} thresholds: contract={contract_threshold:g}, "
            f"demand={demand_threshold:g}"
        )

    glacier, selected_scenarios, climate_rows = original.load_glacier_and_rank(
        data_root / original.GLACIER_FILENAME, scenario_names
    )
    ranked_scenarios = [
        (
            float(row["mean_glacier_melt_2080_2099_mm3"]),
            str(row["scenario_name"]),
        )
        for row in climate_rows
    ]
    rank_lookup = {
        str(row["scenario_name"]): int(row["deterministic_rank"])
        for row in climate_rows
    }
    if args.scenario_name is not None:
        requested = args.scenario_name
        if not requested.endswith("_no_drought_pert"):
            requested += "_no_drought_pert"
        scenarios_to_plot = [
            item for item in ranked_scenarios if item[1] == requested
        ]
        if len(scenarios_to_plot) != 1:
            raise ValueError(f"OOS scenario was not found: {args.scenario_name}")
    elif args.scenario_scope == "all":
        scenarios_to_plot = ranked_scenarios
        if len(scenarios_to_plot) != 108:
            raise RuntimeError(
                "Expected exactly 108 OOS scenarios for --scenario-scope all; "
                f"found {len(scenarios_to_plot)}"
            )
    else:
        scenarios_to_plot = selected_scenarios
    if not args.figures_only:
        write_csv(output_dir / "glacier_melt_scenario_ranking.csv", climate_rows)
    indicator_frame = {
        indicator: original.load_indicator(
            data_root
            / original.PERIOD_CONFIG[PERIOD]["data_folder"]
            / f"{indicator}.csv",
            scenario_names,
        )
        for indicator in ("SPI12", "SRI12")
    }

    summary_rows = []
    rank_width = 3 if args.scenario_scope == "all" or args.scenario_name else 2
    for melt_metric, scenario_name in scenarios_to_plot:
        rank = rank_lookup[scenario_name]
        scenario_index = scenario_lookup[scenario_name]
        precipitation = original.load_spatial_mean_precipitation(
            original.precipitation_path(data_root, scenario_name)
        )
        scenario_glacier = glacier[scenario_name].rename("glacier_melt")
        scenario_indicators = {
            indicator: indicator_frame[indicator][scenario_name]
            for indicator in ("SPI12", "SRI12")
        }
        safe_name = scenario_name.replace("_no_drought_pert", "")
        figure_result = None
        for plot_style in plot_styles:
            style_dir = output_dir if args.flat_output else output_dir / plot_style
            style_dir.mkdir(parents=True, exist_ok=True)
            filename = f"rank{rank:0{rank_width}d}_{safe_name}"
            if args.scenario_name is not None:
                period_tag = original.PERIOD_CONFIG[PERIOD]["label"].replace(
                    "–", "-"
                )
                filename = f"{period_tag}_{filename}"
            alignment_output_base = (
                alignment_dir / filename
                if alignment_dir is not None
                else None
            )
            current_result = make_figure(
                style_dir / filename,
                scenario_name,
                rank,
                melt_metric,
                precipitation,
                scenario_glacier,
                scenario_indicators,
                selections,
                weekly_caches,
                scenario_index,
                plot_style,
                args.dpi,
                output_formats,
                alignment_output_base=alignment_output_base,
            )
            if figure_result is None:
                figure_result = current_result
        if figure_result is None:
            raise RuntimeError("No requested plot style was generated")
        (
            monthly,
            monthly_p,
            monthly_g,
            contract_totals,
            restriction_totals,
            final_reliability,
        ) = figure_result

        if not args.figures_only:
            source_rows = []
            month_index = monthly_p.index.union(monthly_g.index)
            for indicator in INDICATORS:
                month_index = month_index.union(monthly[indicator][0].index)
            month_index = month_index.sort_values()
            for month in month_index:
                row = {
                    "month": month.strftime("%Y-%m-%d"),
                    "precipitation_mm": monthly_p.get(month, np.nan),
                    "glacier_melt_mm3": monthly_g.get(month, np.nan),
                }
                for indicator in INDICATORS:
                    contracts, restriction, unmet = monthly[indicator]
                    row[f"{indicator}_water_contracts"] = contracts.get(
                        month, np.nan
                    )
                    row[f"{indicator}_demand_restriction_mm3"] = restriction.get(
                        month, np.nan
                    )
                    row[f"{indicator}_unmet_urban_demand_mm3"] = unmet.get(
                        month, np.nan
                    )
                source_rows.append(row)
            write_csv(
                source_data_dir / f"rank{rank:0{rank_width}d}_{safe_name}.csv",
                source_rows,
            )

        summary_rows.append(
            {
                "rank": rank,
                "scenario_name": scenario_name,
                "scenario_index": scenario_index,
                "mean_glacier_melt_2080_2099_mm3": melt_metric,
                "SRI12_contract_threshold": policy_thresholds(
                    selections[(PERIOD, "SRI12")]
                )[0],
                "SRI12_demand_threshold": policy_thresholds(
                    selections[(PERIOD, "SRI12")]
                )[1],
                "SPI12_contract_threshold": policy_thresholds(
                    selections[(PERIOD, "SPI12")]
                )[0],
                "SPI12_demand_threshold": policy_thresholds(
                    selections[(PERIOD, "SPI12")]
                )[1],
                "SRI12_total_contracts": contract_totals[0],
                "SPI12_total_contracts": contract_totals[1],
                "SRI12_total_demand_restriction_mm3": restriction_totals[0],
                "SPI12_total_demand_restriction_mm3": restriction_totals[1],
                "SRI12_final_urban_reliability": final_reliability[0],
                "SPI12_final_urban_reliability": final_reliability[1],
            }
        )

    if not args.figures_only:
        write_csv(output_dir / "scenario_summary.csv", summary_rows)
        (output_dir / "method.txt").write_text(
        f"Period-{PERIOD} Sep16 low-glacier actions and reliability figure\n\n"
        "The same deterministic ten lowest-glacier-melt OOS scenarios and the "
        "same highest-urban-reliability SPI12/SRI12 policies are used as in the "
        "existing Sep16 workflow. No simulation or optimization is rerun. "
        "Monthly precipitation is the sum of weekly values and monthly glacier "
        "melt is the mean of available values. Monthly contracts are the sum of "
        "weekly contract actions. Demand-restriction and unmet-demand flows are "
        "converted from m3/s to Mm3 for each week and summed by month. Final "
        "cumulative reliability is the validated official reliability_PT1 value. "
        "Panel B plots the exact contract and demand thresholds for each indicator. "
        "The two SPI12 thresholds share one blue style and the two SRI12 thresholds "
        "share one orange style; one legend entry represents each threshold pair. "
        "The action_bands version removes Panel C and adds full-height transparent "
        "blue or orange bands to Panel B whenever either action is active for the "
        "corresponding policy. The binary_actions version leaves Panel B unchanged "
        "and uses Panel C for four weekly binary on/off series: contracts and demand "
        "restriction under the SRI12 and SPI12 policies. In both versions, Panel E "
        "groups total actions on the x-axis (water contracts first, demand "
        "restriction second), with SRI12 and SPI12 bars within each group.\n",
            encoding="utf-8",
        )
    print(
        f"Saved {len(scenarios_to_plot)} Period-{PERIOD} scenarios for plot styles "
        f"{plot_styles} and formats {output_formats} to: {output_dir}"
    )


if __name__ == "__main__":
    main()
