#!/usr/bin/env python3
"""Compute comparable normalized 3-D hypervolumes for NATWAT policies.

Training performance is read from the seven Borg ``.set`` archives. Test-set
(out-of-sample, OOS) performance is read from the completed 108-scenario NPZ
caches produced by ``simulate_out_of_sample_policies_NATWAT.py``. The script
does not rerun Pywr simulations.

By default, training and OOS fronts are processed together. One global ideal
point, one global reference point, and one normalization are then shared by
all 24 indicator-period-dataset groups, so every reported hypervolume is
directly comparable.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
DATASETS = ("training", "oos")
N_DECISIONS = 4
N_OBJECTIVES = 3
N_OOS_SCENARIOS = 108
EXPECTED_SEEDS = tuple(range(1, 8))
PERIOD_LABELS = {"A": "2020–2040", "B": "2080–2100"}
POLICY_RUN_FOLDERS = {
    "aug28": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug28_MIN",
    "sep09_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep9_MIN_DeCost",
    "sep16_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost",
}
NO_DROUGHT_OOS_DATA_FOLDERS = {
    "A": "transient_108_no_drought_pert_periodA",
    "B": "transient_108_no_drought_pert_periodB",
}
FOLDER_PATTERN = re.compile(
    r"^(?P<indicator>SPI3|SPI6|SPI12|SRI3|SRI6|SRI12)_"
    r"(?P<period>A|B)(?:_|$)",
    re.IGNORECASE,
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Calculate shared-reference normalized 3-D hypervolume for NATWAT "
            "training and/or 108-scenario OOS results."
        )
    )
    parser.add_argument(
        "--policy-run",
        choices=tuple(POLICY_RUN_FOLDERS),
        default="aug28",
        help="Policy archive/cache identity to analyze (default: aug28).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Default: the folder selected by --policy-run beside this script.",
    )
    parser.add_argument(
        "--oos-cache-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/out_of_sample/simulation_cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/hypervolume.",
    )
    parser.add_argument(
        "--datasets",
        choices=("both", "training", "oos"),
        default="both",
        help="Performance datasets to include (default: both).",
    )
    parser.add_argument(
        "--reference-margin",
        type=float,
        default=0.05,
        help="Fraction added beyond the global observed worst point (default: 0.05).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution.")
    return parser.parse_args()


def discover_experiments(results_dir):
    """Return exactly one experiment folder per indicator-period pair."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    found = {}
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        match = FOLDER_PATTERN.match(folder.name)
        if not match:
            continue
        key = (match.group("indicator").upper(), match.group("period").upper())
        if key in found:
            raise RuntimeError(
                f"More than one experiment folder matches {key}:\n"
                f"  {found[key]}\n  {folder}"
            )
        found[key] = folder

    expected = {(indicator, period) for indicator in INDICATORS for period in PERIODS}
    missing = sorted(expected - set(found))
    if missing:
        formatted = ", ".join(f"{indicator}_{period}" for indicator, period in missing)
        raise RuntimeError(f"Missing experiment folder(s): {formatted}")
    return found


def read_seed_file(path):
    """Read four decisions plus three objectives from one Borg archive."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//")):
                continue
            fields = stripped.split()
            if len(fields) < N_DECISIONS + N_OBJECTIVES:
                raise ValueError(
                    f"{path}:{line_number} has {len(fields)} columns; expected at least 7."
                )
            try:
                rows.append([float(value) for value in fields[:7]])
            except ValueError as exc:
                raise ValueError(f"Non-numeric row at {path}:{line_number}") from exc

    if not rows:
        raise RuntimeError(f"No solution rows found in {path}")
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite value found in {path}")
    return values


def load_training_experiment(folder):
    """Pool the seven expected Borg archives for one training group."""
    seed_files = [folder / "sets" / f"Borg_DPS_PySedSim{seed}.set" for seed in EXPECTED_SEEDS]
    missing = [path for path in seed_files if not path.is_file()]
    if missing:
        details = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing expected seed archive(s):\n{details}")
    return np.vstack([read_seed_file(path) for path in seed_files]), seed_files


def reliability_to_positive(raw_values, label):
    """Undo Borg's maximize-to-minimize sign conversion when present."""
    sign_was_negated = float(np.median(raw_values)) < 0.0
    values = -raw_values if sign_was_negated else raw_values.copy()
    if np.min(values) < -0.05 or np.max(values) > 1.05:
        raise ValueError(
            f"{label} values are outside the expected reliability range after sign "
            f"conversion: [{np.min(values):.6g}, {np.max(values):.6g}]"
        )
    return np.clip(values, 0.0, 1.0), sign_was_negated


def training_objectives_to_minimization(raw_rows):
    """Convert training raw objectives to [cost, urban loss, agriculture loss]."""
    raw_objectives = raw_rows[:, N_DECISIONS : N_DECISIONS + N_OBJECTIVES]
    urban, urban_negated = reliability_to_positive(
        raw_objectives[:, 0], "Training urban reliability"
    )
    agricultural, agricultural_negated = reliability_to_positive(
        raw_objectives[:, 1], "Training agricultural reliability"
    )

    raw_cost = raw_objectives[:, 2]
    cost_negated = float(np.median(raw_cost)) < 0.0
    cost = -raw_cost if cost_negated else raw_cost.copy()
    if np.min(cost) < -1e-8:
        raise ValueError(
            f"Training total cost contains negative values after sign conversion: {np.min(cost)}"
        )

    objectives = np.column_stack(
        (cost / 1_000_000.0, 1.0 - urban, 1.0 - agricultural)
    )
    sign_info = {
        "urban_reliability_raw_negated": urban_negated,
        "agricultural_reliability_raw_negated": agricultural_negated,
        "total_cost_raw_negated": cost_negated,
    }
    return objectives, sign_info


def scalar_text(value):
    """Convert a scalar NumPy string/number to plain text."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar metadata; got shape {array.shape}")
    return str(array.reshape(-1)[0])


def load_oos_objectives(cache_dir, indicator, period, expected_policy_run):
    """Load one complete OOS cache and average its 108 scenarios."""
    path = cache_dir / f"{indicator}_{period}_oos_performance.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing OOS cache: {path}\n"
            "Run the matching out-of-sample simulation job first."
        )

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
    with np.load(path, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Cache {path} is missing: {', '.join(sorted(missing))}")
        policies = saved["policies"].copy()
        scenario_names = saved["scenario_names"].astype(str)
        urban = saved["urban_reliability"].copy()
        agricultural = saved["agricultural_reliability"].copy()
        cost = saved["total_cost"].copy()
        completed = int(np.asarray(saved["completed"]).reshape(-1)[0])
        saved_indicator = scalar_text(saved["indicator"])
        saved_period = scalar_text(saved["period"])
        policy_run = scalar_text(saved["policy_run"]).lower()
        data_folder = scalar_text(saved["data_folder"])
        num_scenarios = int(np.asarray(saved["num_scenarios"]).reshape(-1)[0])

    if (saved_indicator, saved_period) != (indicator, period):
        raise ValueError(
            f"Cache identity mismatch in {path}: metadata is "
            f"{saved_indicator}/{saved_period}, expected {indicator}/{period}"
        )
    if policy_run != expected_policy_run:
        raise ValueError(
            f"Cache {path} contains policy_run={policy_run!r}; "
            f"expected {expected_policy_run!r}."
        )
    if expected_policy_run in {"sep09_decost", "sep16_decost"}:
        expected_data_folder = NO_DROUGHT_OOS_DATA_FOLDERS[period]
        if data_folder != expected_data_folder:
            raise ValueError(
                f"Cache {path} was generated from data_folder={data_folder!r}; "
                f"expected {expected_data_folder!r}."
            )
    if num_scenarios != N_OOS_SCENARIOS:
        raise ValueError(
            f"Cache {path} says {num_scenarios} scenarios; expected {N_OOS_SCENARIOS}."
        )
    if scenario_names.size != N_OOS_SCENARIOS:
        raise ValueError(
            f"Cache {path} contains {scenario_names.size} scenario names; "
            f"expected {N_OOS_SCENARIOS}."
        )
    if len(set(scenario_names.tolist())) != N_OOS_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {path}")
    if policies.ndim != 2 or policies.shape[1] != N_DECISIONS:
        raise ValueError(f"Expected n-by-4 policies in {path}; got {policies.shape}")
    if not np.all(np.isfinite(policies)):
        raise ValueError(f"Non-finite policy decision found in {path}")
    if completed != len(policies):
        raise ValueError(f"Incomplete OOS cache {path}: {completed}/{len(policies)} policies")

    expected_shape = (len(policies), N_OOS_SCENARIOS)
    for label, values in (
        ("urban reliability", urban),
        ("agricultural reliability", agricultural),
        ("total cost", cost),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"Unexpected {label} shape {values.shape} in {path}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {label} value found in {path}")

    for label, values in (("urban", urban), ("agricultural", agricultural)):
        if np.min(values) < -0.05 or np.max(values) > 1.05:
            raise ValueError(
                f"OOS {label} reliability in {path} is outside the expected range: "
                f"[{np.min(values):.6g}, {np.max(values):.6g}]"
            )
    if np.min(cost) < -1e-8:
        raise ValueError(f"OOS total cost contains negative values in {path}: {np.min(cost)}")

    mean_urban = np.clip(urban.mean(axis=1), 0.0, 1.0)
    mean_agricultural = np.clip(agricultural.mean(axis=1), 0.0, 1.0)
    mean_cost = cost.mean(axis=1)
    objectives = np.column_stack(
        (mean_cost / 1_000_000.0, 1.0 - mean_urban, 1.0 - mean_agricultural)
    )
    return objectives, path, len(policies), len(scenario_names)


def nondominated(points):
    """Return unique non-dominated rows under all-minimization."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("nondominated() expects an N x 3 array")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    front = []
    for point in points[order]:
        if front and np.any(np.all(np.asarray(front) <= point, axis=1)):
            continue
        front.append(point)
    return np.asarray(front, dtype=float)


def dominated_area_2d(points, reference=(1.0, 1.0)):
    """Exact union area of minimization rectangles ending at the reference."""
    points = np.asarray(points, dtype=float)
    ref_y, ref_z = reference
    valid = points[(points[:, 0] < ref_y) & (points[:, 1] < ref_z)]
    if len(valid) == 0:
        return 0.0

    ordered = valid[np.argsort(valid[:, 0], kind="mergesort")]
    unique_y, first_indices = np.unique(ordered[:, 0], return_index=True)
    min_z_at_y = np.minimum.reduceat(ordered[:, 1], first_indices)
    y_breaks = np.append(unique_y, ref_y)
    area = 0.0
    best_z = ref_z
    for index, (left, right) in enumerate(zip(y_breaks[:-1], y_breaks[1:])):
        best_z = min(best_z, min_z_at_y[index])
        area += (right - left) * (ref_z - best_z)
    return float(area)


def hypervolume_3d(points, reference=(1.0, 1.0, 1.0)):
    """Exact 3-D dominated hypervolume for minimization objectives."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("hypervolume_3d() expects an N x 3 array")
    valid = points[np.all(points < reference, axis=1)]
    if len(valid) == 0:
        return 0.0
    valid = nondominated(valid)

    x_breaks = np.append(np.unique(valid[:, 0]), reference[0])
    volume = 0.0
    for left, right in zip(x_breaks[:-1], x_breaks[1:]):
        active = valid[valid[:, 0] <= left]
        if len(active):
            volume += (right - left) * dominated_area_2d(active[:, 1:], reference[1:])
    return float(volume)


def validate_hypervolume_implementation():
    """Fail early if the exact sweep does not pass known analytical cases."""
    single = np.array([[0.2, 0.3, 0.4]])
    expected_single = (1.0 - 0.2) * (1.0 - 0.3) * (1.0 - 0.4)
    if not np.isclose(hypervolume_3d(single), expected_single, atol=1e-12):
        raise AssertionError("3-D hypervolume self-test failed for a single point")

    pair = np.array([[0.2, 0.4, 0.4], [0.5, 0.2, 0.2]])
    if not np.isclose(hypervolume_3d(pair), 0.428, atol=1e-12):
        raise AssertionError("3-D hypervolume self-test failed for overlapping boxes")


def global_bounds(fronts, margin):
    """Build one ideal and one margin-expanded reference for all selected groups."""
    pooled = np.vstack(list(fronts.values()))
    ideal = np.min(pooled, axis=0)
    worst = np.max(pooled, axis=0)
    observed_span = worst - ideal
    safe_span = np.where(
        observed_span > 0.0, observed_span, np.maximum(np.abs(worst), 1.0)
    )
    reference = worst + margin * safe_span
    if np.any(reference <= ideal):
        raise ValueError("Reference point must be strictly worse than the ideal")
    return ideal, worst, reference


def normalize_front(front, ideal, reference):
    normalized = (front - ideal) / (reference - ideal)
    tolerance = 1e-10
    if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
        raise ValueError("A normalized objective lies outside [0, 1]")
    return np.clip(normalized, 0.0, 1.0)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_reference(output_dir, ideal, worst, reference, margin, selected_datasets):
    rows = []
    for label, values in (
        ("ideal", ideal),
        ("worst_observed", worst),
        ("raw_reference", reference),
        ("normalized_reference", np.ones(3)),
    ):
        rows.append(
            {
                "point": label,
                "cost_million": values[0],
                "urban_unreliability": values[1],
                "agricultural_unreliability": values[2],
                "reference_margin": margin,
                "datasets_used_for_bounds": "+".join(selected_datasets),
            }
        )
    write_csv(
        output_dir / "hypervolume_reference.csv",
        [
            "point",
            "cost_million",
            "urban_unreliability",
            "agricultural_unreliability",
            "reference_margin",
            "datasets_used_for_bounds",
        ],
        rows,
    )


def annotate_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.012, 0.985),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=90,
            rotation_mode="anchor",
        )


def require_matplotlib_panel_alignment(fig, output_base, tolerance_pt=1.5):
    """Record and enforce equal plot-area geometry for comparable panels."""
    fig.canvas.draw()
    axes = [axis for axis in fig.axes if axis.get_visible()]
    width_pt, height_pt = fig.get_size_inches() * 72.0
    panels = []
    for index, axis in enumerate(axes):
        position = axis.get_position()
        panels.append(
            {
                "id": f"panel_{index + 1}",
                "left": float(position.x0 * width_pt),
                "bottom": float(position.y0 * height_pt),
                "right": float(position.x1 * width_pt),
                "top": float(position.y1 * height_pt),
                "width": float(position.width * width_pt),
                "height": float(position.height * height_pt),
            }
        )

    if len(panels) < 2:
        deviations = {}
        verdict = "NOT APPLICABLE"
    else:
        deviations = {
            field: max(panel[field] for panel in panels)
            - min(panel[field] for panel in panels)
            for field in ("top", "bottom", "width", "height")
        }
        verdict = (
            "PASS"
            if all(value <= tolerance_pt for value in deviations.values())
            else "FIX BEFORE DELIVERY"
        )

    report = {
        "schema_version": 1,
        "figure_size_pt": [float(width_pt), float(height_pt)],
        "tolerance_pt": tolerance_pt,
        "panels": panels,
        "deviations_pt": deviations,
        "verdict": verdict,
    }
    alignment_path = Path(f"{output_base}.alignment.json")
    alignment_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verdict == "FIX BEFORE DELIVERY":
        raise RuntimeError(
            f"Hypervolume panel alignment exceeds {tolerance_pt} pt: {deviations}"
        )


def save_figure_bundle(fig, output_base, dpi):
    """Save a PNG preview plus editable PDF/SVG versions and alignment QA."""
    output_base = Path(output_base)
    require_matplotlib_panel_alignment(fig, output_base)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_dataset_figure(output_dir, summaries, dataset, dpi):
    """Plot both date ranges for one performance dataset."""
    rows = [row for row in summaries if row["dataset"] == dataset]
    if not rows:
        return
    lookup = {
        (row["indicator"], row["period"]): row["normalized_hypervolume"] for row in rows
    }
    positions = np.arange(len(INDICATORS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    colors = {"A": "lightgray", "B": "darkgray"}
    for offset, period in ((-width / 2, "A"), (width / 2, "B")):
        values = [lookup[(indicator, period)] for indicator in INDICATORS]
        bars = ax.bar(
            positions + offset,
            values,
            width,
            label=PERIOD_LABELS[period],
            color=colors[period],
            edgecolor="black",
            linewidth=0.5,
        )
        annotate_bars(ax, bars, values)

    label = "Training set" if dataset == "training" else "108-scenario test set"
    ax.set_xticks(positions)
    ax.set_xticklabels(INDICATORS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized 3-D hypervolume")
    ax.set_xlabel("Drought indicator")
    ax.set_title(f"{label}: shared global reference point")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure_bundle(
        fig,
        output_dir / f"normalized_global_hypervolume_{dataset}",
        dpi,
    )


def save_comparison_figure(output_dir, summaries, dpi):
    """Plot training versus OOS in separate, shared-scale period panels."""
    present = {row["dataset"] for row in summaries}
    if set(DATASETS) - present:
        return
    lookup = {
        (row["dataset"], row["indicator"], row["period"]): row["normalized_hypervolume"]
        for row in summaries
    }
    positions = np.arange(len(INDICATORS))
    width = 0.36
    colors = {"training": "lightgray", "oos": "darkgray"}
    labels = {"training": "Training set", "oos": "Test set (108 scenarios)"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3), sharey=True)
    for ax, period in zip(axes, PERIODS):
        for offset, dataset in ((-width / 2, "training"), (width / 2, "oos")):
            values = [lookup[(dataset, indicator, period)] for indicator in INDICATORS]
            bars = ax.bar(
                positions + offset,
                values,
                width,
                label=labels[dataset],
                color=colors[dataset],
                edgecolor="black",
                linewidth=0.5,
            )
            annotate_bars(ax, bars, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(INDICATORS)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Drought indicator")
        ax.set_title(PERIOD_LABELS[period])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized 3-D hypervolume")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Policy hypervolume: training versus test performance", y=0.98)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure_bundle(
        fig,
        output_dir / "normalized_global_hypervolume_training_vs_oos",
        dpi,
    )


def save_method_note(
    output_dir,
    policy_run,
    results_dir,
    oos_cache_dir,
    margin,
    selected_datasets,
    sign_records,
):
    group_count = len(selected_datasets) * len(INDICATORS) * len(PERIODS)
    if sign_records:
        unique_signs = {
            key: sorted({record[key] for record in sign_records}) for key in sign_records[0]
        }
    else:
        unique_signs = "not applicable (training data not selected)"
    note = f"""Normalized global 3-D hypervolume method

Policy run: {policy_run} (discount rate = 0)
Training results: {results_dir}
OOS caches: {oos_cache_dir}
Datasets included: {', '.join(selected_datasets)}
Groups included in shared bounds: {group_count}
Indicators: 6; periods: 2
Training seeds pooled per group: {len(EXPECTED_SEEDS)}
OOS scenarios averaged per policy: {N_OOS_SCENARIOS}

For training, objective values are read from all seven Borg .set archives in
each indicator-period experiment. For OOS, each policy's urban reliability,
agricultural reliability, and total cost are averaged over the 108 test
scenarios in its completed NPZ cache.

All objectives are converted to minimization form: cost in million dollars,
urban unreliability (1 - reliability), and agricultural unreliability
(1 - reliability). Duplicate objective vectors are removed within each group,
then that group's three-objective non-dominated front is retained.

One global ideal and one global worst-observed point are calculated across all
{group_count} selected group fronts. The raw reference is the global worst point
plus a {margin:.6g} ({100 * margin:.3g}%) margin in every objective. Every front
uses these same bounds, making the normalized reference [1, 1, 1]. The exact
3-D hypervolume is the union of the axis-aligned boxes from each non-dominated
point to [1, 1, 1]. Values lie in [0, 1], and larger is better. When both
datasets are selected, training and OOS values are therefore directly
comparable because all 24 groups share the same normalization and reference.

Detected training raw sign conversion flags: {unique_signs}
"""
    (output_dir / "hypervolume_method.txt").write_text(note, encoding="utf-8")


def selected_dataset_names(choice):
    return DATASETS if choice == "both" else (choice,)


def main():
    args = parse_args()
    if args.reference_margin < 0.0:
        raise ValueError("--reference-margin must be non-negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    default_results_dir = Path(__file__).resolve().parent / POLICY_RUN_FOLDERS[args.policy_run]
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else default_results_dir.resolve()
    )
    oos_cache_dir = (
        args.oos_cache_dir.expanduser().resolve()
        if args.oos_cache_dir is not None
        else results_dir / "out_of_sample" / "simulation_cache"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "hypervolume"
    )
    selected_datasets = selected_dataset_names(args.datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    fronts_dir = output_dir / "normalized_fronts"
    fronts_dir.mkdir(parents=True, exist_ok=True)

    validate_hypervolume_implementation()
    experiments = (
        discover_experiments(results_dir) if "training" in selected_datasets else {}
    )
    if "oos" in selected_datasets and not oos_cache_dir.is_dir():
        raise FileNotFoundError(f"OOS cache directory does not exist: {oos_cache_dir}")

    fronts = {}
    counts = {}
    sign_records = []
    print(f"Policy run: {args.policy_run}")
    print(f"Results: {results_dir}")
    print(f"Datasets: {', '.join(selected_datasets)}")

    for dataset in selected_datasets:
        print(f"\nReading {dataset} performance")
        for indicator in INDICATORS:
            for period in PERIODS:
                key = (dataset, indicator, period)
                if dataset == "training":
                    raw_rows, seed_files = load_training_experiment(
                        experiments[(indicator, period)]
                    )
                    objectives, sign_info = training_objectives_to_minimization(raw_rows)
                    sign_records.append(sign_info)
                    pooled_count = len(raw_rows)
                    source = str(experiments[(indicator, period)])
                    seed_count = len(seed_files)
                    scenario_count = ""
                else:
                    objectives, cache_path, pooled_count, scenario_count = load_oos_objectives(
                        oos_cache_dir, indicator, period, args.policy_run
                    )
                    source = str(cache_path)
                    seed_count = ""

                unique_objectives = np.unique(objectives, axis=0)
                front = nondominated(unique_objectives)
                fronts[key] = front
                counts[key] = {
                    "source": source,
                    "seed_count": seed_count,
                    "scenario_count": scenario_count,
                    "pooled_policy_count": pooled_count,
                    "unique_objective_count": len(unique_objectives),
                    "pareto_policy_count": len(front),
                }
                print(
                    f"  {indicator}_{period}: {pooled_count} policies, "
                    f"{len(unique_objectives)} unique, {len(front)} non-dominated"
                )

    ideal, worst, reference = global_bounds(fronts, args.reference_margin)
    summaries = []
    for dataset in selected_datasets:
        dataset_fronts_dir = fronts_dir / dataset
        dataset_fronts_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{dataset.capitalize()} normalized hypervolumes")
        for indicator in INDICATORS:
            for period in PERIODS:
                key = (dataset, indicator, period)
                normalized = normalize_front(fronts[key], ideal, reference)
                hv = hypervolume_3d(normalized)
                if hv < -1e-12 or hv > 1.0 + 1e-12:
                    raise AssertionError(
                        f"Normalized hypervolume outside [0, 1] for {key}: {hv}"
                    )
                hv = float(np.clip(hv, 0.0, 1.0))
                np.savetxt(
                    dataset_fronts_dir / f"{indicator}_{period}_normalized_front.csv",
                    normalized,
                    delimiter=",",
                    header="cost,urban_unreliability,agricultural_unreliability",
                    comments="",
                )
                summaries.append(
                    {
                        "dataset": dataset,
                        "indicator": indicator,
                        "period": period,
                        **counts[key],
                        "normalized_hypervolume": hv,
                    }
                )
                print(f"  {indicator}_{period}: {hv:.8f}")

    summary_fields = [
        "dataset",
        "indicator",
        "period",
        "source",
        "seed_count",
        "scenario_count",
        "pooled_policy_count",
        "unique_objective_count",
        "pareto_policy_count",
        "normalized_hypervolume",
    ]
    combined_name = (
        "normalized_global_hypervolume_training_and_oos.csv"
        if args.datasets == "both"
        else f"normalized_global_hypervolume_{selected_datasets[0]}.csv"
    )
    write_csv(output_dir / combined_name, summary_fields, summaries)
    if args.datasets == "both":
        for dataset in DATASETS:
            write_csv(
                output_dir / f"normalized_global_hypervolume_{dataset}.csv",
                summary_fields,
                [row for row in summaries if row["dataset"] == dataset],
            )

    save_reference(
        output_dir, ideal, worst, reference, args.reference_margin, selected_datasets
    )
    for dataset in selected_datasets:
        save_dataset_figure(output_dir, summaries, dataset, args.dpi)
    save_comparison_figure(output_dir, summaries, args.dpi)
    save_method_note(
        output_dir,
        args.policy_run,
        results_dir,
        oos_cache_dir,
        args.reference_margin,
        selected_datasets,
        sign_records,
    )

    print(f"\nGlobal ideal point:     {ideal}")
    print(f"Global observed worst:  {worst}")
    print(f"Raw reference point:    {reference}")
    print("Normalized reference:   [1. 1. 1.]")
    print(f"Outputs written to:     {output_dir}")


if __name__ == "__main__":
    main()
