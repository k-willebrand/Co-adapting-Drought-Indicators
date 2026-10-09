#!/usr/bin/env python3
"""Classify Period-B OOS climates by SPI/SRI winners and plot percentiles.

Two independent classifications are produced from the completed Sep16 OOS
policy-performance caches. The first compares the maximum urban reliability
attainable by the SPI and SRI policy families in each scenario. The second
compares the largest indicator-specific three-objective hypervolume in each
family. Exact family ties are recorded in the tables and excluded from the
violin plots. No Pywr model is rerun and no performance is averaged across
scenarios.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_diff_climate_oos_pareto_frontier_NATWAT as climate


POLICY_RUN = "sep16_decost"
RESULTS_FOLDER = "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
PERIOD = "B"
PERIOD_LABEL = "2080–2100"
INDICATORS = ("SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12")
FAMILIES = ("SPI", "SRI")
CLASSIFICATION_OUTCOMES = ("SPI", "SRI", "Tie")
CLIMATE_VARIABLES = (
    ("temperature", "Temperature"),
    ("precipitation", "Precipitation"),
    ("glacier", "Glacier melt"),
    ("streamflow", "Streamflow"),
)
FAMILY_COLORS = {"SPI": "#3d85c6", "SRI": "#d47a1f"}
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
    parser.add_argument("--data-root", type=Path, default=script_dir / "data")
    parser.add_argument("--cache-dir", type=Path, default=None)
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
    parser.add_argument("--reference-margin", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.reference_margin <= 0.0:
        parser.error("--reference-margin must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def scalar_text(value):
    return str(np.asarray(value).reshape(-1)[0])


def load_scenario_cache(cache_dir, indicator):
    path = Path(cache_dir) / f"{indicator}_{PERIOD}_oos_performance.npz"
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
            "data_folder",
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
            "data_folder": scalar_text(saved["data_folder"]),
            "num_scenarios": int(
                np.asarray(saved["num_scenarios"]).reshape(-1)[0]
            ),
        }

    if (item["indicator"], item["period"]) != (indicator, PERIOD):
        raise ValueError(f"Cache identity mismatch: {path}")
    if item["policy_run"] != POLICY_RUN:
        raise ValueError(
            f"Cache {path} has policy_run={item['policy_run']!r}; expected {POLICY_RUN!r}"
        )
    if item["data_folder"] != "transient_108_no_drought_pert_periodB":
        raise ValueError(f"Unexpected Period-B data folder in {path}")
    if item["num_scenarios"] != N_SCENARIOS:
        raise ValueError(f"Expected 108 scenarios in {path}")
    if len(item["scenario_names"]) != N_SCENARIOS:
        raise ValueError(f"Expected 108 scenario names in {path}")
    if len(set(item["scenario_names"].tolist())) != N_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {path}")
    policies = item["policies"]
    if policies.ndim != 2 or policies.shape[1] != 4:
        raise ValueError(f"Expected n-by-4 policies in {path}; got {policies.shape}")
    if item["completed"] != len(policies):
        raise ValueError(f"Incomplete cache {path}")
    expected_shape = (len(policies), N_SCENARIOS)
    for label in ("urban", "agriculture", "cost"):
        values = item[label]
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {label} values in {path}")
    if np.min(item["cost"]) < -1e-8:
        raise ValueError(f"Negative total cost in {path}")
    for label in ("urban", "agriculture"):
        if np.min(item[label]) < -0.05 or np.max(item[label]) > 1.05:
            raise ValueError(f"Unexpected reliability range for {label} in {path}")
        item[label] = np.clip(item[label], 0.0, 1.0)
    return item


def align_cache_scenarios(cache_item, reference_names):
    lookup = {
        str(name): index for index, name in enumerate(cache_item["scenario_names"])
    }
    missing = [name for name in reference_names if name not in lookup]
    extra = sorted(set(lookup).difference(reference_names))
    if missing or extra:
        raise ValueError(
            f"Scenario mismatch in {cache_item['path']}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    indices = [lookup[name] for name in reference_names]
    for label in ("urban", "agriculture", "cost"):
        cache_item[label] = cache_item[label][:, indices]
    cache_item["scenario_names"] = np.asarray(reference_names, dtype=str)


def nondominated_min(points):
    """Return unique non-dominated rows for three minimization objectives."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a non-empty N-by-3 objective array")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    front = []
    for point in points[order]:
        if front and np.any(np.all(np.asarray(front) <= point, axis=1)):
            continue
        front.append(point)
    return np.asarray(front, dtype=float)


def dominated_area_2d(points, reference=(1.0, 1.0)):
    points = np.asarray(points, dtype=float)
    ref_y, ref_z = reference
    valid = points[(points[:, 0] < ref_y) & (points[:, 1] < ref_z)]
    if len(valid) == 0:
        return 0.0
    ordered = valid[np.argsort(valid[:, 0], kind="mergesort")]
    unique_y, first = np.unique(ordered[:, 0], return_index=True)
    min_z_at_y = np.minimum.reduceat(ordered[:, 1], first)
    y_breaks = np.append(unique_y, ref_y)
    area = 0.0
    best_z = ref_z
    for index, (left, right) in enumerate(zip(y_breaks[:-1], y_breaks[1:])):
        best_z = min(best_z, min_z_at_y[index])
        area += (right - left) * (ref_z - best_z)
    return float(area)


def hypervolume_3d(points, reference=(1.0, 1.0, 1.0)):
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = points[np.all(points < reference, axis=1)]
    if len(valid) == 0:
        return 0.0
    valid = nondominated_min(valid)
    x_breaks = np.append(np.unique(valid[:, 0]), reference[0])
    volume = 0.0
    for left, right in zip(x_breaks[:-1], x_breaks[1:]):
        active = valid[valid[:, 0] <= left]
        if len(active):
            volume += (right - left) * dominated_area_2d(
                active[:, 1:], reference[1:]
            )
    return float(volume)


def validate_hypervolume():
    single = np.asarray([[0.2, 0.3, 0.4]])
    expected = (1.0 - 0.2) * (1.0 - 0.3) * (1.0 - 0.4)
    if not np.isclose(hypervolume_3d(single), expected, atol=1e-12):
        raise AssertionError("3-D hypervolume single-point test failed")
    pair = np.asarray([[0.2, 0.4, 0.4], [0.5, 0.2, 0.2]])
    if not np.isclose(hypervolume_3d(pair), 0.428, atol=1e-12):
        raise AssertionError("3-D hypervolume two-point test failed")


def scenario_objectives(cache_item, scenario_index):
    return np.column_stack(
        (
            cache_item["cost"][:, scenario_index] / 1_000_000.0,
            1.0 - cache_item["urban"][:, scenario_index],
            1.0 - cache_item["agriculture"][:, scenario_index],
        )
    )


def strict_family_winner(spi_value, sri_value):
    """Classify only a strict family advantage; exact equality is a tie."""
    if spi_value > sri_value:
        return "SPI"
    if sri_value > spi_value:
        return "SRI"
    return "Tie"


def classify_highest_urban(caches, scenario_names):
    rows = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        indicator_winners = []
        for indicator_rank, indicator in enumerate(INDICATORS):
            item = caches[indicator]
            best_index = min(
                range(len(item["policies"])),
                key=lambda policy_index: (
                    -item["urban"][policy_index, scenario_index],
                    item["cost"][policy_index, scenario_index],
                    -item["agriculture"][policy_index, scenario_index],
                    *item["policies"][policy_index].tolist(),
                    policy_index,
                ),
            )
            indicator_winners.append((indicator_rank, indicator, best_index))

        indicator_rank, indicator, policy_index = min(
            indicator_winners,
            key=lambda choice: (
                -caches[choice[1]]["urban"][choice[2], scenario_index],
                caches[choice[1]]["cost"][choice[2], scenario_index],
                -caches[choice[1]]["agriculture"][choice[2], scenario_index],
                *caches[choice[1]]["policies"][choice[2]].tolist(),
                choice[0],
                choice[2],
            ),
        )
        item = caches[indicator]
        policy = item["policies"][policy_index]
        urban_max = item["urban"][policy_index, scenario_index]
        family_maxima = {
            family: max(
                caches[candidate]["urban"][candidate_index, scenario_index]
                for _, candidate, candidate_index in indicator_winners
                if candidate.startswith(family)
            )
            for family in FAMILIES
        }
        strict_family = strict_family_winner(
            family_maxima["SPI"], family_maxima["SRI"]
        )
        tied_indicators = sum(
            caches[candidate]["urban"][candidate_index, scenario_index] == urban_max
            for _, candidate, candidate_index in indicator_winners
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "winner_indicator": indicator if strict_family != "Tie" else "",
                "winner_family": strict_family,
                "deterministic_tiebreak_indicator": indicator,
                "deterministic_tiebreak_family": indicator[:3],
                "best_spi_urban_reliability": family_maxima["SPI"],
                "best_sri_urban_reliability": family_maxima["SRI"],
                "spi_minus_sri_urban_reliability": (
                    family_maxima["SPI"] - family_maxima["SRI"]
                ),
                "source_policy_index": policy_index,
                "contract_threshold": policy[0],
                "contract_action": policy[1],
                "demand_threshold": policy[2],
                "demand_action": policy[3],
                "urban_reliability": urban_max,
                "total_cost": item["cost"][policy_index, scenario_index],
                "total_cost_million": item["cost"][policy_index, scenario_index]
                / 1_000_000.0,
                "agricultural_reliability": item["agriculture"][
                    policy_index, scenario_index
                ],
                "indicators_tied_at_max_urban": tied_indicators,
            }
        )
    return rows


def classify_highest_urban_with_mean_cost_cap(
    caches, scenario_names, cost_cap_million
):
    """Classify strict SPI/SRI winners after an OOS-mean cost constraint.

    Candidate policies are retained when their mean total cost across all 108
    OOS scenarios is no greater than ``cost_cap_million``. Within that fixed
    eligible set, each indicator contributes its maximum-urban-reliability
    policy separately for every scenario. The remaining tie-break rules match
    :func:`classify_highest_urban`.
    """
    if not np.isfinite(cost_cap_million) or cost_cap_million < 0.0:
        raise ValueError("cost_cap_million must be finite and non-negative")

    cap_dollars = float(cost_cap_million) * 1_000_000.0
    eligible = {}
    mean_costs = {}
    for indicator in INDICATORS:
        item = caches[indicator]
        policy_mean_cost = np.mean(item["cost"], axis=1)
        indices = np.flatnonzero(policy_mean_cost <= cap_dollars)
        if len(indices) == 0:
            raise ValueError(
                f"No {indicator} policy has 108-scenario mean total cost "
                f"<= {cost_cap_million:g} M$"
            )
        eligible[indicator] = indices
        mean_costs[indicator] = policy_mean_cost

    rows = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        indicator_winners = []
        for indicator_rank, indicator in enumerate(INDICATORS):
            item = caches[indicator]
            best_index = min(
                eligible[indicator].tolist(),
                key=lambda policy_index: (
                    -item["urban"][policy_index, scenario_index],
                    item["cost"][policy_index, scenario_index],
                    -item["agriculture"][policy_index, scenario_index],
                    *item["policies"][policy_index].tolist(),
                    policy_index,
                ),
            )
            indicator_winners.append((indicator_rank, indicator, best_index))

        indicator_rank, indicator, policy_index = min(
            indicator_winners,
            key=lambda choice: (
                -caches[choice[1]]["urban"][choice[2], scenario_index],
                caches[choice[1]]["cost"][choice[2], scenario_index],
                -caches[choice[1]]["agriculture"][choice[2], scenario_index],
                *caches[choice[1]]["policies"][choice[2]].tolist(),
                choice[0],
                choice[2],
            ),
        )
        item = caches[indicator]
        policy = item["policies"][policy_index]
        urban_max = item["urban"][policy_index, scenario_index]
        family_maxima = {
            family: max(
                caches[candidate]["urban"][candidate_index, scenario_index]
                for _, candidate, candidate_index in indicator_winners
                if candidate.startswith(family)
            )
            for family in FAMILIES
        }
        strict_family = strict_family_winner(
            family_maxima["SPI"], family_maxima["SRI"]
        )
        tied_indicators = sum(
            caches[candidate]["urban"][candidate_index, scenario_index]
            == urban_max
            for _, candidate, candidate_index in indicator_winners
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "winner_indicator": indicator if strict_family != "Tie" else "",
                "winner_family": strict_family,
                "deterministic_tiebreak_indicator": indicator,
                "deterministic_tiebreak_family": indicator[:3],
                "mean_cost_cap_million": float(cost_cap_million),
                "best_spi_urban_reliability": family_maxima["SPI"],
                "best_sri_urban_reliability": family_maxima["SRI"],
                "spi_minus_sri_urban_reliability": (
                    family_maxima["SPI"] - family_maxima["SRI"]
                ),
                "source_policy_index": policy_index,
                "source_policy_mean_total_cost": mean_costs[indicator][policy_index],
                "source_policy_mean_total_cost_million": (
                    mean_costs[indicator][policy_index] / 1_000_000.0
                ),
                "contract_threshold": policy[0],
                "contract_action": policy[1],
                "demand_threshold": policy[2],
                "demand_action": policy[3],
                "urban_reliability": urban_max,
                "scenario_total_cost": item["cost"][policy_index, scenario_index],
                "scenario_total_cost_million": (
                    item["cost"][policy_index, scenario_index] / 1_000_000.0
                ),
                "agricultural_reliability": item["agriculture"][
                    policy_index, scenario_index
                ],
                "indicators_tied_at_max_urban": tied_indicators,
            }
        )
    return rows, eligible


def classify_highest_agricultural(caches, scenario_names):
    rows = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        indicator_winners = []
        for indicator_rank, indicator in enumerate(INDICATORS):
            item = caches[indicator]
            best_index = min(
                range(len(item["policies"])),
                key=lambda policy_index: (
                    -item["agriculture"][policy_index, scenario_index],
                    item["cost"][policy_index, scenario_index],
                    -item["urban"][policy_index, scenario_index],
                    *item["policies"][policy_index].tolist(),
                    policy_index,
                ),
            )
            indicator_winners.append((indicator_rank, indicator, best_index))

        indicator_rank, indicator, policy_index = min(
            indicator_winners,
            key=lambda choice: (
                -caches[choice[1]]["agriculture"][choice[2], scenario_index],
                caches[choice[1]]["cost"][choice[2], scenario_index],
                -caches[choice[1]]["urban"][choice[2], scenario_index],
                *caches[choice[1]]["policies"][choice[2]].tolist(),
                choice[0],
                choice[2],
            ),
        )
        item = caches[indicator]
        policy = item["policies"][policy_index]
        agricultural_max = item["agriculture"][policy_index, scenario_index]
        family_maxima = {
            family: max(
                caches[candidate]["agriculture"][candidate_index, scenario_index]
                for _, candidate, candidate_index in indicator_winners
                if candidate.startswith(family)
            )
            for family in FAMILIES
        }
        strict_family = strict_family_winner(
            family_maxima["SPI"], family_maxima["SRI"]
        )
        tied_indicators = sum(
            caches[candidate]["agriculture"][candidate_index, scenario_index]
            == agricultural_max
            for _, candidate, candidate_index in indicator_winners
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "winner_indicator": indicator if strict_family != "Tie" else "",
                "winner_family": strict_family,
                "deterministic_tiebreak_indicator": indicator,
                "deterministic_tiebreak_family": indicator[:3],
                "best_spi_agricultural_reliability": family_maxima["SPI"],
                "best_sri_agricultural_reliability": family_maxima["SRI"],
                "spi_minus_sri_agricultural_reliability": (
                    family_maxima["SPI"] - family_maxima["SRI"]
                ),
                "source_policy_index": policy_index,
                "contract_threshold": policy[0],
                "contract_action": policy[1],
                "demand_threshold": policy[2],
                "demand_action": policy[3],
                "agricultural_reliability": agricultural_max,
                "urban_reliability": item["urban"][policy_index, scenario_index],
                "total_cost": item["cost"][policy_index, scenario_index],
                "total_cost_million": item["cost"][policy_index, scenario_index]
                / 1_000_000.0,
                "indicators_tied_at_max_agricultural": tied_indicators,
            }
        )
    return rows


def classify_lowest_cost(caches, scenario_names):
    rows = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        indicator_winners = []
        for indicator_rank, indicator in enumerate(INDICATORS):
            item = caches[indicator]
            best_index = min(
                range(len(item["policies"])),
                key=lambda policy_index: (
                    item["cost"][policy_index, scenario_index],
                    -item["urban"][policy_index, scenario_index],
                    -item["agriculture"][policy_index, scenario_index],
                    *item["policies"][policy_index].tolist(),
                    policy_index,
                ),
            )
            indicator_winners.append((indicator_rank, indicator, best_index))

        indicator_rank, indicator, policy_index = min(
            indicator_winners,
            key=lambda choice: (
                caches[choice[1]]["cost"][choice[2], scenario_index],
                -caches[choice[1]]["urban"][choice[2], scenario_index],
                -caches[choice[1]]["agriculture"][choice[2], scenario_index],
                *caches[choice[1]]["policies"][choice[2]].tolist(),
                choice[0],
                choice[2],
            ),
        )
        item = caches[indicator]
        policy = item["policies"][policy_index]
        cost_min = item["cost"][policy_index, scenario_index]
        family_minima = {
            family: min(
                caches[candidate]["cost"][candidate_index, scenario_index]
                for _, candidate, candidate_index in indicator_winners
                if candidate.startswith(family)
            )
            for family in FAMILIES
        }
        if family_minima["SPI"] < family_minima["SRI"]:
            strict_family = "SPI"
        elif family_minima["SRI"] < family_minima["SPI"]:
            strict_family = "SRI"
        else:
            strict_family = "Tie"
        tied_indicators = sum(
            caches[candidate]["cost"][candidate_index, scenario_index] == cost_min
            for _, candidate, candidate_index in indicator_winners
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "winner_indicator": indicator if strict_family != "Tie" else "",
                "winner_family": strict_family,
                "deterministic_tiebreak_indicator": indicator,
                "deterministic_tiebreak_family": indicator[:3],
                "lowest_spi_total_cost": family_minima["SPI"],
                "lowest_sri_total_cost": family_minima["SRI"],
                "spi_minus_sri_total_cost": (
                    family_minima["SPI"] - family_minima["SRI"]
                ),
                "source_policy_index": policy_index,
                "contract_threshold": policy[0],
                "contract_action": policy[1],
                "demand_threshold": policy[2],
                "demand_action": policy[3],
                "total_cost": cost_min,
                "total_cost_million": cost_min / 1_000_000.0,
                "urban_reliability": item["urban"][policy_index, scenario_index],
                "agricultural_reliability": item["agriculture"][
                    policy_index, scenario_index
                ],
                "indicators_tied_at_min_cost": tied_indicators,
            }
        )
    return rows


def build_indicator_fronts(caches, scenario_names):
    fronts = {}
    for scenario_index, scenario_name in enumerate(scenario_names):
        for indicator in INDICATORS:
            fronts[(scenario_name, indicator)] = nondominated_min(
                scenario_objectives(caches[indicator], scenario_index)
            )
        if (scenario_index + 1) % 12 == 0 or scenario_index + 1 == N_SCENARIOS:
            print(
                f"Built indicator-specific fronts for {scenario_index + 1}/108 scenarios",
                flush=True,
            )
    return fronts


def classify_hypervolume(fronts, scenario_names, reference_margin):
    pooled = np.vstack(list(fronts.values()))
    ideal = pooled.min(axis=0)
    worst = pooled.max(axis=0)
    span = worst - ideal
    safe_span = np.where(span > 0.0, span, np.maximum(np.abs(worst), 1.0))
    reference = worst + reference_margin * safe_span
    denominator = reference - ideal
    if np.any(denominator <= 0.0):
        raise ValueError("Invalid shared hypervolume normalization bounds")

    rows = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        values = {}
        for indicator in INDICATORS:
            normalized = (fronts[(scenario_name, indicator)] - ideal) / denominator
            if np.any(normalized < -1e-10) or np.any(normalized > 1.0 + 1e-10):
                raise ValueError("Normalized objective outside [0, 1]")
            values[indicator] = hypervolume_3d(np.clip(normalized, 0.0, 1.0))
        maximum = max(values.values())
        tied = [
            indicator
            for indicator in INDICATORS
            if values[indicator] == maximum
        ]
        deterministic_winner = tied[0]
        family_maxima = {
            family: max(
                value
                for indicator, value in values.items()
                if indicator.startswith(family)
            )
            for family in FAMILIES
        }
        strict_family = strict_family_winner(
            family_maxima["SPI"], family_maxima["SRI"]
        )
        winner = deterministic_winner if strict_family != "Tie" else ""
        ordered_values = sorted(values.values(), reverse=True)
        runner_up = ordered_values[1]
        row = {
            "scenario_name": scenario_name,
            "winner_indicator": winner,
            "winner_family": strict_family,
            "deterministic_tiebreak_indicator": deterministic_winner,
            "deterministic_tiebreak_family": deterministic_winner[:3],
            "best_spi_hypervolume": family_maxima["SPI"],
            "best_sri_hypervolume": family_maxima["SRI"],
            "spi_minus_sri_hypervolume": (
                family_maxima["SPI"] - family_maxima["SRI"]
            ),
            "winner_hypervolume": maximum,
            "runner_up_hypervolume": runner_up,
            "winner_margin": maximum - runner_up,
            "indicators_tied_at_max_hv": len(tied),
        }
        row.update({f"hv_{indicator}": values[indicator] for indicator in INDICATORS})
        rows.append(row)
        if (scenario_index + 1) % 12 == 0 or scenario_index + 1 == N_SCENARIOS:
            print(f"Calculated HV for {scenario_index + 1}/108 scenarios", flush=True)
    reference_rows = []
    for label, values in (
        ("ideal", ideal),
        ("worst_observed", worst),
        ("raw_reference", reference),
        ("normalized_reference", np.ones(3)),
    ):
        reference_rows.append(
            {
                "point": label,
                "cost_million": values[0],
                "urban_unreliability": values[1],
                "agricultural_unreliability": values[2],
                "reference_margin": reference_margin,
            }
        )
    return rows, reference_rows


def empirical_percentiles(values):
    """Average-rank empirical percentiles, scaled so unique extrema are 0 and 1."""
    values = np.asarray(values, dtype=float)
    if values.shape != (N_SCENARIOS,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected 108 finite climate values")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(N_SCENARIOS, dtype=float)
    start = 0
    while start < N_SCENARIOS:
        stop = start + 1
        while stop < N_SCENARIOS and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / (N_SCENARIOS - 1)


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
        "glacier": climate.calculate_glacier(glacier_file, scenario_names),
        "streamflow": climate.calculate_streamflow(streamflow_dir, scenario_names),
    }
    percentiles = {
        variable: empirical_percentiles(metrics[variable])
        for variable, _ in CLIMATE_VARIABLES
    }
    return metrics, percentiles, temperature_sources, precipitation_sources


def add_climate_columns(rows, scenario_names, metrics, percentiles):
    lookup = {name: index for index, name in enumerate(scenario_names)}
    for row in rows:
        index = lookup[row["scenario_name"]]
        for variable, _ in CLIMATE_VARIABLES:
            row[f"{variable}_metric"] = metrics[variable][index]
            row[f"{variable}_percentile"] = percentiles[variable][index]


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def classification_counts(method, rows):
    output = []
    for family in CLASSIFICATION_OUTCOMES:
        count = sum(row["winner_family"] == family for row in rows)
        output.append(
            {
                "classification_method": method,
                "winner_family": family,
                "scenario_count": count,
                "scenario_fraction": count / N_SCENARIOS,
            }
        )
    if sum(row["scenario_count"] for row in output) != N_SCENARIOS:
        raise RuntimeError(f"{method} classification does not cover all 108 scenarios")
    return output


def require_matplotlib_panel_alignment(fig, output_base, tolerance_pt=1.5):
    fig.canvas.draw()
    axes = [axis for axis in fig.axes if axis.get_visible()]
    if len(axes) != 2:
        raise RuntimeError(f"Expected two visible axes; found {len(axes)}")
    width_pt, height_pt = fig.get_size_inches() * 72.0
    panels = []
    for panel_id, axis in zip(("a", "b"), axes):
        position = axis.get_position()
        panels.append(
            {
                "id": panel_id,
                "left": float(position.x0 * width_pt),
                "bottom": float(position.y0 * height_pt),
                "right": float(position.x1 * width_pt),
                "top": float(position.y1 * height_pt),
                "width": float(position.width * width_pt),
                "height": float(position.height * height_pt),
            }
        )
    deviations = {
        field: abs(panels[0][field] - panels[1][field])
        for field in ("top", "bottom", "width", "height")
    }
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
        raise RuntimeError(f"Two-panel alignment failed: {deviations}")


def add_panel_label(axis, label):
    axis.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-25, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        annotation_clip=False,
    )


def draw_distribution(axis, position, values, color):
    values = np.asarray(values, dtype=float)
    if len(values) >= 2 and np.ptp(values) > 0.0:
        violin = axis.violinplot(
            values,
            positions=[position],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method="scott",
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
            body.set_linewidth(0.7)
    elif len(values):
        axis.plot(
            [position - 0.24, position + 0.24],
            [values[0], values[0]],
            color=color,
            alpha=0.35,
            linewidth=5,
            solid_capstyle="round",
        )
    if len(values):
        # Fixed low-discrepancy offsets expose individual scenarios without
        # introducing random or simulated values into the scientific data.
        jitter = (((np.arange(len(values)) * 0.61803398875) % 1.0) - 0.5) * 0.24
        axis.scatter(
            position + jitter,
            values,
            s=10,
            color=color,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.72,
            zorder=3,
        )
        q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
        axis.plot([position, position], [q25, q75], color="0.2", linewidth=2.2, zorder=4)
        axis.scatter(
            [position], [median], s=15, color="white", edgecolors="0.2",
            linewidths=0.6, zorder=5,
        )


def plot_violin(output_base, method_label, rows, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    positions = np.arange(1, len(CLIMATE_VARIABLES) + 1)
    tie_count = sum(row["winner_family"] == "Tie" for row in rows)
    for panel_index, (axis, family) in enumerate(zip(axes, FAMILIES)):
        selected = [row for row in rows if row["winner_family"] == family]
        for position, (variable, _) in zip(positions, CLIMATE_VARIABLES):
            draw_distribution(
                axis,
                position,
                [row[f"{variable}_percentile"] for row in selected],
                FAMILY_COLORS[family],
            )
        axis.set_xticks(
            positions, [label for _, label in CLIMATE_VARIABLES], fontsize=7
        )
        axis.set_xlim(0.45, len(CLIMATE_VARIABLES) + 0.55)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.linspace(0.0, 1.0, 5))
        axis.set_title(
            f"Strictly {family}-favored scenarios (n={len(selected)})", fontsize=8
        )
        if not selected:
            axis.text(
                0.5,
                0.5,
                f"No strict {family} wins",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.35",
                fontsize=7,
            )
        axis.grid(axis="y", color="0.88", linewidth=0.55)
        axis.set_axisbelow(True)
        add_panel_label(axis, chr(ord("a") + panel_index))
    axes[0].set_ylabel("Climate percentile rank across 108 scenarios")
    fig.suptitle(
        f"{PERIOD_LABEL}: {method_label}; exact ties excluded (n={tie_count})",
        fontsize=9,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, bottom=0.16, top=0.82, wspace=0.12
    )
    require_matplotlib_panel_alignment(fig, output_base)
    fig.savefig(f"{output_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    plt.close(fig)


def write_method(output_dir, reference_margin):
    text = f"""Period-B SPI/SRI scenario classification method

All calculations use completed Sep16 OOS caches and are performed separately
for each of the 108 Period-B scenarios. No objective is averaged across climate
scenarios.

Highest-urban classification: each indicator first contributes its policy with
maximum urban reliability in the scenario. The maximum reliability attainable
by any SPI policy is compared directly with the maximum attainable by any SRI
policy. SPI or SRI is assigned only when its value is strictly greater; exact
equality is recorded as Tie and excluded from both violin panels. A
deterministic policy is still retained in the CSV for audit purposes, using
minimum total cost, maximum agricultural reliability, the lexicographically
smallest decision vector, and then the fixed indicator order
{', '.join(INDICATORS)}.

Highest-agricultural classification: the maximum agricultural reliability
attainable by any SPI policy is compared directly with the maximum attainable
by any SRI policy. The family with the strictly greater value is assigned; exact
equality is recorded as Tie and excluded from both violin panels. For an
auditable representative policy, equal-reliability candidates are ordered by
minimum total cost, maximum urban reliability, decision vector, and indicator
order.

Lowest-cost classification: the minimum total cost attainable by any SPI
policy is compared directly with the minimum attainable by any SRI policy. The
family with the strictly lower value is assigned; exact equality is recorded as
Tie and excluded from both violin panels. For an auditable representative
policy, equal-cost candidates are ordered by maximum urban reliability, maximum
agricultural reliability, decision vector, and indicator order.

Hypervolume classification: a separate three-objective Pareto front is computed
for every scenario and every indicator using cost/1e6, 1-urban reliability, and
1-agricultural reliability as minimization objectives. All 648 fronts share one
global Period-B ideal point and one reference point formed from the worst
observed front coordinate plus a {reference_margin:g} span margin. Hypervolume
is calculated independently for SPI3, SPI6, SPI12, SRI3, SRI6, and SRI12. The
largest SPI hypervolume is compared directly with the largest SRI hypervolume.
SPI or SRI is assigned only when its value is strictly greater; exact equality
is recorded as Tie and excluded from both violin panels. The fixed indicator
order above is retained only as an auditable deterministic tie-break field in
the CSV and does not convert a family tie into an SPI or SRI win.

Climate percentiles: the same shifted final-20-year temperature, precipitation,
streamflow, and glacier-melt metrics used by the different-climate workflow are
calculated for all 108 OOS scenarios. Average ranks are assigned to exact ties
and scaled as (rank-1)/(108-1), producing the 0-1 percentile-rank axis. Scenario
matching is by name, never by input column position.
"""
    (Path(output_dir) / "classification_method.txt").write_text(text, encoding="utf-8")


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
        else results_dir / "out_of_sample" / "scenario_spi_sri_climate_violin"
    )
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory missing: {results_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"OOS cache directory missing: {cache_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_hypervolume()
    caches = {indicator: load_scenario_cache(cache_dir, indicator) for indicator in INDICATORS}
    scenario_names = caches[INDICATORS[0]]["scenario_names"].astype(str).tolist()
    for indicator in INDICATORS:
        align_cache_scenarios(caches[indicator], scenario_names)

    print("Classifying scenarios by highest urban reliability...", flush=True)
    urban_rows = classify_highest_urban(caches, scenario_names)
    print("Classifying scenarios by highest agricultural reliability...", flush=True)
    agricultural_rows = classify_highest_agricultural(caches, scenario_names)
    print("Classifying scenarios by lowest total cost...", flush=True)
    cost_rows = classify_lowest_cost(caches, scenario_names)
    print("Building indicator-specific scenario Pareto fronts...", flush=True)
    fronts = build_indicator_fronts(caches, scenario_names)
    print("Classifying scenarios by indicator-specific hypervolume...", flush=True)
    hv_rows, reference_rows = classify_hypervolume(
        fronts, scenario_names, args.reference_margin
    )

    print("Calculating final-20-year climate percentiles...", flush=True)
    metrics, percentiles, _, _ = calculate_climate_metrics(args, scenario_names)
    add_climate_columns(urban_rows, scenario_names, metrics, percentiles)
    add_climate_columns(agricultural_rows, scenario_names, metrics, percentiles)
    add_climate_columns(cost_rows, scenario_names, metrics, percentiles)
    add_climate_columns(hv_rows, scenario_names, metrics, percentiles)

    urban_path = output_dir / "highest_urban_scenario_classification.csv"
    agricultural_path = (
        output_dir / "highest_agricultural_scenario_classification.csv"
    )
    cost_path = output_dir / "lowest_cost_scenario_classification.csv"
    hv_path = output_dir / "indicator_hv_scenario_classification.csv"
    reference_path = output_dir / "indicator_hv_reference.csv"
    counts_path = output_dir / "scenario_family_counts.csv"
    write_csv(urban_path, urban_rows)
    write_csv(agricultural_path, agricultural_rows)
    write_csv(cost_path, cost_rows)
    write_csv(hv_path, hv_rows)
    write_csv(reference_path, reference_rows)
    count_rows = (
        classification_counts("highest_urban", urban_rows)
        + classification_counts("highest_agricultural", agricultural_rows)
        + classification_counts("lowest_cost", cost_rows)
        + classification_counts("indicator_hypervolume", hv_rows)
    )
    write_csv(counts_path, count_rows)

    urban_base = output_dir / "periodB_climate_percentiles_highest_urban_winner"
    agricultural_base = (
        output_dir / "periodB_climate_percentiles_highest_agricultural_winner"
    )
    cost_base = output_dir / "periodB_climate_percentiles_lowest_cost_winner"
    hv_base = output_dir / "periodB_climate_percentiles_indicator_hv_winner"
    plot_violin(
        urban_base,
        "highest-urban-reliability indicator winner",
        urban_rows,
        args.dpi,
    )
    plot_violin(
        agricultural_base,
        "highest agricultural reliability",
        agricultural_rows,
        args.dpi,
    )
    plot_violin(
        cost_base,
        "lowest total cost",
        cost_rows,
        args.dpi,
    )
    plot_violin(
        hv_base,
        "indicator-specific three-objective hypervolume winner",
        hv_rows,
        args.dpi,
    )
    write_method(output_dir, args.reference_margin)

    print(f"Policy run : {POLICY_RUN}")
    print(f"Period     : {PERIOD_LABEL}")
    print(f"Cache      : {cache_dir}")
    print(f"Output     : {output_dir}")
    for row in count_rows:
        print(
            f"  {row['classification_method']:21s} {row['winner_family']}: "
            f"{row['scenario_count']:3d}/108 "
            f"({100.0 * row['scenario_fraction']:.1f}%)"
        )
    print("Saved:")
    for path in (
        urban_path,
        agricultural_path,
        cost_path,
        hv_path,
        reference_path,
        counts_path,
    ):
        print(f"  {path}")
    for base in (urban_base, agricultural_base, cost_base, hv_base):
        for suffix in (".png", ".pdf", ".svg", ".alignment.json"):
            print(f"  {base}{suffix}")


if __name__ == "__main__":
    main()
