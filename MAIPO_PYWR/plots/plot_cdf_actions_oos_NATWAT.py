#!/usr/bin/env python3
"""Simulate and plot OOS action CDFs for NATWAT policies.

The existing out-of-sample (OOS) performance caches contain all unique
policies pooled across Borg seeds and their scenario-level cost and reliability
values, but they do not contain action outcomes. This script:

1. recomputes each indicator-period OOS Pareto mask from 108-scenario mean
   cost (min), urban reliability (max), and agricultural reliability (max);
2. simulates only those OOS Pareto policies to record total contracts and total
   urban demand restricted over the complete period;
3. selects the highest OOS-mean-urban-reliability SPI12 policy in Period A and
   a configured runoff-indicator policy in Period B, and evaluates both fixed
   policies in both periods;
4. plots one 2x2 CDF figure for all per-indicator OOS Pareto policies, one for
   the two existing cross-period selected policies, one for the
   highest-urban-reliability policy of every indicator in each period, and one
   for every indicator-period's OOS Pareto policy closest to, but strictly
   below, a target mean cost.

Use the ``simulate`` subcommand in a 12-task SLURM array, then run ``plot``.
This script does not rerun Borg optimization and does not modify the formal
optimization model file.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np


INDICATORS = ("SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12")
PERIODS = ("A", "B")
COMBINATIONS = tuple((indicator, period) for period in PERIODS for indicator in INDICATORS)
EXPECTED_SEEDS = tuple(range(1, 8))
N_DECISIONS = 4
N_SCENARIOS = 108
SECONDS_PER_WEEK = 7 * 24 * 60 * 60
POLICY_RUN_FOLDERS = {
    "aug28": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug28_MIN",
    "sep09_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep9_MIN_DeCost",
    "sep16_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost",
    "sep29_c05_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_C05_MIN_DeCost",
    "sep29_c15_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_C15_MIN_DeCost",
    "sep29_l20_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_L20_MIN_DeCost",
    "sep29_l40_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_L40_MIN_DeCost",
}
POLICY_RUN_DEMAND_SETTINGS = {
    "sep09_decost": (1.0, 0.3),
    "sep16_decost": (1.0, 0.3),
    "sep29_c05_decost": (0.5, 0.3),
    "sep29_c15_decost": (1.5, 0.3),
    "sep29_l20_decost": (1.0, 0.2),
    "sep29_l40_decost": (1.0, 0.4),
}
DEFAULT_POLICY_RUN = "sep09_decost"
DEFAULT_OOS_DATA_FOLDERS = {
    "A": "transient_108_no_drought_pert_periodA",
    "B": "transient_108_no_drought_pert_periodB",
}
OOS_FLOW_FILES = (
    "YESO.csv",
    "MAIPO.csv",
    "COLORADO.csv",
    "VOLCAN.csv",
    "LAGUNANEGRA.csv",
    "MAIPOEXTRA.csv",
)
RESTRICTED_DEMAND_RECORDER = "restricted demanda_PT1 recorder"

PERIOD_LABELS = {"A": "2020–2040", "B": "2080–2100"}
COLORS = {
    "SPI3": "#9ecae1",
    "SPI6": "#4f8fbd",
    "SPI12": "#145a86",
    "SRI3": "#e7b47a",
    "SRI6": "#bd6d2d",
    "SRI12": "#7f3b08",
}
LINESTYLES = {"SPI3": ":", "SPI6": "--", "SPI12": "-", "SRI3": ":", "SRI6": "--", "SRI12": "-"}


def selected_policy_definitions(policy_run):
    """Return (indicator, selection period, cross-evaluation period) tuples."""
    period_b_indicator = "SRI12" if policy_run == "sep16_decost" else "SRI3"
    return (("SPI12", "A", "B"), (period_b_indicator, "B", "A"))


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-run",
        choices=tuple(POLICY_RUN_FOLDERS),
        default=DEFAULT_POLICY_RUN,
        help=f"Policy archive/cache identity (default: {DEFAULT_POLICY_RUN}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Default: the folder selected by --policy-run beside this script.",
    )
    parser.add_argument(
        "--performance-cache-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/out_of_sample/simulation_cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/out_of_sample/cdf_actions.",
    )

    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate", help="Simulate one indicator-period task.")
    simulate.add_argument("--task-id", type=int, choices=range(1, 13), required=True)
    simulate.add_argument("--data-root", type=Path, default=script_dir / "data")
    simulate.add_argument(
        "--period-a-data-folder", default=DEFAULT_OOS_DATA_FOLDERS["A"]
    )
    simulate.add_argument(
        "--period-b-data-folder", default=DEFAULT_OOS_DATA_FOLDERS["B"]
    )
    simulate.add_argument(
        "--model-script",
        type=Path,
        default=None,
        help="Default: DeCost model for *_decost runs; otherwise the standard model.",
    )
    simulate.add_argument("--checkpoint-every", type=int, default=5)
    simulate.add_argument("--overwrite", action="store_true")

    plot = commands.add_parser("plot", help="Plot after all 12 simulation tasks finish.")
    plot.add_argument("--dpi", type=int, default=300)
    plot.add_argument(
        "--target-cost-million",
        type=float,
        nargs="+",
        default=[650.0, 5000.0],
        help=(
            "One or more upper bounds on mean OOS total cost in M$ for the "
            "additional selected-policy CDFs; for each bound, the closest "
            "policy strictly below it is used (default: 650 5000)."
        ),
    )
    return parser.parse_args()


def resolved_paths(args):
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else (script_dir / POLICY_RUN_FOLDERS[args.policy_run]).resolve()
    )
    performance_cache_dir = (
        args.performance_cache_dir.expanduser().resolve()
        if args.performance_cache_dir is not None
        else results_dir / "out_of_sample" / "simulation_cache"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "out_of_sample" / "cdf_actions"
    )
    return results_dir, performance_cache_dir, output_dir


def scalar_text(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar metadata; got shape {array.shape}")
    return str(array.reshape(-1)[0])


def read_scenario_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None or len(header) < 2:
        raise ValueError(f"Cannot read scenario header from {path}")
    return np.asarray(header[1:], dtype=str)


def validate_data_scenario_order(data_folder, indicator, expected_names):
    """Require the drought index and all inflows to match the cache order."""
    paths = [data_folder / f"{indicator}.csv"] + [
        data_folder / filename for filename in OOS_FLOW_FILES
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required OOS input is missing: {path}")
        names = read_scenario_header(path)
        if not np.array_equal(names, expected_names.astype(str)):
            raise ValueError(
                f"Scenario names/order in {path} do not match the main OOS cache"
            )


def load_performance_cache(cache_dir, indicator, period, expected_policy_run):
    path = cache_dir / f"{indicator}_{period}_oos_performance.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed OOS cache: {path}")
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
            "num_scenarios": int(np.asarray(saved["num_scenarios"]).reshape(-1)[0]),
            "demand_restriction_cost_per_m3": (
                float(
                    np.asarray(
                        saved["demand_restriction_cost_per_m3"]
                    ).reshape(-1)[0]
                )
                if "demand_restriction_cost_per_m3" in saved.files
                else 1.0
            ),
            "max_demand_curtailment_fraction": (
                float(
                    np.asarray(
                        saved["max_demand_curtailment_fraction"]
                    ).reshape(-1)[0]
                )
                if "max_demand_curtailment_fraction" in saved.files
                else 0.3
            ),
        }

    if (item["indicator"], item["period"]) != (indicator, period):
        raise ValueError(
            f"Cache identity mismatch in {path}: {item['indicator']}/{item['period']}"
        )
    if item["policy_run"] != expected_policy_run:
        raise ValueError(
            f"Cache {path} has policy_run={item['policy_run']!r}; "
            f"expected {expected_policy_run!r}"
        )
    expected_settings = POLICY_RUN_DEMAND_SETTINGS.get(expected_policy_run)
    if expected_settings is not None:
        expected_cost_rate, expected_max_curtailment = expected_settings
        if not np.isclose(
            item["demand_restriction_cost_per_m3"],
            expected_cost_rate,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Cache {path} has demand cost "
                f"{item['demand_restriction_cost_per_m3']}; expected "
                f"{expected_cost_rate} for {expected_policy_run}"
            )
        if not np.isclose(
            item["max_demand_curtailment_fraction"],
            expected_max_curtailment,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Cache {path} has maximum curtailment "
                f"{item['max_demand_curtailment_fraction']}; expected "
                f"{expected_max_curtailment} for {expected_policy_run}"
            )
    if item["num_scenarios"] != N_SCENARIOS:
        raise ValueError(f"Cache {path} has {item['num_scenarios']} scenarios; expected 108")
    if item["policies"].ndim != 2 or item["policies"].shape[1] != N_DECISIONS:
        raise ValueError(f"Expected n-by-4 policies in {path}; got {item['policies'].shape}")
    if item["completed"] != len(item["policies"]):
        raise ValueError(f"Incomplete cache {path}: {item['completed']}/{len(item['policies'])}")
    if item["scenario_names"].shape != (N_SCENARIOS,):
        raise ValueError(f"Expected 108 scenario names in {path}")
    if len(set(item["scenario_names"].tolist())) != N_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {path}")
    expected_shape = (len(item["policies"]), N_SCENARIOS)
    for name in ("urban", "agriculture", "cost"):
        values = item[name]
        if values.shape != expected_shape:
            raise ValueError(f"Unexpected {name} shape {values.shape} in {path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} value in {path}")
    if not np.all(np.isfinite(item["policies"])):
        raise ValueError(f"Non-finite policy decision in {path}")
    if np.min(item["urban"]) < -0.05 or np.max(item["urban"]) > 1.05:
        raise ValueError(f"Urban reliability outside expected range in {path}")
    if np.min(item["agriculture"]) < -0.05 or np.max(item["agriculture"]) > 1.05:
        raise ValueError(f"Agricultural reliability outside expected range in {path}")

    item["objectives"] = np.column_stack(
        (item["cost"].mean(axis=1), item["urban"].mean(axis=1), item["agriculture"].mean(axis=1))
    )
    item["pareto_mask"] = nondominated_mask(item["objectives"])
    return item


def nondominated_mask(objectives):
    """Return per-indicator OOS Pareto mask: cost min, reliabilities max."""
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError("Expected an n-by-3 objective array")
    minimized = np.column_stack((objectives[:, 0], -objectives[:, 1], -objectives[:, 2]))
    keep = np.ones(len(minimized), dtype=bool)
    for index, point in enumerate(minimized):
        dominates = np.all(minimized <= point, axis=1) & np.any(minimized < point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return keep


def find_result_folder(results_dir, indicator, period):
    prefix = f"{indicator}_{period}_"
    matches = sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one result folder beginning {prefix!r}; found {len(matches)}"
        )
    return matches[0]


def read_policy_rows(path):
    policies = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith(("#", "//")):
                continue
            fields = text.split()
            if len(fields) < 7:
                raise ValueError(f"Expected at least 7 columns in {path}:{line_number}")
            try:
                policies.append([float(value) for value in fields[:N_DECISIONS]])
            except ValueError as exc:
                raise ValueError(f"Non-numeric policy in {path}:{line_number}") from exc
    if not policies:
        raise RuntimeError(f"No policy rows found in {path}")
    return policies


def load_current_unique_policies(results_dir, indicator, period):
    """Rebuild policy ordering and ensure all seven policy archives exist."""
    folder = find_result_folder(results_dir, indicator, period)
    seed_files = [folder / "sets" / f"Borg_DPS_PySedSim{seed}.set" for seed in EXPECTED_SEEDS]
    missing = [path for path in seed_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing seed archive(s):\n" + "\n".join(map(str, missing)))
    unique = []
    seen = set()
    for path in seed_files:
        for policy in read_policy_rows(path):
            key = tuple(np.round(policy, 12))
            if key not in seen:
                seen.add(key)
                unique.append(policy)
    return np.asarray(unique, dtype=float), folder


def validate_cache_matches_archives(performance, results_dir):
    current, folder = load_current_unique_policies(
        results_dir, performance["indicator"], performance["period"]
    )
    cached = performance["policies"]
    if current.shape != cached.shape or not np.allclose(current, cached, rtol=0, atol=1e-12):
        raise RuntimeError(
            f"OOS cache is stale relative to current seed archives for "
            f"{performance['indicator']}_{performance['period']}. Rebuild that OOS cache first."
        )
    return folder


def import_model_module(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model script not found: {path}")
    spec = importlib.util.spec_from_file_location("natwat_aug28_action_model", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_model"):
        raise AttributeError(f"No make_model() in {path}")
    return module


def add_total_restricted_demand_recorder(model, model_module):
    """Record restricted PT1 demand at every timestep for temporal summation."""
    model_module.NumpyArrayParameterRecorder(
        model,
        name=RESTRICTED_DEMAND_RECORDER,
        param=model.parameters["demand_max_flow_PT1"],
        temporal_agg_func="sum",
        agg_func="SUM",
    )
    return model


def make_model_20yrA(model_module, policy, indicator, data_folder):
    """Build the Period A OOS model with cumulative-action recorders."""
    return make_action_model(model_module, policy, indicator, data_folder, "A")


def make_model_20yrB(model_module, policy, indicator, data_folder):
    """Build the Period B OOS model with cumulative-action recorders."""
    return make_action_model(model_module, policy, indicator, data_folder, "B")


def make_action_model(model_module, policy, indicator, data_folder, period):
    contract_threshold, contract_action, demand_threshold, demand_action = policy
    num_dp = int(getattr(model_module, "num_DP", 6))
    model = model_module.make_model(
        contract_threshold_vals=contract_threshold * np.ones(num_dp),
        contract_action_vals=contract_action * np.ones(num_dp),
        demand_threshold_vals=[demand_threshold * np.ones(12)],
        demand_action_vals=[np.ones(12), demand_action * np.ones(12)],
        indicator=indicator,
        data_folder=str(data_folder),
        period=period,
        num_scenarios=N_SCENARIOS,
    )
    return add_total_restricted_demand_recorder(model, model_module)


def recorder_values(model, name):
    values = np.asarray(model.recorders[name].values(), dtype=float).reshape(-1)
    if values.shape != (N_SCENARIOS,):
        raise ValueError(f"Recorder {name!r} returned shape {values.shape}; expected (108,)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Recorder {name!r} returned non-finite values")
    return values


def simulate_one_policy(model_module, policy, indicator, period, data_folder):
    builder = make_model_20yrA if period == "A" else make_model_20yrB
    model = builder(model_module, policy, indicator, data_folder)
    model.check()
    model.run()

    urban = recorder_values(model, "reliability_PT1")
    agriculture = recorder_values(model, "reliability_Ag")
    cost = recorder_values(model, "TotalCost")
    contracts = recorder_values(model, "Total Contracts Made")
    unrestricted_sum = recorder_values(model, "demanda_PT1 recorder")
    restricted_sum = recorder_values(model, RESTRICTED_DEMAND_RECORDER)
    restricted_volume = (unrestricted_sum - restricted_sum) * SECONDS_PER_WEEK / 1_000_000.0
    if np.min(restricted_volume) < -1e-6:
        raise ValueError(
            f"Negative cumulative urban demand restriction: {np.min(restricted_volume):.6g} Mm3"
        )
    restricted_volume = np.maximum(restricted_volume, 0.0)
    del model
    gc.collect()
    return urban, agriculture, cost, contracts, restricted_volume


def save_atomic(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def state_metadata(
    performance, pareto_indices, pareto_policies, source_folder, policy_run
):
    return {
        "policies": pareto_policies,
        "source_policy_indices": pareto_indices,
        "scenario_names": performance["scenario_names"],
        "indicator": np.asarray(performance["indicator"]),
        "period": np.asarray(performance["period"]),
        "policy_run": np.asarray(policy_run),
        "source_performance_cache": np.asarray(str(performance["path"])),
        "source_result_folder": np.asarray(str(source_folder)),
        "data_folder": np.asarray(performance["data_folder"]),
        "num_scenarios": np.asarray(N_SCENARIOS, dtype=int),
        "seconds_per_timestep": np.asarray(SECONDS_PER_WEEK, dtype=int),
        "source_mean_total_cost": performance["objectives"][pareto_indices, 0],
        "source_mean_urban_reliability": performance["objectives"][pareto_indices, 1],
        "source_mean_agricultural_reliability": performance["objectives"][pareto_indices, 2],
    }


def save_action_state(path, metadata, urban, agriculture, cost, contracts, restriction, completed):
    save_atomic(
        path,
        urban_reliability=urban,
        agricultural_reliability=agriculture,
        total_cost=cost,
        total_contracts=contracts,
        total_urban_demand_restricted_mm3=restriction,
        completed=np.asarray(completed, dtype=int),
        **metadata,
    )


def load_checkpoint(path, policies, source_indices, scenario_names):
    with np.load(path, allow_pickle=False) as saved:
        if not np.allclose(saved["policies"], policies, rtol=0, atol=1e-12):
            raise RuntimeError(f"Checkpoint policies differ: {path}")
        if not np.array_equal(saved["source_policy_indices"], source_indices):
            raise RuntimeError(f"Checkpoint source policy indices differ: {path}")
        if not np.array_equal(saved["scenario_names"].astype(str), scenario_names):
            raise RuntimeError(f"Checkpoint scenario order differs: {path}")
        return (
            saved["urban_reliability"].copy(),
            saved["agricultural_reliability"].copy(),
            saved["total_cost"].copy(),
            saved["total_contracts"].copy(),
            saved["total_urban_demand_restricted_mm3"].copy(),
            int(np.asarray(saved["completed"]).reshape(-1)[0]),
        )


def selected_policy_index(performance, indicator, source_period):
    if (performance["indicator"], performance["period"]) != (indicator, source_period):
        raise ValueError("Selected-policy source cache identity mismatch")
    candidates = np.flatnonzero(performance["pareto_mask"])
    objectives = performance["objectives"]
    policies = performance["policies"]
    # Primary: max OOS mean urban reliability. Ties: min cost, max agriculture,
    # then lexicographically smallest decision vector for deterministic selection.
    return min(
        candidates.tolist(),
        key=lambda index: (
            -objectives[index, 1],
            objectives[index, 0],
            -objectives[index, 2],
            *policies[index].tolist(),
        ),
    )


def target_cost_policy_index(
    performance, indicator, period, target_cost_million
):
    """Select the OOS Pareto policy closest to, but below, the cost target."""
    if (performance["indicator"], performance["period"]) != (indicator, period):
        raise ValueError("Target-cost policy source cache identity mismatch")
    if not np.isfinite(target_cost_million) or target_cost_million < 0.0:
        raise ValueError("Target mean cost must be a finite non-negative value")
    objectives = performance["objectives"]
    policies = performance["policies"]
    target_cost = target_cost_million * 1_000_000.0
    candidates = np.flatnonzero(
        performance["pareto_mask"] & (objectives[:, 0] < target_cost)
    )
    if candidates.size == 0:
        raise RuntimeError(
            f"No {period} {indicator} OOS Pareto policy has mean total cost "
            f"strictly below {target_cost_million:g} M$"
        )
    # Primary: smallest positive distance below the target (equivalently, the
    # largest eligible cost). Exact-cost ties are resolved deterministically by
    # higher urban reliability, higher agricultural reliability, then policy.
    return min(
        candidates.tolist(),
        key=lambda index: (
            target_cost - objectives[index, 0],
            -objectives[index, 1],
            -objectives[index, 2],
            *policies[index].tolist(),
        ),
    )


def simulate_selected_cross_period(
    model_module,
    source_performance,
    target_performance,
    source_period,
    target_period,
    indicator,
    data_root,
    period_folders,
    selected_dir,
    overwrite,
    policy_run,
):
    source_index = selected_policy_index(source_performance, indicator, source_period)
    policy = source_performance["policies"][source_index]
    output = selected_dir / (
        f"{indicator}_selected_in_{source_period}_evaluated_{target_period}_actions.npz"
    )
    if output.is_file() and not overwrite:
        print(f"Selected-policy cross-period cache exists: {output}", flush=True)
        return

    target_data_folder = data_root / period_folders[target_period]
    if not target_data_folder.is_dir():
        raise FileNotFoundError(f"Target OOS data folder does not exist: {target_data_folder}")
    if target_performance["data_folder"] != period_folders[target_period]:
        raise ValueError(
            f"Target performance cache used {target_performance['data_folder']!r}, "
            f"expected {period_folders[target_period]!r}"
        )
    validate_data_scenario_order(
        target_data_folder, indicator, target_performance["scenario_names"]
    )
    print(
        f"Cross-evaluating {indicator} policy selected in {source_period} under Period {target_period}",
        flush=True,
    )
    urban, agriculture, cost, contracts, restriction = simulate_one_policy(
        model_module, policy, indicator, target_period, target_data_folder
    )
    save_atomic(
        output,
        policy=policy,
        source_policy_index=np.asarray(source_index, dtype=int),
        selection_indicator=np.asarray(indicator),
        selection_period=np.asarray(source_period),
        evaluation_period=np.asarray(target_period),
        policy_run=np.asarray(policy_run),
        scenario_names=target_performance["scenario_names"],
        source_mean_total_cost=np.asarray(source_performance["objectives"][source_index, 0]),
        source_mean_urban_reliability=np.asarray(source_performance["objectives"][source_index, 1]),
        source_mean_agricultural_reliability=np.asarray(source_performance["objectives"][source_index, 2]),
        urban_reliability=urban,
        agricultural_reliability=agriculture,
        total_cost=cost,
        total_contracts=contracts,
        total_urban_demand_restricted_mm3=restriction,
        data_folder=np.asarray(period_folders[target_period]),
        num_scenarios=np.asarray(N_SCENARIOS, dtype=int),
    )
    print(f"Saved selected-policy cache: {output}", flush=True)


def run_simulation(args):
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    results_dir, performance_cache_dir, output_dir = resolved_paths(args)
    indicator, period = COMBINATIONS[args.task_id - 1]
    data_root = args.data_root.expanduser().resolve()
    period_folders = {"A": args.period_a_data_folder, "B": args.period_b_data_folder}
    data_folder = data_root / period_folders[period]
    if args.model_script is None:
        model_filename = (
            "sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py"
            if args.policy_run.endswith("_decost")
            else "sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN.py"
        )
        model_script = Path(__file__).resolve().parent / "dps_BORG" / model_filename
    else:
        model_script = args.model_script
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not data_folder.is_dir():
        raise FileNotFoundError(f"OOS data folder does not exist: {data_folder}")

    performance = load_performance_cache(
        performance_cache_dir, indicator, period, args.policy_run
    )
    validate_data_scenario_order(data_folder, indicator, performance["scenario_names"])
    source_folder = validate_cache_matches_archives(performance, results_dir)
    if performance["data_folder"] != period_folders[period]:
        raise ValueError(
            f"Performance cache used {performance['data_folder']!r}, expected {period_folders[period]!r}"
        )
    pareto_indices = np.flatnonzero(performance["pareto_mask"])
    pareto_policies = performance["policies"][pareto_indices]
    if len(pareto_policies) == 0:
        raise RuntimeError(f"No OOS Pareto policies found for {indicator}_{period}")

    action_dir = output_dir / "simulation_cache"
    selected_dir = output_dir / "selected_policy_cache"
    final_path = action_dir / f"{indicator}_{period}_oos_pareto_actions.npz"
    partial_path = action_dir / f"{indicator}_{period}_oos_pareto_actions.partial.npz"
    metadata = state_metadata(
        performance, pareto_indices, pareto_policies, source_folder, args.policy_run
    )

    shape = (len(pareto_policies), N_SCENARIOS)
    arrays = [np.full(shape, np.nan) for _ in range(5)]
    completed = 0
    if final_path.is_file() and not args.overwrite:
        existing = load_action_cache(action_dir, indicator, period, args.policy_run)
        if not np.array_equal(existing["source_policy_indices"], pareto_indices):
            raise RuntimeError(f"Existing action cache uses a different OOS Pareto mask: {final_path}")
        if not np.allclose(existing["policies"], pareto_policies, rtol=0, atol=1e-12):
            raise RuntimeError(f"Existing action cache uses different policies: {final_path}")
        if not np.array_equal(existing["scenario_names"].astype(str), performance["scenario_names"]):
            raise RuntimeError(f"Existing action cache uses different scenarios: {final_path}")
        print(f"Completed action cache exists: {final_path}", flush=True)
        completed = len(pareto_policies)
    else:
        if partial_path.is_file() and not args.overwrite:
            *arrays, completed = load_checkpoint(
                partial_path, pareto_policies, pareto_indices, performance["scenario_names"]
            )
            if not 0 <= completed <= len(pareto_policies):
                raise RuntimeError(f"Invalid checkpoint completed count: {completed}")
        model_module = import_model_module(model_script)
        print(f"Task             : {args.task_id}/12", flush=True)
        print(f"Indicator/period : {indicator}/{period}", flush=True)
        print(f"Unique policies  : {len(performance['policies'])}", flush=True)
        print(f"OOS Pareto       : {len(pareto_policies)}", flush=True)
        print(f"OOS data         : {data_folder}", flush=True)
        print(f"Resume at        : {completed + 1 if completed < len(pareto_policies) else 'complete'}", flush=True)

        urban, agriculture, cost, contracts, restriction = arrays
        for local_index in range(completed, len(pareto_policies)):
            print(
                f"  Policy {local_index + 1}/{len(pareto_policies)} "
                f"(source index {pareto_indices[local_index]})",
                flush=True,
            )
            results = simulate_one_policy(
                model_module, pareto_policies[local_index], indicator, period, data_folder
            )
            for destination, values in zip(arrays, results):
                destination[local_index, :] = values
            completed = local_index + 1
            if completed % args.checkpoint_every == 0 or completed == len(pareto_policies):
                save_action_state(
                    partial_path, metadata, urban, agriculture, cost, contracts, restriction, completed
                )

        save_action_state(
            final_path, metadata, urban, agriculture, cost, contracts, restriction, completed
        )
        if partial_path.is_file():
            partial_path.unlink()
        print(f"Saved action cache: {final_path}", flush=True)

    # The two selection-source tasks also create the one required cross-period cache.
    special = {
        (indicator_name, source_period): target_period
        for indicator_name, source_period, target_period in selected_policy_definitions(
            args.policy_run
        )
    }
    if (indicator, period) in special:
        if "model_module" not in locals():
            model_module = import_model_module(model_script)
        target_period = special[(indicator, period)]
        target_performance = load_performance_cache(
            performance_cache_dir, indicator, target_period, args.policy_run
        )
        simulate_selected_cross_period(
            model_module,
            performance,
            target_performance,
            period,
            target_period,
            indicator,
            data_root,
            period_folders,
            selected_dir,
            args.overwrite,
            args.policy_run,
        )


def load_action_cache(action_dir, indicator, period, expected_policy_run):
    path = action_dir / f"{indicator}_{period}_oos_pareto_actions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed action cache: {path}")
    required = {
        "policies",
        "source_policy_indices",
        "scenario_names",
        "indicator",
        "period",
        "policy_run",
        "num_scenarios",
        "completed",
        "total_contracts",
        "total_urban_demand_restricted_mm3",
        "source_mean_total_cost",
        "source_mean_urban_reliability",
        "source_mean_agricultural_reliability",
    }
    with np.load(path, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Action cache {path} is missing: {', '.join(sorted(missing))}")
        item = {
            name: saved[name].copy()
            for name in required
            if name not in {"indicator", "period", "policy_run", "num_scenarios", "completed"}
        }
        item["indicator"] = scalar_text(saved["indicator"])
        item["period"] = scalar_text(saved["period"])
        item["policy_run"] = scalar_text(saved["policy_run"]).lower()
        item["num_scenarios"] = int(np.asarray(saved["num_scenarios"]).reshape(-1)[0])
        item["completed"] = int(np.asarray(saved["completed"]).reshape(-1)[0])
    if (item["indicator"], item["period"]) != (indicator, period):
        raise ValueError(f"Action cache identity mismatch: {path}")
    if item["policy_run"] != expected_policy_run or item["num_scenarios"] != N_SCENARIOS:
        raise ValueError(
            f"Action cache is not a {expected_policy_run} 108-scenario cache: {path}"
        )
    n_policies = len(item["policies"])
    if n_policies == 0:
        raise ValueError(f"Action cache contains no policies: {path}")
    if item["completed"] != n_policies:
        raise ValueError(f"Incomplete action cache {path}: {item['completed']}/{n_policies}")
    if item["source_policy_indices"].shape != (n_policies,):
        raise ValueError(f"Invalid source_policy_indices in {path}")
    if len(np.unique(item["source_policy_indices"])) != n_policies:
        raise ValueError(f"Duplicate source_policy_indices in {path}")
    if item["policies"].shape != (n_policies, N_DECISIONS):
        raise ValueError(f"Invalid policy array in {path}")
    if item["scenario_names"].shape != (N_SCENARIOS,):
        raise ValueError(f"Invalid scenario_names in {path}")
    if len(set(item["scenario_names"].astype(str).tolist())) != N_SCENARIOS:
        raise ValueError(f"Duplicate scenario names in {path}")
    for name in (
        "source_mean_total_cost",
        "source_mean_urban_reliability",
        "source_mean_agricultural_reliability",
    ):
        if item[name].shape != (n_policies,) or not np.all(np.isfinite(item[name])):
            raise ValueError(f"Invalid {name} in {path}")
    expected = (n_policies, N_SCENARIOS)
    for name in ("total_contracts", "total_urban_demand_restricted_mm3"):
        if item[name].shape != expected or not np.all(np.isfinite(item[name])):
            raise ValueError(f"Invalid {name} array in {path}")
    if np.min(item["total_urban_demand_restricted_mm3"]) < -1e-9:
        raise ValueError(f"Negative total urban demand restriction in {path}")
    item["path"] = path
    return item


def load_cross_cache(
    selected_dir, indicator, source_period, target_period, expected_policy_run
):
    path = selected_dir / (
        f"{indicator}_selected_in_{source_period}_evaluated_{target_period}_actions.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected-policy cross-period cache: {path}")
    with np.load(path, allow_pickle=False) as saved:
        item = {name: saved[name].copy() for name in saved.files}
    required = {
        "policy",
        "selection_indicator",
        "selection_period",
        "evaluation_period",
        "policy_run",
        "scenario_names",
        "num_scenarios",
        "total_contracts",
        "total_urban_demand_restricted_mm3",
    }
    missing = required.difference(item)
    if missing:
        raise ValueError(f"Selected cache {path} is missing: {', '.join(sorted(missing))}")
    if scalar_text(item["selection_indicator"]) != indicator:
        raise ValueError(f"Selected cache indicator mismatch: {path}")
    if scalar_text(item["selection_period"]) != source_period:
        raise ValueError(f"Selected cache source-period mismatch: {path}")
    if scalar_text(item["evaluation_period"]) != target_period:
        raise ValueError(f"Selected cache target-period mismatch: {path}")
    if scalar_text(item["policy_run"]).lower() != expected_policy_run:
        raise ValueError(f"Selected cache policy-run mismatch: {path}")
    if int(np.asarray(item["num_scenarios"]).reshape(-1)[0]) != N_SCENARIOS:
        raise ValueError(f"Selected cache does not contain 108 scenarios: {path}")
    for name in ("total_contracts", "total_urban_demand_restricted_mm3"):
        values = np.asarray(item[name], dtype=float).reshape(-1)
        if values.shape != (N_SCENARIOS,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {name} in {path}")
        item[name] = values
    item["path"] = path
    return item


def empirical_cdf(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot plot an empty CDF")
    sorted_values = np.sort(values)
    probabilities = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
    return sorted_values, probabilities


def finish_cdf_figure(fig, axes, title, legend_handles, legend_labels, output_base, dpi):
    for row in range(2):
        maxima = [axes[row, column].get_xlim()[1] for column in range(2)]
        right = max(maxima)
        for column in range(2):
            axes[row, column].set_xlim(0.0, right)
            axes[row, column].set_ylim(0.0, 1.01)
            axes[row, column].grid(axis="y", alpha=0.2, linewidth=0.6)
    axes[0, 0].set_ylabel("Cumulative probability")
    axes[1, 0].set_ylabel("Cumulative probability")
    axes[0, 0].set_xlabel("Total contracts purchased (thousands)")
    axes[0, 1].set_xlabel("Total contracts purchased (thousands)")
    axes[1, 0].set_xlabel("Total urban demand restricted (Mm³)")
    axes[1, 1].set_xlabel("Total urban demand restricted (Mm³)")
    panel_labels = (("a", "b"), ("c", "d"))
    for row in range(2):
        for column in range(2):
            axes[row, column].text(
                0.015,
                0.96,
                f"({panel_labels[row][column]})",
                transform=axes[row, column].transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
    fig.suptitle(title, y=0.985, fontsize=14)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(legend_labels),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.875), h_pad=2.0, w_pad=1.5)
    require_matplotlib_panel_alignment(fig, axes, output_base)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def require_matplotlib_panel_alignment(fig, axes, output_base, tolerance_pt=1.5):
    """Require aligned plot areas in the repeated 2x2 CDF layout."""
    fig.canvas.draw()
    width_pt, height_pt = fig.get_size_inches() * 72.0
    panels = []
    for row in range(2):
        for column in range(2):
            position = axes[row, column].get_position()
            panels.append(
                {
                    "id": f"panel_{row}_{column}",
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
    deviations = {}
    for row in range(2):
        group = [panel for panel in panels if panel["row"] == row]
        for field in ("top", "bottom", "width", "height"):
            values = [panel[field] for panel in group]
            deviations[f"row_{row}_{field}"] = max(values) - min(values)
    for column in range(2):
        group = [panel for panel in panels if panel["column"] == column]
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
    alignment_path = Path(str(output_base) + ".alignment.json")
    with alignment_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not passed:
        raise RuntimeError(
            f"CDF panel alignment exceeds {tolerance_pt} pt: {deviations}"
        )


def plot_all_policy_cdfs(action_results, output_dir, dpi, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    handles = []
    labels = []
    order = ("SRI3", "SRI6", "SRI12", "SPI3", "SPI6", "SPI12")
    for column, period in enumerate(PERIODS):
        axes[0, column].set_title(PERIOD_LABELS[period])
        for indicator in order:
            item = action_results[(indicator, period)]
            contract_x, contract_y = empirical_cdf(item["total_contracts"] / 1_000.0)
            demand_x, demand_y = empirical_cdf(item["total_urban_demand_restricted_mm3"])
            line = axes[0, column].plot(
                contract_x,
                contract_y,
                color=COLORS[indicator],
                linestyle=LINESTYLES[indicator],
                linewidth=2.0,
                label=indicator,
            )[0]
            axes[1, column].plot(
                demand_x,
                demand_y,
                color=COLORS[indicator],
                linestyle=LINESTYLES[indicator],
                linewidth=2.0,
            )
            if column == 0:
                handles.append(line)
                labels.append(indicator)
    finish_cdf_figure(
        fig,
        axes,
        "Actions across all per-indicator OOS Pareto policies",
        handles,
        labels,
        output_dir / "cdf_actions_all_oos_pareto_policies",
        dpi,
    )
    plt.close(fig)


def matching_selected_row(action_item, source_policy_index):
    matches = np.flatnonzero(action_item["source_policy_indices"] == source_policy_index)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one selected source index {source_policy_index} in {action_item['path']}"
        )
    return int(matches[0])


def build_selected_results(
    action_results, performance_cache_dir, selected_dir, policy_run
):
    selected = {}
    rows = []
    definitions = selected_policy_definitions(policy_run)
    for indicator, source_period, cross_period in definitions:
        performance = load_performance_cache(
            performance_cache_dir, indicator, source_period, policy_run
        )
        source_index = selected_policy_index(performance, indicator, source_period)
        policy = performance["policies"][source_index]
        matching_actions = action_results[(indicator, source_period)]
        local_index = matching_selected_row(matching_actions, source_index)
        selected[(indicator, source_period)] = {
            "total_contracts": matching_actions["total_contracts"][local_index],
            "total_urban_demand_restricted_mm3": matching_actions[
                "total_urban_demand_restricted_mm3"
            ][local_index],
        }
        cross = load_cross_cache(
            selected_dir, indicator, source_period, cross_period, policy_run
        )
        if not np.allclose(cross["policy"], policy, rtol=0, atol=1e-12):
            raise RuntimeError(f"Cross-period cache policy differs for {indicator}")
        selected[(indicator, cross_period)] = cross
        objective = performance["objectives"][source_index]
        rows.append(
            {
                "indicator": indicator,
                "selection_period": source_period,
                "source_policy_index": source_index,
                "contract_threshold": policy[0],
                "contract_action": policy[1],
                "demand_threshold": policy[2],
                "demand_action": policy[3],
                "oos_mean_total_cost": objective[0],
                "oos_mean_urban_reliability": objective[1],
                "oos_mean_agricultural_reliability": objective[2],
            }
        )
    return selected, rows


def plot_selected_policy_cdfs(selected, output_dir, dpi, plt, policy_run):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    definitions = selected_policy_definitions(policy_run)
    styles = {
        indicator: (
            COLORS[indicator],
            LINESTYLES[indicator],
            f"{indicator} policy (selected in Period {source_period})",
        )
        for indicator, source_period, _ in definitions
    }
    handles = []
    labels = []
    for column, period in enumerate(PERIODS):
        axes[0, column].set_title(PERIOD_LABELS[period])
        for indicator, _, _ in definitions:
            item = selected[(indicator, period)]
            color, linestyle, label = styles[indicator]
            contract_x, contract_y = empirical_cdf(item["total_contracts"] / 1_000.0)
            demand_x, demand_y = empirical_cdf(item["total_urban_demand_restricted_mm3"])
            line = axes[0, column].plot(
                contract_x,
                contract_y,
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=label,
            )[0]
            axes[1, column].plot(
                demand_x,
                demand_y,
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
            )
            if column == 0:
                handles.append(line)
                labels.append(label)
    finish_cdf_figure(
        fig,
        axes,
        "Actions of two OOS-selected policies evaluated in both periods",
        handles,
        labels,
        output_dir / "cdf_actions_selected_policies",
        dpi,
    )
    plt.close(fig)


def build_all_indicator_highest_urban_results(
    action_results, performance_cache_dir, policy_run
):
    """Select each indicator-period's highest-urban OOS Pareto policy."""
    selected = {}
    rows = []
    for period in PERIODS:
        for indicator in INDICATORS:
            performance = load_performance_cache(
                performance_cache_dir, indicator, period, policy_run
            )
            source_index = selected_policy_index(performance, indicator, period)
            action_item = action_results[(indicator, period)]
            local_index = matching_selected_row(action_item, source_index)
            selected[(indicator, period)] = {
                "total_contracts": action_item["total_contracts"][local_index],
                "total_urban_demand_restricted_mm3": action_item[
                    "total_urban_demand_restricted_mm3"
                ][local_index],
            }
            policy = performance["policies"][source_index]
            objective = performance["objectives"][source_index]
            rows.append(
                {
                    "indicator": indicator,
                    "period": period,
                    "source_policy_index": source_index,
                    "action_cache_row": local_index,
                    "contract_threshold": policy[0],
                    "contract_action": policy[1],
                    "demand_threshold": policy[2],
                    "demand_action": policy[3],
                    "oos_mean_total_cost": objective[0],
                    "oos_mean_urban_reliability": objective[1],
                    "oos_mean_agricultural_reliability": objective[2],
                }
            )
    return selected, rows


def plot_all_indicator_highest_urban_cdfs(selected, output_dir, dpi, plt):
    """Plot one selected policy per indicator and period (108 samples each)."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    handles = []
    labels = []
    order = ("SRI3", "SRI6", "SRI12", "SPI3", "SPI6", "SPI12")
    for column, period in enumerate(PERIODS):
        axes[0, column].set_title(PERIOD_LABELS[period])
        for indicator in order:
            item = selected[(indicator, period)]
            contract_x, contract_y = empirical_cdf(
                item["total_contracts"] / 1_000.0
            )
            demand_x, demand_y = empirical_cdf(
                item["total_urban_demand_restricted_mm3"]
            )
            line = axes[0, column].plot(
                contract_x,
                contract_y,
                color=COLORS[indicator],
                linestyle=LINESTYLES[indicator],
                linewidth=2.2,
                label=indicator,
            )[0]
            axes[1, column].plot(
                demand_x,
                demand_y,
                color=COLORS[indicator],
                linestyle=LINESTYLES[indicator],
                linewidth=2.2,
            )
            if column == 0:
                handles.append(line)
                labels.append(indicator)
    finish_cdf_figure(
        fig,
        axes,
        "Highest-urban-reliability OOS Pareto policy for each indicator",
        handles,
        labels,
        output_dir / "cdf_actions_highest_urban_reliability_by_indicator",
        dpi,
    )
    plt.close(fig)


def build_all_indicator_target_cost_results(
    action_results,
    performance_cache_dir,
    policy_run,
    target_cost_million,
):
    """Select each Pareto policy closest to, but below, target mean cost."""
    selected = {}
    rows = []
    for period in PERIODS:
        for indicator in INDICATORS:
            performance = load_performance_cache(
                performance_cache_dir, indicator, period, policy_run
            )
            source_index = target_cost_policy_index(
                performance, indicator, period, target_cost_million
            )
            action_item = action_results[(indicator, period)]
            local_index = matching_selected_row(action_item, source_index)
            selected[(indicator, period)] = {
                "total_contracts": action_item["total_contracts"][local_index],
                "total_urban_demand_restricted_mm3": action_item[
                    "total_urban_demand_restricted_mm3"
                ][local_index],
            }
            policy = performance["policies"][source_index]
            objective = performance["objectives"][source_index]
            mean_cost_million = objective[0] / 1_000_000.0
            if not mean_cost_million < target_cost_million:
                raise AssertionError(
                    "Target-cost selector returned a policy that is not "
                    "strictly below the requested cost bound"
                )
            rows.append(
                {
                    "indicator": indicator,
                    "period": period,
                    "source_policy_index": source_index,
                    "action_cache_row": local_index,
                    "contract_threshold": policy[0],
                    "contract_action": policy[1],
                    "demand_threshold": policy[2],
                    "demand_action": policy[3],
                    "target_cost_million": target_cost_million,
                    "distance_below_target_million": (
                        target_cost_million - mean_cost_million
                    ),
                    "oos_mean_total_cost": objective[0],
                    "oos_mean_total_cost_million": mean_cost_million,
                    "oos_mean_urban_reliability": objective[1],
                    "oos_mean_agricultural_reliability": objective[2],
                }
            )
    return selected, rows


def target_cost_tag(target_cost_million):
    if float(target_cost_million).is_integer():
        return f"{int(target_cost_million)}M"
    return f"{target_cost_million:g}M".replace(".", "pt")


def plot_all_indicator_target_cost_cdfs(
    selected, output_dir, dpi, plt, target_cost_million
):
    """Plot one closest-below-target-cost policy per indicator and period."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    handles = []
    labels = []
    order = ("SRI3", "SRI6", "SRI12", "SPI3", "SPI6", "SPI12")
    for column, period in enumerate(PERIODS):
        axes[0, column].set_title(PERIOD_LABELS[period])
        for indicator in order:
            item = selected[(indicator, period)]
            contract_x, contract_y = empirical_cdf(
                item["total_contracts"] / 1_000.0
            )
            demand_x, demand_y = empirical_cdf(
                item["total_urban_demand_restricted_mm3"]
            )
            line = axes[0, column].plot(
                contract_x,
                contract_y,
                color=COLORS[indicator],
                linestyle=LINESTYLES[indicator],
                linewidth=2.2,
                label=indicator,
            )[0]
            axes[1, column].plot(
                demand_x,
                demand_y,
                color=COLORS[indicator],
                linestyle=LINESTYLES[indicator],
                linewidth=2.2,
            )
            if column == 0:
                handles.append(line)
                labels.append(indicator)
    finish_cdf_figure(
        fig,
        axes,
        (
            "OOS Pareto policy closest below "
            f"{target_cost_million:,.0f} M$ mean total cost for each indicator"
        ),
        handles,
        labels,
        output_dir
        / (
            "cdf_actions_cost_below_"
            f"{target_cost_tag(target_cost_million)}_by_indicator"
        ),
        dpi,
    )
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_plot(args):
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
    _, performance_cache_dir, output_dir = resolved_paths(args)
    action_dir = output_dir / "simulation_cache"
    selected_dir = output_dir / "selected_policy_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    action_results = {}
    count_rows = []
    for period in PERIODS:
        for indicator in INDICATORS:
            item = load_action_cache(action_dir, indicator, period, args.policy_run)
            action_results[(indicator, period)] = item
            count_rows.append(
                {
                    "indicator": indicator,
                    "period": period,
                    "oos_pareto_policy_count": len(item["policies"]),
                    "cdf_policy_scenario_samples": len(item["policies"]) * N_SCENARIOS,
                    "source": item["path"],
                }
            )

    selected, selected_rows = build_selected_results(
        action_results, performance_cache_dir, selected_dir, args.policy_run
    )
    all_indicator_selected, all_indicator_rows = (
        build_all_indicator_highest_urban_results(
            action_results, performance_cache_dir, args.policy_run
        )
    )
    target_cost_results = []
    for target_cost in target_costs:
        target_cost_selected, target_cost_rows = (
            build_all_indicator_target_cost_results(
                action_results,
                performance_cache_dir,
                args.policy_run,
                target_cost,
            )
        )
        target_cost_results.append(
            {
                "target_cost_million": target_cost,
                "target_tag": target_cost_tag(target_cost),
                "selected": target_cost_selected,
                "rows": target_cost_rows,
            }
        )
    plot_all_policy_cdfs(action_results, output_dir, args.dpi, plt)
    plot_selected_policy_cdfs(
        selected, output_dir, args.dpi, plt, args.policy_run
    )
    plot_all_indicator_highest_urban_cdfs(
        all_indicator_selected, output_dir, args.dpi, plt
    )
    for target_result in target_cost_results:
        plot_all_indicator_target_cost_cdfs(
            target_result["selected"],
            output_dir,
            args.dpi,
            plt,
            target_result["target_cost_million"],
        )
    write_csv(
        output_dir / "oos_pareto_policy_counts.csv",
        ["indicator", "period", "oos_pareto_policy_count", "cdf_policy_scenario_samples", "source"],
        count_rows,
    )
    write_csv(
        output_dir / "selected_policies.csv",
        [
            "indicator",
            "selection_period",
            "source_policy_index",
            "contract_threshold",
            "contract_action",
            "demand_threshold",
            "demand_action",
            "oos_mean_total_cost",
            "oos_mean_urban_reliability",
            "oos_mean_agricultural_reliability",
        ],
        selected_rows,
    )
    write_csv(
        output_dir / "highest_urban_reliability_policies_by_indicator.csv",
        [
            "indicator",
            "period",
            "source_policy_index",
            "action_cache_row",
            "contract_threshold",
            "contract_action",
            "demand_threshold",
            "demand_action",
            "oos_mean_total_cost",
            "oos_mean_urban_reliability",
            "oos_mean_agricultural_reliability",
        ],
        all_indicator_rows,
    )
    for target_result in target_cost_results:
        write_csv(
            output_dir
            / (
                "cost_below_"
                f"{target_result['target_tag']}_policies_by_indicator.csv"
            ),
            [
                "indicator",
                "period",
                "source_policy_index",
                "action_cache_row",
                "contract_threshold",
                "contract_action",
                "demand_threshold",
                "demand_action",
                "target_cost_million",
                "distance_below_target_million",
                "oos_mean_total_cost",
                "oos_mean_total_cost_million",
                "oos_mean_urban_reliability",
                "oos_mean_agricultural_reliability",
            ],
            target_result["rows"],
        )
    selected_period_b_indicator = selected_policy_definitions(args.policy_run)[1][0]
    target_list_text = ", ".join(f"{value:g}" for value in target_costs)
    note = f"""OOS action CDF method

Policy run: {args.policy_run}

Candidate policies are all unique decision vectors pooled across seven
Borg seeds. For each indicator-period, policies are filtered using the existing
108-scenario OOS mean objectives: total cost is minimized, urban reliability is
maximized, and agricultural reliability is maximized. This is the per-indicator
OOS Pareto mask, not the six-indicator joint mask.

The all-policy CDF flattens every retained OOS Pareto policy by all 108 climate
scenarios. Total contracts are temporally summed by the Pywr recorder. Total
urban demand restricted is the temporal sum of unrestricted demand minus the
temporal sum of policy-restricted demand, converted from weekly m3/s samples to
Mm3 using 604800 seconds per weekly timestep.

The SPI12 policy is selected from the Period A OOS Pareto set by maximum OOS
mean urban reliability. The {selected_period_b_indicator} policy is selected
analogously from Period B.
The two fixed decision vectors are each evaluated in both OOS periods.

The additional all-indicator selected-policy figure independently selects the
maximum-OOS-mean-urban-reliability policy from every indicator-period Pareto
set. Ties are resolved by lower mean total cost, higher mean agricultural
reliability, and then the lexicographically smallest decision vector. Each CDF
in that figure contains the selected policy's 108 scenario-level action values;
it does not pool multiple policies and does not require new simulations.

The target-cost figures use target costs of {target_list_text} M$. For every
target and every indicator-period OOS Pareto set, the policy with the largest
108-scenario mean total cost strictly below the target is selected (i.e., the
closest policy from below). Policies at or above the applicable target are
excluded. Exact-cost ties are resolved by higher mean urban reliability,
higher mean agricultural reliability, and then the lexicographically smallest
decision vector. Each CDF contains that single policy's 108 scenario-level
action values and is extracted from the completed Pareto-policy action caches;
no additional model simulations are required.
"""
    (output_dir / "cdf_actions_method.txt").write_text(note, encoding="utf-8")
    print(f"CDF figures and tables written to: {output_dir}")


def main():
    args = parse_args()
    if args.command == "simulate":
        run_simulation(args)
    else:
        run_plot(args)


if __name__ == "__main__":
    main()
