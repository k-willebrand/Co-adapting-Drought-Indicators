"""Evaluate archived NATWAT policies on 108 out-of-sample scenarios.

The Borg optimization is not rerun. For one indicator-period combination this
program pools the archived policies from all available seeds, removes duplicate
decision vectors, evaluates every policy with Pywr on all 108 out-of-sample
scenarios, and saves scenario-level urban reliability, agricultural reliability,
and total cost. It is intended for a 12-task SLURM array and supports restart
from checkpoints.
"""

import argparse
import csv
import gc
import importlib.util
import os
import re
import sys

import numpy as np


INDICATORS = ("SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12")
PERIODS = ("A", "B")
COMBINATIONS = tuple((indicator, period) for period in PERIODS for indicator in INDICATORS)
N_DECISION_VARS = 4
N_OBJECTIVES = 3
N_SCENARIOS = 108

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_RUN_FOLDERS = {
    "aug24": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug24_MIN",
    "aug28": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Aug28_MIN",
    "sep03_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep3_MIN_DeCost",
    "sep09_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep9_MIN_DeCost",
    "sep16_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost",
    "sep29_c05_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_C05_MIN_DeCost",
    "sep29_c15_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_C15_MIN_DeCost",
    "sep29_l20_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_L20_MIN_DeCost",
    "sep29_l40_decost": "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep29_L40_MIN_DeCost",
}
POLICY_RUN_DEMAND_SETTINGS = {
    "sep03_decost": (1.0, 0.3),
    "sep09_decost": (1.0, 0.3),
    "sep16_decost": (1.0, 0.3),
    "sep29_c05_decost": (0.5, 0.3),
    "sep29_c15_decost": (1.5, 0.3),
    "sep29_l20_decost": (1.0, 0.2),
    "sep29_l40_decost": (1.0, 0.4),
}
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
DEFAULT_PERIOD_A_DATA_FOLDER = "transient_108_no_drought_pert_periodA"
DEFAULT_PERIOD_B_DATA_FOLDER = "transient_108_no_drought_pert_periodB"
OOS_FLOW_FILES = (
    "YESO.csv",
    "MAIPO.csv",
    "COLORADO.csv",
    "VOLCAN.csv",
    "LAGUNANEGRA.csv",
    "MAIPOEXTRA.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--task-id",
        type=int,
        choices=range(1, len(COMBINATIONS) + 1),
        metavar="1-12",
        help="SLURM task: A/SPI3..SRI12 are 1-6; B/SPI3..SRI12 are 7-12",
    )
    selection.add_argument("--indicator", choices=INDICATORS)
    parser.add_argument("--period", choices=PERIODS)
    parser.add_argument(
        "--policy-run",
        choices=tuple(POLICY_RUN_FOLDERS),
        default="aug24",
        help=(
            "optimization policy archive to evaluate (default: aug24); "
            "*_decost labels select runs whose demand-restriction settings "
            "are supplied separately"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="optional explicit override for the folder selected by --policy-run",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--period-a-data-folder", default=DEFAULT_PERIOD_A_DATA_FOLDER
    )
    parser.add_argument(
        "--period-b-data-folder", default=DEFAULT_PERIOD_B_DATA_FOLDER
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: <selected-results-dir>/out_of_sample/simulation_cache",
    )
    parser.add_argument(
        "--model-script",
        default=None,
        help="default: <project-root>/dps_BORG/sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN.py",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=None,
        help="fail unless this many Borg seed archives are available",
    )
    parser.add_argument(
        "--require-demand-restriction-cost",
        action="store_true",
        help="record DemandRestrictionCost and fail if the model does not provide it",
    )
    parser.add_argument(
        "--demand-restriction-cost-per-m3",
        type=float,
        default=1.0,
        help="Demand-curtailment cost used by the policy run (default: 1.0 USD/m3).",
    )
    parser.add_argument(
        "--max-demand-curtailment-fraction",
        type=float,
        default=0.3,
        help="Maximum weekly demand curtailment used in training (default: 0.3).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.indicator and not args.period:
        parser.error("--period is required with --indicator")
    if args.period and not args.indicator:
        parser.error("--period can only be used with --indicator")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.expected_seeds is not None and args.expected_seeds < 1:
        parser.error("--expected-seeds must be at least 1")
    if (
        not np.isfinite(args.demand_restriction_cost_per_m3)
        or args.demand_restriction_cost_per_m3 < 0.0
    ):
        parser.error("--demand-restriction-cost-per-m3 must be finite and non-negative")
    if (
        not np.isfinite(args.max_demand_curtailment_fraction)
        or not 0.0 <= args.max_demand_curtailment_fraction <= 1.0
    ):
        parser.error("--max-demand-curtailment-fraction must be between 0 and 1")
    expected_settings = POLICY_RUN_DEMAND_SETTINGS.get(args.policy_run)
    if expected_settings is not None:
        expected_cost, expected_max_curtailment = expected_settings
        if not np.isclose(
            args.demand_restriction_cost_per_m3,
            expected_cost,
            rtol=0.0,
            atol=1e-12,
        ):
            parser.error(
                f"{args.policy_run} requires "
                f"--demand-restriction-cost-per-m3 {expected_cost:g}"
            )
        if not np.isclose(
            args.max_demand_curtailment_fraction,
            expected_max_curtailment,
            rtol=0.0,
            atol=1e-12,
        ):
            parser.error(
                f"{args.policy_run} requires "
                "--max-demand-curtailment-fraction "
                f"{expected_max_curtailment:g}"
            )
    return args


def selected_combination(args):
    if args.task_id is not None:
        return COMBINATIONS[args.task_id - 1]
    return args.indicator, args.period


def find_result_folder(results_dir, indicator, period):
    prefix = f"{indicator}_{period}_"
    matches = [
        os.path.join(results_dir, name)
        for name in sorted(os.listdir(results_dir))
        if name.startswith(prefix) and os.path.isdir(os.path.join(results_dir, name))
    ]
    if not matches:
        raise FileNotFoundError(
            f"No optimization result folder beginning with {prefix!r} under {results_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple optimization folders match this indicator-period; use a results "
            "directory containing only the intended run:\n  " + "\n  ".join(matches)
        )
    return matches[0]


def read_policy_rows(path):
    policies = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                values = [float(value) for value in text.split()]
            except ValueError as exc:
                raise ValueError(f"Non-numeric value in {path}:{line_number}") from exc
            if len(values) < N_DECISION_VARS + N_OBJECTIVES:
                raise ValueError(
                    f"Expected at least 7 columns in {path}:{line_number}; got {len(values)}"
                )
            policies.append(values[:N_DECISION_VARS])
    return policies


def load_unique_policies(result_folder):
    sets_dir = os.path.join(result_folder, "sets")
    if not os.path.isdir(sets_dir):
        raise FileNotFoundError(f"Missing sets directory: {sets_dir}")
    pattern = re.compile(r"^Borg_DPS_PySedSim(\d+)\.set$")
    archives = []
    for name in os.listdir(sets_dir):
        match = pattern.match(name)
        if match:
            archives.append((int(match.group(1)), os.path.join(sets_dir, name)))
    if not archives:
        raise FileNotFoundError(f"No Borg_DPS_PySedSim*.set files in {sets_dir}")

    pooled = []
    for _, path in sorted(archives):
        pooled.extend(read_policy_rows(path))
    if not pooled:
        raise RuntimeError(f"No policies found in {sets_dir}")

    unique = []
    seen = set()
    for policy in pooled:
        key = tuple(np.round(policy, 12))
        if key not in seen:
            seen.add(key)
            unique.append(policy)
    return (
        np.asarray(unique, dtype=float),
        len(pooled),
        [seed_index for seed_index, _ in sorted(archives)],
    )


def import_model_module(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model script not found: {path}")
    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("natwat_3obj_oos_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_model"):
        raise AttributeError(f"{path} does not define make_model()")
    return module


def read_scenario_names(indicator_csv):
    import pandas as pd

    frame = pd.read_csv(indicator_csv, index_col=0, nrows=1)
    names = [str(name) for name in frame.columns]
    if len(names) != N_SCENARIOS:
        raise ValueError(
            f"Expected {N_SCENARIOS} columns in {indicator_csv}; got {len(names)}"
        )
    if len(set(names)) != N_SCENARIOS:
        raise ValueError(f"Scenario names are not unique in {indicator_csv}")
    return np.asarray(names, dtype="U120")


def read_csv_header(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc


def validate_oos_scenario_order(data_folder, indicator):
    """Require the selected drought index and all inflows to use one scenario order."""
    indicator_path = os.path.join(data_folder, f"{indicator}.csv")
    indicator_header = read_csv_header(indicator_path)
    indicator_names = indicator_header[1:]
    if len(indicator_names) != N_SCENARIOS:
        raise ValueError(
            f"Expected {N_SCENARIOS} scenario columns in {indicator_path}; "
            f"got {len(indicator_names)}"
        )

    for filename in OOS_FLOW_FILES:
        path = os.path.join(data_folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required OOS input file missing: {path}")
        names = read_csv_header(path)[1:]
        if len(names) != N_SCENARIOS:
            raise ValueError(
                f"Expected {N_SCENARIOS} scenario columns in {path}; got {len(names)}"
            )
        if names != indicator_names:
            mismatches = [
                index + 1
                for index, (indicator_name, flow_name) in enumerate(
                    zip(indicator_names, names)
                )
                if indicator_name != flow_name
            ]
            first = mismatches[0]
            raise ValueError(
                "OOS scenario-column order mismatch between "
                f"{os.path.basename(indicator_path)} and {filename}. "
                f"First mismatch is scenario column {first}: "
                f"{indicator_names[first - 1]!r} versus {names[first - 1]!r}. "
                f"Total mismatched positions: {len(mismatches)}. Reorder columns "
                "before simulation; Pywr pairs scenarios by column position."
            )

    extra_data_path = os.path.join(data_folder, "Extra data.csv")
    if not os.path.isfile(extra_data_path):
        raise FileNotFoundError(f"Required OOS input file missing: {extra_data_path}")


def recorder_values(model, name):
    values = np.asarray(model.recorders[name].values(), dtype=float).reshape(-1)
    if values.size != N_SCENARIOS:
        raise ValueError(
            f"Recorder {name!r} returned {values.size} values; expected {N_SCENARIOS}. "
            "Confirm that the uploaded make_model accepts num_scenarios=108."
        )
    if not np.all(np.isfinite(values)):
        bad = np.where(~np.isfinite(values))[0].tolist()
        raise ValueError(f"Recorder {name!r} has non-finite scenarios: {bad}")
    return values


def save_atomic(path, **arrays):
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_checkpoint(
    path, metadata, policies, urban, agriculture, cost, demand_cost, completed
):
    arrays = dict(
        policies=policies,
        urban_reliability=urban,
        agricultural_reliability=agriculture,
        total_cost=cost,
        completed=np.asarray(completed, dtype=int),
        **metadata,
    )
    if demand_cost is not None:
        arrays["demand_restriction_cost"] = demand_cost
    save_atomic(path, **arrays)


def load_checkpoint(
    path,
    policies,
    scenario_names,
    require_demand_cost,
    expected_policy_run,
    expected_indicator,
    expected_period,
    expected_data_folder,
    expected_demand_restriction_cost_per_m3,
    expected_max_demand_curtailment_fraction,
):
    with np.load(path, allow_pickle=False) as saved:
        required_metadata = {
            "policy_run",
            "indicator",
            "period",
            "data_folder",
            "num_scenarios",
        }
        missing = required_metadata.difference(saved.files)
        if missing:
            raise RuntimeError(
                f"Checkpoint {path} lacks metadata: {', '.join(sorted(missing))}"
            )
        saved_identity = (
            str(np.asarray(saved["policy_run"]).reshape(-1)[0]).lower(),
            str(np.asarray(saved["indicator"]).reshape(-1)[0]),
            str(np.asarray(saved["period"]).reshape(-1)[0]),
            str(np.asarray(saved["data_folder"]).reshape(-1)[0]),
            int(np.asarray(saved["num_scenarios"]).reshape(-1)[0]),
        )
        expected_identity = (
            expected_policy_run,
            expected_indicator,
            expected_period,
            expected_data_folder,
            N_SCENARIOS,
        )
        if saved_identity != expected_identity:
            raise RuntimeError(
                f"Checkpoint identity differs from this run: {path}\n"
                f"saved={saved_identity}; expected={expected_identity}"
            )
        saved_demand_cost_rate = (
            float(np.asarray(saved["demand_restriction_cost_per_m3"]).reshape(-1)[0])
            if "demand_restriction_cost_per_m3" in saved.files
            else 1.0
        )
        saved_max_curtailment = (
            float(
                np.asarray(saved["max_demand_curtailment_fraction"]).reshape(-1)[0]
            )
            if "max_demand_curtailment_fraction" in saved.files
            else 0.3
        )
        if not np.isclose(
            saved_demand_cost_rate,
            expected_demand_restriction_cost_per_m3,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Checkpoint demand-restriction cost differs in {path}: "
                f"saved={saved_demand_cost_rate}, "
                f"expected={expected_demand_restriction_cost_per_m3}"
            )
        if not np.isclose(
            saved_max_curtailment,
            expected_max_demand_curtailment_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Checkpoint maximum curtailment differs in {path}: "
                f"saved={saved_max_curtailment}, "
                f"expected={expected_max_demand_curtailment_fraction}"
            )
        saved_policies = saved["policies"]
        saved_scenarios = saved["scenario_names"].astype(str)
        if saved_policies.shape != policies.shape or not np.allclose(
            saved_policies, policies, rtol=0, atol=1e-12
        ):
            raise RuntimeError(
                f"Checkpoint policies differ from the current Borg archives: {path}"
            )
        if not np.array_equal(saved_scenarios, scenario_names.astype(str)):
            raise RuntimeError(f"Checkpoint scenario order differs: {path}")
        demand_cost = None
        if require_demand_cost:
            if "demand_restriction_cost" not in saved.files:
                raise RuntimeError(
                    f"Checkpoint lacks DemandRestrictionCost values: {path}"
                )
            demand_cost = saved["demand_restriction_cost"].copy()
        return (
            saved["urban_reliability"].copy(),
            saved["agricultural_reliability"].copy(),
            saved["total_cost"].copy(),
            demand_cost,
            int(saved["completed"]),
        )


def main():
    args = parse_args()
    indicator, period = selected_combination(args)
    selected_results_dir = os.path.join(SCRIPT_DIR, POLICY_RUN_FOLDERS[args.policy_run])
    results_dir = os.path.abspath(args.results_dir or selected_results_dir)
    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(results_dir, "out_of_sample", "simulation_cache")
    )
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    data_folder_name = (
        args.period_a_data_folder if period == "A" else args.period_b_data_folder
    )
    data_folder = os.path.join(data_root, data_folder_name)
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Out-of-sample data folder missing: {data_folder}")
    indicator_csv = os.path.join(data_folder, f"{indicator}.csv")
    if not os.path.isfile(indicator_csv):
        raise FileNotFoundError(f"Indicator file missing: {indicator_csv}")
    validate_oos_scenario_order(data_folder, indicator)
    scenario_names = read_scenario_names(indicator_csv)

    result_folder = find_result_folder(results_dir, indicator, period)
    policies, pooled_count, seed_indices = load_unique_policies(result_folder)
    seed_count = len(seed_indices)
    if args.expected_seeds is not None and seed_indices != list(
        range(1, args.expected_seeds + 1)
    ):
        raise RuntimeError(
            f"Expected seed archives 1..{args.expected_seeds} in {result_folder}; "
            f"found {seed_indices}"
        )
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"{indicator}_{period}_oos_performance.npz")
    partial_path = os.path.join(output_dir, f"{indicator}_{period}_oos_performance.partial.npz")

    print(f"Policy run         : {args.policy_run}", flush=True)
    print(f"Indicator / period : {indicator} / {period}", flush=True)
    print(f"Optimization folder: {result_folder}", flush=True)
    print(f"OOS data folder    : {data_folder}", flush=True)
    print(f"Scenarios          : {len(scenario_names)}", flush=True)
    print(f"Seed archives      : {seed_count}", flush=True)
    print(f"Seed indices       : {seed_indices}", flush=True)
    print(f"Pooled policies    : {pooled_count}", flush=True)
    print(f"Unique policies    : {len(policies)}", flush=True)
    print(f"Final cache        : {final_path}", flush=True)
    print(
        "Demand cost rate   : "
        f"{args.demand_restriction_cost_per_m3:g} USD/m3 curtailed",
        flush=True,
    )
    print(
        "Maximum curtailment: "
        f"{100.0 * args.max_demand_curtailment_fraction:g}%",
        flush=True,
    )

    if os.path.exists(final_path) and not args.overwrite:
        existing = load_checkpoint(
            final_path,
            policies,
            scenario_names,
            args.require_demand_restriction_cost,
            args.policy_run,
            indicator,
            period,
            data_folder_name,
            args.demand_restriction_cost_per_m3,
            args.max_demand_curtailment_fraction,
        )
        existing_arrays = list(existing[:3])
        if existing[3] is not None:
            existing_arrays.append(existing[3])
        if existing[4] != len(policies) or not all(
            np.all(np.isfinite(values)) for values in existing_arrays
        ):
            raise RuntimeError(f"Existing completed cache is invalid: {final_path}")
        print("Validated completed cache exists; nothing to do.", flush=True)
        return

    shape = (len(policies), N_SCENARIOS)
    urban = np.full(shape, np.nan)
    agriculture = np.full(shape, np.nan)
    cost = np.full(shape, np.nan)
    demand_cost = (
        np.full(shape, np.nan) if args.require_demand_restriction_cost else None
    )
    completed = 0
    if os.path.exists(partial_path) and not args.overwrite:
        urban, agriculture, cost, demand_cost, completed = load_checkpoint(
            partial_path,
            policies,
            scenario_names,
            args.require_demand_restriction_cost,
            args.policy_run,
            indicator,
            period,
            data_folder_name,
            args.demand_restriction_cost_per_m3,
            args.max_demand_curtailment_fraction,
        )
        if not 0 <= completed <= len(policies):
            raise RuntimeError(f"Invalid checkpoint count {completed}: {partial_path}")
        print(f"Resuming at policy {completed + 1}/{len(policies)}", flush=True)

    default_model_name = (
        "sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py"
        if args.policy_run.endswith("_decost")
        else "sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN.py"
    )
    model_script = args.model_script or os.path.join(
        SCRIPT_DIR, "dps_BORG", default_model_name
    )
    model_script = os.path.abspath(model_script)
    model_module = import_model_module(model_script)
    num_dp = int(getattr(model_module, "num_DP", 6))

    metadata = {
        "scenario_names": scenario_names,
        "indicator": np.asarray(indicator),
        "period": np.asarray(period),
        "policy_run": np.asarray(args.policy_run),
        "data_folder": np.asarray(data_folder_name),
        "result_folder": np.asarray(result_folder),
        "model_script": np.asarray(model_script),
        "num_scenarios": np.asarray(N_SCENARIOS, dtype=int),
        "demand_restriction_cost_required": np.asarray(
            args.require_demand_restriction_cost, dtype=bool
        ),
        "demand_restriction_cost_per_m3": np.asarray(
            args.demand_restriction_cost_per_m3, dtype=float
        ),
        "max_demand_curtailment_fraction": np.asarray(
            args.max_demand_curtailment_fraction, dtype=float
        ),
    }

    for policy_index in range(completed, len(policies)):
        contract_threshold, contract_action, demand_threshold, demand_action = policies[
            policy_index
        ]
        model = model_module.make_model(
            contract_threshold_vals=contract_threshold * np.ones(num_dp),
            contract_action_vals=contract_action * np.ones(num_dp),
            demand_threshold_vals=[demand_threshold * np.ones(12)],
            demand_action_vals=[np.ones(12), demand_action * np.ones(12)],
            indicator=indicator,
            data_folder=data_folder,
            period=period,
            num_scenarios=N_SCENARIOS,
            demand_restriction_cost_per_m3=(
                args.demand_restriction_cost_per_m3
            ),
            min_demand_retention_factor=(
                1.0 - args.max_demand_curtailment_fraction
            ),
        )
        model.check()
        model.run()
        urban[policy_index, :] = recorder_values(model, "reliability_PT1")
        agriculture[policy_index, :] = recorder_values(model, "reliability_Ag")
        cost[policy_index, :] = recorder_values(model, "TotalCost")
        if demand_cost is not None:
            try:
                demand_cost_values = recorder_values(model, "DemandRestrictionCost")
            except KeyError as exc:
                raise RuntimeError(
                    "The selected model does not provide the required "
                    "DemandRestrictionCost recorder"
                ) from exc
            demand_cost[policy_index, :] = demand_cost_values
        del model
        gc.collect()

        completed = policy_index + 1
        print(f"Completed policy {completed}/{len(policies)}", flush=True)
        if completed % args.checkpoint_every == 0 or completed == len(policies):
            save_checkpoint(
                partial_path,
                metadata,
                policies,
                urban,
                agriculture,
                cost,
                demand_cost,
                completed,
            )

    completed_arrays = [urban, agriculture, cost]
    if demand_cost is not None:
        completed_arrays.append(demand_cost)
    if not all(np.all(np.isfinite(values)) for values in completed_arrays):
        raise RuntimeError("Completed simulation contains non-finite objective values")
    os.replace(partial_path, final_path)
    print(f"Complete: {final_path}", flush=True)


if __name__ == "__main__":
    main()
