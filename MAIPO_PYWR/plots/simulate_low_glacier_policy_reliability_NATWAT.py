#!/usr/bin/env python3
"""Cache weekly reliability mechanisms for Sep16 SPI12/SRI12 policies.

Six independent configurations are supported: no policy, the selected SPI12
policy, and the selected SRI12 policy in each of Periods A and B. Each model
run evaluates all 108 OOS scenarios and saves weekly actions, demand, delivery,
failure, and cumulative urban reliability. Borg optimization is never run.
"""

import argparse
import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import plot_cdf_actions_oos_NATWAT as cdf_actions


POLICY_RUN = "sep16_decost"
INDICATORS = ("SPI12", "SRI12")
PERIODS = ("A", "B")
CONFIGURATIONS = tuple(
    (period, policy_type)
    for period in PERIODS
    for policy_type in ("no_policy", "SPI12", "SRI12")
)
N_SCENARIOS = 108
PERIOD_DATES = {
    "A": pd.date_range("2020-03-12", "2040-02-16", freq="7D"),
    "B": pd.date_range("2078-12-22", "2098-11-27", freq="7D"),
}
RECORDER_SPECS = {
    "contract_action": ("parameter", "contract_value"),
    "demand_restriction_factor": ("parameter", "demand_restriction_factor"),
    "unrestricted_demand": ("parameter", "demanda_PT1"),
    "restricted_demand": ("parameter", "demand_max_flow_PT1"),
    "actual_pt1_delivery": ("node", "PT1_output"),
}
SEP16_DEMAND_COST_PER_M3 = 1.0
SEP16_MIN_DEMAND_RETENTION = 0.7


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--task-id", type=int, choices=range(1, len(CONFIGURATIONS) + 1)
    )
    selection.add_argument(
        "--configuration",
        choices=[f"{period}_{kind}" for period, kind in CONFIGURATIONS],
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            script_dir
            / "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=script_dir / "data")
    parser.add_argument("--performance-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model-script",
        type=Path,
        default=(
            script_dir
            / "dps_BORG"
            / "sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py"
        ),
    )
    parser.add_argument(
        "--period-a-data-folder", default="transient_108_no_drought_pert_periodA"
    )
    parser.add_argument(
        "--period-b-data-folder", default="transient_108_no_drought_pert_periodB"
    )
    parser.add_argument("--failure-epsilon", type=float, default=1e-6)
    parser.add_argument("--reliability-atol", type=float, default=1e-10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.failure_epsilon < 0.0 or not np.isfinite(args.failure_epsilon):
        parser.error("--failure-epsilon must be finite and non-negative")
    if args.reliability_atol < 0.0 or not np.isfinite(args.reliability_atol):
        parser.error("--reliability-atol must be finite and non-negative")
    return args


def selected_configuration(args):
    if args.task_id is not None:
        return CONFIGURATIONS[args.task_id - 1]
    period, policy_type = args.configuration.split("_", 1)
    return period, policy_type


def import_model_module(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model script missing: {path}")
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(
        "natwat_sep16_low_glacier_reliability_model", str(path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_model"):
        raise AttributeError(f"{path} does not define make_model()")
    required_parameters = {
        "demand_restriction_cost_per_m3",
        "min_demand_retention_factor",
    }
    available = set(inspect.signature(module.make_model).parameters)
    missing = required_parameters.difference(available)
    if missing:
        raise RuntimeError(
            "The model cannot safely reproduce Sep16 demand settings; "
            f"make_model() lacks: {', '.join(sorted(missing))}"
        )
    return module, path


def selected_policy(performance_cache_dir, period, indicator):
    item = cdf_actions.load_performance_cache(
        performance_cache_dir, indicator, period, POLICY_RUN
    )
    source_index = cdf_actions.selected_policy_index(item, indicator, period)
    policy = item["policies"][source_index].astype(float).copy()
    if policy[3] < SEP16_MIN_DEMAND_RETENTION - 1e-12 or policy[3] > 1.0 + 1e-12:
        raise ValueError(
            f"Selected {period}/{indicator} demand retention {policy[3]} is "
            "incompatible with the Sep16 30% curtailment bound"
        )
    return item, int(source_index), policy


def add_weekly_recorders(model, model_module):
    recorder_names = {}
    for output_name, (kind, object_name) in RECORDER_SPECS.items():
        recorder_name = f"reviewer_reliability_{output_name}"
        if kind == "parameter":
            try:
                parameter = model.parameters[object_name]
            except KeyError as exc:
                raise KeyError(f"Model parameter missing: {object_name}") from exc
            model_module.NumpyArrayParameterRecorder(
                model,
                name=recorder_name,
                param=parameter,
                temporal_agg_func="sum",
                agg_func="SUM",
            )
        else:
            try:
                node = model.nodes[object_name]
            except KeyError as exc:
                raise KeyError(f"Model node missing: {object_name}") from exc
            model_module.NumpyArrayNodeRecorder(
                model, name=recorder_name, node=node, agg_func="SUM"
            )
        recorder_names[output_name] = recorder_name
    return recorder_names


def weekly_recorder_data(model, recorder_name, n_weeks):
    data = np.asarray(model.recorders[recorder_name].data, dtype=float)
    if data.shape == (N_SCENARIOS, n_weeks):
        data = data.T
    if data.shape != (n_weeks, N_SCENARIOS):
        raise ValueError(
            f"Recorder {recorder_name!r} shape {data.shape}; expected "
            f"{(n_weeks, N_SCENARIOS)}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Recorder {recorder_name!r} contains non-finite data")
    return data


def official_reliability(model):
    values = np.asarray(
        model.recorders["reliability_PT1"].values(), dtype=float
    ).reshape(-1)
    if values.shape != (N_SCENARIOS,) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid reliability_PT1 values")
    return values


def build_and_run(
    model_module,
    policy,
    indicator,
    period,
    data_folder,
    failure_epsilon,
    reliability_atol,
):
    num_dp = int(getattr(model_module, "num_DP", 6))
    model = model_module.make_model(
        contract_threshold_vals=float(policy[0]) * np.ones(num_dp),
        contract_action_vals=float(policy[1]) * np.ones(num_dp),
        demand_threshold_vals=[float(policy[2]) * np.ones(12)],
        demand_action_vals=[np.ones(12), float(policy[3]) * np.ones(12)],
        indicator=indicator,
        data_folder=str(data_folder),
        period=period,
        num_scenarios=N_SCENARIOS,
        demand_restriction_cost_per_m3=SEP16_DEMAND_COST_PER_M3,
        min_demand_retention_factor=SEP16_MIN_DEMAND_RETENTION,
    )
    recorder_names = add_weekly_recorders(model, model_module)
    model.check()
    print(
        f"Running {period}/{indicator} across all {N_SCENARIOS} OOS scenarios...",
        flush=True,
    )
    model.run()
    dates = PERIOD_DATES[period]
    weekly = {
        key: weekly_recorder_data(model, name, len(dates))
        for key, name in recorder_names.items()
    }
    official = official_reliability(model)

    over_delivery = np.max(
        weekly["actual_pt1_delivery"] - weekly["restricted_demand"]
    )
    if over_delivery > failure_epsilon:
        raise RuntimeError(
            "PT1 delivery exceeds restricted demand by "
            f"{over_delivery:.12g}, above tolerance {failure_epsilon:g}"
        )
    failure = (
        np.abs(
            weekly["actual_pt1_delivery"] - weekly["restricted_demand"]
        )
        > failure_epsilon
    )
    elapsed = np.arange(1, len(dates) + 1, dtype=float)[:, None]
    cumulative = 1.0 - np.cumsum(failure, axis=0) / elapsed
    if not np.allclose(
        cumulative[-1], official, rtol=0.0, atol=reliability_atol
    ):
        difference = np.max(np.abs(cumulative[-1] - official))
        raise RuntimeError(
            "Final cumulative reliability does not match reliability_PT1; "
            f"maximum difference={difference:.12g}"
        )
    trigger_union = (
        (weekly["contract_action"] > failure_epsilon)
        | (weekly["demand_restriction_factor"] < 1.0 - failure_epsilon)
    )
    weekly["weekly_failure"] = failure.astype(np.int8)
    weekly["cumulative_reliability"] = cumulative
    weekly["trigger_union"] = trigger_union.astype(np.int8)
    return dates, weekly, official


def load_completed_cache(path, period, policy_type, scenario_names, policy):
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "completed",
            "period",
            "policy_type",
            "policy_run",
            "scenario_names",
            "policy",
            "cumulative_reliability",
            "demand_restriction_cost_per_m3",
            "min_demand_retention_factor",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Existing cache {path} is missing {sorted(missing)}")
        if int(np.asarray(saved["completed"]).reshape(-1)[0]) != 1:
            raise ValueError(f"Existing cache is incomplete: {path}")
        identity = (
            str(np.asarray(saved["period"]).reshape(-1)[0]),
            str(np.asarray(saved["policy_type"]).reshape(-1)[0]),
            str(np.asarray(saved["policy_run"]).reshape(-1)[0]).lower(),
        )
        if identity != (period, policy_type, POLICY_RUN):
            raise ValueError(f"Existing cache identity mismatch: {path}")
        if not np.array_equal(saved["scenario_names"].astype(str), scenario_names):
            raise ValueError(f"Existing cache scenario order changed: {path}")
        if not np.allclose(saved["policy"], policy, rtol=0.0, atol=1e-12):
            raise ValueError(f"Existing cache policy changed: {path}")
        saved_demand_cost = float(
            np.asarray(saved["demand_restriction_cost_per_m3"]).reshape(-1)[0]
        )
        saved_retention_floor = float(
            np.asarray(saved["min_demand_retention_factor"]).reshape(-1)[0]
        )
        if not np.isclose(
            saved_demand_cost,
            SEP16_DEMAND_COST_PER_M3,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Existing cache demand cost is not the Sep16 value: {path}"
            )
        if not np.isclose(
            saved_retention_floor,
            SEP16_MIN_DEMAND_RETENTION,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Existing cache retention floor is not the Sep16 value: {path}"
            )
    print(f"Completed cache already exists; skipping: {path}")


def main():
    args = parse_args()
    period, policy_type = selected_configuration(args)
    results_dir = args.results_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    performance_cache_dir = (
        args.performance_cache_dir.expanduser().resolve()
        if args.performance_cache_dir is not None
        else results_dir / "out_of_sample" / "simulation_cache"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir
        / "out_of_sample"
        / "low_glacier_optimal_policy_timeseries"
        / "reviewer_reliability_cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_folder_name = (
        args.period_a_data_folder if period == "A" else args.period_b_data_folder
    )
    data_folder = data_root / data_folder_name
    if not data_folder.is_dir():
        raise FileNotFoundError(f"OOS data folder missing: {data_folder}")

    if policy_type == "no_policy":
        reference, _, _ = selected_policy(
            performance_cache_dir, period, "SPI12"
        )
        indicator = "SPI12"
        source_policy_index = -1
        policy = np.asarray((-4.0, 0.0, -4.0, 1.0), dtype=float)
    else:
        indicator = policy_type
        reference, source_policy_index, policy = selected_policy(
            performance_cache_dir, period, indicator
        )
    scenario_names = reference["scenario_names"].astype(str)
    cdf_actions.validate_data_scenario_order(data_folder, indicator, scenario_names)

    cache_path = output_dir / f"{period}_{policy_type}_weekly_reliability.npz"
    if cache_path.is_file() and not args.overwrite:
        load_completed_cache(
            cache_path, period, policy_type, scenario_names, policy
        )
        return

    model_module, model_path = import_model_module(args.model_script)
    dates, weekly, official = build_and_run(
        model_module,
        policy,
        indicator,
        period,
        data_folder,
        args.failure_epsilon,
        args.reliability_atol,
    )
    if policy_type == "no_policy":
        if np.max(np.abs(weekly["contract_action"])) > args.failure_epsilon:
            raise RuntimeError("No-policy simulation contains contract actions")
        if np.max(
            np.abs(weekly["demand_restriction_factor"] - 1.0)
        ) > args.failure_epsilon:
            raise RuntimeError("No-policy simulation contains demand restrictions")

    np.savez_compressed(
        cache_path,
        completed=np.asarray(1, dtype=np.int8),
        policy_run=np.asarray(POLICY_RUN),
        period=np.asarray(period),
        policy_type=np.asarray(policy_type),
        indicator=np.asarray(indicator),
        source_policy_index=np.asarray(source_policy_index, dtype=np.int64),
        policy=policy,
        scenario_names=scenario_names,
        dates=dates.to_numpy(dtype="datetime64[ns]"),
        official_reliability=official,
        demand_restriction_cost_per_m3=np.asarray(
            SEP16_DEMAND_COST_PER_M3, dtype=float
        ),
        min_demand_retention_factor=np.asarray(
            SEP16_MIN_DEMAND_RETENTION, dtype=float
        ),
        failure_epsilon=np.asarray(args.failure_epsilon, dtype=float),
        model_script=np.asarray(str(model_path)),
        **weekly,
    )
    print(f"Saved completed weekly cache: {cache_path}")


if __name__ == "__main__":
    main()
