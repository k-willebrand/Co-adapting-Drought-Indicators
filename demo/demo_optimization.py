"""
Maipo Basin Demo Optimization

This script runs a small SERIAL (non-MPI) Borg optimization using the demo dataset.
It is intended to be run on a local computer without MPI/parallel setup.

- Data: demo/data_demo
- Indicator: SPI3
- Scenarios: 10
- Optimization: 2 seeds, 100 evaluations

IMPORTANT:
The make_model() function is the primary function used to construct the Pywr model.
To improve readability of this demo script, it is imported from:
    make_model_function.py

Outputs are saved to:
  Co-adapting-Drought-Indicators/demo/outputs_demo/{data_folder}_{indicator}_2020-2040/
"""


import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from make_model_function import make_model

# Add Borg Python plugin path.
# Repository structure:
# Co-adapting-Drought-Indicators/
# ├── demo/
# └── MAIPO_PYWR/
#
# This script assumes it lives inside:
#   Co-adapting-Drought-Indicators/demo/

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = REPO_ROOT / "demo"
MAIPO_ROOT = REPO_ROOT / "MAIPO_PYWR"
BORG_PLUGIN_PATH = MAIPO_ROOT / "dps_BORG" / "BorgMOEA_master" / "plugins" / "Python"
sys.path.append(str(BORG_PLUGIN_PATH))

try:
    from MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg import Borg
except ImportError:
    from dps_BORG.BorgMOEA_master.plugins.Python.borg import Borg


# -----------------------------
# Demo configuration
# -----------------------------
BASE_DATA_DIR = DEMO_ROOT

CONFIG = {
    "base_data_dir": BASE_DATA_DIR,
    "data_folder": "data_demo",
    "indicator": "SPI3",
    "num_dp": 6,
    "num_scenarios": 10,
    "num_seeds": 2,
    "num_func_evals": 100,
    "runtime_freq": 100,
    "decision_var_range": [[-4, 0], [0, 2100], [-4, 0], [0, 1]],
    "epsilon_list": [0.01, 0.01, 0.05e9],
    "time_label": "2020-2040",
}


def evaluate_policy(contract_threshold_val, contract_action_val, demand_threshold_val, demand_action_val):
    """Run the Pywr model once and return objective values for Borg."""

    contract_threshold_vals = contract_threshold_val * np.ones(CONFIG["num_dp"])
    contract_action_vals = contract_action_val * np.ones(CONFIG["num_dp"])
    demand_threshold_vals = [demand_threshold_val * np.ones(12)]
    demand_action_vals = [np.ones(12), demand_action_val * np.ones(12)]

    model = make_model(
        contract_threshold_vals=contract_threshold_vals,
        contract_action_vals=contract_action_vals,
        demand_threshold_vals=demand_threshold_vals,
        demand_action_vals=demand_action_vals,
        indicator=CONFIG["indicator"],
        data_folder=CONFIG["data_folder"],
    )

    model.check()
    model.check_graph()
    model.find_orphaned_parameters()
    model.run()

    def get_performance(recorder_name):
        recorder = model.recorders[recorder_name]
        values = recorder.values()

        if recorder.agg_func == "mean":
            return np.mean(values)
        if recorder.agg_func == "max":
            return np.max(values)
        if recorder.agg_func == "min":
            return np.min(values)

        # Safe fallback for recorders without a recognized agg_func string.
        return np.mean(values)

    # Borg minimizes by default in this setup, so cost is negated to retain
    # the same convention used in the original script.
    return [
        get_performance("reliability_PT1"),
        get_performance("reliability_Ag"),
        -get_performance("TotalCost"),
    ]


def run_optimization():
    """Run the serial demo optimization and write result files."""

    data_folder = CONFIG["data_folder"]
    indicator = CONFIG["indicator"]
    label = f"{data_folder}_{indicator}_{CONFIG['time_label']}"

    output_dir = DEMO_ROOT / "outputs_demo" / label
    sets_dir = output_dir / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    num_dec_vars = 4
    num_objs = 3
    num_constraints = 0

    for seed in range(CONFIG["num_seeds"]):
        borg = Borg(num_dec_vars, num_objs, num_constraints, evaluate_policy)
        borg.setBounds(*CONFIG["decision_var_range"])
        borg.setEpsilons(*CONFIG["epsilon_list"])

        runtime_file = output_dir / f"{data_folder}_runtime_seed_{seed + 1}.runtime"

        result = borg.solve({
            "maxEvaluations": CONFIG["num_func_evals"],
            "runtimeformat": "borg",
            "frequency": CONFIG["runtime_freq"],
            "runtimefile": str(runtime_file),
        })

        if result:
            result.display()

            full_results_file = sets_dir / f"{data_folder}_Borg_results_seed_{seed + 1}.set"
            objectives_file = sets_dir / f"{data_folder}_objectives_seed_{seed + 1}.set"

            with open(full_results_file, "w") as f:
                f.write("# Borg Optimization Results\n")
                f.write(
                    f"# First {num_dec_vars} values are decision variables; "
                    f"last {num_objs} values are objective values.\n"
                )

                for solution in result:
                    values = [
                        *solution.getVariables(),
                        *solution.getObjectives(),
                    ]
                    f.write(" ".join(str(value) for value in values) + "\n")

                f.write("#")

            with open(objectives_file, "w") as f:
                for solution in result:
                    f.write(" ".join(str(value) for value in solution.getObjectives()) + "\n")
                f.write("#")

            print(f"Seed {seed + 1} complete")
            print(f"Saved full results to: {full_results_file}")
            print(f"Saved objectives to: {objectives_file}")


if __name__ == "__main__":
    print("Running Maipo Basin demo optimization in serial mode...")
    print(f"Dataset: {CONFIG['data_folder']}")
    print(f"Indicator: {CONFIG['indicator']}")
    print(f"Function evaluations per seed: {CONFIG['num_func_evals']}")
    print(f"Seeds: {CONFIG['num_seeds']}")
    run_optimization()
