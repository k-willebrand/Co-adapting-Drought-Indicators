#!/usr/bin/env python3
"""Reproduce the Sep16 Nature manuscript OOS figures from one entry point.

This driver deliberately separates expensive model simulations from plotting.
It calls the existing, validated Sep16 analysis programs instead of duplicating
their model or objective logic.  Run ``preflight`` first, use the three
``simulate-*`` stages on compute nodes (or arrays), and then use ``plot`` to
build and collect the manuscript figure bundle.  ``all-local`` is provided for
a complete sequential run inside a sufficiently large compute allocation.

Main products
-------------
Figure 2: 108-scenario OOS Pareto frontier and normalized 3-D hypervolume.
Figure 3: action-band time series for the configured OOS scenario in both
          planning periods, using highest-urban-reliability SPI12/SRI12 plans.
Figure 4: action CDFs for (i) each indicator's highest-urban-reliability plan
          and (ii) its highest-urban-reliability plan with mean cost < 5000 M$.
Figure 6: high/low climate-quartile performance and the Period-B SPI/SRI
          highest-urban-winner climate-percentile violin plot.

No optimization is performed by this script.  Fixed trained Sep16 policies are
evaluated with ``sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


POLICY_RUN = "sep16_decost"
RESULTS_FOLDER = (
    "MAIPO_outputs_NATWAT_3obj_2paramGamma_disc0_rerun_Sep16_MIN_DeCost"
)
PERIOD_A_DATA_FOLDER = "transient_108_no_drought_pert_periodA"
PERIOD_B_DATA_FOLDER = "transient_108_no_drought_pert_periodB"
FIGURE3_SCENARIO = "ACCESS-CM2_ssp585_scen03"
CDF_COST_CAP_MILLION = 5000.0
FIGURE3_CONFIGURATIONS = ("A_SPI12", "A_SRI12", "B_SPI12", "B_SRI12")
OOS_TASKS = tuple(range(1, 13))
ACTION_TASKS = tuple(range(1, 13))

REQUIRED_SCRIPTS = (
    "simulate_out_of_sample_policies_NATWAT.py",
    "plot_out_of_sample_pareto_frontier_NATWAT.py",
    "plot_hypervolume_NATWAT.py",
    "simulate_low_glacier_policy_reliability_NATWAT.py",
    "plot_periodB_low_glacier_policy_actions_reliability_NATWAT.py",
    "plot_cdf_actions_oos_NATWAT.py",
    "plot_figure5_sep16_oos_quartiles_NATWAT.py",
    "plot_scenario_spi_sri_climate_violin_NATWAT.py",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "simulate-oos",
            "simulate-actions",
            "simulate-figure3",
            "plot-figure2",
            "plot-figure3",
            "plot-figure4",
            "plot-figure6",
            "plot",
            "collect",
            "all-local",
        ),
        help="One reproducible workflow stage.",
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--results-dir", type=Path, default=root / RESULTS_FOLDER)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument(
        "--model-script",
        type=Path,
        default=(
            root
            / "dps_BORG"
            / "sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py"
        ),
    )
    parser.add_argument(
        "--manuscript-output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/manuscript_NatureV2.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--task-id", type=int, choices=OOS_TASKS)
    parser.add_argument("--configuration", choices=FIGURE3_CONFIGURATIONS)
    parser.add_argument("--scenario-name", default=FIGURE3_SCENARIO)
    parser.add_argument("--cost-cap-million", type=float, default=CDF_COST_CAP_MILLION)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.cost_cap_million <= 0:
        parser.error("--cost-cap-million must be positive")
    return args


class Workflow:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.project_root.expanduser().resolve()
        self.results = args.results_dir.expanduser().resolve()
        self.data = args.data_root.expanduser().resolve()
        self.model = args.model_script.expanduser().resolve()
        self.bundle = (
            args.manuscript_output_dir.expanduser().resolve()
            if args.manuscript_output_dir is not None
            else self.results / "manuscript_NatureV2"
        )
        self.oos_root = self.results / "out_of_sample"
        self.oos_cache = self.oos_root / "simulation_cache"
        self.cdf_root = self.oos_root / "cdf_actions"
        self.reliability_cache = (
            self.oos_root
            / "low_glacier_optimal_policy_timeseries"
            / "reviewer_reliability_cache"
        )
        self.history = self.bundle / "pipeline_run_history.jsonl"

    def script(self, name: str) -> Path:
        return self.root / name

    def preflight(self) -> None:
        required = [self.script(name) for name in REQUIRED_SCRIPTS]
        required.extend((self.results, self.data))
        simulation_stages = {
            "simulate-oos", "simulate-actions", "simulate-figure3", "all-local"
        }
        if self.args.stage in simulation_stages:
            required.append(self.model)
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Required NatureV2 inputs are missing:\n{formatted}")

        data_stages = {
            "simulate-oos", "simulate-actions", "simulate-figure3",
            "plot-figure3", "plot-figure6", "plot", "all-local",
        }
        if self.args.stage in data_stages:
            for folder in (PERIOD_A_DATA_FOLDER, PERIOD_B_DATA_FOLDER):
                path = self.data / folder
                if not path.is_dir():
                    raise FileNotFoundError(f"Missing 108-scenario OOS folder: {path}")

        if self.args.stage in {"simulate-oos", "all-local"}:
            expected = {
                f"{indicator}_{period}"
                for period in ("A", "B")
                for indicator in ("SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12")
            }
            discovered = {
                prefix
                for path in self.results.rglob("*")
                if path.is_dir()
                for prefix in expected
                if path.name.startswith(prefix)
            }
            if discovered != expected:
                absent = ", ".join(sorted(expected - discovered))
                raise RuntimeError(
                    "Sep16 training archives are incomplete; missing "
                    f"indicator-period folders for: {absent}"
                )
        print("Preflight passed")
        print(f"  project root : {self.root}")
        print(f"  results      : {self.results}")
        print(f"  OOS data     : {self.data}")
        print(f"  model        : {self.model}")
        print(f"  final bundle : {self.bundle}")

    def _record(self, command: list[str], status: str, returncode: int | None) -> None:
        if self.args.dry_run:
            return
        self.bundle.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": self.args.stage,
            "status": status,
            "returncode": returncode,
            "command": command,
        }
        with self.history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def run(self, command: list[str]) -> None:
        printable = " ".join(subprocess.list2cmdline([part]) for part in command)
        print(f"\n$ {printable}", flush=True)
        if self.args.dry_run:
            return
        self._record(command, "started", None)
        try:
            subprocess.run(command, cwd=self.root, check=True)
        except subprocess.CalledProcessError as exc:
            self._record(command, "failed", exc.returncode)
            raise
        self._record(command, "completed", 0)

    def python(self, script: str, *arguments: object) -> list[str]:
        return [self.args.python_executable, str(self.script(script)), *(str(x) for x in arguments)]

    def simulate_oos(self, task_id: int | None = None) -> None:
        tasks = (task_id,) if task_id is not None else OOS_TASKS
        if not self.args.dry_run:
            self.oos_cache.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            command = self.python(
                "simulate_out_of_sample_policies_NATWAT.py",
                "--task-id", task,
                "--policy-run", POLICY_RUN,
                "--results-dir", self.results,
                "--data-root", self.data,
                "--period-a-data-folder", PERIOD_A_DATA_FOLDER,
                "--period-b-data-folder", PERIOD_B_DATA_FOLDER,
                "--output-dir", self.oos_cache,
                "--model-script", self.model,
                "--checkpoint-every", self.args.checkpoint_every,
                "--expected-seeds", 7,
                "--require-demand-restriction-cost",
                "--demand-restriction-cost-per-m3", 1.0,
                "--max-demand-curtailment-fraction", 0.3,
            )
            if self.args.overwrite:
                command.append("--overwrite")
            self.run(command)

    def simulate_actions(self, task_id: int | None = None) -> None:
        tasks = (task_id,) if task_id is not None else ACTION_TASKS
        for task in tasks:
            command = self.python(
                "plot_cdf_actions_oos_NATWAT.py",
                "--policy-run", POLICY_RUN,
                "--results-dir", self.results,
                "--performance-cache-dir", self.oos_cache,
                "--output-dir", self.cdf_root,
                "simulate",
                "--task-id", task,
                "--data-root", self.data,
                "--period-a-data-folder", PERIOD_A_DATA_FOLDER,
                "--period-b-data-folder", PERIOD_B_DATA_FOLDER,
                "--model-script", self.model,
                "--checkpoint-every", self.args.checkpoint_every,
            )
            if self.args.overwrite:
                command.append("--overwrite")
            self.run(command)

    def simulate_figure3(self, configuration: str | None = None) -> None:
        configurations = (
            (configuration,) if configuration is not None else FIGURE3_CONFIGURATIONS
        )
        for current in configurations:
            command = self.python(
                "simulate_low_glacier_policy_reliability_NATWAT.py",
                "--configuration", current,
                "--results-dir", self.results,
                "--data-root", self.data,
                "--performance-cache-dir", self.oos_cache,
                "--output-dir", self.reliability_cache,
                "--model-script", self.model,
                "--period-a-data-folder", PERIOD_A_DATA_FOLDER,
                "--period-b-data-folder", PERIOD_B_DATA_FOLDER,
            )
            if self.args.overwrite:
                command.append("--overwrite")
            self.run(command)

    def plot_figure2(self) -> None:
        self.run(self.python(
            "plot_out_of_sample_pareto_frontier_NATWAT.py",
            "--policy-run", POLICY_RUN,
            "--results-dir", self.results,
            "--output-dir", self.oos_root,
            "--dpi", self.args.dpi,
        ))
        hv_dir = self.bundle / "Figure2" / "hypervolume"
        self.run(self.python(
            "plot_hypervolume_NATWAT.py",
            "--policy-run", POLICY_RUN,
            "--results-dir", self.results,
            "--oos-cache-dir", self.oos_cache,
            "--output-dir", hv_dir,
            "--datasets", "oos",
            "--reference-margin", 0.05,
            "--dpi", self.args.dpi,
        ))

    def plot_figure3(self) -> None:
        for period, label in (("A", "2020-2040"), ("B", "2080-2100")):
            output = self.bundle / "Figure3" / label
            self.run(self.python(
                "plot_periodB_low_glacier_policy_actions_reliability_NATWAT.py",
                "--period", period,
                "--results-dir", self.results,
                "--data-root", self.data,
                "--reliability-cache-dir", self.reliability_cache,
                "--output-dir", output,
                "--scenario-name", self.args.scenario_name,
                "--plot-style", "action_bands",
                "--output-formats", "png", "pdf", "svg", "tiff",
                "--flat-output",
                "--dpi", self.args.dpi,
            ))

    def plot_figure4(self) -> None:
        self.run(self.python(
            "plot_cdf_actions_oos_NATWAT.py",
            "--policy-run", POLICY_RUN,
            "--results-dir", self.results,
            "--performance-cache-dir", self.oos_cache,
            "--output-dir", self.cdf_root,
            "plot",
            "--dpi", self.args.dpi,
            "--target-cost-million", self.args.cost_cap_million,
            "--cost-cap-selection", "highest-urban",
            "--figure-set", "manuscript-v2",
        ))

    def plot_figure6(self) -> None:
        quartile_dir = self.bundle / "Figure6" / "quartile_performance"
        violin_dir = self.bundle / "Figure6" / "violin_analysis"
        common = (
            "--results-dir", self.results,
            "--cache-dir", self.oos_cache,
            "--data-root", self.data,
            "--temperature-folder", "knn_weap_transient_no_drought_pert",
            "--precipitation-folder", "knn_weap_transient_no_drought_pert",
            "--temperature-prefix", "temp_transient_",
            "--precipitation-prefix", "precip_transient_",
            "--climate-file-suffix", "_1979_2099",
            "--glacier-file", "CMIP6_Glacier_Melt_108scenarios_no_drought_pert.csv",
            "--period-b-oos-folder", PERIOD_B_DATA_FOLDER,
            "--temperature-column", "LBR_2375",
            "--precipitation-column", "LBR_2375",
            "--dpi", self.args.dpi,
        )
        self.run(self.python(
            "plot_figure5_sep16_oos_quartiles_NATWAT.py",
            *common,
            "--output-dir", quartile_dir,
        ))
        self.run(self.python(
            "plot_scenario_spi_sri_climate_violin_NATWAT.py",
            *common,
            "--output-dir", violin_dir,
            "--reference-margin", 0.1,
            "--analysis", "highest-urban-only",
        ))

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Expected manuscript artifact is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def collect(self) -> None:
        """Collect the precise main-text artifacts and their source tables."""
        if self.args.dry_run:
            print(f"[dry-run] would collect final artifacts in: {self.bundle}")
            return
        cost_tag = (
            str(int(self.args.cost_cap_million))
            if float(self.args.cost_cap_million).is_integer()
            else f"{self.args.cost_cap_million:g}".replace(".", "pt")
        ) + "M"
        artifacts: list[tuple[str, str, Path, Path]] = []

        for suffix in ("png", "pdf", "svg"):
            artifacts.append((
                "Figure2", "OOS cost-urban Pareto frontier",
                self.oos_root / "figures" / f"oos_3obj_pareto_cost_urban.{suffix}",
                self.bundle / "Figure2" / f"oos_3obj_pareto_cost_urban.{suffix}",
            ))
            artifacts.append((
                "Figure2", "OOS normalized hypervolume",
                self.bundle / "Figure2" / "hypervolume" / f"normalized_global_hypervolume_oos.{suffix}",
                self.bundle / "Figure2" / f"normalized_global_hypervolume_oos.{suffix}",
            ))
            artifacts.append((
                "Figure4", "Highest urban reliability action CDF",
                self.cdf_root / f"cdf_actions_highest_urban_reliability_by_indicator.{suffix}",
                self.bundle / "Figure4" / f"cdf_actions_highest_urban_reliability_by_indicator.{suffix}",
            ))
            capped_name = f"cdf_actions_highest_urban_cost_below_{cost_tag}_by_indicator"
            artifacts.append((
                "Figure4", f"Highest urban reliability action CDF below {self.args.cost_cap_million:g} M$",
                self.cdf_root / f"{capped_name}.{suffix}",
                self.bundle / "Figure4" / f"{capped_name}.{suffix}",
            ))
            artifacts.append((
                "Figure6", "High/low climate-quartile performance",
                self.bundle / "Figure6" / "quartile_performance" / f"figure5_sep16_oos_high_low_quartile_performance.{suffix}",
                self.bundle / "Figure6" / f"quartile_performance.{suffix}",
            ))
            artifacts.append((
                "Figure6", "Period-B highest-urban SPI/SRI winner climate violin",
                self.bundle / "Figure6" / "violin_analysis" / f"periodB_climate_percentiles_highest_urban_winner.{suffix}",
                self.bundle / "Figure6" / f"highest_urban_winner_climate_violin.{suffix}",
            ))
        artifacts.append((
            "Figure6", "High/low climate-quartile performance (publication TIFF)",
            self.bundle / "Figure6" / "quartile_performance" / "figure5_sep16_oos_high_low_quartile_performance.tiff",
            self.bundle / "Figure6" / "quartile_performance.tiff",
        ))

        data_artifacts = (
            ("Figure2", "Pareto source table", self.oos_root / "pareto_data" / "out_of_sample_policy_means_and_pareto.csv", self.bundle / "Figure2" / "source_data" / "out_of_sample_policy_means_and_pareto.csv"),
            ("Figure2", "Hypervolume source table", self.bundle / "Figure2" / "hypervolume" / "normalized_global_hypervolume_oos.csv", self.bundle / "Figure2" / "source_data" / "normalized_global_hypervolume_oos.csv"),
            ("Figure4", "Highest-urban policy selection", self.cdf_root / "highest_urban_reliability_policies_by_indicator.csv", self.bundle / "Figure4" / "source_data" / "highest_urban_reliability_policies_by_indicator.csv"),
            ("Figure4", "Cost-capped highest-urban policy selection", self.cdf_root / f"highest_urban_cost_below_{cost_tag}_policies_by_indicator.csv", self.bundle / "Figure4" / "source_data" / f"highest_urban_cost_below_{cost_tag}_policies_by_indicator.csv"),
            ("Figure4", "CDF method", self.cdf_root / "cdf_actions_method.txt", self.bundle / "Figure4" / "source_data" / "cdf_actions_method.txt"),
            ("Figure6", "Quartile performance source table", self.bundle / "Figure6" / "quartile_performance" / "figure5_sep16_oos_quartile_performance.csv", self.bundle / "Figure6" / "source_data" / "quartile_performance.csv"),
            ("Figure6", "Highest-urban winner classification", self.bundle / "Figure6" / "violin_analysis" / "highest_urban_scenario_classification.csv", self.bundle / "Figure6" / "source_data" / "highest_urban_scenario_classification.csv"),
            ("Figure6", "Quartile classification method", self.bundle / "Figure6" / "quartile_performance" / "figure5_sep16_oos_method.txt", self.bundle / "Figure6" / "source_data" / "quartile_method.txt"),
            ("Figure6", "Violin classification method", self.bundle / "Figure6" / "violin_analysis" / "classification_method.txt", self.bundle / "Figure6" / "source_data" / "violin_method.txt"),
        )
        artifacts.extend(data_artifacts)

        manifest_rows = []
        for figure, role, source, destination in artifacts:
            self._copy(source, destination)
            manifest_rows.append({
                "figure": figure,
                "role": role,
                "file": str(destination.relative_to(self.bundle)),
                "source": str(source),
            })

        figure3_files = []
        for label in ("2020-2040", "2080-2100"):
            period_dir = self.bundle / "Figure3" / label
            period_files = [path for path in period_dir.rglob("*") if path.is_file()]
            if not any(path.suffix == ".png" for path in period_files):
                raise FileNotFoundError(
                    f"No Figure 3 PNG was found for {label} under {period_dir}"
                )
            figure3_files.extend(period_files)
        for path in sorted(figure3_files):
            manifest_rows.append({
                "figure": "Figure3",
                "role": "Time-series figure or source data",
                "file": str(path.relative_to(self.bundle)),
                "source": str(path),
            })

        config = {
            "policy_run": POLICY_RUN,
            "trained_policy_results": str(self.results),
            "simulation_model": str(self.model),
            "oos_scenarios": 108,
            "period_A_data_folder": PERIOD_A_DATA_FOLDER,
            "period_B_data_folder": PERIOD_B_DATA_FOLDER,
            "figure3_scenario": self.args.scenario_name,
            "figure3_policy_rule": "highest OOS mean urban reliability within SPI12 or SRI12 Period-specific Pareto set",
            "figure4_unconstrained_rule": "highest OOS mean urban reliability within each indicator-period Pareto set",
            "figure4_cost_cap_million_USD": self.args.cost_cap_million,
            "figure4_cost_cap_rule": "highest OOS mean urban reliability among Pareto policies with mean total cost strictly below cap",
            "figure6_quartiles": "Q1 ranks 1-27 and Q4 ranks 82-108 using final-20-year climate metrics, scenario-name matched",
            "figure6_violin": "Period-B strict SPI-versus-SRI highest-urban-reliability winner; exact ties excluded",
            "dpi": self.args.dpi,
        }
        self.bundle.mkdir(parents=True, exist_ok=True)
        (self.bundle / "pipeline_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        with (self.bundle / "artifact_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("figure", "role", "file", "source"))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Collected {len(manifest_rows)} artifacts in: {self.bundle}")

    def plot_all(self) -> None:
        self.plot_figure2()
        self.plot_figure3()
        self.plot_figure4()
        self.plot_figure6()
        self.collect()


def main() -> None:
    args = parse_args()
    workflow = Workflow(args)
    workflow.preflight()
    if args.stage == "preflight":
        return
    if args.stage == "simulate-oos":
        workflow.simulate_oos(args.task_id)
    elif args.stage == "simulate-actions":
        workflow.simulate_actions(args.task_id)
    elif args.stage == "simulate-figure3":
        workflow.simulate_figure3(args.configuration)
    elif args.stage == "plot-figure2":
        workflow.plot_figure2()
    elif args.stage == "plot-figure3":
        workflow.plot_figure3()
    elif args.stage == "plot-figure4":
        workflow.plot_figure4()
    elif args.stage == "plot-figure6":
        workflow.plot_figure6()
    elif args.stage == "plot":
        workflow.plot_all()
    elif args.stage == "collect":
        workflow.collect()
    elif args.stage == "all-local":
        workflow.simulate_oos()
        workflow.simulate_actions()
        workflow.simulate_figure3()
        workflow.plot_all()
    else:  # pragma: no cover - guarded by argparse
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
