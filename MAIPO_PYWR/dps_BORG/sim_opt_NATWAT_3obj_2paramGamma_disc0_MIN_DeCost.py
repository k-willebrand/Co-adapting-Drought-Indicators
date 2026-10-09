"""
sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN.py

3-objective simulation-optimization of drought management policy for the
Maipo Basin using the TDP_scenarios_DS_2paramGamma data folder.

Converted from example_sim_opt_20yrA_SPI12_nFunc2000.py to be suitable
for SLURM array job submission. Key changes from the original:

  - indicator is now a command-line argument (no longer hardcoded as SPI12)
  - data_folder defaults to TDP_scenarios_DS_2paramGamma (overridable via CLI)
  - data_root, output_root, nfunc, seed are command-line arguments
  - one seed per array task (--seed flag); no more for-loop over seeds
  - MPI start/stop handled once per task, not inside a seed loop
  - output directory name is auto-constructed from indicator + nfunc + eps

Objectives (3, signs as returned to Borg):
  1. -reliability_PT1  (minimize; equivalent to maximizing reliability)
  2. -reliability_Ag   (minimize; equivalent to maximizing reliability)
  3. TotalCost         (minimize; includes a configurable demand-restriction
                        cost, default $1/m3)

Decision variables (4):
  1. contract_threshold  [-4, 0]
  2. contract_action     [0, 2100]
  3. demand_threshold    [-4, 0]
  4. demand_action       [1-max_curtailment, 1]
                         (default [0.7, 1], curtail at most 30%)

Epsilons (3):  [0.01, 0.01, 0.05e9]  (default, overridable via --epsilon)

Usage (SLURM array job, one seed per task):
  mpirun -np $SLURM_NTASKS python3 sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py \\
      SRI3 A \\
      --nfunc       10000 \\
      --seed        $SEED_IDX \\
      --data_folder TDP_scenarios_DS_2paramGamma \\
      --data_root   /home/groups/smfletch/keaniw/MAIPO_PYWR/data \\
      --output_root /home/groups/smfletch/keaniw/MAIPO_PYWR/MAIPO_outputs
"""

# general package imports
import numpy as np
import pandas as pd
import platform  # helps identify directory locations on different types of OS
import sys
import os

# try importing mpi4py library
#import mpi4py
#import mpich

# Resolve paths from this file so the group-folder installation is portable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
sys.path.append(PROJECT_PARENT)
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(
    PROJECT_ROOT, 'dps_BORG', 'BorgMOEA_master', 'plugins', 'Python'
))
import MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg as bg  # full package path avoids picking up wrong borg.py

# pywr imports
from pywr.core import *
from pywr.parameters import *
from pywr.parameters._thresholds import StorageThresholdParameter, ParameterThresholdParameter
from pywr.recorders import DeficitFrequencyNodeRecorder, TotalDeficitNodeRecorder, MeanFlowNodeRecorder, \
    NumpyArrayParameterRecorder, NumpyArrayStorageRecorder, NumpyArrayNodeRecorder, RollingMeanFlowNodeRecorder, \
    AggregatedRecorder
# from MAIPO_searcher import *
from pywr.dataframe_tools import *


#from MAIPO_PYWR.MAIPO_parameters import *
try:
    from MAIPO_PYWR.MAIPO_parameters_DS_V2 import *
except:
    from MAIPO_parameters_DS_V2 import *
# from MAIPO_PYWR.dps_BORG.MAIPO_DPS import *

import importlib
import argparse
# BorgMOEA = __import__("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\dps_BORG\\BorgMOEA_master")
# BorgMOEA = importlib.import_module("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\dps_BORG\\BorgMOEA_master")
# from MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg import borg as bg  # Import borg wrapper


num_k = 1  # number of levels in policy tree
num_DP = 6  # number of decision periods

# Urban demand-restriction valuation. Model demand is expressed as m3/s and
# every model timestep is one week, so multiplying by 604800 converts the
# curtailed weekly flow to m3. The resulting recorder is therefore in dollars.
SECONDS_PER_WEEK = 7 * 24 * 60 * 60
DEMAND_RESTRICTION_COST_PER_M3 = 1.0
MIN_DEMAND_RETENTION_FACTOR = 0.7


# ---------------------------------------------------------------------------
# Global state set by Optimization() before Borg starts.
# Simulation_Caller reads these so the closure captures the correct values.
# ---------------------------------------------------------------------------
_INDICATOR   = None   # e.g. "SRI3" -- set from CLI arg
_DATA_FOLDER = "TDP_scenarios_DS_2paramGamma"  # default; overridden by --data_folder
_DATA_ROOT   = None   # None = derive path from project root
_PERIOD      = "A"    # "A" or "B"; set from CLI arg

def make_model(contract_threshold_vals=-999999 * np.ones(num_DP), contract_action_vals=np.zeros(num_DP),
               demand_threshold_vals=[], demand_action_vals=[np.ones(12)], indicator="SRI3",
               drought_status_agg="drought_status_single_week",
               data_folder="TDP_scenarios_DS_2paramGamma",
               period="A", num_scenarios=36,
               demand_restriction_cost_per_m3=None,
               min_demand_retention_factor=None):
    '''
    Purpose: Creates a Pywr model with the specified number and values for policy thresholds/actions. Intended for use with MOEAs.

    Args:
        threshold_vals: an array of policy thresholds for drought index
        action_vals: an array of policy actions corresponding to policy thresholds
        num_scenarios: number of climate-scenario columns in the input files.
            Defaults to 36 for the original optimization; out-of-sample
            evaluation passes 108 explicitly.

    Returns:
        model: a Pywr Model object
    '''

    if demand_restriction_cost_per_m3 is None:
        demand_restriction_cost_per_m3 = DEMAND_RESTRICTION_COST_PER_M3
    if min_demand_retention_factor is None:
        min_demand_retention_factor = MIN_DEMAND_RETENTION_FACTOR
    demand_restriction_cost_per_m3 = float(demand_restriction_cost_per_m3)
    min_demand_retention_factor = float(min_demand_retention_factor)
    if (
        not np.isfinite(demand_restriction_cost_per_m3)
        or demand_restriction_cost_per_m3 < 0.0
    ):
        raise ValueError(
            "demand_restriction_cost_per_m3 must be finite and non-negative"
        )
    if (
        not np.isfinite(min_demand_retention_factor)
        or min_demand_retention_factor < 0.0
        or min_demand_retention_factor > 1.0
    ):
        raise ValueError(
            "min_demand_retention_factor must be between 0 and 1"
        )

    # demand_action is the fraction of unrestricted urban demand retained
    # after a restriction is triggered. Validate it here as well as in the
    # Borg bounds so direct make_model/OOS calls cannot exceed the configured
    # maximum weekly curtailment.
    for profile_index, profile in enumerate(demand_action_vals):
        profile_values = np.asarray(profile, dtype=float)
        if profile_values.size == 0:
            raise ValueError(f"demand_action_vals[{profile_index}] is empty")
        if not np.all(np.isfinite(profile_values)):
            raise ValueError(
                f"demand_action_vals[{profile_index}] contains non-finite values"
            )
        if np.any(profile_values < min_demand_retention_factor) or np.any(
            profile_values > 1.0
        ):
            raise ValueError(
                "Every demand action must retain between "
                f"{100.0 * min_demand_retention_factor:g}% and 100% of "
                "unrestricted demand (weekly curtailment cannot exceed "
                f"{100.0 * (1.0 - min_demand_retention_factor):g}%); "
                f"profile {profile_index} ranges from {profile_values.min()} "
                f"to {profile_values.max()}"
            )

    # set current working directory
    # os.chdir(os.path.abspath(os.path.dirname(__file__)))

    # create a Pywr model (including an empty network)
    model = Model()

    # create a dictionary object to keep track of key parameters and nodes
    paramIndex = {}
    recorderIndex = {}

    # METADATA [UPDATE THIS!!!]
    model.metadata = {
        "title": "Maipo Basin Model",
        "description": "Simulation-only AGU schematic of the model in JSON format for simulated flow used in the WEAP. 15 climate change scenarios between 2020 and 2050. KW",
        "minimum_version": "0.1"
    }

    # TIME STEPPER
    # Time window depends on period argument:
    #   "A" = 2020-2040 (pre-infrastructure baseline window)
    #   "B" = 2078-2098 (post-infrastructure equilibrium window)
    if period == "B":
        _start, _end = '2078-12-22', '2098-11-27'
    else:  # period == "A" (default)
        _start, _end = '2020-03-12', '2040-02-16'
    model.timestepper = Timestepper(
        start=pd.to_datetime(_start),
        end=pd.to_datetime(_end),
        delta=datetime.timedelta(7)
    )

    # SCENARIOS
    if not isinstance(num_scenarios, (int, np.integer)) or num_scenarios < 1:
        raise ValueError("num_scenarios must be a positive integer")
    num_scenarios = int(num_scenarios)
    Scenario(model, name="climate change", size=num_scenarios)

    # REQUIRED NODES FOR PARAMETERS
    # DP_index -- which development period we're in
    if period == "B":
        datestr = ["2075-01-03", "2079-12-28", "2084-12-21", "2089-12-15", "2094-12-09", "2099-12-03"]
    else:  # period == "A"
        datestr = ["2020-03-12", "2025-03-06", "2030-02-28", "2035-02-22", "2040-02-16", "2045-02-09"]
    FakeYearIndexParameter(
        model,
        name="DP_index",
        dates=[datetime.datetime.strptime(i, '%Y-%m-%d') for i in datestr],
        comment="convert a specific date to integer, from 0 to 16, depending on the 5-year development plan period 2020-2098"
    )
    paramIndex["DP_index"] = model.parameters.__len__() - 1

    # Embalse
    Embalse = Storage(
        model,
        name="Embalse",
        min_volume=15,
        max_volume=220,
        initial_volume=220,
        cost=-1000
    )

    # Maipo_capacity
    ConstantParameter(
        model,
        name="Maipo_capacity",
        value=0,
        is_variable=False,
        lower_bounds=0,
        upper_bounds=300
    )
    paramIndex['Maipo_capacity'] = model.parameters.__len__() - 1

    # Maipo_current_capacity
    ConstantParameter(
        model,
        name="Maipo_current_capacity",
        value=0
    )
    paramIndex['Maipo_current_capacity'] = model.parameters.__len__() - 1

    # Maipo_construction_dp
    ConstantParameter(
        model,
        name="Maipo_construction_dp",
        value=17,
        is_variable=False,
        lower_bounds=2,
        upper_bounds=17,
        comment="Choose any dp from 2025 to 2099. 17 means never constructed"
    )
    paramIndex['Maipo_construction_dp'] = model.parameters.__len__() - 1

    # Maipo_constructed
    ParameterThresholdParameter(
        model,
        param=model.parameters["DP_index"],
        threshold=model.parameters["Maipo_construction_dp"],
        predicate="GE",  # JSON: ">="
        name="Maipo_constructed",
        comment="indicates if the reservoir is active in a specific DP period"
    )
    paramIndex['Maipo_constructed'] = model.parameters.__len__() - 1

    # Maipo_max_volume
    IndexedArrayParameter(
        model,
        index_parameter=model.parameters["Maipo_constructed"],
        params=[
            model.parameters["Maipo_current_capacity"],
            model.parameters["Maipo_capacity"]
        ],
        name="Maipo_max_volume"
    )
    paramIndex['Maipo_max_volume'] = model.parameters.__len__() - 1

    # Embalse_Maipo
    Embalse_Maipo = Storage(
        model,
        name="Embalse Maipo",
        min_volume=0,
        max_volume=model.parameters["Maipo_max_volume"],  # Was 240
        initial_volume=0.0,
        initial_volume_pc=0.0,
        cost=-800
    )

    # requisito_embalse_Maipo
    StorageThresholdParameter(
        model,
        storage=Embalse_Maipo,
        threshold=140,
        predicate="LT",
        values=[1, 0],
        name="requisito_embalse_Maipo"
    )
    paramIndex['requisito_embalse_Maipo'] = model.parameters.__len__() - 1

    # flujo_excedentes_Yeso
    StorageThresholdParameter(
        model,
        storage=Embalse_Maipo,
        threshold=220,
        predicate="LT",
        values=[60.48, 0],
        name="flujo_excedentes_Yeso"
    )
    paramIndex['flujo_excedentes_Yeso'] = model.parameters.__len__() - 1

    # El Manzano
    El_Manzano = River(
        model,
        name="El Manzano"
    )

    # PARAMETERS
    # multiple usable drought_status parameters:
    # drought_status_single_week (uses just the first week of april/october)
    df = {
        'url': os.path.join(data_folder, '{}.csv'.format(indicator)),  # 'data/SRI6.csv'
        "parse_dates": True,
        "index_col": "Timestamp",
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        name="drought_status_single_week",
        scenario=model.scenarios.scenarios[0],
    )
    paramIndex["drought_status_single_week"] = model.parameters.__len__() - 1
    model.parameters._objects[
        paramIndex["drought_status_single_week"]].name = "drought_status_single_week"  # add name to parameter

    # Below: template for more general aggregator functions

    # df = {
    #     'url': './MAIPO_PYWR/data/{}.csv'.format(indicator),  # 'data/SRI6.csv'
    #     "parse_dates": True,
    #     "index_col": "Timestamp",
    #     "dayfirst": True}
    # DroughtStatusAggregationParameter(
    #     model,
    #     dataframe=read_dataframe(model, df),
    #     name="drought_status_single_week_using_agg",
    #     agg_func=lambda x: x[len(x) - 1],
    #     num_weeks=1,
    #     scenario=model.scenarios.scenarios[0]
    # )
    # paramIndex["drought_status_single_week_using_agg"] = model.parameters.__len__() - 1
    # model.parameters._objects[paramIndex[
    #     "drought_status_single_week_using_agg"]].name = "drought_status_single_week_using_agg"  # add name to parameter

    # april_threshold
    april_thresholds = []
    for i, k in enumerate(contract_threshold_vals):
        april_thresholds.append(
            ConstantParameter(model, name=f"april_threshold{i}", value=k, is_variable=False, upper_bounds=0))
        paramIndex[f"april_threshold{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="april_threshold",
        index_parameter=model.parameters["DP_index"],  # DP_index
        params=april_thresholds,
        comment="variable parameter that set the drought threshold for contracts in april"
    )
    paramIndex["april_threshold"] = model.parameters.__len__() - 1

    # october_threshold
    october_thresholds = []
    for i, k in enumerate(contract_threshold_vals):
        october_thresholds.append(
            ConstantParameter(model, name=f"october_threshold{i}", value=k, is_variable=False, upper_bounds=0))
        paramIndex[f"october_threshold{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="october_threshold",
        index_parameter=model.parameters["DP_index"],  # DP_index
        params=october_thresholds,
        comment="variable parameter that set the drought threshold for contracts in october"
    )
    paramIndex["october_threshold"] = model.parameters.__len__() - 1

    # april_contract
    april_contracts = []
    for i, k in enumerate(contract_action_vals):
        april_contracts.append(
            ConstantParameter(model, name=f"april_contract{i}", value=k, is_variable=False, upper_bounds=1500))
        paramIndex[f"april_contract{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="april_contract",
        index_parameter=model.parameters["DP_index"],  # DP_index
        params=april_contracts,
        comment="variable parameter that set the contract shares in april for a determined dp"
    )
    paramIndex["april_contract"] = model.parameters.__len__() - 1

    # october_contract
    october_contracts = []
    for i, k in enumerate(contract_action_vals):
        october_contracts.append(
            ConstantParameter(model, name=f"october_contract{i}", value=k, is_variable=False, upper_bounds=1500))
        paramIndex[f"april_contract{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="october_contract",
        index_parameter=model.parameters["DP_index"],  # DP_index parameter
        params=october_contracts,
        comment="variable parameter that set the contract shares in october for a determined dp"
    )
    paramIndex["october_contract"] = model.parameters.__len__() - 1

    # contract_value
    PolicyTreeTriggerHardCoded(
        model,
        name="contract_value",
        thresholds={
            1: model.parameters["april_threshold"],  # april_threshold parameter
            27: model.parameters["october_threshold"]  # october_threshold parameter
        },
        contracts={
            1: model.parameters["april_contract"],  # april_threshold parameter
            27: model.parameters["october_contract"]  # october_threshold parameter
        },
        drought_status=model.parameters[drought_status_agg],  # drought_status parameter
        comment="Receive two dates where the drought status is evaluated, the contract and the reservoir evaluated, and gives back the amount of shares transferred in that specific week"
    )
    paramIndex["contract_value"] = model.parameters.__len__() - 1

    # purchases_value
    purchases = []
    for i in range(len(contract_action_vals)):
        purchases.append(ConstantParameter(model, name=f"purchase{i}", value=0, is_variable=False, upper_bounds=813))
        paramIndex[f"purchase{i}"] = model.parameters.__len__() - 1
    AccumulatedIndexedArrayParameter(
        model,
        name="purchases_value",
        index_parameter=model.parameters["DP_index"],  # DP_index parameter
        params=purchases,
        comment="parameter that set the shares bought at a determined dp, accumulating past purchases"
    )
    paramIndex["purchases_value"] = model.parameters.__len__() - 1

    demand_control_curves = []
    for i in range(len(demand_threshold_vals)):
        # Assume we only pass in monthly profiles
        demand_control_curves.append(
            MonthlyProfileParameter(
                model, name=f"demand_control_curve{i}", values=demand_threshold_vals[i]
            )
        )
        paramIndex[f"demand_control_curve{i}"] = model.parameters.__len__() - 1

    # demand restriction level (done with indicators)
    IndicatorControlCurveIndexParameter(
        model,
        name="demand_restriction_level",
        indicator=model.parameters["drought_status_single_week"],
        control_curves=demand_control_curves
    )
    paramIndex["demand_restriction_level"] = model.parameters.__len__() - 1

    monthly_demand_restrictions = []
    for i in range(len(demand_action_vals)):
        # Assume we only pass in monthly profiles
        monthly_demand_restrictions.append(
            MonthlyProfileParameter(
                model, name=f"monthly_demand_restriction{i}", values=demand_action_vals[i]
            )
        )
        paramIndex[f"monthly_demand_restriction{i}"] = model.parameters.__len__() - 1

    # Demand restriction factor
    IndexedArrayParameter(
        model,
        name="demand_restriction_factor",
        index_parameter=model.parameters["demand_restriction_level"],
        params=monthly_demand_restrictions
    )
    paramIndex["demand_restriction_factor"] = model.parameters.__len__() - 1

    # flow_Yeso
    df = {
        'url': os.path.join(data_folder, 'YESO.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Yeso"
    )
    paramIndex['flow_Yeso'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Yeso']].name = 'flow_Yeso'  # add name to parameter

    # flow_Maipo
    df = {
        'url': os.path.join(data_folder, 'MAIPO.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Maipo"
    )
    paramIndex['flow_Maipo'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Maipo']].name = 'flow_Maipo'  # add name to parameter

    # flow_Colorado
    df = {
        'url': os.path.join(data_folder, 'COLORADO.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Colorado"
    )
    paramIndex['flow_Colorado'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Colorado']].name = 'flow_Colorado'  # add name to parameter

    # flow_Volcan
    df = {
        'url': os.path.join(data_folder, 'VOLCAN.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Volcan"
    )
    paramIndex['flow_Volcan'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Volcan']].name = 'flow_Volcan'  # add name to parameter

    # flow_Laguna Negra
    df = {
        'url': os.path.join(data_folder, 'LAGUNANEGRA.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Laguna Negra"
    )
    paramIndex['flow_Laguna Negra'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Laguna Negra']].name = 'flow_Laguna Negra'  # add name to parameter

    # flow_Maipo extra
    df = {
        'url': os.path.join(data_folder, 'MAIPOEXTRA.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Maipo extra"
    )
    paramIndex['flow_Maipo extra'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Maipo extra']].name = 'flow_Maipo extra'  # add name to parameter

    # aux_acueductoln
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Acueducto Laguna Negra']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_acueductoln"
    )
    paramIndex['aux_acueductoln'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_acueductoln']].name = 'aux_acueductoln'  # add name to parameter

    # aux_extraccionln
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Extraccion Laguna Negra']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_extraccionln"
    )
    paramIndex['aux_extraccionln'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_extraccionln']].name = 'aux_extraccionln'  # add name to parameter

    # aux_acueductoyeso
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Acueducto El Yeso']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_acueductoyeso"
    )
    paramIndex['aux_acueductoyeso'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_acueductoyeso']].name = 'aux_acueductoyeso'  # add name to parameter

    # aux_filtraciones
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Filtraciones']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_filtraciones"
    )
    paramIndex['aux_filtraciones'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_filtraciones']].name = 'aux_filtraciones'  # add name to parameter

    # threshold_laobra
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Threshold']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="threshold_laobra"
    )
    paramIndex['threshold_laobra'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['threshold_laobra']].name = 'threshold_laobra'  # add name to parameter

    # discount_rate_factor
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Discount rate factor']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="discount_rate_factor"
    )
    paramIndex['discount_rate_factor'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['discount_rate_factor']].name = 'discount_rate_factor'  # add name to parameter

    # descarga_adicional
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'AdicionalEmbalse']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="descarga_adicional"
    )
    paramIndex['descarga_adicional'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['descarga_adicional']].name = 'descarga_adicional'  # add name to parameter

    # estacionalidad_distribucion
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Estacionalidad']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="estacionalidad_distribucion"
    )
    paramIndex['estacionalidad_distribucion'] = model.parameters.__len__() - 1
    model.parameters._objects[
        paramIndex['estacionalidad_distribucion']].name = 'estacionalidad_distribucion'  # add name to parameter

    # demanda_PT1
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'PT1']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="demanda_PT1"
    )
    paramIndex['demanda_PT1'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['demanda_PT1']].name = 'demanda_PT1'  # add name to parameter

    # Restricted demand through PT1
    AggregatedParameter(
        model,
        name="demand_max_flow_PT1",
        parameters=[
            model.parameters['demanda_PT1'],
            model.parameters['demand_restriction_factor']
        ],
        agg_func="product"
    )
    paramIndex["demand_max_flow_PT1"] = model.parameters.__len__() - 1

    # Weekly economic cost of intentionally curtailed urban demand:
    # max(unrestricted demand - policy-restricted demand, 0) [m3/s]
    # * 604800 [s/week] * configured unit cost [$/m3]. This is distinct from
    # an involuntary supply deficit below the already restricted demand.
    AggregatedParameter(
        model,
        name="demand_restriction_cost_per_week",
        parameters=[
            model.parameters["demanda_PT1"],
            model.parameters["demand_max_flow_PT1"]
        ],
        agg_func=lambda values: max(values[0] - values[1], 0.0)
        * SECONDS_PER_WEEK
        * demand_restriction_cost_per_m3
    )
    paramIndex["demand_restriction_cost_per_week"] = model.parameters.__len__() - 1

    # Below: agricultural demand on monthly cycle

    # MonthlyProfileParameter(
    #     model,
    #     name="agricultural_demand",
    #     # values approximated by scaling water rights profile by ratio of demand to rights
    #     values=[18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
    #             17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186]
    # )
    # paramIndex['agricultural_demand'] = model.parameters.__len__() - 1

    # Agricultural demand (doesn't vary much, treated as constant for now)
    ConstantParameter(
        model,
        name="agricultural_demand_constant",
        value=16.978
    )
    paramIndex['agricultural_demand_constant'] = model.parameters.__len__() - 1

    # Below: agricultural water rights on monthly cycle

    # MonthlyProfileParameter(
    #     model,
    #     name="agricultural_water_rights",
    #     # values taken from DGA paper, changing from m3/s to Mm3/week
    #     values=[144.60768, 144.06336, 143.21664, 142.91424, 142.24896, 142.4304,
    #             142.61184, 142.67232, 142.7328, 143.76096, 144.30528, 145.69632]
    # )
    # paramIndex['agricultural_water_rights'] = model.parameters.__len__() - 1

    # Agricultural shares (doesn't vary much, treated as constant for now)
    ConstantParameter(
        model,
        name="agricultural_shares_constant",
        value=3408.639
    )
    paramIndex['agricultural_shares_constant'] = model.parameters.__len__() - 1

    # demanda_PT2
    df = {
        'url': os.path.join(data_folder, 'Extra data.csv'),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'PT2']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="demanda_PT2"
    )
    paramIndex['demanda_PT2'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['demanda_PT2']].name = 'demanda_PT2'  # add name to parameter

    # demanda_PT2_negativa
    NegativeParameter(
        model,
        parameter=model.parameters["demanda_PT2"],
        name="demanda_PT2_negativa"
    )
    paramIndex['demanda_PT2_negativa'] = model.parameters.__len__() - 1

    # requisito_embalse
    StorageThresholdParameter(
        model,
        storage=Embalse,
        threshold=140,
        predicate="LT",
        values=[1, 0],
        name="requisito_embalse"
    )
    paramIndex['requisito_embalse'] = model.parameters.__len__() - 1

    # caudal_naturalizado
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["flow_Maipo"],
            model.parameters["flow_Yeso"],
            model.parameters["flow_Colorado"],
            model.parameters["flow_Laguna Negra"],
            model.parameters["flow_Maipo extra"],
            model.parameters["flow_Volcan"]
        ],
        agg_func="sum",
        name="caudal_naturalizado"
    )
    paramIndex['caudal_naturalizado'] = model.parameters.__len__() - 1

    # flow_Volcan+Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["flow_Maipo"],
            model.parameters["flow_Volcan"]
        ],
        agg_func="sum",
        name="flow_Volcan+Maipo"
    )
    paramIndex['flow_Volcan+Maipo'] = model.parameters.__len__() - 1

    # descarga_embalse
    ParameterThresholdParameter(
        model,
        param=model.parameters["caudal_naturalizado"],
        threshold=60.48,
        predicate="LT",
        values=[0, 1],
        name="descarga_embalse"
    )
    paramIndex['descarga_embalse'] = model.parameters.__len__() - 1

    # descarga_embalse_real
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_embalse"],
            model.parameters["flow_Yeso"]
        ],
        agg_func="product",
        name="descarga_embalse_real"
    )
    paramIndex['descarga_embalse_real'] = model.parameters.__len__() - 1

    # descarga_embalse_real_Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["flow_Volcan+Maipo"],
            model.parameters["descarga_embalse"]
        ],
        agg_func="product",
        name="descarga_embalse_real_Maipo"
    )
    paramIndex['descarga_embalse_real_Maipo'] = model.parameters.__len__() - 1

    # descarga_adicional2
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional"],
            model.parameters["requisito_embalse"]
        ],
        agg_func="product",
        name="descarga_adicional2"
    )
    paramIndex['descarga_adicional2'] = model.parameters.__len__() - 1

    # descarga_adicional_Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional"],
            model.parameters["requisito_embalse_Maipo"]
        ],
        agg_func="product",
        name="descarga_adicional_Maipo"
    )
    paramIndex['descarga_adicional_Maipo'] = model.parameters.__len__() - 1

    # descarga_adicional_real
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional2"],
            model.parameters["aux_acueductoyeso"]
        ],
        agg_func="product",
        name="descarga_adicional_real"
    )
    paramIndex['descarga_adicional_real'] = model.parameters.__len__() - 1

    # descarga_regla_Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional_Maipo"],
            model.parameters["aux_acueductoyeso"]
        ],
        agg_func="product",
        name="descarga_regla_Maipo"
    )
    paramIndex['descarga_regla_Maipo'] = model.parameters.__len__() - 1

    # AA_total_shares_constant
    ConstantParameter(
        model,
        name="AA_total_shares_constant",
        value=1917
    )
    paramIndex['AA_total_shares_constant'] = model.parameters.__len__() - 1

    # AA_total_shares
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["AA_total_shares_constant"],
            model.parameters["purchases_value"],
            model.parameters["contract_value"],
            model.parameters["agricultural_shares_constant"]
        ],
        # Shares bought can't be more than what ag has to give
        agg_func=lambda x: np.min([x[0] + x[1] + x[2], x[0] + x[3]]),
        name="AA_total_shares",
        comment="expressed as absolute value of total shares"
    )
    paramIndex['AA_total_shares'] = model.parameters.__len__() - 1

    # Agriculture_total_shares
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["agricultural_shares_constant"],
            model.parameters["purchases_value"],
            model.parameters["contract_value"]
        ],
        agg_func=lambda x: np.max([x[0] - x[1] - x[2], 0]),
        name="ag_total_shares",
        comment="expressed as absolute value of total shares"
    )
    paramIndex['ag_total_shares'] = model.parameters.__len__() - 1

    # AA_total_shares_fraction_constant
    ConstantParameter(
        model,
        name="AA_total_shares_fraction_constant",
        value=.0001229558588466740
    )
    paramIndex['AA_total_shares_fraction_constant'] = model.parameters.__len__() - 1

    # AA_total_shares_fraction
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["AA_total_shares"],
            model.parameters["AA_total_shares_fraction_constant"]
        ],
        agg_func="product",
        name="AA_total_shares_fraction",
        comment="expressed as fraction of total shares"
    )
    paramIndex['AA_total_shares_fraction'] = model.parameters.__len__() - 1

    # ag_total_shares_fraction
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["ag_total_shares"],
            model.parameters["AA_total_shares_fraction_constant"]
        ],
        agg_func="product",
        name="ag_total_shares_fraction",
        comment="expressed as fraction of total shares"
    )
    paramIndex['ag_total_shares_fraction'] = model.parameters.__len__() - 1

    # max_flow_perdicez
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["caudal_naturalizado"],
            model.parameters["estacionalidad_distribucion"],
            model.parameters["AA_total_shares_fraction"]
        ],
        agg_func="product",
        name="max_flow_perdicez"
    )
    paramIndex['max_flow_perdicez'] = model.parameters.__len__() - 1

    # derechos_sobrantes
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["max_flow_perdicez"],
            model.parameters["demanda_PT2_negativa"]
        ],
        agg_func="sum",
        name="derechos_sobrantes"
    )
    paramIndex['derechos_sobrantes'] = model.parameters.__len__() - 1

    # contrato
    ConstantParameter(
        model,
        name="contrato",
        value=0
    )
    paramIndex['contrato'] = model.parameters.__len__() - 1

    # derechos_sobrantes_contrato
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["derechos_sobrantes"],
            model.parameters["contrato"]
        ],
        agg_func=lambda x: np.max([x[0] + x[1], 0]),  # was "sum", now lower-bounding at 0
        name="derechos_sobrantes_contrato"
    )
    paramIndex['derechos_sobrantes_contrato'] = model.parameters.__len__() - 1

    # ag max flow
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["caudal_naturalizado"],
            model.parameters["estacionalidad_distribucion"],
            model.parameters["ag_total_shares_fraction"]
        ],
        agg_func="product",
        name="ag_max_flow"
    )
    paramIndex['ag_max_flow'] = model.parameters.__len__() - 1


    # REMAINING NODES
    # Yeso
    Yeso = Catchment(
        model,
        name="Yeso",
        flow=model.parameters["flow_Yeso"]
    )

    # Maipo
    Maipo = Catchment(
        model,
        name="Maipo",
        flow=model.parameters["flow_Maipo"]
    )

    # Colorado
    Colorado = Catchment(
        model,
        name="Colorado",
        flow=model.parameters["flow_Colorado"]
    )

    # Volcan
    Volcan = Catchment(
        model,
        name="Volcan",
        flow=model.parameters["flow_Volcan"]
    )

    # Laguna negra
    Laguna_negra = Catchment(
        model,
        name="Laguna negra",
        flow=model.parameters["flow_Laguna Negra"]
    )

    # Maipo extra
    Maipo_extra = Catchment(
        model,
        name="Maipo extra",
        flow=model.parameters["flow_Maipo extra"]
    )

    # Regla Embalse Maipo
    Regla_Embalse_Maipo = River(
        model,
        name="Regla Embalse Maipo",
        min_flow=model.parameters["descarga_regla_Maipo"],
        cost=100
    )

    # Rio Yeso Alto
    Rio_Yeso_Alto = River(
        model,
        name="Rio Yeso Alto"
    )

    # Rio Yeso Alto
    Rio_Yeso_Bajo = River(
        model,
        name="Rio Yeso Bajo"
    )

    # Rio Maipo Alto
    Rio_Maipo_Alto = River(
        model,
        name="Rio Maipo Alto"
    )

    # Rio Colorado
    Rio_Colorado = River(
        model,
        name="Rio Colorado"
    )

    # Estero del Manzanito
    Estero_del_Manzanito = River(
        model,
        name="Estero del Manzanito"
    )

    # aux_Maipo Extra
    aux_Maipo_Extra = River(
        model,
        name="aux_Maipo Extra"
    )

    # Acueducto Laguna Negra
    Acueducto_Laguna_Negra = River(
        model,
        name="Acueducto Laguna Negra",
        max_flow=model.parameters["aux_acueductoln"],
        cost=-1000
    )

    # Acueducto El Yeso
    Acueducto_El_Yeso = River(
        model,
        name="Acueducto El Yeso"
    )

    # Acueducto Maipo
    Acueducto_Maipo = River(
        model,
        name="Acueducto Maipo",
        cost=-200
    )

    # Acueducto El Yeso 2
    Acueducto_El_Yeso_2 = River(
        model,
        name="Acueducto El Yeso 2",
        min_flow=model.parameters["descarga_adicional_real"],
        cost=100
    )

    # Retorno al Maipo
    Retorno_al_Maipo = River(
        model,
        name="Retorno al Maipo",
        cost=-100
    )

    # aux_Acueducto Yeso
    aux_Acueducto_Yeso = River(
        model,
        name="aux_Acueducto Yeso"
    )

    # Rio Maipo bajo El Manzano
    Rio_Maipo_bajo_El_Manzano = River(
        model,
        name="Rio Maipo bajo El Manzano"
    )

    # Captacion PT
    Captacion_PT = River(
        model,
        name="Captacion PT"
    )

    # Extraccion a Acueducto Laguna Negra
    Extraccion_a_Acueducto_Laguna_Negra = River(
        model,
        max_flow=model.parameters['aux_extraccionln'],
        name="Extraccion a Acueducto Laguna Negra"
    )

    # Toma a PT1
    Toma_a_PT1 = River(
        model,
        name="Toma a PT1",
        max_flow=model.parameters["derechos_sobrantes_contrato"],
        cost=-300
    )

    # PT1 (unrestricted, lets us find stats with true demand)
    PT1 = Link(
        model,
        name="PT1",
        max_flow=model.parameters["demanda_PT1"]
    )

    # Below: implementing demand restriction in a single node (instead of two seperate nodes, as it is now)

    # PT1 = RestrictedOutput(
    #     model,
    #     name="PT1",
    #     desired_flow=model.parameters["demanda_PT1"],
    #     restriction_factor=model.parameters["demand_restriction_factor"],
    #     cost=-10000
    # )

    # PT1 output node representing restricted demand
    PT1_output = Output(
        model,
        name="PT1_output",
        max_flow=model.parameters["demand_max_flow_PT1"],
        cost=-10000
    )

    # PT2
    PT2 = Output(
        model,
        name="PT2",
        max_flow=model.parameters["demanda_PT2"],
        cost=-10000
    )

    # Las Perdicez
    Las_Perdicez = River(
        model,
        name="Las Perdicez"
    )

    # Filtraciones Embalse
    Filtraciones_Embalse = River(
        model,
        name="Filtraciones Embalse",
        max_flow=model.parameters["aux_filtraciones"],
        cost=-9999
    )

    # Filtraciones Embalse Maipo
    Filtraciones_Embalse_Maipo = River(
        model,
        name="Filtraciones Embalse Maipo",
        max_flow=model.parameters["aux_filtraciones"],
        cost=-9999
    )

    # Descarga Embalse
    Descarga_Embalse = River(
        model,
        name="Descarga Embalse",
        min_flow=model.parameters["descarga_embalse_real"],
        cost=-100
    )

    # Descarga Embalse Maipo
    Descarga_Embalse_Maipo = River(
        model,
        name="Descarga Embalse Maipo",
        min_flow=model.parameters["descarga_embalse_real_Maipo"]
    )

    # aux_Salida Maipo
    aux_Salida_Maipo = River(
        model,
        name="aux_Salida Maipo",
        min_flow=model.parameters["flujo_excedentes_Yeso"]
    )

    # aux_PT1
    aux_PT1 = River(
        model,
        name="aux_PT1"
    )

    # Salida_Maipo (leftover leaving the basin)
    Salida_Maipo = Output(
        model,
        name="Salida Maipo",
        cost=-500
    )

    # Agriculture node with true demand
    Agriculture = Link(
        model,
        name="Agriculture",
        max_flow=model.parameters["agricultural_demand_constant"]
    )

    # Agriculture output node with demand restricted by water rights
    Agriculture_output = Output(
        model,
        name="Agriculture_output",
        max_flow=model.parameters["ag_max_flow"],
        cost=-600  # More negative than Salida_Maipo but not enough to take from Embalse
    )


    # EDGES
    # from catchment inflows
    Yeso.connect(Rio_Yeso_Alto)
    Maipo.connect(Rio_Maipo_Alto)
    Colorado.connect(Rio_Colorado)
    Volcan.connect(Rio_Maipo_Alto)
    Laguna_negra.connect(Acueducto_Laguna_Negra)
    Laguna_negra.connect(Estero_del_Manzanito)
    Maipo_extra.connect(aux_Maipo_Extra)
    # from storage nodes
    Embalse.connect(Acueducto_El_Yeso_2)
    Embalse.connect(Descarga_Embalse)
    Embalse.connect(Filtraciones_Embalse)
    Embalse_Maipo.connect(Descarga_Embalse_Maipo)
    Embalse_Maipo.connect(Acueducto_Maipo)
    Embalse_Maipo.connect(Regla_Embalse_Maipo)
    Embalse_Maipo.connect(Filtraciones_Embalse_Maipo)
    # from river nodes
    Regla_Embalse_Maipo.connect(El_Manzano)
    Rio_Yeso_Alto.connect(Embalse)
    Rio_Yeso_Bajo.connect(El_Manzano)
    Rio_Maipo_Alto.connect(Embalse_Maipo)
    Rio_Colorado.connect(El_Manzano)
    Estero_del_Manzanito.connect(Rio_Yeso_Bajo)
    aux_Maipo_Extra.connect(El_Manzano)
    Acueducto_Laguna_Negra.connect(aux_PT1)
    Acueducto_El_Yeso.connect(aux_Acueducto_Yeso)
    Acueducto_El_Yeso.connect(Retorno_al_Maipo)
    Acueducto_Maipo.connect(aux_PT1)
    Acueducto_El_Yeso_2.connect(Acueducto_El_Yeso)
    Retorno_al_Maipo.connect(El_Manzano)
    aux_Acueducto_Yeso.connect(aux_PT1)
    El_Manzano.connect(Rio_Maipo_bajo_El_Manzano)
    Rio_Maipo_bajo_El_Manzano.connect(Extraccion_a_Acueducto_Laguna_Negra)
    Rio_Maipo_bajo_El_Manzano.connect(Las_Perdicez)
    Rio_Maipo_bajo_El_Manzano.connect(Captacion_PT)
    Captacion_PT.connect(Toma_a_PT1)
    Captacion_PT.connect(aux_Salida_Maipo)
    Extraccion_a_Acueducto_Laguna_Negra.connect(Acueducto_Laguna_Negra)
    Toma_a_PT1.connect(aux_PT1)
    Las_Perdicez.connect(PT2)
    Filtraciones_Embalse.connect(Rio_Yeso_Bajo)
    Filtraciones_Embalse_Maipo.connect(El_Manzano)
    Descarga_Embalse.connect(Rio_Yeso_Bajo)
    Descarga_Embalse_Maipo.connect(El_Manzano)
    aux_Salida_Maipo.connect(Agriculture)
    aux_Salida_Maipo.connect(Salida_Maipo)
    aux_PT1.connect(PT1)
    PT1.connect(PT1_output)  # add demand restriction as a new node
    Agriculture.connect(Agriculture_output)  # add water rights limit as a new node

    # RECORDERS
    # RollingMeanFlowElManzano
    RollingMeanFlowNodeRecorder(
        model,
        node=model.nodes["El Manzano"],  # El_Manzano
        timesteps=520,
        name="RollingMeanFlowElManzano"
    )
    recorderIndex['RollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # failure_frequency_PT1 (measured with restricted demand)
    DeficitFrequencyNodeRecorder(
        model,
        node=model.nodes["PT1_output"],  # PT1
        is_objective="minimize",
        comment="Frequency deficit recorded on PT1 output",
        name="failure_frequency_PT1"
    )
    recorderIndex['failure_frequency_PT1'] = model.recorders.__len__() - 1

    def reliability_agg_func(x, axis=0):
        return (1 - np.array(x)).reshape((num_scenarios,))
    # reliability_PT1 (measured with restricted demand)
    AggregatedRecorder(
        model,
        recorders=[model.recorders['failure_frequency_PT1']],
        recorder_agg_func=reliability_agg_func,
        name="reliability_PT1"
    )
    recorderIndex['reliability_PT1'] = model.recorders.__len__() - 1

    # failure_frequency_Ag
    DeficitFrequencyNodeRecorder(
        model,
        node=model.nodes["Agriculture"],  # Agriculture
        is_objective="minimize",
        comment="Frequency deficit recorded on Agriculture output",
        name="failure_frequency_Ag"
    )
    recorderIndex['failure_frequency_Ag'] = model.recorders.__len__() - 1

    # reliability_Ag
    AggregatedRecorder(
        model,
        recorders=[model.recorders['failure_frequency_Ag']],
        recorder_agg_func=reliability_agg_func,
        name="reliability_Ag"
    )
    recorderIndex['reliability_Ag'] = model.recorders.__len__() - 1

    # ReservoirCost
    ReservoirCostRecorder(
        model,
        capacity=model.parameters["Maipo_capacity"],
        construction_dp=model.parameters["Maipo_construction_dp"],
        discount_rate=0.0,
        unit_costs=[9999, 20, 30, 40, 50, 60, 0],
        fixed_cost=100,
        name="ReservoirCost"
    )
    recorderIndex['ReservoirCost'] = model.recorders.__len__() - 1

    # PurchasesCost
    PurchasesCostRecorder(
        model,
        purchases_value=model.parameters["purchases_value"],
        meanflow=model.recorders['RollingMeanFlowElManzano'],
        discount_rate=0.0,
        coeff=1,
        name="PurchasesCost"
    )
    recorderIndex['PurchasesCost'] = model.recorders.__len__() - 1

    # AprilSeasonRollingMeanFlowElManzano
    SeasonRollingMeanFlowNodeRecorder(
        model,
        node=El_Manzano,
        first_week=1,
        last_week=27,
        years=5,
        name='AprilSeasonRollingMeanFlowElManzano'
    )
    recorderIndex['AprilSeasonRollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # OctoberSeasonRollingMeanFlowElManzano
    SeasonRollingMeanFlowNodeRecorder(
        model,
        node=El_Manzano,
        first_week=27,
        last_week=53,
        years=5,
        name='OctoberSeasonRollingMeanFlowElManzano'
    )
    recorderIndex['OctoberSeasonRollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # PremiumAprilCost
    PurchasesCostRecorder(
        model,
        purchases_value=model.parameters["april_contract"],
        meanflow=model.recorders['RollingMeanFlowElManzano'],
        discount_rate=0.0,
        coeff=0.1,
        name="PremiumAprilCost"
    )
    recorderIndex['PremiumAprilCost'] = model.recorders.__len__() - 1

    # PremiumOctoberCost
    PurchasesCostRecorder(
        model,
        purchases_value=model.parameters["october_contract"],
        meanflow=model.recorders["RollingMeanFlowElManzano"],
        discount_rate=0.0,
        coeff=0.1,
        name="PremiumOctoberCost"
    )
    recorderIndex['PremiumOctoberCost'] = model.recorders.__len__() - 1

    # AprilContractCost
    ContractCostRecorder(
        model,
        contract_value=model.parameters["april_contract"],
        meanflow=model.recorders["AprilSeasonRollingMeanFlowElManzano"],
        purchases_value=model.parameters["purchases_value"],
        discount_rate=0.0,
        max_cost=100,
        gradient=-1,
        coeff=1,
        week_no=1,
        name="AprilContractCost"
    )
    recorderIndex['AprilContractCost'] = model.recorders.__len__() - 1

    # OctoberContractCost
    ContractCostRecorder(
        model,
        contract_value=model.parameters["october_contract"],
        meanflow=model.recorders["OctoberSeasonRollingMeanFlowElManzano"],
        purchases_value=model.parameters["purchases_value"],
        discount_rate=0.0,
        max_cost=100,
        gradient=-1,
        coeff=1,
        week_no=27,
        name="OctoberContractCost"
    )
    recorderIndex['OctoberContractCost'] = model.recorders.__len__() - 1

    # Total demand-restriction cost in dollars for each scenario. No discount
    # is applied, consistent with the disc0 configuration of the other costs.
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["demand_restriction_cost_per_week"],
        temporal_agg_func="sum",
        agg_func="mean",
        name="DemandRestrictionCost"
    )
    recorderIndex['DemandRestrictionCost'] = model.recorders.__len__() - 1

    # TotalCost
    AggregatedRecorder(
        model,
        agg_func="mean",
        recorder_agg_func="sum",
        recorders=[
            model.recorders["ReservoirCost"],
            model.recorders["PurchasesCost"],
            model.recorders["PremiumAprilCost"],
            model.recorders["PremiumOctoberCost"],
            model.recorders["AprilContractCost"],
            model.recorders["OctoberContractCost"],
            model.recorders["DemandRestrictionCost"]
        ],
        is_objective="minimize",
        name="TotalCost"
    )
    recorderIndex['TotalCost'] = model.recorders.__len__() - 1

    # deficit PT1 (measured on original demand)
    TotalDeficitNodeRecorder(
        model,
        node=PT1,
        is_objective="min",
        comment="Total deficit recorded on PT1",
        name="deficit PT1"
    )
    recorderIndex['deficit PT1'] = model.recorders.__len__() - 1

    # deficit Ag
    TotalDeficitNodeRecorder(
        model,
        node=Agriculture,
        is_objective="min",
        comment="Total deficit recorded on Agriculture",
        name="deficit Ag"
    )
    recorderIndex['deficit Ag'] = model.recorders.__len__() - 1

    # Caudal en salida promedio
    MeanFlowNodeRecorder(
        model,
        node=Salida_Maipo,
        is_objective="max",
        comment="Mean flow at system output",
        name="Caudal en salida promedio"
    )
    recorderIndex['Caudal en salida promedio'] = model.recorders.__len__() - 1

    # Maximum Deficit on PT1 (measured with original demand)
    MaximumDeficitNodeRecorder(
        model,
        node=PT1,
        is_objective="min",
        name="Maximum Deficit PT1"
    )
    recorderIndex['Maximum Deficit PT1'] = model.recorders.__len__() - 1

    # Maximum Deficit on Agriculture
    MaximumDeficitNodeRecorder(
        model,
        node=Agriculture,
        is_objective="min",
        name="Maximum Deficit Ag"
    )
    recorderIndex['Maximum Deficit Ag'] = model.recorders.__len__() - 1

    # Total Contracts Made
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["contract_value"],
        temporal_agg_func="sum",
        agg_func="mean",
        is_objective="min",
        name="Total Contracts Made"
    )
    recorderIndex['Total Contracts Made'] = model.recorders.__len__() - 1

    # InstanstaneousDeficit (measured with original demand)
    InstantaneousDeficictNodeRecorder(
        model,
        node=PT1,
        name="InstanstaneousDeficit"
    )
    recorderIndex['InstanstaneousDeficit'] = model.recorders.__len__() - 1

    # PT1 flow
    NumpyArrayNodeRecorder(
        model,
        node=PT1,
        name="PT1 flow",
        agg_func="SUM"
    )
    recorderIndex['PT1 flow'] = model.recorders.__len__() - 1

    # PT2 flow
    NumpyArrayNodeRecorder(
        model,
        node=PT2,
        name="PT2 flow",
        agg_func="SUM"
    )
    recorderIndex['PT2 flow'] = model.recorders.__len__() - 1

    # PT1 demand (tracking demand as parameter to compare to data's demand)
    NumpyArrayParameterRecorder(
        model,
        name="demanda_PT1 recorder",
        param=model.parameters["demanda_PT1"],
        temporal_agg_func="sum",
        agg_func="SUM"
    )
    recorderIndex['demanda_PT1 recorder'] = model.recorders.__len__() - 1

    # PT2 demand (tracking demand as parameter to compare to data's demand)
    NumpyArrayParameterRecorder(
        model,
        name="demanda_PT2 recorder",
        param=model.parameters["demanda_PT2"],
        temporal_agg_func="sum",
        agg_func="SUM"
    )
    recorderIndex['demanda_PT2 recorder'] = model.recorders.__len__() - 1

    # Agriculture flow
    NumpyArrayNodeRecorder(
        model,
        node=Agriculture,
        name="Agriculture flow",
        agg_func="SUM"
    )
    recorderIndex['Agriculture flow'] = model.recorders.__len__() - 1

    # Salida Maipo flow
    NumpyArrayNodeRecorder(
        model,
        node=Salida_Maipo,
        name="Salida Maipo flow",
        agg_func="SUM"
    )
    recorderIndex['Salida Maipo flow'] = model.recorders.__len__() - 1

    # Embalse storage
    NumpyArrayStorageRecorder(
        model,
        node=Embalse,
        name="Embalse storage"
    )
    recorderIndex['Embalse storage'] = model.recorders.__len__() - 1

    # Total inflow from reservoirs
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["caudal_naturalizado"],
        name="Total reservoir inflow"
    )
    recorderIndex['Total reservoir inflow'] = model.recorders.__len__() - 1

    # remaining water rights per week
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["ag_max_flow"],
        name="Remaining water rights per week"
    )
    recorderIndex['Remaining water rights per week'] = model.recorders.__len__() - 1

    # Agricultural demand
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["agricultural_demand_constant"],
        name="Agricultural demand recorder",
    )
    recorderIndex['Agricultural demand recorder'] = model.recorders.__len__() - 1

    # check model validity
    model.check_graph()  # check the connectivity of the graph
    model.check()  # check the validity of the model

    return model


def Simulation_Caller(contract_threshold_val, contract_action_val,
                      demand_threshold_val, demand_action_val):
    """
    Called by Borg MOEA at each function evaluation.

    Uses globals _INDICATOR, _DATA_FOLDER, _DATA_ROOT set by Optimization()
    so this function needs no changes when sweeping indicators via CLI.

    Returns
    -------
    list of float : [-reliability_PT1, -reliability_Ag, TotalCost]
        (3 objectives minimized by Borg)
    """
    contract_threshold_vals = contract_threshold_val * np.ones(num_DP)
    contract_action_vals    = contract_action_val    * np.ones(num_DP)
    demand_threshold_vals   = [demand_threshold_val  * np.ones(12)]
    demand_action_vals      = [np.ones(12), demand_action_val * np.ones(12)]

    # Resolve data path
    if _DATA_ROOT is not None:
        data_folder_path = _DATA_ROOT
    else:
        data_folder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )

    model = make_model(
        contract_threshold_vals=contract_threshold_vals,
        contract_action_vals=contract_action_vals,
        demand_threshold_vals=demand_threshold_vals,
        demand_action_vals=demand_action_vals,
        indicator=_INDICATOR,
        data_folder=os.path.join(data_folder_path, _DATA_FOLDER),
        period=_PERIOD,
        demand_restriction_cost_per_m3=DEMAND_RESTRICTION_COST_PER_M3,
        min_demand_retention_factor=MIN_DEMAND_RETENTION_FACTOR,
    )
    model.check()
    model.check_graph()
    model.find_orphaned_parameters()
    model.run()

    func_list = {
        'mean': np.mean,
        'max':  np.max,
        'min':  np.min,
    }

    def get_performance(recorder):
        func = func_list[recorder.agg_func]
        return func(recorder.values())

    return [
        -get_performance(model.recorders['reliability_PT1']),   # assume borg minimize, so negate to maximize 
        -get_performance(model.recorders['reliability_Ag']),    # 
        get_performance(model.recorders['TotalCost']),        # 
    ]


def Optimization(indicator, period, nfunc, seed, epsilon=None,
                 data_folder=None, data_root=None, output_root=None,
                 demand_restriction_cost_per_m3=1.0,
                 max_demand_curtailment_fraction=0.3):
    """
    Run Borg MOEA for a single (indicator, seed) combination.

    Parameters
    ----------
    indicator   : str   e.g. "SRI3"
    nfunc       : int   number of Borg function evaluations
    seed        : int   1-based seed index (one per SLURM array task)
    epsilon     : str or None  comma-separated, e.g. "0.01,0.01,50000000"
    data_folder : str or None  overrides _DATA_FOLDER global
    data_root   : str or None  root directory containing data folders
    output_root : str or None  root directory for outputs
    demand_restriction_cost_per_m3 : float
        Cost assigned to each cubic metre of curtailed urban demand.
    max_demand_curtailment_fraction : float
        Maximum fraction of unrestricted urban demand that may be curtailed
        in any week. The demand-action lower bound is one minus this value.
    """
    import sys

    global _INDICATOR, _DATA_FOLDER, _DATA_ROOT, _PERIOD
    global DEMAND_RESTRICTION_COST_PER_M3, MIN_DEMAND_RETENTION_FACTOR
    _INDICATOR = indicator
    _PERIOD    = period
    if data_folder is not None:
        _DATA_FOLDER = data_folder
    if data_root is not None:
        _DATA_ROOT = data_root

    demand_restriction_cost_per_m3 = float(
        demand_restriction_cost_per_m3
    )
    max_demand_curtailment_fraction = float(
        max_demand_curtailment_fraction
    )
    if (
        not np.isfinite(demand_restriction_cost_per_m3)
        or demand_restriction_cost_per_m3 < 0.0
    ):
        raise ValueError(
            "--demand-restriction-cost-per-m3 must be finite and non-negative"
        )
    if (
        not np.isfinite(max_demand_curtailment_fraction)
        or max_demand_curtailment_fraction < 0.0
        or max_demand_curtailment_fraction > 1.0
    ):
        raise ValueError(
            "--max-demand-curtailment-fraction must be between 0 and 1"
        )

    DEMAND_RESTRICTION_COST_PER_M3 = demand_restriction_cost_per_m3
    MIN_DEMAND_RETENTION_FACTOR = 1.0 - max_demand_curtailment_fraction

    # ------------------------------------------------------------------
    # Borg settings
    # ------------------------------------------------------------------
    parallel     = 1   # 1 = master-slave MPI, 0 = serial
    num_dec_vars = 4
    n_objs       = 3
    n_constrs    = 0
    runtime_freq = 100

    decision_var_range = [
        [-4, 0], [0, 2100], [-4, 0],
        [MIN_DEMAND_RETENTION_FACTOR, 1]
    ]  # contract threshold/action, demand threshold/retention factor

    if epsilon is not None:
        epsilon_list = [float(e) for e in epsilon.split(',')]
        if len(epsilon_list) != n_objs:
            raise ValueError(
                f"--epsilon must have {n_objs} values, got {len(epsilon_list)}"
            )
    else:
        epsilon_list = [0.01, 0.01, 0.05e9]  # original defaults

    print(f"Indicator   : {_INDICATOR}")
    print(f"Period      : {_PERIOD}")
    print(f"Data folder : {_DATA_FOLDER}")
    print(f"NFE         : {nfunc}")
    print(f"Seed        : {seed}")
    print(f"Epsilons    : {epsilon_list}")
    print(
        "Demand restriction cost: "
        f"{DEMAND_RESTRICTION_COST_PER_M3:g} USD/m3 curtailed"
    )
    print(
        "Maximum weekly curtailment: "
        f"{100.0 * max_demand_curtailment_fraction:g}% "
        f"(demand retention >= {MIN_DEMAND_RETENTION_FACTOR:g})"
    )

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    os_fold = Op_Sys_Folder_Operator()

    # Build run label from indicator + nfunc + epsilon + data folder
    eps_str    = epsilon.replace(',', '_').replace('.', 'pt') if epsilon else 'default'
    folder_tag = f"_{_DATA_FOLDER}" if _DATA_FOLDER != "TDP_scenarios_DS_2paramGamma" else ""
    run_label  = f"{_INDICATOR}_{_PERIOD}_nFunc{nfunc}_eps{eps_str}{folder_tag}"

    if output_root is not None:
        base_output = output_root
    else:
        proj_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_output = os.path.join(proj_dir, 'outputs')

    output_dir    = os.path.join(base_output, run_label)
    sets_dir      = os.path.join(output_dir, 'sets')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sets_dir,   exist_ok=True)

    # ------------------------------------------------------------------
    # Run optimization -- one seed per task (SLURM array mode)
    # ------------------------------------------------------------------
    j = seed - 1  # convert 1-based CLI seed to 0-based file index

    if parallel == 1:
        bg.Configuration.startMPI()

    borg = bg.Borg(num_dec_vars, n_objs, n_constrs, Simulation_Caller)
    borg.setBounds(*decision_var_range)
    borg.setEpsilons(*epsilon_list)

    runtime_file = os.path.join(output_dir, f'runtime_file_seed_{j + 1}.runtime')
    if not os.path.exists(runtime_file):
        open(runtime_file, 'w').close()

    if parallel == 1:
        result = borg.solveMPI(maxEvaluations=nfunc, runtime=runtime_file)
    else:
        result = borg.solve({
            "maxEvaluations": nfunc,
            "runtimeformat":  "borg",
            "frequency":      runtime_freq,
            "runtimefile":    runtime_file,
        })

    if result:
        result.display()

        # Write full solution set (decision vars + objectives)
        set_path = os.path.join(sets_dir, f'Borg_DPS_PySedSim{j + 1}.set')
        with open(set_path, 'w') as f:
            f.write('#Borg Optimization Results\n')
            f.write(
                f'#First {num_dec_vars} cols: decision variables; '
                f'last {n_objs} cols: objectives\n'
            )
            f.write(
                '#Objectives: -reliability_PT1 -reliability_Ag TotalCost; '
                'TotalCost includes demand restriction at '
                f'{DEMAND_RESTRICTION_COST_PER_M3:g} USD/m3; '
                'maximum weekly demand curtailment is '
                f'{100.0 * max_demand_curtailment_fraction:g}%\n'
            )
            for solution in result:
                vals = (list(solution.getVariables()) +
                        list(solution.getObjectives()))
                f.write(' '.join(str(v) for v in vals) + '\n')
            f.write('#')

        # Write objectives-only file (for MOEAFramework post-processing)
        set_no_vars_path = os.path.join(
            sets_dir, f'Borg_DPS_PySedSim_no_vars{j + 1}.set'
        )
        with open(set_no_vars_path, 'w') as f:
            for solution in result:
                f.write(
                    ' '.join(str(v) for v in solution.getObjectives()) + '\n'
                )
            f.write('#')

        print(f"[{run_label}] Seed {j + 1} complete.")

    if parallel == 1:
        bg.Configuration.stopMPI()


def Op_Sys_Folder_Operator():
    '''
    Function to determine whether operating system is (1) Windows, or (2) Linux

    Returns folder operator for use in specifying directories (file locations) for reading/writing data pre- and
    post-simulation.
    '''

    if platform.system() == 'Windows':
        os_fold_op = '\\'
    elif platform.system() == 'Linux':
        os_fold_op = '/'
    else:
        os_fold_op = '/'  # Assume unix OS if it can't be identified

    return os_fold_op


#%% Function to find value of parameters -- helpful for debugging

def param_evaluator(param, timestep, scenario):
    try:
        if isinstance(param, NegativeParameter):
            return -param_evaluator(param.parameter, timestep, scenario)
        elif isinstance(param, AggregatedParameter):
            param_values = np.array([param_evaluator(p, timestep, scenario) for p in param.parameters])
            agg_func = param.agg_func
            if callable(agg_func):
                return agg_func(param_values)
            else:
                if agg_func == "sum":
                    return np.sum(param_values)
                elif agg_func == "min":
                    return np.min(param_values)
                elif agg_func == "max":
                    return np.max(param_values)
                elif agg_func == "mean":
                    return np.mean(param_values)
                elif agg_func == "product":
                    return np.prod(param_values)
                else:
                    raise Exception("Agg function not found")
        else:
            return param.value(timestep, scenario)
    except:
        return "Value not found"


#%% practice running
# num_k = 1  # number of levels in policy tree
# num_DP = 7  # number of decision periods
#
# thresh = np.ones(num_DP) * -0.84
# acts = np.ones(num_DP) * 300
#
# Simulation_Caller({
#     "contract_threshold_vals": -999999 * np.ones(num_DP),
#     "contract_action_vals": np.zeros(num_DP),
#     "demand_threshold_vals": [],
#     "demand_action_vals": [np.ones(12)],
#     "indicator": "SRI3",
#     "drought_status_agg": "drought_status_single_day_using_agg"
# })


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "3-objective Borg MOEA optimization of drought management policy "
            "for the Maipo Basin using TDP scenarios. "
            "Designed for SLURM array job submission: run once per "
            "(indicator, seed) combination."
        )
    )
    parser.add_argument(
        "indicator",
        choices=["SPI3", "SPI6", "SPI12", "SRI3", "SRI6", "SRI12"],
        help="Drought indicator to use as policy trigger.",
    )
    parser.add_argument(
        "period",
        choices=["A", "B"],
        help=(
            "Simulation window: "
            "A = 2020-2040 (pre-infrastructure baseline), "
            "B = 2078-2098 (post-infrastructure equilibrium)."
        ),
    )
    parser.add_argument(
        "--nfunc", type=int, default=2000,
        help="Number of Borg function evaluations per seed (default: 2000).",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help=(
            "1-based index of the seed to run. "
            "In SLURM array jobs, derive this from $SLURM_ARRAY_TASK_ID. "
            "Each array task must run exactly one seed."
        ),
    )
    parser.add_argument(
        "--epsilon", type=str, default=None,
        help=(
            "Comma-separated epsilon values for Borg archive. "
            "Must have 3 values matching the 3 objectives "
            "(-reliability_PT1, -reliability_Ag, TotalCost). "
            "Default: '0.01,0.01,50000000'."
        ),
    )
    parser.add_argument(
        "--data_folder", type=str, default="TDP_scenarios_DS_2paramGamma",
        help=(
            "Name of the data folder under data_root containing indicator "
            "CSV files (e.g. SPI3.csv, MAIPO.csv, etc.). "
            "Default: 'TDP_scenarios_DS_2paramGamma'."
        ),
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help=(
            "Root directory containing the data folder. "
            "Overrides the default path derived from the project root. "
            "Example: /home/groups/smfletch/keaniw/MAIPO_PYWR/data"
        ),
    )
    parser.add_argument(
        "--output_root", type=str, default=None,
        help=(
            "Root directory for output folders. "
            "Defaults to <proj_root>/outputs if not specified. "
            "Example: /home/groups/smfletch/keaniw/MAIPO_PYWR/MAIPO_outputs"
        ),
    )
    parser.add_argument(
        "--demand-restriction-cost-per-m3", type=float, default=1.0,
        help=(
            "Cost in USD per cubic metre of curtailed urban demand "
            "(default: 1.0)."
        ),
    )
    parser.add_argument(
        "--max-demand-curtailment-fraction", type=float, default=0.3,
        help=(
            "Maximum fraction of unrestricted urban demand that may be "
            "curtailed in each week (default: 0.3)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    Optimization(
        indicator   = args.indicator,
        period      = args.period,
        nfunc       = args.nfunc,
        seed        = args.seed,
        epsilon     = args.epsilon,
        data_folder = args.data_folder,
        data_root   = args.data_root,
        output_root = args.output_root,
        demand_restriction_cost_per_m3 = (
            args.demand_restriction_cost_per_m3
        ),
        max_demand_curtailment_fraction = (
            args.max_demand_curtailment_fraction
        ),
    )
