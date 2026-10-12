"""
SPI_calculator_TDP_2paramGamma.py

SPI-3, SPI-6 and SPI-12 for the 36 TDP CMIP6 GCM/SSP scenarios (9 models x 4 SSPs), with a
2-parameter gamma fit (location fixed at 0), for two baselines:

  PERIOD A -> data/TDP_scenarios_DS_2paramGamma/SPI3.csv, SPI6.csv, SPI12.csv
      One shared fit per calendar month on 1988-2018
  PERIOD B -> data/TDP_scenarios_DS_2paramGamma_periodB/SPI3.csv, SPI6.csv, SPI12.csv
      Each scenario fit per calendar month on its own 2049-2079, applied to that scenario

Precipitation series:
  basin precipitation = umean of PP_semanal_{model}_{ssp}.csv,
  weekly, summed to monthly totals (mm).

LOCATION AND INPUTS
  data/UQM_weap_TempC_new/PP_semanal_{model}_{ssp}.csv
Update file input paths for new scenarios
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

# ============================== SETTINGS ==============================
PROJECT = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA = PROJECT + 'data/'

PERIODS_TO_RUN = ['A', 'B']        # e.g. ['A'] to write only the Period A files
WINDOWS = [3, 6, 12]

# inputs (weekly precipitation)
PRECIP_DIR = DATA + 'UQM_weap_TempC_new/'
MODELS = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
EXPERIMENTS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

# baseline months for the gamma fits (first and last month, inclusive)
BASELINE_MONTHS = {'A': ('1988-01', '2018-12'),    # shared fit on the first scenario
                   'B': ('2049-12', '2079-11')}    # each scenario's own fit

# outputs (same folders as the SRI files)
OUT_DIR = {'A': DATA + 'TDP_scenarios_DS_2paramGamma/',
           'B': DATA + 'TDP_scenarios_DS_2paramGamma_periodB/'}


# ============================== PRECIPITATION SERIES ==============================
def load_basin_precip(scenario, period):
    """Weekly basin-average precipitation for one scenario, indexed by week start date."""
    df = pd.read_csv(PRECIP_DIR + f'PP_semanal_{scenario}.csv')
    cols = [c for c in df.columns if c not in ('year', 'week')]
    if period == 'B':
        cols = [c for c in cols if not c.startswith(('ARR_', 'HUA_')) and '.' not in c]
    ts = pd.to_datetime(df['year'].astype(str) + df['week'].astype(str).str.zfill(2) + '-1',
                        format='%Y%W-%w')
    p = pd.Series(df[cols].mean(axis=1).values, index=ts).sort_index()
    return p[~p.index.duplicated(keep='first')]


def build_monthly_precip(period):
    """Monthly basin precipitation (mm) for every scenario"""
    scenarios = [f'{m}_{x}' for m in MODELS for x in EXPERIMENTS]
    monthly = pd.DataFrame({s: load_basin_precip(s, period).resample('ME').sum() for s in scenarios})
    return monthly, scenarios                      # 'ME' = month end (pandas >= 2.2)


# ============================== SPI ==============================
def fit_2p_by_month(series):
    """2-parameter gamma (floc=0) fit for each calendar month"""
    params = {}
    for m in range(1, 13):
        data = series[series.index.month == m].dropna()
        params[m] = gamma.fit(data, floc=0)
    return params


def apply_params(rolled, params):
    """SPI = Phi^-1(F)"""
    out = pd.Series(index=rolled.index, dtype=float)
    for m in range(1, 13):
        k = rolled.index.month == m
        out[k] = norm.ppf(np.minimum(gamma.cdf(rolled[k], *params[m]), 1 - 1e-6))
    return out


def monthly_to_weekly(monthly, out_ts):
    filled = monthly.bfill()
    lookup = pd.Series(filled.values, index=filled.index.to_period('M'))
    return pd.Series(out_ts).dt.to_period('M').map(lookup).values


def baseline(series, period):
    """Months of `series` inside the period's BASELINE_MONTHS range (inclusive)."""
    first, last = (pd.Period(m, 'M') for m in BASELINE_MONTHS[period])
    months = series.index.to_period('M')
    return series[(months >= first) & (months <= last)]


def compute_spi(P_mo, scenarios, w, period):
    rolled = {c: P_mo[c].shift(1).rolling(w, min_periods=w).sum() for c in scenarios}

    if period == 'A':
        shared = fit_2p_by_month(baseline(rolled[scenarios[0]], 'A'))
        params = {c: shared for c in scenarios}
    else:
        params = {c: fit_2p_by_month(baseline(rolled[c], 'B')) for c in scenarios}

    # standard Pywr weekly output
    out_ts = pd.date_range('2004-04-01', '2098-11-27', freq='7D')
    cols = {'Timestamp': out_ts.strftime('%d-%m-%Y')}
    for c in scenarios:
        cols[c] = monthly_to_weekly(apply_params(rolled[c], params[c]), out_ts)
    return pd.DataFrame(cols)


def main():
    for period in PERIODS_TO_RUN:
        P_mo, scenarios = build_monthly_precip(period)
        os.makedirs(OUT_DIR[period], exist_ok=True)
        for w in WINDOWS:
            df = compute_spi(P_mo, scenarios, w, period)
            df.to_csv(os.path.join(OUT_DIR[period], f'SPI{w}.csv'), index=False)


if __name__ == '__main__':
    main()
