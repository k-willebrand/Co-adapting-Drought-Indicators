"""
SRI_calculator_TDP_2paramGamma.py

SRI-3, SRI-6 and SRI-12 for the 36 TDP CMIP6 GCM/SSP scenarios (9 models x 4 SSPs), with a
2-parameter gamma fit (location fixed at 0), for two baselines:

  PERIOD A -> data/TDP_scenarios_DS_2paramGamma/SRI3.csv, SRI6.csv, SRI12.csv
      One shared fit per calendar month on 1988-2018
  PERIOD B -> data/TDP_scenarios_DS_2paramGamma_periodB/SRI3.csv, SRI6.csv, SRI12.csv
      Each scenario fit per calendar month on its own 2049-2079, applied to that scenario

Flow series:
  basin inflow = 0.85 x (Yeso + Volcan + Volcan_Canal + Colorado_before_Maipo + Laguna_Negra
                         + Maipo_Las_Hualtatas + Macul + Ramon + Mapocho)
                 + n x glacier melt,   n = 0.15 x 0.85 x headflow / melt

LOCATION AND INPUTS
  data/WEAP_CMIP6_PEGH_Maipo_Cordillera_2021_Results_1980-2099_3June2024_TempC_new/
      <basin>_1980-2099_3June2024_TempC_new.csv  (3 header rows skipped)
  data/CMIP6_Glacier_Melt_TDP.csv
Update file input paths for new scenarios
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

# ============================== SETTINGS ==============================
PROJECT = os.path.dirname(os.path.abspath(__file__)) + os.sep
DATA = PROJECT + 'data' + os.sep

PERIODS_TO_RUN = ['A', 'B']        # e.g. ['A'] to write only the Period A files
WINDOWS = [3, 6, 12]

# inputs (WEAP headflows and glacier melt)
WEAP_DIR = DATA + 'WEAP_CMIP6_PEGH_Maipo_Cordillera_2021_Results_1980-2099_3June2024_TempC_new' + os.sep
WEAP_SUFFIX = '_1980-2099_3June2024_TempC_new.csv'
GLACIER_FILE = DATA + 'CMIP6_Glacier_Melt_TDP.csv'

# baseline months for the gamma fits
BASELINE_MONTHS = {'A': ('1988-01', '2018-12'),
                   'B': ('2049-12', '2079-11')}

# outputs
OUT_DIR = {'A': DATA + 'TDP_scenarios_DS_2paramGamma' + os.sep,
           'B': DATA + 'TDP_scenarios_DS_2paramGamma_periodB' + os.sep}


# ============================== FLOW SERIES ==============================
def load_weap(name):
    return pd.read_csv(WEAP_DIR + name + WEAP_SUFFIX, skiprows=3)


def build_monthly_inflow():
    """Monthly basin inflow for every scenario: scaled headflows + scaled glacier melt."""
    names = ['Yeso', 'Volcan', 'Volcan_Canal', 'Colorado_before_Maipo', 'Laguna_Negra',
             'Maipo_Las_Hualtatas', 'Macul', 'Ramon', 'Mapocho']
    raw = {name: load_weap(name) for name in names}
    scenarios = [c for c in raw['Yeso'].columns if c != 'Scenario']

    parts = raw['Yeso']['Scenario'].str.split()          # e.g. "Wk 12 1979"
    ts = pd.to_datetime(parts.str[2] + parts.str[1].str.zfill(2) + '-1', format='%Y%W-%w')

    r = {name: df[scenarios] for name, df in raw.items()}
    headflows = [r['Yeso'],
                 r['Volcan'] + r['Volcan_Canal'],
                 r['Colorado_before_Maipo'],
                 r['Maipo_Las_Hualtatas'],
                 r['Laguna_Negra'],
                 r['Macul'] + r['Ramon'] + r['Mapocho']]

    # glacier melt scaling factor n (first scenario, 2010-2018 to match historical estimates)
    ref_sc = scenarios[0]
    ref = (ts.dt.year >= 2010) & (ts.dt.year <= 2018)
    I_tot_ref = sum(h.loc[ref, ref_sc].sum() for h in headflows)

    glac = pd.read_csv(GLACIER_FILE, index_col=0).reset_index(drop=True).copy()
    glac['Timestamp'] = pd.to_datetime(glac['Date'])
    glac_ref = (glac['Timestamp'].dt.year >= 2010) & (glac['Timestamp'].dt.year <= 2018)
    # glacier melt = 15% of the scaled (x 0.85) headflow total over 2010-2018
    n_sc = 0.15 * 0.85 * I_tot_ref / glac.loc[glac_ref, ref_sc].sum()

    # align to WEAP timestamps
    hits = np.flatnonzero(glac['Timestamp'].values == ts.iloc[0].to_datetime64())
    glac_aligned = glac[scenarios].iloc[hits[0]:hits[0] + len(ts)].reset_index(drop=True)

    weekly = 0.85 * sum(headflows).reset_index(drop=True) + n_sc * glac_aligned
    weekly.index = pd.DatetimeIndex(ts)
    return weekly.resample('ME').sum(), scenarios


# ============================== SRI ==============================
def fit_2p_by_month(series):
    """2-parameter gamma (floc=0) fit for each calendar month"""
    params = {}
    for m in range(1, 13):
        data = series[series.index.month == m].dropna()
        params[m] = gamma.fit(data, floc=0)
    return params


def apply_params(rolled, params):
    """SRI = Phi^-1(F)"""
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


def compute_sri(I_mo, scenarios, w, period):
    rolled = {c: I_mo[c].shift(1).rolling(w, min_periods=w).sum() for c in scenarios}

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
    I_mo, scenarios = build_monthly_inflow()
    for period in PERIODS_TO_RUN:
        os.makedirs(OUT_DIR[period], exist_ok=True)
        for w in WINDOWS:
            df = compute_sri(I_mo, scenarios, w, period)
            df.to_csv(os.path.join(OUT_DIR[period], f'SRI{w}.csv'), index=False)


if __name__ == '__main__':
    main()
