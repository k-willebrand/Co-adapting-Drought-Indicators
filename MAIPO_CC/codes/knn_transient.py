"""
KNN transient scenario generator
======================================================================================
Generates 108 transient precipitation/temperature scenarios:

    9 GCMs x 4 SSPs  =  36 climate-change trajectories
    x 3 independent stationary KNN base series per trajectory  =  108 scenarios

Each of the 108 scenarios is built from its OWN independently-sampled
stationary KNN base series -- no stationary series is reused across
scenarios. Internal (weather) variability is independently resampled
for every one of the 108 scenarios.

wet_threshold is fixed at -4 (precipitation can never be <= -4), which
disables the KNN drought-walk logic.

OUTPUTS (all under OUTPUT_DIR):
  temp_stationary_scenNNN_1979_2099.csv / precip_stationary_scenNNN_1979_2099.csv
      108 files each. NNN = 001..108, global scenario id.
  temp_transient_{model}_{ssp}_scenNN_1979_2099.csv / precip_transient_...
      108 files each (36 combos x 3 replicates). NN = 01..03, replicate within that combo.

Some diagnostic plots included for GCM-vs-KNN validation figures included.
"""
import re
import warnings
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(invalid="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*divide.*")

# ============================================================
# CONFIG -- point these at your files
# ============================================================
REFERENCE_DIR = Path("/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/Climate_series/Hist")
REFERENCE_TEMP_FILE   = REFERENCE_DIR / "MPC_Catchment_tm_1979-2019_week.csv"
REFERENCE_PRECIP_FILE = REFERENCE_DIR / "MPC_Catchment_pp_1979-2019_week.csv"
REFERENCE_YEARS = (1979, 2019)

GCM_DIR = Path("/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/Climate_series/UQM_weap_TempC_new")

# ── New output folder
OUTPUT_DIR = Path("/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/Climate_series/KNN_resampling/knn_weap_transient_no_drought_pert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_OUT_DIR = Path("/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/plots/transient_no_drought_pert")
PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

SHP_DIR       = Path("/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/shp/PEGH_Maipo_Cordillera_2021-vCCG_CC")
NODES_SHP     = SHP_DIR / "WEAPNode.shp"
SITE_ID_FIELD = "Name"

MODELS = [
    "ACCESS-CM2", "CNRM-CM6-1", "INM-CM4-8", "INM-CM5-0",
    "MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-MM",
]
EXPERIMENTS = ["ssp126", "ssp245", "ssp370", "ssp585"]

PERIODS = [
    ("Start (2020-2040)", 2020, 2040),
    ("Mid (2050-2070)",   2050, 2070),
    ("End (2080-2100)",   2080, 2100),
]

RANDOM_SEED = 42
N_REPLICATES_PER_COMBO = 3
N_STATIONARY_TOTAL = len(MODELS) * len(EXPERIMENTS) * N_REPLICATES_PER_COMBO  # = 108

n_neighbors = 10
block_max = 12
occ_weight = 2.0

WET_THRESHOLD  = -4     # precip can't be <= -4 -> disables dry-state/drought-walk logic (block max and occ_weight also unused in this case)
DRY_FRAC_ENTER = 0.7    # unreachable given WET_THRESHOLD, kept for signature compatibility
FIXED = dict(
    succ_dry_match=0.8,
    max_dry_walk=52,
    dry_frac_continue=0.5,
)

BASELINE_START = 1979
BASELINE_END   = 2019
HALF_WINDOW    = 10
TARGET_YEAR_START = 2020
T_STATIONARY   = 80     # 2020-2099 inclusive
TARGET_YEARS   = np.arange(TARGET_YEAR_START, TARGET_YEAR_START + T_STATIONARY)
T_OUT          = T_STATIONARY * 52
PRECIP_LOG_EPS = 1.0        # mm, avoids log(0)

GCM_TEMP_RE   = re.compile(r"^Tprom_semanal_(.+?)_(ssp\d+)\.csv$", re.IGNORECASE)
GCM_PRECIP_RE = re.compile(r"^PP_semanal_(.+?)_(ssp\d+)\.csv$", re.IGNORECASE)

# Output-file regexes (used by the diagnostic block to re-discover what was written)
KNN_TEMP_RE   = re.compile(r"^temp_transient_(.+?)_(ssp\d+)_scen(\d+)_\d+_\d+\.csv$",   re.IGNORECASE)
KNN_PRECIP_RE = re.compile(r"^precip_transient_(.+?)_(ssp\d+)_scen(\d+)_\d+_\d+\.csv$", re.IGNORECASE)

N_KNN_SCENARIOS_FOR_PLOTS = 3   # = N_REPLICATES_PER_COMBO; how many transient scens/combo per GCM

# ============================================================
# PART A -- KNN stationary generator
# ============================================================

def load_data(temp_file, precip_file):
    temp_data   = pd.read_csv(temp_file)
    precip_data = pd.read_csv(precip_file)
    if not (temp_data[['year', 'week']] == precip_data[['year', 'week']]).all().all():
        raise ValueError("Temp/precip files must have matching year/week columns.")
    years         = temp_data['year'].values
    weeks         = temp_data['week'].values
    temp_values   = temp_data.drop(columns=['year', 'week']).values
    precip_values = precip_data.drop(columns=['year', 'week']).values
    site_names    = temp_data.columns[2:].tolist()
    return years, weeks, temp_values, precip_values, site_names


def frac_dry(P_row, wet_threshold=0.0):
    return np.mean(P_row <= wet_threshold)


def is_dry_state(P_row, wet_threshold=0.0, dry_frac=0.8):
    return frac_dry(P_row, wet_threshold) >= dry_frac


def knn_sample(temp_series, precip_series, temp_values, precip_values,
               week_constraint, weeks_all, expected_next_week,
               n_neighbors=10, occ_weight=2.0, wet_threshold=0.5,
               dry_frac=0.7, succ_dry_match=0.8):
    week_indices = np.where(week_constraint)[0]
    if week_indices.size == 0:
        raise ValueError("week_constraint selected 0 rows; cannot sample.")
    nHist = temp_values.shape[0]

    succ = week_indices + 1
    in_bounds = succ < nHist
    week_indices_ib = week_indices[in_bounds]
    succ_ib = succ[in_bounds]
    valid_week = (weeks_all[succ_ib] == expected_next_week)
    week_indices_valid = week_indices_ib[valid_week]

    if week_indices_valid.size == 0:
        prev_weeks_present = np.unique(weeks_all[week_indices])
        if prev_weeks_present.size > 0:
            prev_week  = int(prev_weeks_present[0])
            prev_minus = 52 if prev_week == 1  else (prev_week - 1)
            prev_plus  = 1  if prev_week == 52 else (prev_week + 1)
            next_minus = 52 if expected_next_week == 1  else (expected_next_week - 1)
            next_plus  = 1  if expected_next_week == 52 else (expected_next_week + 1)
            widened_mask = ((weeks_all == prev_minus) | (weeks_all == prev_week) | (weeks_all == prev_plus))
            widened_idx = np.where(widened_mask)[0]
            succ2 = widened_idx + 1
            inb2 = succ2 < nHist
            widened_idx_ib = widened_idx[inb2]
            succ2_ib = succ2[inb2]
            ok2 = ((weeks_all[succ2_ib] == next_minus) | (weeks_all[succ2_ib] == expected_next_week) | (weeks_all[succ2_ib] == next_plus))
            week_indices_valid = widened_idx_ib[ok2]

    if week_indices_valid.size == 0:
        candidates = np.where(weeks_all == expected_next_week)[0]
        if candidates.size == 0:
            raise ValueError(f"No rows found for expected_next_week={expected_next_week}")
        return int(np.random.choice(candidates))

    prevP = np.asarray(precip_series[-1])
    dry_sites_now = (prevP <= wet_threshold)
    n_dry_sites = int(dry_sites_now.sum())

    if n_dry_sites > 0:
        succ_idx = week_indices_valid + 1
        succP = precip_values[succ_idx]
        keep_frac = np.mean(succP[:, dry_sites_now] <= wet_threshold, axis=1)
        keep_mask = keep_frac >= succ_dry_match
        if keep_mask.sum() >= max(5, n_neighbors):
            week_indices_valid = week_indices_valid[keep_mask]

    ft = temp_values[week_indices_valid]
    fp = precip_values[week_indices_valid]

    current_is_dry = (np.mean(prevP <= wet_threshold) >= dry_frac)
    if current_is_dry:
        cand_frac_dry = np.mean(fp <= wet_threshold, axis=1)
        dry_mask = cand_frac_dry >= dry_frac
        if dry_mask.sum() >= max(5, n_neighbors):
            week_indices_valid = week_indices_valid[dry_mask]
            ft = temp_values[week_indices_valid]
            fp = precip_values[week_indices_valid]

    eps = 1e-8
    temp_mean = ft.mean(axis=0); temp_std = ft.std(axis=0); temp_std = np.where(temp_std == 0, eps, temp_std)
    precip_mean = fp.mean(axis=0); precip_std = fp.std(axis=0); precip_std = np.where(precip_std == 0, eps, precip_std)

    z_amt = np.hstack([(ft - temp_mean) / temp_std, (fp - precip_mean) / precip_std])
    x_amt = np.hstack([(np.asarray(temp_series[-1]) - temp_mean) / temp_std, (prevP - precip_mean) / precip_std])
    occ_cand = (fp > wet_threshold).astype(float)
    occ_x = (prevP > wet_threshold).astype(float)
    z = np.hstack([z_amt, occ_weight * occ_cand])
    x = np.hstack([x_amt, occ_weight * occ_x])
    dist = np.linalg.norm(z - x, axis=1)

    k = min(n_neighbors, week_indices_valid.size)
    nn_sub = np.argsort(dist)[:k]
    ranks = np.arange(1, k + 1, dtype=float)
    weights = 1.0 / ranks; weights /= weights.sum()
    j_sub = np.random.choice(nn_sub, p=weights)
    j_full_prev = week_indices_valid[j_sub]
    return int(j_full_prev + 1)


def generate_stochastic_series_block(years, weeks_all, temp_values, precip_values, N, T,
                                      n_neighbors=10, block_max=12, occ_weight=2.0,
                                      wet_threshold=0.5, dry_frac_enter=0.7, dry_frac_continue=0.5,
                                      succ_dry_match=0.8, max_dry_walk=52):
    total_steps = T * 52
    nHist = temp_values.shape[0]
    week1_idx = np.where(weeks_all == 1)[0]
    if week1_idx.size == 0:
        raise ValueError("No week==1 rows found.")

    def sample_block_len(Lmax):
        if Lmax <= 1:
            return 1
        lens = np.arange(1, Lmax + 1, dtype=int)
        probs = 1.0 / lens.astype(float); probs /= probs.sum()
        return int(np.random.choice(lens, p=probs))

    stochastic_temp_series = []
    stochastic_precip_series = []

    for scen_i in range(N):
        temp_series = []
        precip_series = []
        hist_ptr = int(np.random.choice(week1_idx))
        temp_series.append(temp_values[hist_ptr])
        precip_series.append(precip_values[hist_ptr])
        t = 1
        dry_walk = 0

        while t < total_steps:
            current_week = (t % 52) + 1
            prev_week = 52 if current_week == 1 else (current_week - 1)
            curr_dry = is_dry_state(precip_series[-1], wet_threshold=wet_threshold, dry_frac=dry_frac_enter)

            if curr_dry and (hist_ptr + 1 < nHist) and (dry_walk < max_dry_walk):
                next_idx = hist_ptr + 1
                if weeks_all[next_idx] == current_week:
                    next_dry = is_dry_state(precip_values[next_idx], wet_threshold=wet_threshold, dry_frac=dry_frac_continue)
                    if next_dry:
                        hist_ptr = next_idx
                        temp_series.append(temp_values[hist_ptr])
                        precip_series.append(precip_values[hist_ptr])
                        t += 1; dry_walk += 1
                        continue

            dry_walk = 0
            week_constraint = (weeks_all == prev_week)
            start_index = knn_sample(temp_series=temp_series, precip_series=precip_series,
                                      temp_values=temp_values, precip_values=precip_values,
                                      week_constraint=week_constraint, weeks_all=weeks_all,
                                      expected_next_week=current_week, n_neighbors=n_neighbors,
                                      occ_weight=occ_weight, wet_threshold=wet_threshold,
                                      dry_frac=dry_frac_enter, succ_dry_match=succ_dry_match)

            Lmax_hist = nHist - start_index
            Lmax_syn = total_steps - t
            Lmax = int(min(block_max, Lmax_hist, Lmax_syn))

            if Lmax <= 0:
                candidates = np.where(weeks_all == current_week)[0]
                if candidates.size == 0:
                    warnings.warn(f"No candidates for week {current_week} at t={t}; terminating early.", RuntimeWarning, stacklevel=2)
                    break
                start_index = int(np.random.choice(candidates))
                Lmax_hist = nHist - start_index
                Lmax = int(min(block_max, Lmax_hist, Lmax_syn))
                if Lmax <= 0:
                    warnings.warn(f"Lmax still 0 after fallback at t={t}; terminating early.", RuntimeWarning, stacklevel=2)
                    break

            block_len = sample_block_len(Lmax)
            temp_series.extend(temp_values[start_index:start_index + block_len])
            precip_series.extend(precip_values[start_index:start_index + block_len])
            hist_ptr = start_index + block_len - 1
            t += block_len

        stochastic_temp_series.append(np.array(temp_series))
        stochastic_precip_series.append(np.array(precip_series))
        print(f"    stationary scenario {scen_i + 1}/{N} generated ({len(temp_series)} weeks)")

    return np.array(stochastic_temp_series), np.array(stochastic_precip_series)


def rescale_series(stochastic_series, target_values, weeks_all, N):
    rescaled_series = []
    weeks_all = np.asarray(weeks_all)
    for n in range(N):
        series = stochastic_series[n]
        rescaled = np.copy(series)
        weeks_syn = (np.arange(len(series)) % 52) + 1
        for w in np.unique(weeks_all):
            target_mask = (weeks_all == w)
            target_mean = target_values[target_mask].mean(axis=0)
            target_std  = target_values[target_mask].std(axis=0)
            series_mask = (weeks_syn == w)
            series_mean = series[series_mask].mean(axis=0)
            series_std  = series[series_mask].std(axis=0)
            good = series_std > 0
            rescaled[series_mask] = target_mean
            if np.any(good):
                vals = rescaled[series_mask]
                vals[:, good] = (
                    target_mean[good]
                    + ((series[series_mask][:, good] - series_mean[good]) / series_std[good]) * target_std[good]
                )
                rescaled[series_mask] = vals
        rescaled_series.append(rescaled)
    return np.array(rescaled_series)

# ============================================================
# PART B -- GCM delta-forcing signal + application
# ============================================================

def load_gcm_weekly(temp_gcm_file, precip_gcm_file):
    temp_df   = pd.read_csv(temp_gcm_file)
    precip_df = pd.read_csv(precip_gcm_file)
    if not (temp_df[['year', 'week']] == precip_df[['year', 'week']]).all().all():
        raise ValueError(f"GCM temp/precip files must align in year/week:\n{temp_gcm_file}\n{precip_gcm_file}")
    years = temp_df['year'].to_numpy()
    weeks = temp_df['week'].to_numpy()
    temp = temp_df.iloc[:, 2:].to_numpy()
    precip = precip_df.iloc[:, 2:].to_numpy()
    site_names = temp_df.columns[2:].tolist()
    return years, weeks, temp, precip, site_names


def weekly_climatology(values, years, weeks, year_mask):
    out = np.zeros((52, values.shape[1]))
    for w in range(1, 53):
        mask = (weeks == w) & year_mask
        if not np.any(mask):
            raise ValueError(f"No data for week {w} in selected period.")
        out[w - 1, :] = values[mask].mean(axis=0)
    return out


STD_RATIO_CAP = 5.0
STD_RATIO_FLOOR = 0.2


def weekly_climatology_mean_std(values, years, weeks, year_mask):
    """Per (week, site): mean AND std across the years selected by year_mask."""
    out_mean = np.zeros((52, values.shape[1]))
    out_std = np.zeros((52, values.shape[1]))
    for w in range(1, 53):
        mask = (weeks == w) & year_mask
        if not np.any(mask):
            raise ValueError(f"No data for week {w} in selected period.")
        vals = values[mask]
        out_mean[w - 1, :] = vals.mean(axis=0)
        out_std[w - 1, :] = vals.std(axis=0)
    return out_mean, out_std


def moving_window_weekly_signal_mva(gcm_years, gcm_weeks, gcm_temp, gcm_precip,
                                     baseline_start, baseline_end, target_years,
                                     half_window=10, precip_log_eps=PRECIP_LOG_EPS):
    """MVA (Mean and Variance Adjustment) change-factor vectors: a mean delta AND a
    std ratio, computed from the GCM's 21-year moving-window climatology, for both
    temperature and precipitation. """
    baseline_mask = (gcm_years >= baseline_start) & (gcm_years <= baseline_end)
    if not np.any(baseline_mask):
        raise ValueError(f"No GCM data found in baseline period {baseline_start}-{baseline_end}.")

    temp_base_mean, temp_base_std = weekly_climatology_mean_std(gcm_temp, gcm_years, gcm_weeks, baseline_mask)
    log_precip = np.log(np.maximum(gcm_precip, 0.0) + precip_log_eps)
    logp_base_mean, logp_base_std = weekly_climatology_mean_std(log_precip, gcm_years, gcm_weeks, baseline_mask)

    nY = len(target_years)
    nL = gcm_temp.shape[1]
    temp_win_mean = np.zeros((nY, 52, nL)); temp_win_std = np.zeros((nY, 52, nL))
    logp_win_mean = np.zeros((nY, 52, nL)); logp_win_std = np.zeros((nY, 52, nL))

    for i, y in enumerate(target_years):
        win_mask = (gcm_years >= y - half_window) & (gcm_years <= y + half_window)
        if not np.any(win_mask):
            raise ValueError(f"No GCM data found in moving window around year {y}.")
        tm, ts = weekly_climatology_mean_std(gcm_temp, gcm_years, gcm_weeks, win_mask)
        temp_win_mean[i] = tm; temp_win_std[i] = ts
        lm, ls = weekly_climatology_mean_std(log_precip, gcm_years, gcm_weeks, win_mask)
        logp_win_mean[i] = lm; logp_win_std[i] = ls

    temp_delta = temp_win_mean - temp_base_mean[None, :, :]
    temp_std_ratio = temp_win_std / np.maximum(temp_base_std[None, :, :], 1e-6)
    logp_delta = logp_win_mean - logp_base_mean[None, :, :]
    logp_std_ratio = logp_win_std / np.maximum(logp_base_std[None, :, :], 1e-6)

    return temp_delta, temp_std_ratio, logp_delta, logp_std_ratio


def apply_transient_signal_mva(temp_knn, precip_knn, temp_delta, temp_std_ratio,
                                logp_delta, logp_std_ratio, hist_temp_mean, hist_logp_mean,
                                target_years, precip_log_eps=PRECIP_LOG_EPS,
                                std_ratio_cap=STD_RATIO_CAP, std_ratio_floor=STD_RATIO_FLOOR):
    """Apply change-factor vectors"""
    N, T_knn, L = temp_knn.shape
    n_target_years = len(target_years)
    T_out = n_target_years * 52
    if T_knn < T_out:
        raise ValueError(f"k-NN scenarios shorter than transient horizon: T_knn={T_knn}, need {T_out}.")

    temp_base = temp_knn[:, :T_out, :].reshape(N, n_target_years, 52, L)
    precip_base = precip_knn[:, :T_out, :].reshape(N, n_target_years, 52, L)

    std_ratio_t = np.clip(temp_std_ratio, std_ratio_floor, std_ratio_cap)
    hist_temp_mean_b = hist_temp_mean[None, None, :, :]
    temp_out = hist_temp_mean_b + (temp_base - hist_temp_mean_b) * std_ratio_t[None, :, :, :] + temp_delta[None, :, :, :]

    log_base = np.log(np.maximum(precip_base, 0.0) + precip_log_eps)
    std_ratio_p = np.clip(logp_std_ratio, std_ratio_floor, std_ratio_cap)
    hist_logp_mean_b = hist_logp_mean[None, None, :, :]
    log_out = hist_logp_mean_b + (log_base - hist_logp_mean_b) * std_ratio_p[None, :, :, :] + logp_delta[None, :, :, :]
    precip_out = np.maximum(np.exp(log_out) - precip_log_eps, 0.0)

    return temp_out.reshape(N, T_out, L), precip_out.reshape(N, T_out, L)


def list_gcm_pairs(ssp):
    temp_files, precip_files = {}, {}
    for p in GCM_DIR.glob(f"Tprom_semanal_*_{ssp}.csv"):
        m = GCM_TEMP_RE.match(p.name)
        if m and m.group(2).lower() == ssp.lower() and m.group(1) in MODELS:
            temp_files[m.group(1)] = p
    for p in GCM_DIR.glob(f"PP_semanal_*_{ssp}.csv"):
        m = GCM_PRECIP_RE.match(p.name)
        if m and m.group(2).lower() == ssp.lower() and m.group(1) in MODELS:
            precip_files[m.group(1)] = p
    return [(model, temp_files[model], precip_files[model]) for model in MODELS
            if model in temp_files and model in precip_files]

# ============================================================
# PART C -- CSV formatting for WEAP compatability
# ============================================================

def append_hist_to_transient_week(obs_df, transient_df, end_hist):
    df_obs = obs_df[obs_df.year <= end_hist].copy()
    df_trans = transient_df[transient_df.year > end_hist].copy()
    return pd.concat([df_obs, df_trans], axis=0, ignore_index=True)


def scenario_array_to_weekly_df(arr_2d, years_out, site_names):
    arr_2d = np.asarray(arr_2d)
    n_weeks, n_sites = arr_2d.shape
    n_years = len(years_out)
    expected_weeks = n_years * 52
    if n_weeks != expected_weeks:
        raise ValueError(f"Scenario length mismatch: got {n_weeks} weeks, expected {expected_weeks}")
    year_col = np.repeat(years_out, 52)
    week_col = np.tile(np.arange(1, 53), n_years)
    df = pd.DataFrame(arr_2d, columns=site_names)
    df.insert(0, "week", week_col)
    df.insert(0, "year", year_col)
    return df

# ============================================================
# PART D -- Some diagnostic plots (GCMs vs the new plain transient scenarios)
# ============================================================

CACHE_DIR = Path(tempfile.gettempdir()) / "maipo_cc_scenario_cache_v4"


def _local_path(remote: Path) -> Path:
    if CACHE_DIR is None:
        return remote
    return CACHE_DIR / remote.parent.name / remote.name


def _copy_with_retry(src: Path, dst: Path, retries: int = 5, chunk_bytes: int = 4 * 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                while True:
                    chunk = fsrc.read(chunk_bytes)
                    if not chunk:
                        break
                    fdst.write(chunk)
            if dst.stat().st_size == 0:
                dst.unlink()
                raise OSError(f"Empty after copy: {dst.name}")
            return
        except (TimeoutError, OSError) as e:
            if dst.exists():
                dst.unlink()
            if attempt == retries:
                raise
            print(f"    Retry {attempt}/{retries} for {src.name} ({e})")


def warm_cache(files):
    if CACHE_DIR is None:
        return files
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    need_copy = [p for p in files if not _local_path(p).exists()]
    stale = [p for p in files if _local_path(p).exists() and _local_path(p).stat().st_size == 0]
    for p in stale:
        _local_path(p).unlink()
    need_copy = sorted(set(need_copy + stale), key=lambda p: p.name)
    if need_copy:
        print(f"  Caching {len(need_copy)} files to {CACHE_DIR} ...")
        for i, src in enumerate(need_copy, 1):
            _copy_with_retry(src, _local_path(src))
    return [_local_path(p) for p in files]


def list_gcm_files(var):
    pattern = GCM_TEMP_RE if var == "temp" else GCM_PRECIP_RE
    prefix  = "Tprom_semanal_" if var == "temp" else "PP_semanal_"
    files = sorted([p for p in GCM_DIR.glob(f"{prefix}*_ssp*.csv") if pattern.match(p.name)], key=lambda p: p.name)
    return warm_cache(files)


def list_knn_files(var):
    pattern = KNN_TEMP_RE if var == "temp" else KNN_PRECIP_RE
    prefix  = "temp_transient_" if var == "temp" else "precip_transient_"
    groups = defaultdict(list)
    for p in OUTPUT_DIR.glob(f"{prefix}*_ssp*_scen*.csv"):
        m = pattern.match(p.name)
        if m:
            model, ssp, scen_n = m.group(1), m.group(2), int(m.group(3))
            groups[(model, ssp)].append((scen_n, p))
    files = []
    for key in sorted(groups.keys()):
        gf = sorted(groups[key], key=lambda x: x[0])[:N_KNN_SCENARIOS_FOR_PLOTS]
        files.extend(p for _, p in gf)
    return warm_cache(sorted(files, key=lambda p: p.name))


def safe_read_csv(path):
    try:
        if path.stat().st_size == 0:
            print(f"  WARNING: skipping empty file {path.name}")
            return None
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as e:
        print(f"  WARNING: skipping unreadable file {path.name} ({e})")
        return None


def iso_week_to_datetime(year, week):
    y = year.astype(int).astype(str)
    w = week.astype(int).astype(str).str.zfill(2)
    return pd.to_datetime(y + "-" + w + "-1", format="%G-%V-%u", errors="coerce")


def load_weekly_climate_csv(path):
    df = pd.read_csv(path)
    if not {"year", "week"}.issubset(df.columns):
        raise KeyError(f"{path.name} must contain 'year' and 'week' columns")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-W" + df["week"].astype(str) + "-1", format="%G-W%V-%u")
    df = df.drop(columns=["year", "week"]).set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", regex=True)]
    df.columns = df.columns.astype(str)
    return df


def read_location_cols(path):
    return [c for c in pd.read_csv(path, nrows=0).columns if c not in {"year", "week"}]


def common_locations(file_lists):
    loc_sets = [set(read_location_cols(lst[0])) for lst in file_lists if lst]
    return sorted(set.intersection(*loc_sets)) if loc_sets else []


def block1_temp_heatmaps():
    print("\n=== Diagnostic 1: Temperature spatial-correlation heatmaps ===")
    gcm_files = list_gcm_files("temp")
    knn_files = list_knn_files("temp")
    common_locs = common_locations([gcm_files, knn_files])
    print(f"GCM: {len(gcm_files)}  KNN: {len(knn_files)}  Common locs: {len(common_locs)}")

    def corr_matrix(path, y0, y1):
        df = safe_read_csv(path)
        if df is None: return None
        df["date"] = iso_week_to_datetime(df["year"], df["week"])
        df = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]
        X = df[common_locs].apply(pd.to_numeric, errors="coerce")
        corr = X.corr(method="pearson", min_periods=10)
        valid = ~corr.isna().all(axis=0)
        corr_clean = corr.loc[valid, valid]
        prefix = corr_clean.index.to_series().str[:3]
        keep = corr_clean.index[~prefix.duplicated()]
        return X[keep].corr(method="pearson", min_periods=10).to_numpy(dtype=float)

    def group_mean(files, y0, y1):
        mats = [m for m in (corr_matrix(p, y0, y1) for p in files) if m is not None]
        return np.nanmean(np.stack(mats, axis=0), axis=0)

    gcm_means = [group_mean(gcm_files, y0, y1) for _, y0, y1 in PERIODS]
    knn_means = [group_mean(knn_files, y0, y1) for _, y0, y1 in PERIODS]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    vmin, vmax = 0.9, 1.0
    for c, (label, _, _) in enumerate(PERIODS):
        for r, (mats, row_label) in enumerate([(gcm_means, "GCMs (all 36)"),
                                                 (knn_means, f"KNN transient (scen01-{N_KNN_SCENARIOS_FOR_PLOTS:02d})")]):
            ax = axes[r, c]
            im = ax.imshow(mats[c], vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("Location index"); ax.set_ylabel("Location index")
            ax.set_title(f"{label}\n{row_label}" if r == 0 else row_label)

    fig.subplots_adjust(left=0.12)
    cax = fig.add_axes([0.04, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax).set_label("Mean Pearson r")
    fig.suptitle("Mean cross-site temperature correlation (Tprom)", y=0.98)
    fig.tight_layout()
    out = PLOT_OUT_DIR / "mean_spatial_corr_6panel_Tprom.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out}")


def block2_precip_heatmaps():
    print("\n=== Diagnostic 2: Precipitation spatial-correlation heatmaps ===")
    gcm_files = list_gcm_files("precip")
    knn_files = list_knn_files("precip")
    common_locs = common_locations([gcm_files, knn_files])
    print(f"GCM: {len(gcm_files)}  KNN: {len(knn_files)}  Common locs: {len(common_locs)}")

    def corr_matrix(path, y0, y1):
        df = safe_read_csv(path)
        if df is None: return None
        df["date"] = iso_week_to_datetime(df["year"], df["week"])
        df = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]
        X = df[common_locs].apply(pd.to_numeric, errors="coerce")
        corr = X.corr(method="pearson", min_periods=10)
        valid = ~corr.isna().all(axis=0)
        corr_clean = corr.loc[valid, valid]
        prefix = corr_clean.index.to_series().str[:3]
        keep = corr_clean.index[~prefix.duplicated()]
        return X[keep].corr(method="pearson", min_periods=10).to_numpy(dtype=float)

    def group_mean(files, y0, y1):
        mats = [m for m in (corr_matrix(p, y0, y1) for p in files) if m is not None]
        return np.nanmean(np.stack(mats, axis=0), axis=0)

    gcm_means = [group_mean(gcm_files, y0, y1) for _, y0, y1 in PERIODS]
    knn_means = [group_mean(knn_files, y0, y1) for _, y0, y1 in PERIODS]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    vmin, vmax = 0.9, 1.0
    for c, (label, _, _) in enumerate(PERIODS):
        for r, (mats, row_label) in enumerate([(gcm_means, "GCMs (all 36)"),
                                                 (knn_means, f"KNN transient (scen01-{N_KNN_SCENARIOS_FOR_PLOTS:02d})")]):
            ax = axes[r, c]
            im = ax.imshow(mats[c], vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("Location index"); ax.set_ylabel("Location index")
            ax.set_title(f"{label}\n{row_label}" if r == 0 else row_label)

    fig.subplots_adjust(left=0.12)
    cax = fig.add_axes([0.04, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax).set_label("Mean Pearson r")
    fig.suptitle("Mean cross-site precipitation correlation (PP)", y=0.98)
    fig.tight_layout()
    out = PLOT_OUT_DIR / "mean_spatial_corr_6panel_PP.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out}")


def block3_monthly_tp_boxplots():
    """NOTE: this is a SPATIAL cross-variable diagnostic -- for each week it
    correlates temperature-across-sites against precip-across-sites. It does
    NOT test temporal T-P coupling (whether hot years/weeks trend dry over
    the century). Since this run applies no shuffle/conditioning, this mostly
    reflects whatever spatial T-P structure the KNN historical pool already had."""
    print("\n=== Diagnostic 3: Monthly T-P spatial correlation boxplots ===")
    MIN_LOCATIONS_FOR_CORR = 3

    gcm_temp_files   = list_gcm_files("temp")
    gcm_precip_files = list_gcm_files("precip")
    knn_temp_files   = list_knn_files("temp")
    knn_precip_files = list_knn_files("precip")

    def gcm_key(p):
        m = GCM_TEMP_RE.match(p.name) or GCM_PRECIP_RE.match(p.name)
        return f"{m.group(1)}_{m.group(2)}" if m else p.stem

    def knn_key(p):
        m = KNN_TEMP_RE.match(p.name) or KNN_PRECIP_RE.match(p.name)
        return f"{m.group(1)}_{m.group(2)}_scen{m.group(3)}" if m else p.stem

    gcm_temp_dict = {gcm_key(p): p for p in gcm_temp_files}
    gcm_prec_dict = {gcm_key(p): p for p in gcm_precip_files}
    knn_temp_dict = {knn_key(p): p for p in knn_temp_files}
    knn_prec_dict = {knn_key(p): p for p in knn_precip_files}

    gcm_pairs = [(gcm_temp_dict[k], gcm_prec_dict[k]) for k in sorted(gcm_temp_dict) if k in gcm_prec_dict]
    knn_pairs = [(knn_temp_dict[k], knn_prec_dict[k]) for k in sorted(knn_temp_dict) if k in knn_prec_dict]
    print(f"GCM pairs: {len(gcm_pairs)}  KNN pairs: {len(knn_pairs)}")

    common_locs = common_locations([gcm_temp_files + knn_temp_files, gcm_precip_files + knn_precip_files])

    def weekly_tp_corr(tp_path, pp_path):
        df_t = load_weekly_climate_csv(tp_path)
        df_p = load_weekly_climate_csv(pp_path)
        idx = df_t.index.intersection(df_p.index)
        df_t = df_t.loc[idx, [c for c in common_locs if c in df_t.columns]]
        df_p = df_p.loc[idx, [c for c in common_locs if c in df_p.columns]]
        shared = [c for c in df_t.columns if c in df_p.columns]
        r_vals = []
        for d in idx:
            a = df_t.loc[d, shared].to_numpy(dtype=float)
            b = df_p.loc[d, shared].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            r_vals.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= MIN_LOCATIONS_FOR_CORR else np.nan)
        return pd.Series(r_vals, index=idx)

    def monthly_pooled(pairs, y0, y1):
        monthly = [[] for _ in range(12)]
        for tp, pp in pairs:
            rts = weekly_tp_corr(tp, pp)
            sel = rts[(rts.index.year >= y0) & (rts.index.year <= y1)].dropna()
            for m in range(1, 13):
                v = sel[sel.index.month == m].to_numpy()
                if v.size:
                    monthly[m - 1].append(v)
        return [np.concatenate(v) if v else np.array([]) for v in monthly]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    months = np.arange(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    offsets = [-0.18, +0.18]; width = 0.28

    for ax, (label, y0, y1) in zip(axes, PERIODS):
        g1_vals = monthly_pooled(gcm_pairs, y0, y1)
        g2_vals = monthly_pooled(knn_pairs, y0, y1)
        data, pos, colors = [], [], []
        for i, m in enumerate(months):
            data.append(g1_vals[i]); pos.append(m + offsets[0]); colors.append("chocolate")
            data.append(g2_vals[i]); pos.append(m + offsets[1]); colors.append("steelblue")
        bp = ax.boxplot(data, positions=pos, widths=width, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.2)
        ax.set_xticks(months); ax.set_xticklabels(month_labels)
        ax.set_title(label); ax.set_xlabel("Month"); ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Weekly spatial corr: corr(T_locations, P_locations)")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="chocolate", edgecolor="black", alpha=0.75),
        plt.Rectangle((0, 0), 1, 1, facecolor="steelblue", edgecolor="black", alpha=0.75),
    ]
    fig.legend(legend_handles, ["GCMs (all 36)", f"KNN transient (scen01-{N_KNN_SCENARIOS_FOR_PLOTS:02d})"],
               loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Monthly distribution of weekly T-PP spatial correlation\n(GCMs vs KNN transient, plain delta-forcing)", y=1.05)
    plt.tight_layout()
    out = PLOT_OUT_DIR / "monthly_tp_corr_boxplots.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def block4_diff_heatmaps():
    print("\n=== Diagnostic 4: Difference heatmaps (GCMs minus KNN transient) ===")
    gcm_files = list_gcm_files("temp")
    knn_files = list_knn_files("temp")
    common_locs = common_locations([gcm_files, knn_files])
    print(f"GCM: {len(gcm_files)}  KNN: {len(knn_files)}  Common locs: {len(common_locs)}")

    def _kept_labels(path, y0, y1):
        df = pd.read_csv(path)
        df["date"] = iso_week_to_datetime(df["year"], df["week"])
        df = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]
        X = df[common_locs].apply(pd.to_numeric, errors="coerce")
        corr = X.corr(method="pearson", min_periods=10)
        valid = ~corr.isna().all(axis=0)
        corr_clean = corr.loc[valid, valid]
        prefix = corr_clean.index.to_series().str[:3]
        return corr_clean.index[~prefix.duplicated()].tolist()

    def corr_matrix(path, y0, y1, kept):
        df = pd.read_csv(path)
        df["date"] = iso_week_to_datetime(df["year"], df["week"])
        df = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]
        X = df[[c for c in kept if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        return X.corr(method="pearson", min_periods=10).to_numpy(dtype=float)

    def group_mean(files, y0, y1, kept):
        mats = [corr_matrix(p, y0, y1, kept) for p in files]
        return np.nanmean(np.stack(mats, axis=0), axis=0)

    kept = _kept_labels(gcm_files[0], PERIODS[0][1], PERIODS[0][2])
    diff_mats = []
    for label, y0, y1 in PERIODS:
        diff_mats.append(group_mean(gcm_files, y0, y1, kept) - group_mean(knn_files, y0, y1, kept))
        print(f"Computed diff for {label}")

    all_vals = np.concatenate([m[np.isfinite(m)].ravel() for m in diff_mats])
    vmax = max(float(np.quantile(np.abs(all_vals), 0.99)) if all_vals.size else 1.0, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (label, _, _), mat in zip(axes, PERIODS, diff_mats):
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{label}\n(GCMs - KNN transient)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Location index"); ax.set_ylabel("Location index")

    fig.subplots_adjust(left=0.14, right=0.98, wspace=0.25)
    cax = fig.add_axes([0.06, 0.18, 0.02, 0.64])
    fig.colorbar(im, cax=cax).set_label("Delta Mean Pearson r  (GCMs - KNN transient)")
    fig.suptitle("Difference in mean cross-site temperature correlation", y=1.02)
    out = PLOT_OUT_DIR / "diff_spatial_corr_3panel_Tprom.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def block5_distance_decay_kde(var_flag=1):
    """Requires geopandas + WEAPNode.shp. var_flag: 0=temp, 1=precip."""
    import geopandas as gpd
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform

    print("\n=== Diagnostic 5: Distance-decay correlation KDE ===")
    VAR_NAME = {0: "Temperature", 1: "Precipitation"}
    X_MAX_KM = 125
    Y_MIN, Y_MAX = (0.85, 1.0) if var_flag == 0 else (0.5, 1.0)
    MIN_PAIRS = 30
    MONTH_ABBR = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}

    sites = gpd.read_file(NODES_SHP)
    if not all(sites.geometry.geom_type == "Point"):
        sites = sites.copy(); sites["geometry"] = sites.geometry.centroid
    sites = sites[[SITE_ID_FIELD, "geometry"]].rename(columns={SITE_ID_FIELD: "site_id"}).dropna()
    sites["site_id"] = sites["site_id"].astype(str)
    site_ids = sites["site_id"].to_numpy()
    n_sites = len(site_ids)
    site_to_idx = {sid: i for i, sid in enumerate(site_ids)}
    coords = np.column_stack([sites.geometry.x.to_numpy(), sites.geometry.y.to_numpy()])
    dist_full_km = squareform(pdist(coords)) / 1000.0

    var_key = "temp" if var_flag == 0 else "precip"
    all_files = list_gcm_files(var_key) + list_knn_files(var_key)
    print(f"Total files: {len(all_files)}")

    def pair_records(df):
        common = [c for c in df.columns if c in site_to_idx]
        if len(common) < 2: return pd.DataFrame()
        idx = np.array(sorted(site_to_idx[s] for s in common), dtype=int)
        cols = [site_ids[i] for i in idx]
        X = df[cols].to_numpy(dtype=float)
        month_arr = df.index.month.to_numpy()
        dist_sub = dist_full_km[np.ix_(idx, idx)]
        tri = np.triu_indices(len(idx), k=1)
        pair_id = idx[tri[0]] * n_sites + idx[tri[1]]
        dist_upper = dist_sub[tri]
        out = []
        for m in range(1, 13):
            Xm = X[month_arr == m]
            if Xm.shape[0] < MIN_PAIRS: continue
            C = np.corrcoef(Xm, rowvar=False)
            corr = C[tri]
            mask = ~np.isnan(corr)
            out.append(pd.DataFrame({"month": m, "pair_id": pair_id[mask], "distance_km": dist_upper[mask], "corr": corr[mask]}))
        return pd.concat(out) if out else pd.DataFrame()

    all_records = []
    for path in all_files:
        df = load_weekly_climate_csv(path)
        rec = pair_records(df)
        if not rec.empty:
            all_records.append(rec)

    all_df = pd.concat(all_records, ignore_index=True)
    avg_df = (all_df.groupby(["month", "pair_id"], as_index=False)
              .agg(distance_km=("distance_km", "first"), corr_mean=("corr", "mean")))
    avg_df["month_name"] = avg_df["month"].map(MONTH_ABBR)

    g = sns.FacetGrid(avg_df, col="month_name", col_wrap=4, height=3.2, aspect=1.1, sharex=True, sharey=True)
    filled_handles = []

    def panel(data, **kwargs):
        filled = sns.kdeplot(data=data, x="distance_km", y="corr_mean", fill=True, cmap="YlOrRd", thresh=0.02, levels=60)
        filled_handles.append(filled)
        sns.kdeplot(data=data, x="distance_km", y="corr_mean", fill=False, levels=6, linewidths=0.7, color="black", alpha=0.6)

    g.map_dataframe(panel)
    for ax in g.axes.flatten():
        ax.set_title(ax.get_title().split(" = ")[-1]); ax.grid(True, alpha=0.6)
    g.set_axis_labels("Distance [km]", "Mean correlation")
    g.set(xlim=(0, X_MAX_KM), ylim=(Y_MIN, Y_MAX))
    g.fig.subplots_adjust(right=0.88, top=0.92)
    if filled_handles and filled_handles[0].collections:
        cax = g.fig.add_axes([0.90, 0.20, 0.02, 0.60])
        g.fig.colorbar(filled_handles[0].collections[0], cax=cax).set_label("Relative density")

    g.fig.suptitle(f"Mean correlation vs. distance -- {VAR_NAME[var_flag]} (GCMs + KNN transient)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    out = PLOT_OUT_DIR / f"seasonal_kde_all_{var_key}.png"
    g.fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(g.fig)
    print(f"Saved: {out}")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print(f"Config check: {len(MODELS)} models x {len(EXPERIMENTS)} ssps x "
          f"{N_REPLICATES_PER_COMBO} replicates = {N_STATIONARY_TOTAL} scenarios "
          f"(target: 108) -> {'OK' if N_STATIONARY_TOTAL == 108 else 'MISMATCH -- check MODELS/EXPERIMENTS lists'}")

    print("\nLoading historical reference and GCM pairs ...")
    years, weeks_all, temp_values, precip_values, site_names = load_data(REFERENCE_TEMP_FILE, REFERENCE_PRECIP_FILE)
    temp_values_K = temp_values + 273.15   # internal convention carried over from prior scripts;
                                            # converted back to the reference's own units on write-out below
    obs_temp   = pd.read_csv(REFERENCE_TEMP_FILE)
    obs_precip = pd.read_csv(REFERENCE_PRECIP_FILE)

    combos = [(m, s) for m in MODELS for s in EXPERIMENTS]
    assert len(combos) * N_REPLICATES_PER_COMBO == N_STATIONARY_TOTAL

    print(f"\nGenerating all {N_STATIONARY_TOTAL} stationary KNN base series "
          f"(seed={RANDOM_SEED}, single call, independently drawn) ...")
    np.random.seed(RANDOM_SEED)
    raw_temp, raw_precip = generate_stochastic_series_block(
        years=years, weeks_all=weeks_all, temp_values=temp_values_K, precip_values=precip_values,
        N=N_STATIONARY_TOTAL, T=T_STATIONARY, n_neighbors=n_neighbors, block_max=block_max, occ_weight=occ_weight,
        wet_threshold=WET_THRESHOLD, dry_frac_enter=DRY_FRAC_ENTER,
        dry_frac_continue=FIXED["dry_frac_continue"], succ_dry_match=FIXED["succ_dry_match"],
        max_dry_walk=FIXED["max_dry_walk"],
    )
    stationary_temp = rescale_series(raw_temp, temp_values_K, weeks_all, N_STATIONARY_TOTAL)
    stationary_precip = rescale_series(raw_precip, precip_values, weeks_all, N_STATIONARY_TOTAL)

    if stationary_temp.shape[1] < T_OUT or stationary_precip.shape[1] < T_OUT:
        raise RuntimeError(f"Stationary series shorter than target horizon "
                            f"(got {stationary_precip.shape[1]}, need {T_OUT}).")
    stationary_temp = stationary_temp[:, :T_OUT, :]
    stationary_precip = stationary_precip[:, :T_OUT, :]

    print(f"\nWriting {N_STATIONARY_TOTAL} stationary (unforced) CSV pairs to {OUTPUT_DIR} ...")
    for gid in range(N_STATIONARY_TOTAL):
        temp_df_future = scenario_array_to_weekly_df(stationary_temp[gid] - 273.15, TARGET_YEARS, site_names)
        precip_df_future = scenario_array_to_weekly_df(stationary_precip[gid], TARGET_YEARS, site_names)
        temp_df_full = append_hist_to_transient_week(obs_temp, temp_df_future, BASELINE_END)
        precip_df_full = append_hist_to_transient_week(obs_precip, precip_df_future, BASELINE_END)
        temp_df_full.to_csv(OUTPUT_DIR / f"temp_stationary_scen{gid+1:03d}_1979_2099.csv", index=False)
        precip_df_full.to_csv(OUTPUT_DIR / f"precip_stationary_scen{gid+1:03d}_1979_2099.csv", index=False)
        if (gid + 1) % 20 == 0 or gid + 1 == N_STATIONARY_TOTAL:
            print(f"  {gid + 1}/{N_STATIONARY_TOTAL} stationary pairs written")

    print("\nComputing historical per-week-per-site mean anchors for MVA transform ...")
    hist_full_mask = np.ones_like(years, dtype=bool)  # years/weeks_all already span exactly REFERENCE_YEARS
    hist_temp_mean = weekly_climatology(temp_values_K, years, weeks_all, hist_full_mask)
    hist_log_precip = np.log(np.maximum(precip_values, 0.0) + PRECIP_LOG_EPS)
    hist_logp_mean = weekly_climatology(hist_log_precip, years, weeks_all, hist_full_mask)

    print("\nForcing each combo's 3 assigned stationary bases with that GCM/SSP's MVA change-factor vector "
          "(mean delta + std ratio; precip in log-space) ...")
    manifest_rows = []
    for ci, (model, ssp) in enumerate(combos):
        idxs = list(range(ci * N_REPLICATES_PER_COMBO, (ci + 1) * N_REPLICATES_PER_COMBO))
        temp_path = GCM_DIR / f"Tprom_semanal_{model}_{ssp}.csv"
        precip_path = GCM_DIR / f"PP_semanal_{model}_{ssp}.csv"
        if not temp_path.exists() or not precip_path.exists():
            warnings.warn(f"Missing GCM files for {model}/{ssp}; skipping this combo "
                          f"(stationary scenarios {[i+1 for i in idxs]} were still written above).",
                          RuntimeWarning, stacklevel=2)
            continue

        gcm_years_m, gcm_weeks_m, gcm_temp_m, gcm_precip_m, _ = load_gcm_weekly(temp_path, precip_path)
        temp_delta, temp_std_ratio, logp_delta, logp_std_ratio = moving_window_weekly_signal_mva(
            gcm_years_m, gcm_weeks_m, gcm_temp_m, gcm_precip_m,
            baseline_start=BASELINE_START, baseline_end=BASELINE_END,
            target_years=TARGET_YEARS, half_window=HALF_WINDOW,
        )

        temp_knn_sub = stationary_temp[idxs]
        precip_knn_sub = stationary_precip[idxs]
        temp_transient, precip_transient = apply_transient_signal_mva(
            temp_knn=temp_knn_sub, precip_knn=precip_knn_sub,
            temp_delta=temp_delta, temp_std_ratio=temp_std_ratio,
            logp_delta=logp_delta, logp_std_ratio=logp_std_ratio,
            hist_temp_mean=hist_temp_mean, hist_logp_mean=hist_logp_mean,
            target_years=TARGET_YEARS,
        )

        for r in range(N_REPLICATES_PER_COMBO):
            gid = idxs[r]
            temp_df_future = scenario_array_to_weekly_df(temp_transient[r] - 273.15, TARGET_YEARS, site_names)
            precip_df_future = scenario_array_to_weekly_df(precip_transient[r], TARGET_YEARS, site_names)
            temp_df_full = append_hist_to_transient_week(obs_temp, temp_df_future, BASELINE_END)
            precip_df_full = append_hist_to_transient_week(obs_precip, precip_df_future, BASELINE_END)

            temp_out = OUTPUT_DIR / f"temp_transient_{model}_{ssp}_scen{r+1:02d}_1979_2099.csv"
            precip_out = OUTPUT_DIR / f"precip_transient_{model}_{ssp}_scen{r+1:02d}_1979_2099.csv"
            temp_df_full.to_csv(temp_out, index=False)
            precip_df_full.to_csv(precip_out, index=False)

            manifest_rows.append(dict(
                model=model, ssp=ssp, replicate=r + 1,
                global_stationary_id=gid + 1,
                stationary_temp_file=f"temp_stationary_scen{gid+1:03d}_1979_2099.csv",
                stationary_precip_file=f"precip_stationary_scen{gid+1:03d}_1979_2099.csv",
                transient_temp_file=temp_out.name,
                transient_precip_file=precip_out.name,
                random_seed=RANDOM_SEED,
            ))
        print(f"  [{ci+1}/{len(combos)}] {model}/{ssp}: wrote replicates {[r+1 for r in range(N_REPLICATES_PER_COMBO)]} "
              f"(stationary ids {[i+1 for i in idxs]})")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUTPUT_DIR / "scenario_manifest.csv", index=False)
    print(f"\nWrote manifest for {len(manifest_df)} transient scenarios to "
          f"{OUTPUT_DIR / 'scenario_manifest.csv'}")

    print("\n" + "=" * 70)
    print("Generation complete. Running diagnostics ...")
    print("=" * 70)
    block1_temp_heatmaps()
    block2_precip_heatmaps()
    block3_monthly_tp_boxplots()
    block4_diff_heatmaps()
    # block5_distance_decay_kde(var_flag=0)   # temperature
    # block5_distance_decay_kde(var_flag=1)   # precipitation

    print("\nDone.")
