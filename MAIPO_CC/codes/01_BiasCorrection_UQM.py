import matplotlib.pylab as plt

from climQMBC.methods import UQM
import pandas as pd
import numpy as np

def format_mod(mod):
    mod_format = mod.copy()
    mod_format.rename(columns={'Mes':'month'}, inplace=True)
    mod_format.index = pd.to_datetime(mod_format[['Year', 'month', 'day']])
    mod_format = mod_format.iloc[:,3:]
    return mod_format
    
def get_tmed_mod(path_mod_tx, path_mod_tn):
    mod_tx = pd.read_csv(path_mod_tx)
    # mod_tx = format_mod(mod_tx)
    mod_tx = format_mod(mod_tx[mod_tx['Year'] <= 2100])

    
    mod_tn = pd.read_csv(path_mod_tn)
    # mod_tn = format_mod(mod_tn)
    mod_tn = format_mod(mod_tn[mod_tn['Year'] <= 2100])
    
    return 0.5*(mod_tn+mod_tx)
    

path_clima = '../Climate_series/'
bc_start = '1986'
bc_end = '2014'
# bc_start = '1979'
# bc_end = '2100'

# %% Modelo Maipo Cordillera
def update_matching_week_datetimeindex(df, resample):
    df_mod = df.copy()
    # Infer first and final week and starting week
    first_year_int = df_mod.iloc[0,0]
    final_year_int = df_mod.iloc[-1,0]
    first_week_int = df_mod.iloc[0,1]
    
    # Get a list with datetime indexes restarting every year to match
    # starting date for the starting week
    date_li = []
    for year in range(first_year_int, final_year_int+1):
        df_index_temp = pd.date_range(start=str(year),
                                      end=str(year+1),
                                      freq='W')
    
        first_date_bool = df_index_temp.isocalendar().week.isin([first_week_int])
        first_date = df_index_temp[first_date_bool][0]
    
        df_index = pd.date_range(start=first_date,
                                  freq='W',
                                  periods=52)
        
        date_li = date_li + df_index.tolist()
    
    # Update index and resample as mean
    df_mod.index = date_li[:len(df_mod)]
    if resample=='sum':
        df_mod = df_mod.resample('M').sum()
    elif resample=='mean':
        df_mod = df_mod.resample('M').mean()
    else:
        df_mod = df_mod.resample('M').mean()
    
    return df_mod


path_obs_pp = path_clima + 'Hist/MPC_Catchment_pp_1979-2019_week.csv'
path_obs_tm = path_clima + 'Hist/MPC_Catchment_tm_1979-2019_week.csv'

obs_pp = pd.read_csv(path_obs_pp)
obs_tm = pd.read_csv(path_obs_tm)

models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # note: 'historical' experiment exists
models = ['MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # note: 'historical' experiment exists

for m in models:
    for x in experiments:
        path_mod_pp = path_clima + f'ARCLIM/pr_{m}_{x}_Agua_1970-2100_day.csv'
        path_mod_tx = path_clima + f'ARCLIM/tasmax_{m}_{x}_Agua_1970-2100_day.csv'
        path_mod_tn = path_clima + f'ARCLIM/tasmin_{m}_{x}_Agua_1970-2100_day.csv'

        mod_pp = pd.read_csv(path_mod_pp)
        # mod_pp = format_mod(mod_pp)
        mod_pp = format_mod(mod_pp[mod_pp['Year'] <= 2100])
        mod_pp = mod_pp.resample('W').sum()

        mod_iso = mod_pp.index.isocalendar()
        mod_pp = mod_pp[mod_iso.week<53]

        mod_tm = get_tmed_mod(path_mod_tx, path_mod_tn)
        mod_tm = mod_tm.resample('W').mean()

        mod_iso = mod_tm.index.isocalendar()
        mod_tm = mod_tm[mod_iso.week < 53]

        for var in [1,0]:
            cor = pd.DataFrame(index=range(52*(2100-int(bc_start))))
            cor['year'] = mod_tm.loc[bc_start:'2099'].index.isocalendar().year.tolist()
            cor['week'] = mod_tm.loc[bc_start:'2099'].index.isocalendar().week.tolist()

            if var == 1:
                obs = obs_pp
                mod = mod_pp
            else:
                obs = obs_tm
                mod = mod_tm

            for catch in obs.keys():
            # for catch in mod.keys():
                if ('year' not in catch) and ('week' not in catch):
                    if 'HUA' not in catch:
                        if catch in mod.keys():  # add this statement since mod has subset of catchments from obs
                            catch_mod = catch[:3]
                            print(catch, catch_mod in mod.keys())

                            obs_bc = obs[catch][obs.year.isin([i for i in range(int(bc_start), int(bc_end)+1)])].to_numpy()
                            # mod_bc = mod[catch_mod].loc[bc_start:'2099'].to_numpy()
                            mod_bc = mod[catch].loc[bc_start:'2099'].to_numpy()

                            if var==1:
                                th = 1/100000
                                obs_bc[obs_bc<th] = np.random.rand(len(obs_bc[obs_bc<th]))*th
                                mod_bc[mod_bc<th] = np.random.rand(len(mod_bc[mod_bc<th]))*th

                            cor[catch] = UQM(obs_bc,mod_bc,var,frq='W')

            if var==1:
                cor.to_csv(path_clima + f'UQM/MPC_pr_{m}_{x}_{bc_start}_2100_week_UQM.csv', index=False)
                print(f'Precipitation downscaling completed for {m}_{x}')
            else:
                cor.to_csv(path_clima + f'UQM/MPC_t2m_{m}_{x}_{bc_start}_2100_week_UQM.csv', index=False)
                print(f'Temperature downscaling completed for {m}_{x}')
