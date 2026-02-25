# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:23:23 2022

@author: Sebastian_Aedo
"""

import pandas as pd

def format_clima_HistCor_week(obs, cor, end_hist):
    df_obs = obs[obs.year < (end_hist + 1)]
    df_cor = cor[cor.year > end_hist]
    df_h = pd.concat([df_obs, df_cor])

    df_h.rename(columns={'year': 'year'}, inplace=True)

    df_h.index = range(len(df_h))
    #df_h.dropna(axis=1, how='any', inplace=True)  # remove missing columns

    return df_h


path_clima = '../Climate_series/'
end_hist = 2018

obs_3p = pd.read_csv(path_clima + 'Hist/MPC_Catchment_pp_1979-2019_week.csv')
obs_3t = pd.read_csv(path_clima + 'Hist/MPC_Catchment_tm_1979-2019_week.csv')

models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

for m in models:
    for x in experiments:

        df3_p = pd.read_csv(path_clima + f'UQM_TempC/MPC_pr_{m}_{x}_1986_2100_week_UQM.csv')
        df3_t = pd.read_csv(path_clima + f'UQM_TempC/MPC_t2m_{m}_{x}_1986_2100_week_UQM.csv')

        df3_ph = format_clima_HistCor_week(obs_3p, df3_p, end_hist)
        df3_th = format_clima_HistCor_week(obs_3t, df3_t, end_hist)

        df3_ph.to_csv(path_clima + f'UQM_hist_TempC_new/MPC_pr_{m}_{x}_1979_2100_week_UQM_hist.csv', index=False)
        df3_th.to_csv(path_clima + f'UQM_hist_TempC_new/MPC_t2m_{m}_{x}_1979_2100_week_UQM_hist.csv', index=False)



