# format UQM_TempC files for use with WEAP Cordillera model
import numpy as np
import pandas as pd

path_clima = '../Climate_series/'

models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

Tprom_ref =pd.read_csv(path_clima + 'Tprom_semanal.csv')
PP_ref = pd.read_csv(path_clima + 'PP_semanal.csv')

for m in models:
    for x in experiments:

        df3_p = pd.read_csv(path_clima + f'UQM_hist_TempC_new/MPC_pr_{m}_{x}_1979_2100_week_UQM_hist.csv')
        df3_p.fillna(method='ffill', inplace=True)
        PP_df = pd.concat([pd.DataFrame(columns=PP_ref.columns), df3_p])
        PP_df.to_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv', index=False)

        df3_t = pd.read_csv(path_clima + f'UQM_hist_TempC_new/MPC_t2m_{m}_{x}_1979_2100_week_UQM_hist.csv')
        df3_t.fillna(method='ffill', inplace=True)
        Tprom_df = pd.concat([pd.DataFrame(columns=Tprom_ref.columns), df3_t])
        Tprom_df.to_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv', index=False)


#%% test empty columns

models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

cols_req = pd.read_csv(path_clima + f'/UQM/MPC_pr_ACCESS-CM2_ssp126_1986_2100_week_UQM.csv').columns[2::]

for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        #t = pd.read_csv(path_clima + f'UQM_hist_TempC_new/MPC_t2m_{m}_{x}_1979_2100_week_UQM_hist.csv')

        #t.fillna(axis=0, method='ffill', inplace=True)
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC/PP_semanal_{m}_{x}.csv')

        print(f'{m}_{x}:')
        print(f'Columns: {t.shape[1]}')
        print(f'Any Cols with Nans: {t[cols_req].isna().any().sum()}')
        print(f'Req Cols num Nans: {t[cols_req].isna().any().sum()}')
        print(f'Req Cols with Nans: {cols_req[t[cols_req].isna().any() == True]}')
        print(f'Min, max: {t[cols_req].min().min()}, {t[cols_req].max().max()}')




#%% make time series plots
import matplotlib.pyplot as plt

plt.figure()
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        plt.plot(ts, t.iloc[:, 3], label=f'{m}_{x}')

#plt.legend()
plt.show()

#%% Rolling time series individual figures
import matplotlib.pyplot as plt
import datetime

fig = plt.figure(figsize=(6, 3))
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        l, = plt.plot(ts, t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean(), label=f'{m}_{x}', color='steelblue')

#plt.legend()
hist_start = ts[ts.dt.year >= 1999].index[0]
hist_end = ts[ts.dt.year >= 2020].index[0]
h, = plt.plot(ts.iloc[hist_start:hist_end], t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
plt.title('20-Year Mean Temperature', fontsize=20)
plt.ylabel(r'20-Year Moving Average \n Temperature ($\degree$ C)', fontsize=15)
legend = plt.legend([l, h], ['GCM projections', "Historical"], frameon=False)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor('0.9')
plt.xlabel('Date', fontsize=15)
plt.xlim([ts[ts.dt.year >= 1999].iloc[0], ts[ts.dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
#plt.ylim([0, 13])
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.tight_layout()
plt.show()


fig = plt.figure(figsize=(6, 3))
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        l, = plt.plot(ts, t.iloc[:, 3].shift().rolling(5*52, min_periods=3).mean(), label=f'{m}_{x}', color='steelblue')

#plt.legend()
hist_start = ts[ts.dt.year >= 1999].index[0]
hist_end = ts[ts.dt.year >= 2020].index[0]
h, = plt.plot(ts.iloc[hist_start:hist_end], t.iloc[:, 3].shift().rolling(5*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
plt.title('20-Year Mean Precipitation')
plt.ylabel(r'20-Year Mean Precipitation (mm/week)')
#plt.legend([l, h], ['GCM projections', "Historical"])
plt.xlabel('Date')
plt.xlim([ts[ts.dt.year >= 1999].iloc[0], ts[ts.dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
#plt.xlim([0, 52*20])
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()

#%% #%% Rolling time series as subplot (temp vs. precip)
import matplotlib.pyplot as plt
import datetime

fig = plt.figure(figsize=(9, 6))
plt.subplot(3, 1, 1)
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        l, = plt.plot(ts, t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean(), label=f'{m}_{x}', color='steelblue')

#plt.legend()
hist_start = ts[ts.dt.year >= 1999].index[0]
hist_end = ts[ts.dt.year >= 2020].index[0]
h, = plt.plot(ts.iloc[hist_start:hist_end], t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
plt.title('Temperature vs. Time', fontsize=15)
plt.ylabel(r'Temperature ($\degree$ C)', fontsize=15)
legend = plt.legend([l, h], ['GCM projections', "Historical"], frameon=False, fontsize=12)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor('0.9')
#plt.xlabel('Date', fontsize=15)
plt.xlim([ts[ts.dt.year >= 1999].iloc[0], ts[ts.dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
#plt.ylim([0, 13])
plt.xticks([])
plt.yticks(fontsize=12)
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.tight_layout()


plt.subplot(3, 1, 2)
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        l, = plt.plot(ts, t.iloc[:, 3].shift().rolling(5*52, min_periods=3).mean(), label=f'{m}_{x}', color='steelblue')

#plt.legend()
hist_start = ts[ts.dt.year >= 1999].index[0]
hist_end = ts[ts.dt.year >= 2020].index[0]
plt.xticks(fontsize=10)
plt.yticks([0, 5, 10, 20], [0, 5, 10, 20], fontsize=12)
h, = plt.plot(ts.iloc[hist_start:hist_end], t.iloc[:, 3].shift().rolling(5*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
#plt.title('20-Year Mean Precipitation')
plt.title('Precipitation vs. Time', fontsize=15)
#plt.ylabel(r'20-Year Mean Precipitation (mm/week)')
plt.ylabel(r'Precip. (mm/week)', fontsize=15)
plt.ylim([0,20])
#plt.legend([l, h], ['GCM projections', "Historical"])
plt.xlim([ts[ts.dt.year >= 1999].iloc[0], ts[ts.dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
#plt.xlim([0, 52*20])
plt.xlabel('Time', fontsize=15)
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#plt.suptitle('20-Year Moving Average of Climate Variable', fontsize=12)
fig.supylabel('20-Year Moving Average', fontsize=15)

fig.tight_layout()
plt.show()


#%% Showing how glacier melt contributions to streamflow will change over the century

files = ['COLORADO', 'MAIPO', 'YESO', 'LAGUNANEGRA', 'VOLCAN', 'MAIPOEXTRA']
scenarios = pd.read_csv('../TDP_scenarios/2004-2040/COLORADO_2004Wk1-2040Wk52.csv').columns[1::]
#ssp_colors = ['midnightblue', 'steelblue', 'chocolate', 'firebrick']
ssp_colors = ['midnightblue', 'steelblue', 'chocolate', 'firebrick']
ssp_colors = ['#B4CAD2', '#597797', '#8C2620', '#C67B78']
ssp_colors = ['#597797', '#8C2620']

I_tot =pd.read_csv(f'../TDP_scenarios/{files[0]}_2004Wk14-2099Wk13.csv', index_col=0)
for f in files[1::]:
    I_tot[scenarios] = I_tot[scenarios] + pd.read_csv(f'../TDP_scenarios/{f}_2004Wk14-2099Wk13.csv', index_col=0)[scenarios]

# lookup dataframe to convert between time series conventions
Timestamp_convert = pd.DataFrame({'Timestamp': I_tot.index.values})
start_date = datetime.datetime.strptime(Timestamp_convert['Timestamp'][0], '%d-%m-%Y')
week_no = np.array(range(len(Timestamp_convert)))
Timestamp_convert.insert(0, 'Timestamp_', start_date + week_no * datetime.timedelta(days=7))

# scaled glacier melt
# plt.figure()
# plt.plot(I_tot)
# plt.show()

n = 0.019405960290035287
gla = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/CMIP6_Glacier_Melt_TDP.csv', index_col=0)
gla[scenarios] = n* gla[scenarios]
gla_subsettime = gla.iloc[1313:1313+52*95]

f = plt.figure(figsize=(14, 4.5))

# shaded, rolling mean
# m_rolling_ssp126 = gla.iloc[:,gla.columns.str.contains("ssp126")].shift().rolling(20*52, min_periods=5).mean()
# m_rolling_ssp585 = gla.iloc[:,gla.columns.str.contains("ssp585")].shift().rolling(20*52, min_periods=5).mean()
#
# s_rolling_ssp126 = I_tot.iloc[:, I_tot.columns.str.contains("ssp126")].shift().rolling(20*52, min_periods=5).mean()
# s_rolling_ssp585 = I_tot.iloc[:, I_tot.columns.str.contains("ssp585")].shift().rolling(20*52, min_periods=5).mean()

m_rolling = gla.iloc[:,1::].shift().rolling(20*52, min_periods=5).mean()
s_rolling = I_tot.iloc[:, :].shift().rolling(20*52, min_periods=5).mean()


plt.subplot(1, 2, 1)

# std and mean lines
plt.plot(0, 0, color='black', label='Mean')
plt.plot(0, 0, color='black', linestyle='--', label='Std. dev.')

# plot glacier melt and uncertainty bars (ssp126)
plt.plot(gla['Date'].iloc[1313:1313+52*95], m_rolling.iloc[1313:1313+52*95].T.mean(), color=ssp_colors[1], linewidth=2.5, label='Glacier melt')
plt.plot(gla['Date'].iloc[1313:1313+52*95],
                 m_rolling.iloc[1313:1313+52*95].T.mean() - m_rolling.iloc[1313:1313+52*95].T.std(),
         color=ssp_colors[1], linestyle='--', linewidth=1.5)
plt.plot(gla['Date'].iloc[1313:1313+52*95],
                 m_rolling.iloc[1313:1313+52*95].T.mean() + m_rolling.iloc[1313:1313+52*95].T.std(),
         color=ssp_colors[1], linestyle='--', linewidth=1.5)
plt.fill_between(gla['Date'].iloc[1313:1313+52*95],
                 m_rolling.iloc[1313:1313+52*95].T.mean() - m_rolling.iloc[1313:1313+52*95].T.std(),
                 m_rolling.iloc[1313:1313+52*95].T.mean() + m_rolling.iloc[1313:1313+52*95].T.std(),
                 color=ssp_colors[1], alpha=0.2)

# plt total flow and uncertainty bars (ssp126)
plt.plot(gla['Date'].iloc[1313:1313+52*95], s_rolling.T.mean(), color=ssp_colors[0], linewidth=2.5, label='Total streamflow')
plt.plot(gla['Date'].iloc[1313:1313+52*95],
                 s_rolling.T.mean() - s_rolling.T.std(),
         color=ssp_colors[0], linestyle='--', linewidth=1.5)
plt.plot(gla['Date'].iloc[1313:1313+52*95],
                 s_rolling.T.mean() + s_rolling.T.std(),
         color=ssp_colors[0], linestyle='--', linewidth=1.5)
plt.fill_between(gla['Date'].iloc[1313:1313+52*95],
                 s_rolling.T.mean() - s_rolling.T.std(),
                 s_rolling.T.mean() + s_rolling.T.std(),
                 color=ssp_colors[0], alpha=0.35)

# fill between glacier melt and total flow (ssp126)
plt.fill_between(gla['Date'].iloc[1313:1313+52*95],
                 s_rolling.T.mean() + s_rolling.T.std(),
                 m_rolling.iloc[1313:1313+52*95].T.mean() + m_rolling.iloc[1313:1313+52*95].T.std(),
                 color=ssp_colors[0], alpha=0.35)
plt.fill_between(gla['Date'].iloc[1313:1313+52*95],
                 m_rolling.iloc[1313:1313+52*95].T.mean() + m_rolling.iloc[1313:1313+52*95].T.std(),
                 np.zeros(len(m_rolling.iloc[1313:1313+52*95])),
                 color=ssp_colors[1], alpha=0.35)
#plt.title('Glacier Melt and Streamflow vs. Time', fontsize=15)
#plt.xlabel('Time', fontsize=14)
plt.xticks(np.unique(pd.to_datetime(gla['Date'].iloc[1313:1313+52*95]).dt.year, return_index=True)[1][6::20],
           np.unique(pd.to_datetime(gla['Date'].iloc[1313:1313+52*95]).dt.year, return_index=True)[0][6::20])
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([gla['Date'].iloc[1611], gla['Date'].iloc[-1]])  # 2010-2099
#plt.ylabel('20-year Moving Average of Flow (Mm3/week)', fontsize=14)
plt.ylabel('Streamflow (Mm3/week)', fontsize=14)
legend = plt.legend(fontsize=12, frameon=False)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor('0.9')
plt.tight_layout()
# plt.grid(True, color='w')
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax = plt.gca()
ymax = ax.get_ylim()[1]
plt.ylim([0, ymax])
plt.grid(False)

plt.show()
