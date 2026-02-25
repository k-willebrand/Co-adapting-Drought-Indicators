import numpy as np
import pandas as pd
import random
from sklearn import linear_model
import matplotlib.pyplot as plt

#%% Preprocessing Glacier_series from Ayala et al. (2022)

scenarios = np.arange(10) + 1  # committed ice loss scenarios
glaciers = pd.read_csv('../Glacier_geometry.csv')

# initialize master dataframes
df_ref = pd.read_csv(f'../Glacier_series/{glaciers.OBJECTID[0]}_{scenarios[0]}.csv')
df_T = pd.DataFrame()
df_P = pd.DataFrame()
df_S = pd.DataFrame()
df_G = pd.DataFrame()

for i, s in enumerate(scenarios):
    # initialize intermediate dataframes
    df_Tij = df_ref[['YYYY', 'MM', 'DD']]
    df_Tij['CC'] = s * np.ones(len(df_Tij)).astype(int)
    df_Pij = df_ref[['YYYY', 'MM', 'DD']]
    df_Pij['CC'] = s * np.ones(len(df_Pij)).astype(int)
    df_Sij = df_ref[['YYYY', 'MM', 'DD']]
    df_Sij['CC'] = s * np.ones(len(df_Sij)).astype(int)
    df_Gij = df_ref[['YYYY', 'MM', 'DD']]
    df_Gij['CC'] = s * np.ones(len(df_Gij)).astype(int)

    for j, g in enumerate(glaciers['OBJECTID']):
        # load data
        df = pd.read_csv(f'../Glacier_series/{g}_{s}.csv')

        # add glacier data to dataframes
        df_Tij[g] = df['RefHAirTemp'].values + 273.15  # deg. C to K
        df_Pij[g] = np.log(df['Prec'].values + 1E-6)
        #df_Pij[g] = df['Prec'].values
        df_Sij[g] = df['MeltS'].values  # mm to m
        df_Gij[g] = df['MeltG'].values  # mm to m

    # append master dataframes
    df_T = pd.concat([df_T, df_Tij], ignore_index=True)
    df_P = pd.concat([df_P, df_Pij], ignore_index=True)
    df_S = pd.concat([df_S, df_Sij], ignore_index=True)
    df_G = pd.concat([df_G, df_Gij], ignore_index=True)

    del df_Tij, df_Pij, df_Sij, df_Gij

# aggregate across 31 glaciers to monthly
df_T = df_T.groupby(['YYYY', 'MM', 'CC']).mean().drop(['DD'], axis=1).mean(axis=1).reset_index()
df_P = df_P.groupby(['YYYY', 'MM', 'CC']).mean().drop(['DD'], axis=1).mean(axis=1).reset_index()
df_S = df_S.groupby(['YYYY', 'MM', 'CC']).sum().drop(['DD'], axis=1).mean(axis=1).reset_index()
df_G = df_G.groupby(['YYYY', 'MM', 'CC']).sum().drop(['DD'], axis=1).mean(axis=1).reset_index()
# df_G = df_G.groupby(['YYYY', 'MM', 'CC']).mean().drop(['DD'], axis=1).mean(axis=1).reset_index()

df_G[0]  = np.log(df_G[0] + 1E-6)  # for log transform

# df_G_cumsum = df_G
# for i, s in enumerate(scenarios):
#     df_G_cumsum.iloc[df_G.groupby(['CC']).get_group(s)[0].index, -1] = df_G.groupby(['CC']).get_group(s)[0].cumsum()

# also make table by climate change scenario
df_P_CC = pd.concat((pd.DataFrame(data=pd.to_datetime(df_P.groupby('CC').get_group(1)['YYYY'].astype(str) + df_P.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(df_P.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)
df_T_CC = pd.concat((pd.DataFrame(data=pd.to_datetime(df_P.groupby('CC').get_group(1)['YYYY'].astype(str) + df_P.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(df_P.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)
for i, s in enumerate(scenarios):
    df_T_CC[s] = df_T.groupby('CC').get_group(s)[0].reset_index(drop=True)
    df_P_CC[s] = df_P.groupby('CC').get_group(s)[0].reset_index(drop=True)

# dataframe to train regression model
T_ref = 5 + 273.15  # annual estimate of mean T in K from 1955-1970 from Ayala et al. (2020)
P_ref = 2.74  # annual estimate of mean P in mm/d from 1955-1970 from Ayala et al. (2020)

# alternatively, consider monthly estimate from data
T_ref = df_T.loc[df_T['YYYY'] <= 2003].groupby(['MM']).mean()[0]
P_ref = df_P.loc[df_T['YYYY'] <= 2003].groupby(['MM']).mean()[0]
#P_ref =np.log(df_P.loc[df_T['YYYY'] <= 2003]).groupby(['MM']).mean()[0]
P_std = df_P.loc[df_T['YYYY'] <= 2003].groupby(['MM']).std()[0]

# plot monthly estimates
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(P_ref, c='steelblue', label='Precipitation')
ax1.scatter(np.arange(12)+1, P_ref, c='steelblue', label='Precipitation')
plt.ylabel('Precipitation (mm/d)', c='steelblue')
plt.xlabel('Month')
plt.xticks(np.arange(12)+1)
ax1.spines['left'].set_color('steelblue')
ax1.yaxis.label.set_color('steelblue')
ax1.tick_params(axis='y', colors='steelblue')
ax2 = ax1.twinx()
ax2.plot(T_ref-273.15, c='chocolate', label='Temperature')
ax2.scatter(np.arange(12)+1, T_ref-273.15, c='chocolate', label='Precipitation')
plt.ylabel('Temperature (deg. C)')
plt.title('Reference Temperature and Precipitation (2000-2003)')
ax2.spines['right'].set_color('chocolate')
ax2.yaxis.label.set_color('chocolate')
ax2.tick_params(axis='y', colors='chocolate')
plt.show()

# calculate anomalies
df_T['Anom'] = df_T[0]
df_P['Anom'] = df_P[0]
for m in range(12):
    df_T.iloc[df_T.groupby('MM').get_group(m+1).index, -1] = df_T.groupby('MM').get_group(m+1)[0] - T_ref.iloc[m]
    #df_P.iloc[df_P.groupby('MM').get_group(m + 1).index, -1] = np.log(df_P.groupby('MM').get_group(m + 1)[0])
    df_P.iloc[df_P.groupby('MM').get_group(m+1).index, -1] = (df_P.groupby('MM').get_group(m+1)[0] - P_ref.iloc[m])/P_ref.iloc[m]

# df_P['Anom'] = np.log(df_P['Anom'].values - min(df_P['Anom'].values) + 1E-6)

# df_ = pd.DataFrame([df_T['CC'], df_T['MM'], df_T[0], df_P[0], df_G[0]]).transpose()
#df_ = pd.DataFrame([df_T['CC'], df_T['YYYY'], df_T['MM'], df_T['Anom'], df_P['Anom'], df_G[0]]).transpose()
df_ = pd.DataFrame([df_T['CC'], df_T['YYYY'], df_T['MM'], df_T['Anom'], df_P['Anom'], df_G[0]]).transpose()
df_.columns = ['CC', 'YYYY', 'MM', 'Temp', 'Prec', 'MeltG']
df_['YYYY'] = df_['YYYY'].astype(int)
df_['MM'] = df_['MM'].astype(int)
df_['CC'] = df_['CC'].astype(int)

#%% plot of melt G over time for each CC scenario
MeltG_plot = pd.concat((pd.DataFrame(data=pd.to_datetime(df_.groupby('CC').get_group(1)['YYYY'].astype(str) + df_.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(df_.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)
for i, s in enumerate(scenarios):
    MeltG_plot[s] = df_.groupby('CC').get_group(s)['MeltG'].reset_index(drop=True)

plt.figure(figsize=(15, 3))
plt.subplot(1, 2, 1)
plt.plot(MeltG_plot['Date'], MeltG_plot.iloc[:, 1::], label=MeltG_plot.columns[1::])
plt.ylabel(r'$\Delta$ b')
plt.xlabel('Time')
plt.legend(title="Climate Scenario", ncol=2, bbox_to_anchor=(-0.12, 0.5), loc="center right", borderaxespad=0)
plt.title('Total MeltG vs. Time')
plt.subplot(1, 2, 2)
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']
bplot1 = plt.boxplot(MeltG_plot.iloc[:, 1::], patch_artist=True)
plt.ylabel(r'$\Delta$ b')
plt.xlabel('Committed Ice Loss Scenario')
plt.title('Total MeltG vs. Committed Ice Loss Scenario')
for bplot in bplot1:
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        patch._set_edgecolor(color)
    for i in range(10):
        bplot1['whiskers'][2*i].set_color(colors[i])
        bplot1['whiskers'][2*i +1].set_color(colors[i])
    for i in range(10):
        bplot1['caps'][2*i].set_c(colors[i])
        bplot1['caps'][2*i +1].set_c(colors[i])
    for i in range(10):
        bplot1['fliers'][i].set_markeredgecolor(colors[i])
    for i in range(10):
        bplot1['medians'][i].set_c('black')
plt.gcf().subplots_adjust(bottom=0.06, left=0.08)
plt.tight_layout()
plt.show()


#%% correlation plots/anomalies
plt.figure()

plt.subplot(2, 2, 1)
plt.scatter(df_T[0], df_G[0], c='chocolate')
plt.xlabel(r'$T_i$')
#plt.ylabel(r'$\Delta$ b')
plt.ylabel(r'$log \Delta$ b')

plt.subplot(2, 2, 2)
plt.scatter(df_P[0], df_G[0], c='steelblue')
plt.xlabel(r'log $P_i$')
#plt.xlabel(r'$P_i$')
#plt.ylabel(r'$\Delta$ b')
plt.ylabel(r'$log \Delta$ b')

plt.subplot(2, 2, 3)
plt.scatter(df_T['Anom'], df_G[0], c='chocolate')
plt.xlabel(r'$T_{i} - T_{ref}$')
# plt.ylabel(r'$\Delta$ b')
plt.ylabel(r'$log \Delta$ b')

plt.subplot(2, 2, 4)
plt.scatter(df_P['Anom'], df_G[0], c='steelblue')
#plt.xlabel(r'$log ((P_i - P_{ref})/ P_{ref})$')
plt.xlabel(r'$(log P_i - log P_{ref})/log P_{ref}$')
#plt.xlabel(r'$(P_i - P_{ref})/P_{ref}$')
#plt.ylabel(r'$\Delta$ b')
plt.ylabel(r'$log \Delta$ b')

plt.tight_layout()
plt.show()



#%% fit linear regression with 80-20 train/test

#df_ = df_.groupby('CC').get_group(1)

gamma = 1.375  # constant from literature
c = 0.034  # global average V-A parameter
rho_I = 850  # density of ice in kg/m3
rho_w = 997  # density of water in kg/m3

cm = pd.DataFrame(columns=['cT', 'cP'])

random.seed(0)
n = len(df_.groupby('MM').get_group(1))  # number of observations for month
train_idx = random.sample(range(n), k=round(0.8*n))
test_idx = list(set(np.arange(n)) - set(train_idx))

train_pred = np.zeros([12, len(train_idx), 2])
test_pred = np.zeros([12, len(test_idx), 2])

for m in range(12):

    X_train = df_.groupby('MM').get_group(m + 1).reset_index().loc[train_idx, ['Temp','Prec']]
    y_train = df_.groupby('MM').get_group(m + 1).reset_index().loc[train_idx, ['MeltG']]
    X_test = df_.groupby('MM').get_group(m + 1).reset_index().loc[test_idx, ['Temp', 'Prec']]
    y_test = df_.groupby('MM').get_group(m + 1).reset_index().loc[test_idx, ['MeltG']]

    # fit model and get coeffs
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    cm = pd.concat([cm, pd.DataFrame(data={'cT': [regr.coef_[0][0]], 'cP': [regr.coef_[0][1]]})])

    # apply to train data
    train_pred[m, :, :] = np.array([y_train.values, regr.predict(X_train)]).transpose().squeeze()
    test_pred[m, :, :] = np.array([y_test.values, regr.predict(X_test)]).transpose().squeeze()
cm.index = np.arange(12) + 1

# plot fitted parameters
fig = plt.figure()
w = 0.4  # bar width
plt.bar(np.arange(12)+1-w/2, cm['cT'], w, label=r'$c_i^T$', color='chocolate')
plt.bar(np.arange(12)+1+w/2, cm['cP'], w, label=r'$c_i^P$', color='steelblue')
l = plt.axhline(0, color='black')
l.set_linewidth(1)
plt.xticks(np.arange(12)+1)
plt.xlabel('Month')
plt.ylabel('Coefficient')
plt.title('Fitted SSC Coefficients')
plt.legend()
plt.xlim([1-1.5*w, 12+1.5*w])
plt.show()

#%% perfomance of fitted model

#train_pred[:, :, :] = np.exp(train_pred[:, :, :]) - 1E-6
#test_pred[:, :, :] = np.exp(test_pred[:, :, :]) - 1E-6

# train error
fig = plt.figure(figsize=[9, 9])
for m in range(12):
    plt.subplot(4, 3, m + 1)
    plt.plot(train_pred[m, :, 0], train_pred[m, :, 0], label='perfect', c='chocolate')
    #plt.scatter(train_pred[m, :, 0], train_pred[m, :, 0], label='perfect', c='chocolate', s=7)
    plt.scatter(train_pred[m, :, 0], train_pred[m, :, 1], label='predicted', c='steelblue', s=4)
    #plt.xlabel(r'Target $\Delta$ b')
    #plt.ylabel(r'Predicted $\Delta$ b')
    plt.title(f'Climate Change: {m+1}')
    plt.annotate('n=792', xy=(0.1, 0.8), xycoords='axes fraction', fontsize=10)
plt.suptitle('Training Data by Month')
fig.tight_layout()
# fig.supxlabel(r'Target $\Delta$ b')
# fig.supylabel(r'Predicted $\Delta$ b')
fig.supxlabel(r'Target log $\Delta$ b')
fig.supylabel(r'Predicted log $\Delta$ b')
plt.gcf().subplots_adjust(bottom=0.06, left=0.08)
plt.show()

plt.figure(figsize=[6, 6])
plt.plot(train_pred[:, :, 0], train_pred[:, :, 0], label='perfect', c='chocolate')
#plt.scatter(train_pred[:, :, 0], train_pred[:, :, 0], label='perfect', c='chocolate', s=7)
plt.scatter(train_pred[:, :, 0], train_pred[:, :, 1], label='predicted', c='steelblue', s=4)
# plt.xlabel(r'Target $\Delta$ b')
# plt.ylabel(r'Predicted $\Delta$ b')
plt.xlabel(r'Target log $\Delta$ b')
plt.ylabel(r'Predicted log $\Delta$ b')
plt.title(f'Training Data')
plt.annotate('n=9504', xy=(0.2, 0.8), xycoords='figure fraction', fontsize=14)
plt.tight_layout()
plt.show()

# training R2
RSS_train = np.sum((train_pred[:, :, 0] - train_pred[:, :, 1])**2)  # actual - pred
TSS_train = np.sum((train_pred[:, :, 0] - np.mean(train_pred[:, :, 0]))**2)  # actual - mean
R2_train = 1 - (RSS_train/TSS_train)
print(f'R2_train: {R2_train}')

# test data
fig = plt.figure(figsize=[9, 9])
for m in range(12):
    plt.subplot(4, 3, m + 1)
    plt.plot(test_pred[m, :, 0], test_pred[m, :, 0], label='perfect', c='chocolate')
    #plt.scatter(test_pred[m, :, 0], test_pred[m, :, 0], label='perfect', c='chocolate', s=7)
    plt.scatter(test_pred[m, :, 0], test_pred[m, :, 1], label='predicted', c='steelblue', s=4)
    #plt.xlabel(r'Target $\Delta$ b')
    #plt.ylabel(r'Predicted $\Delta$ b')
    plt.title(f'Climate Change: {m+1}')
    plt.annotate('n=198', xy=(0.1, 0.8), xycoords='axes fraction', fontsize=10)
plt.suptitle('Testing Data by Month')
fig.tight_layout()
# fig.supxlabel(r'Target $\Delta$ b')
# fig.supylabel(r'Predicted $\Delta$ b')
fig.supxlabel(r'Target log $\Delta$ b')
fig.supylabel(r'Predicted log $\Delta$ b')
plt.gcf().subplots_adjust(bottom=0.06, left=0.08)
plt.show()

plt.figure(figsize=[6, 6])
plt.plot(test_pred[:, :, 0], test_pred[:, :, 0], label='perfect', c='chocolate')
#plt.scatter(test_pred[:, :, 0], test_pred[:, :, 0], label='perfect', c='chocolate', s=7)
plt.scatter(test_pred[:, :, 0], test_pred[:, :, 1], label='predicted', c='steelblue', s=4)
# plt.xlabel(r'Target $\Delta$ b')
# plt.ylabel(r'Predicted $\Delta$ b')
plt.xlabel(r'Target log $\Delta$ b')
plt.ylabel(r'Predicted log $\Delta$ b')
plt.title(f'Testing Data')
plt.annotate('n=2376',  xy=(0.2, 0.8), xycoords='figure fraction', fontsize=14)
plt.tight_layout()
plt.show()

# test R2
RSS_test = np.sum((test_pred[:, :, 0] - test_pred[:, :, 1])**2)  # actual - pred
TSS_test = np.sum((test_pred[:, :, 0] - np.mean(test_pred[:, :, 0]))**2)  # actual - mean
R2_test = 1 - (RSS_test/TSS_test)
print(f'R2_test: {R2_test}')

#%% Sample simulations of glacier melting over time

# load glacier data
glaciers = pd.read_csv('../Glacier_geometry.csv')

# Step 1. Estimate c

gamma = 1.357  # from Ayala et al. (2020)
c_m2 = glaciers['mean_ice_t']/(glaciers['Shape_Area']**(gamma-1))
c_km2 = glaciers['mean_ice_t']/(glaciers['area']**(gamma-1))
c = 28.5  # based on estimate from Ayala et al. (2020)
c_glob = 0.034  # global average value

# Step 2. Estimate initial total glacier area, A0 and thickness b0
A0 = glaciers['area'].sum()  # km2
b0 = c * A0**(gamma-1)  # m
# b0 = glaciers['mean_ice_t'].mean()
# c = b0/(A0**(gamma-1))

# Step 2: Calculate delta b from fitted regression
sim_deltab = pd.DataFrame(data=np.zeros((len(df_.groupby(['CC']).get_group(1)), 10)), columns=np.arange(10)+1)
sim_deltab.insert(0,'Month', df_T.groupby('CC').get_group(1)['MM'].to_numpy())
sim_deltab.insert(0,'Year', df_T.groupby('CC').get_group(1)['YYYY'].to_numpy())

sim_deltab = pd.DataFrame(data={'YYYY': df_['YYYY'], 'MM': df_['MM'], 'CC': df_['CC'], 'delta b': np.zeros((len(df_)))})

for i, s in enumerate(scenarios):

    df_i = df_.groupby(['CC']).get_group(s)[['MM', 'Prec', 'Temp']]

    for j, m in enumerate(np.arange(12)+1):
        df_ij = df_i.groupby('MM').get_group(m)
        TP = df_i.groupby('MM').get_group(m)[['Temp', 'Prec']].to_numpy()
        cTcP = cm.loc[m].to_numpy()
        sim_deltab.iloc[df_ij.index, -1] = np.matmul(TP, cTcP)


# Step 3: for each climate change scenario, simulate change in glacier area and melt volume
delta_b = pd.concat((pd.DataFrame(data=pd.to_datetime(sim_deltab.groupby('CC').get_group(1)['YYYY'].astype(str) + sim_deltab.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(sim_deltab.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)
b_t = pd.concat((pd.DataFrame(data=pd.to_datetime(sim_deltab.groupby('CC').get_group(1)['YYYY'].astype(str) + sim_deltab.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(sim_deltab.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)
A_t = pd.concat((pd.DataFrame(data=pd.to_datetime(sim_deltab.groupby('CC').get_group(1)['YYYY'].astype(str) + sim_deltab.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(sim_deltab.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)
MeltGw_t = pd.concat((pd.DataFrame(data=pd.to_datetime(sim_deltab.groupby('CC').get_group(1)['YYYY'].astype(str) + sim_deltab.groupby('CC').get_group(1)['MM'].astype(str).str.zfill(2), format='%Y%m').reset_index(drop=True), columns=['Date']), pd.DataFrame(np.zeros((len(sim_deltab.groupby('CC').get_group(1)), 10)), columns=np.arange(10)+1)), axis=1)

for i, s in enumerate(scenarios):

    delta_b[s] = sim_deltab.groupby('CC').get_group(s)['delta b'].reset_index(drop=True)
    b_t[s] = np.maximum(b0 - delta_b[s].cumsum(), 0)
    A_t[s] = np.maximum((b_t[s] / c) ** (1 / (gamma - 1)), 0)
    delta_A = A_t[s].diff()
    # delta_A.iloc[0] = A0 - A_t.iloc[0]
    A_t_mean = A_t[s].rolling(window=2).mean()
    delta_V = delta_b[s]/ 1000 * A_t_mean  # volume of ice lost km3
    delta_V[b_t[s] <= 0] = 0
    delta_V[A_t[s] <= 0] = 0
    MeltGw_t[s] = np.maximum((delta_V * 1000 ** 3) * (rho_I / rho_w) / 1E6, 0)  # volume of water Mm3

#%% plot simulation results

# precipitation vs. time
plt.figure(figsize=(15, 3))
plt.subplot(1, 2, 1)
plt.plot(df_P_CC['Date'], df_P_CC.iloc[:, 1::], label=df_P_CC.columns[1::])
plt.ylabel('Precipitation (mm/d)')
plt.xlabel('Time')
plt.legend(title="Climate Scenario", ncol=2, bbox_to_anchor=(-0.1, 0.5), loc="center right", borderaxespad=0)
plt.title('Precipitation vs. Time')
plt.subplot(1, 2, 2)
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']
bplot1 = plt.boxplot(df_P_CC.iloc[:, 1::], patch_artist=True)
plt.ylabel(r'Precipitation (mm/d)')
plt.xlabel('Committed Ice Loss Scenario')
plt.title('Precipitation vs. Committed Ice Loss Scenario')
for bplot in bplot1:
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        patch._set_edgecolor(color)
    for i in range(10):
        bplot1['whiskers'][2*i].set_color(colors[i])
        bplot1['whiskers'][2*i +1].set_color(colors[i])
    for i in range(10):
        bplot1['caps'][2*i].set_c(colors[i])
        bplot1['caps'][2*i +1].set_c(colors[i])
    for i in range(10):
        bplot1['fliers'][i].set_markeredgecolor(colors[i])
    for i in range(10):
        bplot1['medians'][i].set_c('black')
plt.gcf().subplots_adjust(bottom=0.06, left=0.08)
plt.tight_layout()
plt.show()

# temperature vs. time
plt.figure(figsize=(15, 3))
plt.subplot(1, 2, 1)
plt.plot(df_T_CC['Date'], df_T_CC.iloc[:, 1::]-273.15, label=df_T_CC.columns[1::])
plt.ylabel('Temperature (deg. C)')
plt.xlabel('Time')
plt.legend(title="Climate Scenario", ncol=2, bbox_to_anchor=(-0.12, 0.5), loc="center right", borderaxespad=0)
plt.title('Temperature vs. Time')
plt.subplot(1, 2, 2)
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']
bplot1 = plt.boxplot(df_T_CC.iloc[:, 1::]-273.15, patch_artist=True)
plt.ylabel(r'Temperature (deg. C)')
plt.xlabel('Committed Ice Loss Scenario')
plt.title('Temperature vs. Committed Ice Loss Scenario')
for bplot in bplot1:
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        patch._set_edgecolor(color)
    for i in range(10):
        bplot1['whiskers'][2*i].set_color(colors[i])
        bplot1['whiskers'][2*i +1].set_color(colors[i])
    for i in range(10):
        bplot1['caps'][2*i].set_c(colors[i])
        bplot1['caps'][2*i +1].set_c(colors[i])
    for i in range(10):
        bplot1['fliers'][i].set_markeredgecolor(colors[i])
    for i in range(10):
        bplot1['medians'][i].set_c('black')
plt.gcf().subplots_adjust(bottom=0.06, left=0.08)
plt.tight_layout()
plt.show()

# glacier area vs. time
plt.figure()
plt.plot(A_t['Date'], A_t.iloc[:, 1::], label=A_t.columns[1::])
plt.ylabel(r'Total Glacier Area (km$^2$) (?)')
plt.xlabel('Time')
plt.legend(title="Climate Scenario", ncol=2)
plt.title('Simulated Total Glacier Area vs. Time')
plt.show()

# glacier thickness vs. time
plt.figure()
plt.plot(b_t['Date'], b_t.iloc[:, 1::], label=b_t.columns[1::])
plt.ylabel('Mean Ice Thickness (m) (?)')
plt.xlabel('Time')
plt.legend(title="Climate Scenario", ncol=2)
plt.title('Simulated Mean Ice Thickness vs. Time')
plt.show()

# melt water vs. time (most important for our study) - MONTHLY
plt.figure()
plt.plot(MeltGw_t['Date'], MeltGw_t.iloc[:, 1::], label=MeltGw_t.columns[1::])
plt.ylabel(r'Total Glacial Melt (Mm$^3$/month) (?)')
plt.xlabel('Time')
plt.legend(title="Climate Scenario", ncol=2)
plt.title('Simulated Total Glacial Melt vs. Time')
plt.show()


# by scenario: melt water vs. time (most important for our study)
fig = plt.figure(figsize=(9, 9))
for i in range(10):
    plt.subplot(5, 2, i+1)
    plt.plot(MeltGw_t['Date'], MeltGw_t.iloc[:, i+1], c=colors[i])
    plt.title(f'Climate Scenario: {i+1}')
    #plt.ylim((0, 6000))
plt.suptitle('Simulated Total Glacial Melt vs. Time')
fig.tight_layout()
fig.supylabel(r'Total Glacial Melt (Mm$^3$/month) (?)')
fig.supxlabel('Time')
plt.gcf().subplots_adjust(bottom=0.06, left=0.1)
plt.show()

# comparison over 2004-2010
fig = plt.figure(figsize=(15, 6))
plt.grid(axis='y')
for i in range(10):
    plt.bar(MeltGw_t.columns[i+1], MeltGw_t.iloc[47:119, i+1].sum(axis=0), color=colors[i])
    #plt.ylim((0, 6000))
plt.bar(0, 22001.21, color='#800080', label='Pywr (RCP_45_Pp50_Temp_50)')
plt.xticks(np.arange(11), labels=['Pywr (RCP4.5)', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.title('Simulated Total Glacial Melt vs. Time \n(March 2004 - March 2010)')
fig.tight_layout()
plt.ylabel(r'Total Glacial Melt (Mm$^3$)')
plt.xlabel('Scenario')
plt.gcf().subplots_adjust(bottom=0.15, left=0.08)
plt.show()


#%% Apply fitted model to new scenarios (not monthly aggregated)
import datetime
from datetime import datetime

def year_week_to_datetime(row):
    return pd.to_datetime(f"{row['year']}-{row['week']}-1", format="%Y-%W-%w")

# load glacier data
glaciers = pd.read_csv('../Glacier_geometry.csv')
scenarios = np.arange(10) + 1  # committed ice loss scenarios

# initialize master dataframes
df_ref = pd.read_csv(f'../Glacier_series/{glaciers.OBJECTID[0]}_{scenarios[0]}.csv')
df_T = pd.DataFrame()
df_P = pd.DataFrame()
df_S = pd.DataFrame()
df_G = pd.DataFrame()

for i, s in enumerate(scenarios):
    # initialize intermediate dataframes
    df_Tij = df_ref[['YYYY', 'MM', 'DD']]
    df_Tij['CC'] = s * np.ones(len(df_Tij)).astype(int)
    df_Pij = df_ref[['YYYY', 'MM', 'DD']]
    df_Pij['CC'] = s * np.ones(len(df_Pij)).astype(int)
    df_Sij = df_ref[['YYYY', 'MM', 'DD']]
    df_Sij['CC'] = s * np.ones(len(df_Sij)).astype(int)
    df_Gij = df_ref[['YYYY', 'MM', 'DD']]
    df_Gij['CC'] = s * np.ones(len(df_Gij)).astype(int)

    for j, g in enumerate(glaciers['OBJECTID']):
        # load data
        df = pd.read_csv(f'../Glacier_series/{g}_{s}.csv')

        # add glacier data to dataframes
        df_Tij[g] = df['RefHAirTemp'].values + 273.15  # deg. C to K
        # df_Pij[g] = np.log(df['Prec'].values + 1E-6)
        df_Pij[g] = df['Prec'].values
        df_Sij[g] = df['MeltS'].values  # mm to m
        df_Gij[g] = df['MeltG'].values  # mm to m

    # append master dataframes
    df_T = pd.concat([df_T, df_Tij], ignore_index=True)
    df_P = pd.concat([df_P, df_Pij], ignore_index=True)

# Step 1. Estimate c
gamma = 1.357  # from Ayala et al. (2020)
c_m2 = glaciers['mean_ice_t']/(glaciers['Shape_Area']**(gamma-1))
c_km2 = glaciers['mean_ice_t']/(glaciers['area']**(gamma-1))
c = 28.5  # based on estimate from Ayala et al. (2020)
c_glob = 0.034  # global average value

# Step 2. Estimate initial total glacier area, A0 and thickness b0
A0 = glaciers['area'].sum()  # km2
b0 = c * A0**(gamma-1)  # m

# specify scenarios
models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

locations = pd.read_csv('../Climate_series/MPC_pr_ACCESS-CM2_ssp126_1986_2100_week_UQM.csv').columns[2::]
dates = pd.read_csv(f'../Climate_series/UQM_weap_TempC_new/Tprom_semanal_ACCESS-CM2_ssp126.csv')


# initialize dataframes
delta_b_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
b_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
A_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
MeltGw_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])

for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        delta_b_df[f'{mo}_{x}'] = np.zeros(len(delta_b))
        b_t_df[f'{mo}_{x}'] = np.zeros(len(delta_b))
        A_t_df[f'{mo}_{x}'] = np.zeros(len(delta_b))
        MeltGw_t_df[f'{mo}_{x}'] = np.zeros(len(delta_b))

for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        df__ = pd.DataFrame()

        T = pd.read_csv(f'../Climate_series/UQM_weap_TempC_new/Tprom_semanal_{mo}_{x}.csv')
        df__['Temp'] = T[locations].mean(axis=1) + 273.15
        P = pd.read_csv(f'../Climate_series/UQM_weap_TempC_new/PP_semanal_{mo}_{x}.csv')
        df__.insert(0, 'MM', pd.to_datetime(P['year'].astype(str) + '-' + P['week'].astype(str).str.zfill(2) + '-1', format="%Y-%W-%w").dt.month)
        df__.insert(0, 'YYYY', pd.to_datetime(P['year'].astype(str) + '-' + P['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w").dt.year)

        # calibrate average T to historical period 2001-2018
        T_mu = df__.loc[(df__['YYYY'] <= 2018) & (df__['YYYY'] >= 2001)]['Temp'].mean().mean()
        T_mu_ref = df_T.loc[(df_T['YYYY'] <= 2018) & (df_T['YYYY'] >= 2001)].iloc[:, 4::].mean().mean()
        #T_mu_ref = df_T.loc[(df_T['YYYY'] <= 2018) & (df_T['YYYY'] >= 2001)][0].mean()
        T_calibrate = T_mu - T_mu_ref
        df__['Temp'] = df__['Temp'] - T_calibrate

        # calibrate average P to historical period 2001-2018
        df__['Prec'] = P[locations].mean(axis=1)
        P_mu = df__.loc[(df__['YYYY'] <= 2018) & (df__['YYYY'] >= 2001)]['Prec'].mean().mean()
        P_mu_ref = df_P.loc[(df_P['YYYY'] <= 2018) & (df_P['YYYY'] >= 2001)].iloc[:, 4::].mean().mean()
        #P_mu_ref = df_P.loc[(df_P['YYYY'] <= 2018) & (df_P['YYYY'] >= 2001)][0].mean()
        #P_calibrate = P_mu_ref/P_mu
        P_calibrate =  P_mu - P_mu_ref
        #df__['Prec'] = df__['Prec'] * P_calibrate
        df__['Prec'] = np.maximum(df__['Prec'] - P_calibrate, 0)
        df__['Prec'] = np.log(df__['Prec'] + 1E-6)


        # replace Temp and Prec columns with T and P monthly anomalies
        for m in range(12):
            df__.iloc[df__.groupby('MM').get_group(m + 1).index, 2] = df__.groupby('MM').get_group(m + 1)['Temp'] - \
                                                                       T_ref.iloc[m]
            # df_P.iloc[df_P.groupby('MM').get_group(m + 1).index, -1] = np.log(df_P.groupby('MM').get_group(m + 1)[0])
            df__.iloc[df__.groupby('MM').get_group(m + 1).index, -1] = (df__.groupby('MM').get_group(m + 1)['Prec'] -
                                                                        P_ref.iloc[m]) / P_ref.iloc[m]

        # Step 3. simualte delta b using fitted relation
        sim_deltab = pd.DataFrame(
            data={'YYYY': df__['YYYY'], 'MM': df__['MM'], 'delta b': np.zeros((len(df__)))})
        for j, m in enumerate(np.arange(12)+1):
            df_ij = df__.groupby('MM').get_group(m)
            TP = df__.groupby('MM').get_group(m)[['Temp', 'Prec']].to_numpy()
            cTcP = cm.loc[m].to_numpy()
            sim_deltab.iloc[df_ij.index, -1] = np.matmul(TP, cTcP)

        # Step 4: simulate change in glacier area and melt volume
        delta_b = sim_deltab['delta b'].reset_index(drop=True)
        #delta_b = np.exp(sim_deltab['delta b']).reset_index(drop=True)
        b_t = np.maximum(b0 - delta_b.cumsum(), 0)
        A_t = np.maximum((b_t / c) ** (1 / (gamma - 1)), 0)
        delta_A = A_t.diff()
        # delta_A.iloc[0] = A0 - A_t.iloc[0]
        A_t_mean = A_t.rolling(window=2).mean()
        delta_V = delta_b / 1000 * A_t_mean  # volume of ice lost km3
        delta_V[b_t <= 0] = 0
        delta_V[A_t <= 0] = 0
        MeltGw_t = np.maximum((delta_V * 1000 ** 3) * (rho_I / rho_w) / 1E6, 0)  # volume of water Mm3

        # add to dataframes
        delta_b_df[f'{mo}_{x}'] = delta_b
        b_t_df[f'{mo}_{x}'] = b_t
        A_t_df[f'{mo}_{x}'] = A_t_mean
        MeltGw_t_df[f'{mo}_{x}'] = MeltGw_t

# PLOT thickness
plt.figure()
for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        plt.plot(b_t_df['Date'], b_t_df[f'{mo}_{x}'], label=f'{mo}_{x}')
plt.xlabel('Date')
plt.ylabel('Mean Ice Thickness')
plt.legend()
plt.show()

# all glacier melt
plt.figure()
for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{mo}_{x}')
plt.xlabel('Date')
plt.ylabel('MeltGw_t')
plt.legend()
plt.show()

# subplot glacier melt experiment
fig = plt.figure(figsize=(10, 10))
for j, x in enumerate(experiments):
    plt.subplot(4, 1, j+1)
    for i, mo in enumerate(models):
        plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{mo}')
    plt.title(f'Experiment: {x}')
fig.supxlabel('Date')
fig.supylabel('Glacier Melt (Mm3/month)')
plt.legend(ncols=2)
plt.suptitle('Glacial Melt vs. Time for Climate Change Scenarios')
fig.tight_layout()
plt.show()


#%% Apply fitted model to new scenarios (not monthly aggregated)

import datetime
from datetime import datetime

def year_week_to_datetime(row):
    return pd.to_datetime(f"{row['year']}-{row['week']}-1", format="%Y-%W-%w")

# load glacier data
glaciers = pd.read_csv('../Glacier_geometry.csv')
scenarios = np.arange(10) + 1  # committed ice loss scenarios

# initialize master dataframes
df_ref = pd.read_csv(f'../Glacier_series/{glaciers.OBJECTID[0]}_{scenarios[0]}.csv')
df_T = pd.DataFrame()
df_P = pd.DataFrame()
df_S = pd.DataFrame()
df_G = pd.DataFrame()

for i, s in enumerate(scenarios):
    # initialize intermediate dataframes
    df_Tij = df_ref[['YYYY', 'MM', 'DD']]
    df_Tij['CC'] = s * np.ones(len(df_Tij)).astype(int)
    df_Pij = df_ref[['YYYY', 'MM', 'DD']]
    df_Pij['CC'] = s * np.ones(len(df_Pij)).astype(int)
    df_Sij = df_ref[['YYYY', 'MM', 'DD']]
    df_Sij['CC'] = s * np.ones(len(df_Sij)).astype(int)
    df_Gij = df_ref[['YYYY', 'MM', 'DD']]
    df_Gij['CC'] = s * np.ones(len(df_Gij)).astype(int)

    for j, g in enumerate(glaciers['OBJECTID']):
        # load data
        df = pd.read_csv(f'../Glacier_series/{g}_{s}.csv')

        # add glacier data to dataframes
        df_Tij[g] = df['RefHAirTemp'].values + 273.15  # deg. C to K
        # df_Pij[g] = np.log(df['Prec'].values + 1E-6)
        df_Pij[g] = df['Prec'].values
        df_Sij[g] = df['MeltS'].values  # mm to m
        df_Gij[g] = df['MeltG'].values  # mm to m

    # append master dataframes
    df_T = pd.concat([df_T, df_Tij], ignore_index=True)
    df_P = pd.concat([df_P, df_Pij], ignore_index=True)

# Step 1. Estimate c
gamma = 1.357  # from Ayala et al. (2020)
c_m2 = glaciers['mean_ice_t']/(glaciers['Shape_Area']**(gamma-1))
c_km2 = glaciers['mean_ice_t']/(glaciers['area']**(gamma-1))
c = 28.5  # based on estimate from Ayala et al. (2020)
c_glob = 0.034  # global average value

# Step 2. Estimate initial total glacier area, A0 and thickness b0 if SINGLE MASSIVE GLACIER
A0 = glaciers['area'].sum()  # km2
b0 = c * A0**(gamma-1)  # m
V0 = A0 * b0 #

# specify scenarios
models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

locations = pd.read_csv('../Climate_series/MPC_pr_ACCESS-CM2_ssp126_1986_2100_week_UQM.csv').columns[2::]
dates = pd.read_csv(f'../Climate_series/UQM_weap_TempC_new/Tprom_semanal_ACCESS-CM2_ssp126.csv')


# initialize dataframes
delta_b_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
b_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
A_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
MeltGw_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])
V_t_df = pd.DataFrame(data=pd.to_datetime(dates['year'].astype(str) + '-' + dates['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w"), columns=['Date'])


for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        delta_b_df[f'{mo}_{x}'] = np.zeros(len(delta_b_df))
        b_t_df[f'{mo}_{x}'] = np.zeros(len(delta_b_df))
        A_t_df[f'{mo}_{x}'] = np.zeros(len(delta_b_df))
        MeltGw_t_df[f'{mo}_{x}'] = np.zeros(len(delta_b_df))

for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        df__ = pd.DataFrame()

        T = pd.read_csv(f'../Climate_series/UQM_weap_TempC_new/Tprom_semanal_{mo}_{x}.csv')
        df__['Temp'] = T[locations].mean(axis=1) + 273.15
        P = pd.read_csv(f'../Climate_series/UQM_weap_TempC_new/PP_semanal_{mo}_{x}.csv')
        df__.insert(0, 'MM', pd.to_datetime(P['year'].astype(str) + '-' + P['week'].astype(str).str.zfill(2) + '-1', format="%Y-%W-%w").dt.month)
        df__.insert(0, 'YYYY', pd.to_datetime(P['year'].astype(str) + '-' + P['week'].astype(str).str.zfill(2) + '-1',
                                    format="%Y-%W-%w").dt.year)
        df__['Prec'] = P[locations].mean(axis=1)

        # monthly aggregation
        df__ = df__.groupby(['YYYY', 'MM']).mean().reset_index()

        # calibrate average T to historical period 2001-2018
        T_mu = df__.loc[(df__['YYYY'] <= 2018) & (df__['YYYY'] >= 2001)]['Temp'].mean().mean()
        T_mu_ref = df_T.loc[(df_T['YYYY'] <= 2018) & (df_T['YYYY'] >= 2001)].iloc[:, 4::].mean().mean()
        #T_mu_ref = df_T.loc[(df_T['YYYY'] <= 2018) & (df_T['YYYY'] >= 2001)][0].mean()
        T_calibrate = T_mu - T_mu_ref
        df__['Temp'] = df__['Temp'] - T_calibrate

        # calibrate average P to historical period 2001-2018
        P_mu = df__.loc[(df__['YYYY'] <= 2018) & (df__['YYYY'] >= 2001)]['Prec'].mean().mean()
        P_mu_ref = df_P.loc[(df_P['YYYY'] <= 2018) & (df_P['YYYY'] >= 2001)].iloc[:, 4::].mean().mean()
        #P_mu_ref = df_P.loc[(df_P['YYYY'] <= 2018) & (df_P['YYYY'] >= 2001)][0].mean()
        #P_calibrate = P_mu_ref/P_mu
        P_calibrate =  P_mu - P_mu_ref
        #df__['Prec'] = df__['Prec'] * P_calibrate
        df__['Prec'] = np.maximum(df__['Prec'] - P_calibrate, 0)
        df__['Prec'] = np.log(df__['Prec'] + 1E-6)


        # replace Temp and Prec columns with T and P monthly anomalies
        for m in range(12):
            df__.iloc[df__.groupby('MM').get_group(m + 1).index, 2] = df__.groupby('MM').get_group(m + 1)['Temp'] - \
                                                                       T_ref.iloc[m]
            # df_P.iloc[df_P.groupby('MM').get_group(m + 1).index, -1] = np.log(df_P.groupby('MM').get_group(m + 1)[0])
            df__.iloc[df__.groupby('MM').get_group(m + 1).index, -1] = (df__.groupby('MM').get_group(m + 1)['Prec'] -
                                                                        P_ref.iloc[m]) / P_ref.iloc[m]

        # Step 3. simualte delta b using fitted relation
        sim_deltab = pd.DataFrame(
            data={'YYYY': df__['YYYY'], 'MM': df__['MM'], 'delta b': np.zeros((len(df__)))})
        for j, m in enumerate(np.arange(12)+1):
            df_ij = df__.groupby('MM').get_group(m)
            TP = df__.groupby('MM').get_group(m)[['Temp', 'Prec']].to_numpy()
            cTcP = cm.loc[m].to_numpy()
            sim_deltab.iloc[df_ij.index, -1] = np.matmul(TP, cTcP)

        # Step 4: simulate change in glacier area and melt volume
        delta_b = sim_deltab['delta b'].reset_index(drop=True)
        #delta_b = np.exp(sim_deltab['delta b']).reset_index(drop=True)
        b_t = np.maximum(b0 - delta_b.cumsum(), 0)
        A_t = np.maximum((b_t / c) ** (1 / (gamma - 1)), 0)
        delta_A = A_t.diff()
        # delta_A.iloc[0] = A0 - A_t.iloc[0]
        A_t_mean = A_t.rolling(window=2).mean()
        delta_V = delta_b / 1000 * A_t_mean  # volume of ice lost km3
        V_t_df[f'{mo}_{x}'] = delta_V.cumsum() + V0
        delta_V[b_t <= 0] = 0
        delta_V[A_t <= 0] = 0

        MeltGw_t = pd.DataFrame(data=np.maximum((delta_V * 1000 ** 3) * (rho_I / rho_w) / 1E6, 0).values, columns=['MeltG'])  # volume of water Mm3
        MeltGw_t.insert(0, 'Month', sim_deltab['MM'])
        MeltGw_t.insert(0, 'Year', sim_deltab['YYYY'])

        for m in MeltGw_t['Month'].unique():
            for y in MeltGw_t['Year'].unique():
                n_weeks = MeltGw_t_df.loc[(MeltGw_t_df['Date'].dt.year == y) & (MeltGw_t_df['Date'].dt.month == m)].shape[0]
                MeltGw_t_df[f'{mo}_{x}'].loc[(MeltGw_t_df['Date'].dt.year == y) & (MeltGw_t_df['Date'].dt.month == m)] = \
                    MeltGw_t.loc[(MeltGw_t['Year'] == y) & (MeltGw_t['Month'] == m)][
                        'MeltG'].values / n_weeks * np.ones(n_weeks)

        # add to dataframes
        delta_b_df[f'{mo}_{x}'] = delta_b
        b_t_df[f'{mo}_{x}'] = b_t
        A_t_df[f'{mo}_{x}'] = A_t_mean
        #MeltGw_t_df[f'{mo}_{x}'] = MeltGw_t

#MeltGw_t_df.to_csv('CMIP6_Glacier_Melt_TDP.csv')  # save to csv

#%% Plots
# PLOT thickness
plt.figure()
for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        plt.plot(b_t_df['Date'], b_t_df[f'{mo}_{x}'], label=f'{mo}_{x}')
plt.xlabel('Date')
plt.ylabel('Mean Ice Thickness')
#plt.legend()
plt.show()

#%% plot annual glacier VOLUME (SSP 5-8.5 for 2020-2100)
plt.figure()
thick = b_t_df.iloc[0:1452, 1::]
area = A_t_df.iloc[0:1452, 1::]
volume = np.array(thick * area)
volume_annual = volume.reshape(int(volume.shape[0]/12), 12, 36).sum(axis=1)
#pd.DataFrame(volume_annual, columns=A_t_df.columns[1::]).to_csv('CMIP6_Annual_Glacier_Volume_agg_TDP.csv')
plt.plot(volume_annual[41::, 3::4])  # only SSP5-8.5 scenario
plt.ylabel('Glacier Volume (Mm3)')
plt.xlabel('Year')
plt.title('Annual Total Glacier Volume vs. Time for SSP5-8.5 scenario')
plt.show()

#%% plot all glacier melt on same axis
plt.figure()
for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{mo}_{x}')
plt.xlabel('Date')
plt.xticks()
plt.ylabel('MeltGw_t')
plt.legend()
plt.show()

#%% plot annual glacier melt (SSP 5-8.5 for 2020-2100)
plt.figure()
plt.plot(MeltGw_t_df.groupby(MeltGw_t_df['Date'].dt.year).sum(numeric_only=True).iloc[41::, 3::4])  # only SSP5-8.5 scenario
plt.ylabel('Glacier Melt (MCM/year)')
plt.xlabel('Year')
plt.title('Annual Glacier Melt vs. Time for SSP5-8.5 scenario')
plt.show()

#%%  Smoothed out glacier melt by experiment
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

colors = ['cadetblue', 'steelblue', 'chocolate', 'darkred']
fig = plt.figure(figsize=(6, 3))

for j, x in enumerate(experiments):
    for mo in models:
        #plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{x}', color=colors[j])
        plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean(), label=f'{x}', color=colors[j])
hist_start = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].index[0]
hist_end = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2020].index[0]
h, = plt.plot(MeltGw_t_df['Date'].iloc[hist_start:hist_end], MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
plt.xlabel('Time')
plt.ylabel('MeltGw_t')
plt.xlim([MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].iloc[0], MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
plt.ylim([0, np.ceil(plt.gca().get_ylim()[1])])
#plt.legend()
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
fig.tight_layout()
plt.show()


#%% all on same plots, with a low and high GCM highlights

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

colors = ['cadetblue', 'steelblue', 'chocolate', 'darkred']
fig = plt.figure(figsize=(6, 3))

for j, x in enumerate(experiments):
    for i, mo in enumerate(models):
        #plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{x}', color=colors[j])
        l1, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean(), label=f'{x}', color='steelblue')

# l2, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{models[3]}_{experiments[0]}'].shift().rolling(20 * 52, min_periods=3).mean(),
#                label=f'{x}', color='midnightblue')
#
# l3, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{models[-3]}_{experiments[-1]}'].shift().rolling(20 * 52, min_periods=3).mean(),
#                label=f'{x}', color='midnightblue', linewidth=2)
hist_start = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].index[0]
hist_end = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2020].index[0]
h, = plt.plot(MeltGw_t_df['Date'].iloc[hist_start:hist_end], MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end],
              label=f'Historical', color='black', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Glacier Melt (MCM)')
plt.xlim([MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].iloc[0], MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
plt.ylim([0, np.ceil(plt.gca().get_ylim()[1])])
#plt.legend()
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
fig.tight_layout()
plt.show()

#%% #%% Rolling time series as subplot (temp vs. glacier melt)
import matplotlib.pyplot as plt
import datetime

path_clima = '../Climate_series/'

fig = plt.figure(figsize=(6, 6))
fig = plt.figure(figsize=(6, 7))
plt.subplot(2, 1, 1)
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        l, = plt.plot(ts, t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean(), label=f'{m}_{x}', color='#50789A') # color='#8ABDC9') #color='steelblue')

#plt.legend()
hist_start = ts[ts.dt.year >= 1999].index[0]
hist_end = ts[ts.dt.year >= 2020].index[0]
h, = plt.plot(ts.iloc[hist_start:hist_end], t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
plt.title('Temperature vs. Time', fontsize=18)
plt.ylabel(r'Temperature ($\degree$ C)', fontsize=16)
legend = plt.legend([l, h], ['GCM projections', "Historical"], frameon=False, fontsize=14)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor('0.9')
#plt.xlabel('Date', fontsize=15)
plt.xlim([ts[ts.dt.year >= 1999].iloc[0], ts[ts.dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
#plt.ylim([0, 13])
plt.xticks([])
plt.yticks(fontsize=14)
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.tight_layout()


plt.subplot(2, 1, 2)
MeltGw_t_df = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/CMIP6_Glacier_Melt_TDP.csv')
MeltGw_t_df['Date'] = pd.to_datetime(MeltGw_t_df['Date'])
n = 0.019405960290035287
for j, x in enumerate(experiments):
    for i, mo in enumerate(models):
        #plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{x}', color=colors[j])
        l1, = plt.plot(MeltGw_t_df['Date'], n * MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean(), label=f'{x}', color='#50789A')  #color='#8ABDC9') # 'steelblue')

# l2, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{models[3]}_{experiments[0]}'].shift().rolling(20 * 52, min_periods=3).mean(),
#                label=f'{x}', color='midnightblue')
#
# l3, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{models[-3]}_{experiments[-1]}'].shift().rolling(20 * 52, min_periods=3).mean(),
#                label=f'{x}', color='midnightblue', linewidth=2)
hist_start = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].index[0]
hist_end = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2020].index[0]
h, = plt.plot(MeltGw_t_df['Date'].iloc[hist_start:hist_end], n* MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end],
              label=f'Historical', color='black', linewidth=2)
#plt.xlabel('Time', fontsize=16)
plt.ylabel('Glacier Melt (Mm3/week)', fontsize=16)
plt.xticks(fontsize=14)
plt.xlim([MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].iloc[0], MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
plt.ylim([0, np.ceil(plt.gca().get_ylim()[1])])
plt.yticks(fontsize=14)
plt.title('Glacier Melt vs. Time', fontsize=18)

#plt.legend()
plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.supylabel('20-Year Moving Average', fontsize=16)
fig.tight_layout()

#plt.savefig("../GlacierMeltandTempVsTime_SSPs.pdf", format="pdf", bbox_inches="tight")
plt.show()

#%% #%% V2 Rolling time series as subplot (temp vs. glacier melt)
import matplotlib.pyplot as plt
import datetime

path_clima = '../Climate_series/'

fig = plt.figure(figsize=(6, 6))
fig = plt.figure(figsize=(6, 7))
plt.subplot(2, 1, 1)
for m in models:
    for x in experiments:

        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        #t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')

        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')
        l, = plt.plot(ts, t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean(), label=f'{m}_{x}', color='firebrick') # color='#8ABDC9') #color='steelblue')

#plt.legend()
hist_start = ts[ts.dt.year >= 1999].index[0]
hist_end = ts[ts.dt.year >= 2020].index[0]
h, = plt.plot(ts.iloc[hist_start:hist_end], t.iloc[:, 3].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end], label=f'Historical', color='black')
plt.title('Temperature vs. Time', fontsize=18)
plt.ylabel(r'Temperature ($\degree$ C)', fontsize=16)
# legend = plt.legend([l, h], ['GCM projections', "Historical"], frameon=False, fontsize=14)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor('0.9')
#plt.xlabel('Date', fontsize=15)
plt.xlim([ts[ts.dt.year >= 1999].iloc[0], ts[ts.dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
#plt.ylim([0, 13])
plt.xticks([])
plt.yticks(fontsize=14)
# plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
fig.tight_layout()


plt.subplot(2, 1, 2)
MeltGw_t_df = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/CMIP6_Glacier_Melt_TDP.csv')
MeltGw_t_df['Date'] = pd.to_datetime(MeltGw_t_df['Date'])
n = 0.019405960290035287
for j, x in enumerate(experiments):
    for i, mo in enumerate(models):
        #plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{x}', color=colors[j])
        l1, = plt.plot(MeltGw_t_df['Date'], n * MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean(), label=f'{x}', color='darkolivegreen')  #color='#8ABDC9') # 'steelblue')

# l2, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{models[3]}_{experiments[0]}'].shift().rolling(20 * 52, min_periods=3).mean(),
#                label=f'{x}', color='midnightblue')
#
# l3, = plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{models[-3]}_{experiments[-1]}'].shift().rolling(20 * 52, min_periods=3).mean(),
#                label=f'{x}', color='midnightblue', linewidth=2)
hist_start = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].index[0]
hist_end = MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2020].index[0]
h, = plt.plot(MeltGw_t_df['Date'].iloc[hist_start:hist_end], n* MeltGw_t_df[f'{mo}_{x}'].shift().rolling(20*52, min_periods=3).mean()[hist_start:hist_end],
              label=f'Historical', color='black', linewidth=2)
#plt.xlabel('Time', fontsize=16)
plt.ylabel('Glacier Melt (Mm3/week)', fontsize=16)
plt.xticks(fontsize=14)
plt.xlim([MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 1999].iloc[0], MeltGw_t_df['Date'][MeltGw_t_df['Date'].dt.year >= 2000].iloc[-1]+datetime.timedelta(days=30)])
plt.ylim([0, np.ceil(plt.gca().get_ylim()[1])])
plt.yticks(fontsize=14)
plt.title('Glacier Melt vs. Time', fontsize=18)

#plt.legend()
#plt.gca().patch.set_facecolor('0.9')
ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
fig.supylabel('20-Year Moving Average', fontsize=16)
fig.tight_layout()

#plt.savefig("../GlacierMeltandTempVsTime_SSPs.pdf", format="pdf", bbox_inches="tight")
plt.show()

#%% subplots glacier of melt experiment
fig = plt.figure(figsize=(10, 10))
for j, x in enumerate(experiments):
    plt.subplot(4, 1, j+1)
    for i, mo in enumerate(models):
        plt.plot(MeltGw_t_df['Date'], MeltGw_t_df[f'{mo}_{x}'], label=f'{mo}')
    plt.title(f'Experiment: {x}')
fig.supxlabel('Date')
fig.supylabel('Glacier Melt (Mm3/week)')
plt.legend(ncols=2)
plt.suptitle('Glacial Melt vs. Time for Climate Change Scenarios')
fig.tight_layout()
plt.show()
