#%% Plots for Paper 1 and/or HydroSeminar 2025
# This V2 version uses outputs from optimzed policies with nFunc=2000

import matplotlib
# imports
import numpy as np
import matplotlib.pyplot as plt
#import mne
import json
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from itertools import product
import matplotlib.gridspec as gridspec
import datetime
import matplotlib.colors

#%% Helpful plotting functions from the Reed Group
#source: https://reedgroup.github.io/FigureLibrary/ParallelCoordinatesPlots.html#:~:text=Parallel%20coordinates%20plots%20(also%20known,Diagnostics%20page%20of%20this%20manual.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates


#%% FIGURE 2
# NEW FIGURE FROM SARAH V2: Scatter Plot objective Best performing policy for each indicator type

#fol = 'DPS_results_Validation'  # folder convention where results are saved
fol = 'DPS_results_Validation_combined'  # folder convention where results are saved

periods = ['2020_2040', '2080_2100']  # time periods for optimizations -> use for file readings
drought_indices = ['SRI3', 'SRI6', 'SRI12',
                   'SPI3', 'SPI6', 'SPI12']
#seeds = np.arange(5) + 1  # for each seed
#date = '27Jan26'  # date in filename
#date = '6Feb26'  # date in filename
date = '10Feb26'
seeds = [1,2,3,4,5,6,7] #[4]  # plot 1 seed for now
nFunc = 2000
disc = 35  # format is the % without the decimal, so 3.5 is 35

# plotting fields
ideal_direction = 'top'  # top or bottom
zorder_direction = 'ascending'  # ascending or descending
minmaxs = ['min', 'max', 'max']  # min or max for each objective
color_by_categorical = True  # color by indicator type, SRI or SPI
color_by_continuous = None
columns_axes = ['Total Cost', 'Urban Rel.', 'Agr. Rel.']
column_titles = ['Total Cost', 'Urban Reliability', 'Agricultural Reliability']
axis_labels = ['Total Cost (M$)', 'Urban Reliability', 'Agricultural Reliability']
alpha_base = 0.6
lw_base = 1
fontsize = 13

### initialize figure for plotting
fig = plt.figure(figsize=(13, 4), constrained_layout=True)
widths = [0.1, 0.2, 1, 0.2, 1, 0.2, 1]
heights = [1]
gs = fig.add_gridspec(ncols=7, nrows=1, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
fig.tight_layout(h_pad=0)

#  load objective results and store in master dictionary
objs_dict = {}
for p in periods:
    objs_dict[p] = {}
    for d in drought_indices:
        objs_dict[p][d] = {}
for t, p in enumerate(periods):
    for j, d in enumerate(drought_indices):
        # Concatenate solutions across all seeds for this scenario
        all_objs = []
        for i in seeds:
            # edit this to read for different files
            file = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/outputs/{fol}/{d}_{p}_CMIP6_{date}_nFunc{nFunc}_disc{disc}/sets/Borg_DPS_PySedSim{i}.set'
            # formulate the table
            with open(file, 'r') as f:
                # data = pd.read_csv(file, skiprows=2)
                data = pd.read_csv(file, skiprows=2, skipfooter=1, header=None, engine='python').explode(0)
                data = pd.DataFrame(data.iloc[:, 0].str.split(' ').explode())
            num_vars = np.sum(data.index == 0)
            num_rows = data.index.unique().shape[0]
            data['vars'] = np.tile(np.arange(num_vars), num_rows)
            data = data.reset_index().pivot(index='index', columns='vars')
            urban_rel = data.iloc[:, 4].values.astype('float')
            ag_rel = data.iloc[:, 5].values.astype('float')
            cost = data.iloc[:, 6].values.astype('float') * -1 / 1E6  # millions of dollars
            # % data syntax for plotting
            objs = pd.DataFrame(data=np.vstack([cost, urban_rel, ag_rel]).T,
                                columns=['Total Cost', 'Urban Rel.', 'Agr. Rel.'])
            objs = objs.loc[:, columns_axes]
            all_objs.append(objs)

        # Store pooled objective set (all seeds) for this scenario
        objs_dict[p][d] = pd.concat(all_objs, ignore_index=True) if len(all_objs) else pd.DataFrame(columns=columns_axes)


# plot best performance for each objective
for o in range(3):
    sc = np.zeros([len(periods), len(drought_indices)])
    ax = fig.add_subplot(gs[0, 2 + 2 * o]) # intialize subplot
    for t, p in enumerate(periods):
        for j, d in enumerate(drought_indices):
            if minmaxs[o] == 'min':
                sc[t, j] = np.min(objs_dict[p][d].iloc[:, o])
            elif minmaxs[o] == 'max':
                sc[t, j] = np.max(objs_dict[p][d].iloc[:, o])
    ax.scatter(np.arange(len(drought_indices)), sc[0], label='2020-2040', color='black', marker='o', s=40)
    ax.scatter(np.arange(len(drought_indices)), sc[1], label='2080-2100', color='black', marker='x', s=45)
    ax.set_title(column_titles[o])
    ax.set_xticks(np.arange(len(drought_indices)))
    ax.set_xticklabels(drought_indices)
    ax.set_xlim(-1, len(drought_indices))
    if o == 0:
        import matplotlib.patches as mpatches

        leg1 = ax.legend(frameon=False, loc='lower left')

        # --- Custom legend handles --
        mean_patch = mpatches.Patch(color='black', label='Mean Performance')
        best_patch = mpatches.Patch(color='chocolate', label='Best Performance')

        # Add a shared legend for the figure
        leg2 = fig.legend(handles=[mean_patch, best_patch],
                  loc='lower left', bbox_to_anchor=(0.21, 0.1),
                  ncol=1, frameon=False, fontsize=9)

        ax.set_ylabel(axis_labels[o])

    else:
        ax.set_ylim(top=1)
        ax.set_ylabel(axis_labels[o])
    ax.set_ylim(bottom=0)

    # highlight best performing policy
    if minmaxs[o] == 'min':
        ax.scatter(sc[0].argmin(), sc[0].min(), color='chocolate', marker='o', s=45)
        ax.scatter(sc[1].argmin(), sc[1].min(), color='chocolate', marker='x', s=45)
    else:
        ax.scatter(sc[0].argmax(), sc[0].max(), color='chocolate', marker='o', s=45)
        ax.scatter(sc[1].argmax(), sc[1].max(), color='chocolate', marker='x', s=45)

    # if minmaxs[o] == 'min': # omit for now
    #     ax.invert_yaxis()

#fig.suptitle('2020-2100', fontsize=15)
plt.tight_layout()
fig.show()


#%% Sample time series of climate variables (pair with figure 4 of drought indicator time series)

drought_indices = ['SPI12', 'SRI6']  # which indicators to plot
climate_scenario = [5]  # pick climate scenario 0-35 from CMIP6
periods = ['2020_2040', '2080_2100']
climate_scenario = [5]  # pick climate scenario 0-35 from CMIP6
periods = ['2020_2040', '2080_2100']
T = 1/52  # number of years for climate smoothing/time steps

for z in climate_scenario:

    # load the drought index data
    index1 = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/TDP_scenarios_DS/{drought_indices[0]}.csv')
    index2 = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/TDP_scenarios_DS/{drought_indices[1]}.csv')

    # load the corresponding climate data
    scenario_name = index1.columns[z]
    t = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/UQM_weap_TempC_new/Tprom_semanal_{scenario_name}.csv')
    p = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/UQM_weap_TempC_new/PP_semanal_{scenario_name}.csv')
    melt = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/CMIP6_Glacier_Melt_TDP.csv')
    #melt = melt[scenario_name]
    ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                        format='%Y%W-%w')

    # calculate rolling mean of climate variables
    # indices to store 20-year average (i.e., starting 1999-12-27)
    index_ts = np.arange(len(ts))[::-1][0::int(52 * T)][::-1][1::]
    p_means = p.iloc[:, 3].shift().rolling(int(T * 52), min_periods=int(T * 52)).mean()[index_ts].values
    t_means = t.iloc[:, 3].shift().rolling(int(T * 52), min_periods=int(T * 52)).mean()[index_ts].values
    # check if index is correct for melt!
    melt_means = melt[scenario_name].shift().rolling(int(T * 52), min_periods=int(T * 52)).mean()[index_ts].values

    for p in periods:

        # initialize figure
        fig = plt.figure(figsize=(15, 7), constrained_layout=True)
        widths = [1]
        heights = [1, 0.3, 1, 0.25, 0.1]
        gs = fig.add_gridspec(ncols=1, nrows=5, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
        fig.tight_layout(h_pad=0)

        # upper panel: climate dynamics
        ax_A = fig.add_subplot(gs[0, 0])
        ax_A.patch.set_facecolor('0.9')
        ax_A.spines['top'].set_visible(False)
        ax_A.spines['right'].set_visible(False)
        ax_A.spines['bottom'].set_visible(False)
        ax_A.tick_params(axis='x', labelsize=13)
        ax_A.tick_params(axis='y', labelsize=13)
        ax_A.grid(False)

        # twin y axes for panel a
        twin1 = ax_A.twinx()
        #twin2 = ax_A.twinx()

        # format twin axes
        twin1.spines['top'].set_visible(False)
        twin1.spines['left'].set_visible(False)
        twin1.spines['bottom'].set_visible(False)
        twin1.tick_params(axis='x', labelsize=13)
        twin1.tick_params(axis='y', labelsize=13)
        twin1.grid(False)

        # twin2.spines['right'].set_visible(True)
        # twin2.spines['right'].set_linewidth(1.5)
        # twin2.spines['top'].set_visible(False)
        # twin2.spines['left'].set_visible(False)
        # twin2.spines['bottom'].set_visible(False)
        #twin2.tick_params(axis='x', labelsize=13)
        #twin2.tick_params(axis='y', labelsize=13)
        #twin2.grid(False)

        # Offset the right spine of twin2.  The ticks and label have already been
        # placed on the right by twinx above.
        #twin2.spines['right'].set_position(("axes", 1.07))

        # plot on panel a
        rows = np.where((ts[index_ts].dt.year >= int(p.split('_')[0])) & (ts[index_ts].dt.year <= int(p.split('_')[1])))[0]
        if p == '2020_2040':
            rows = np.where((ts[index_ts].dt.year >= int(p.split('_')[0])) & (ts[index_ts].dt.year < int(p.split('_')[1])))[0]
        else:
            rows = np.where((ts[index_ts].dt.year >= int(p.split('_')[0])) & (ts[index_ts].dt.year < int(p.split('_')[1])))[0]
        # p1, = ax_A.plot(index_ts[rows], t_means[rows], c="#771913", lw=3, label="Temperature")
        # p2, = twin1.plot(index_ts[rows], p_means[rows], c="#6094A5", lw=3, label="Precipitation")
        # p3, = twin2.plot(index_ts[rows], melt_means[rows], c="#9680AF", lw=3, label="Glacier Melt")
        #p3, = twin2.plot(index_ts[rows], t_means[rows], c="#771913", lw=3, label="Temperature", zorder=4)

        p1, = ax_A.plot(index_ts[rows], p_means[rows], c="#6094A5", lw=3, label="Precipitation", zorder=3)
        p2, = twin1.plot(index_ts[rows], melt_means[rows], c="#9680AF", lw=3, label="Glacier Melt", zorder=2)

        ax_A.set_xlim(index_ts[rows][0], index_ts[rows][-1])
        # ax_A.set_ylim(0, 15)
        # twin1.set_ylim(0, 30)
        # twin2.set_ylim(-20, 1750)
        ax_A.set_ylim(0, 500)
        #twin1.set_ylim(0, 30)
        twin1.set_ylim(-100, 5500)
        #twin2.set_ylim(-20, 1750)

        #ax_A.set_xlabel("Distance", fontsize=15)
        # ax_A.get_xaxis().set_visible(False)
        # ax_A.set_ylabel(r"Temperature (C)", fontsize=15)
        # twin1.set_ylabel("Precipitation (mm)", fontsize=15)
        # twin2.set_ylabel("Glacier Melt (BCM)", fontsize=15)
        ax_A.set_ylabel(r"Precipitation (mm)", fontsize=15)
        twin1.set_ylabel("Glacier Melt (MCM)", fontsize=15)
        #twin2.set_ylabel("Temperature (C)", fontsize=15)
        ax_A.set_title(f'{p.replace("_", "-")}', fontsize=16)

        ax_A.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        #twin2.yaxis.label.set_color(p3.get_color())

        # legend for panel a
        tkw = dict(size=4, width=1.5)
        ax_A.tick_params(axis='y', colors=p1.get_color(), **tkw)
        twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        #twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        #ax_A.tick_params(axis='x', **tkw)
        ax_A.set_xticks([])
        #legend = ax_A.legend(handles=[p1, p2, p3], fontsize=13, loc='upper center', ncol=3)
        legend = ax_A.legend(handles=[p1, p2], fontsize=13, loc='upper center', ncol=3)
        legend.get_frame().set_alpha(0)

        # lower panel: indices
        ax_B = fig.add_subplot(gs[2, 0])
        ax_B.patch.set_facecolor('0.9')
        ax_B.spines['top'].set_visible(False)
        ax_B.spines['right'].set_visible(False)
        ax_B.spines['bottom'].set_visible(False)
        ax_B.tick_params(axis='x', labelsize=13)
        ax_B.tick_params(axis='y', labelsize=13)
        ax_B.grid(False)

        # legend axis
        ax_leg = fig.add_subplot(gs[4, 0])
        ax_leg.set_xticks([])
        ax_leg.set_yticks([])
        for spine in ['top', 'bottom', 'left', 'right']:
            ax_leg.spines[spine].set_visible(False)

        # lower panel: drought indices
        ylims = [np.min([index1.iloc[:, z].min(), index2.iloc[:, z].min()]),
                 np.max([index1.iloc[:, z].max(), index2.iloc[:, z].max()])]

        # plot on lower panel
        # rows = np.where((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year >= int(p.split('_')[0])) &
        #                 ((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year <= int(p.split('_')[1]))))[0]
        rows = np.where((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year >= int(p.split('_')[0])) &
                        ((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year < int(p.split('_')[1]))))[0]
        ax_B.plot(index1.iloc[rows, z], color='black', label=drought_indices[0], lw=3, ls=":")
        ax_B.plot(index2.iloc[rows, z], color='#516E8C', label=drought_indices[0], lw=2.5)
        ax_B.set_ylabel('Drought Index (-)', fontsize=15)
        try:
            ax_B.set_xticks(np.arange(rows[0], rows[-1], 2*52),
                                  labels=np.arange(int(p.split("_")[0]), int(p.split("_")[1])+1, 2).astype(str))
        except:
            ax_B.set_xticks(np.arange(rows[0], rows[-1] + 2*52, 2 * 52),
                            labels=np.arange(int(p.split("_")[0]), int(p.split("_")[1]) + 1, 2).astype(str))
        ax_B.set_xlim(rows[0], rows[-1])
        ax_B.set_ylim(ylims)
        ax_B.axhline(-0, ls='--', label='Policy threshold', color='#D09972', lw=4)
        ax_B.annotate('k', [rows[-1]+10, 0], ha='center', va='center', weight='bold',
                            zorder=5, color='#D09972', fontsize=20, annotation_clip=False)
        for i in np.arange(rows[0], rows[-1], 0.5 * 52):
            ax.axvline(x=i, color='#B3BFC4', linestyle='-', alpha=0.8, lw=1)

        for i in np.arange(36 * 52 + 0, 36 * 52 + 10 * 52 + 1, 0.5 * 52):
            ax.axvline(x=i, color='#A498A3', linestyle='-', alpha=0.3, lw=1.5)

        # legend
        handles = [Line2D([0], [0], color=color, lw=2, ls=ls) for color, ls in zip(['black', '#516E8C', '#D09972'], [':', '-', '--'])]
        labels = drought_indices
        leg = ax_leg.legend(handles, labels, loc='center',
                            ncol=max(3, len(drought_indices)), frameon=False, fontsize=fontsize)
        leg.get_frame().set_alpha(0)  # Make the background transparent
        leg.get_frame().set_linewidth(0)

        fig.tight_layout()
        fig.show()

#%% Sample time series of all 3 climate variables (pair with figure 4 of drought indicator time series)

drought_indices = ['SPI12', 'SRI6']  # which indicators to plot
climate_scenario = [5]  # pick climate scenario 0-35 from CMIP6
periods = ['2020_2040', '2080_2100']
climate_scenario = [5]  # pick climate scenario 0-35 from CMIP6
periods = ['2020_2040', '2080_2100']
T = 1/52  # number of years for climate smoothing/time steps
T = 20/52  # number of years for climate smoothing/time steps

ax_periodB = fig.add_subplot(gs[2, 0])
#ax_periodB.patch.set_facecolor(color='#E7D1D4')
ax_periodB.patch.set_facecolor('0.9')
ax_periodB.spines['top'].set_visible(False)
ax_periodB.spines['right'].set_visible(False)
ax_periodB.spines['bottom'].set_visible(False)
ax_periodB.tick_params(axis='x', labelsize=13)
ax_periodB.tick_params(axis='y', labelsize=13)
ax_periodB.grid(False)

for z in climate_scenario:

    # load the drought index data
    index1 = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/TDP_scenarios_DS/{drought_indices[0]}.csv')
    index2 = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/TDP_scenarios_DS/{drought_indices[1]}.csv')

    # load the corresponding climate data
    scenario_name = index1.columns[z]
    t = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/UQM_weap_TempC_new/Tprom_semanal_{scenario_name}.csv')
    p = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/UQM_weap_TempC_new/PP_semanal_{scenario_name}.csv')
    melt = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/CMIP6_Glacier_Melt_TDP.csv')
    #melt = melt[scenario_name]
    ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                        format='%Y%W-%w')

    # calculate rolling mean of climate variables
    # indices to store 20-year average (i.e., starting 1999-12-27)
    index_ts = np.arange(len(ts))[::-1][0::int(52 * T)][::-1][1::]
    p_means = p.iloc[:, 3].shift().rolling(int(T * 52), min_periods=int(T * 52)).mean()[index_ts].values
    t_means = t.iloc[:, 3].shift().rolling(int(T * 52), min_periods=int(T * 52)).mean()[index_ts].values
    # check if index is correct for melt!
    melt_means = melt[scenario_name].shift().rolling(int(T * 52), min_periods=int(T * 52)).mean()[index_ts].values

    for p in periods:

        # initialize figure
        fig = plt.figure(figsize=(15, 7), constrained_layout=True)
        widths = [1]
        heights = [1, 0.3, 1, 0.25, 0.1]
        gs = fig.add_gridspec(ncols=1, nrows=5, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
        fig.tight_layout(h_pad=0)

        # upper panel: climate dynamics
        ax_A = fig.add_subplot(gs[0, 0])
        ax_A.patch.set_facecolor('0.9')
        ax_A.spines['top'].set_visible(False)
        ax_A.spines['right'].set_visible(False)
        ax_A.spines['bottom'].set_visible(False)
        ax_A.tick_params(axis='x', labelsize=13)
        ax_A.tick_params(axis='y', labelsize=13)
        ax_A.grid(False)

        # twin y axes for panel a
        twin1 = ax_A.twinx()
        twin2 = ax_A.twinx()

        # format twin axes
        twin1.spines['top'].set_visible(False)
        twin1.spines['left'].set_visible(False)
        twin1.spines['bottom'].set_visible(False)
        twin1.tick_params(axis='x', labelsize=13)
        twin1.tick_params(axis='y', labelsize=13)
        twin1.grid(False)

        twin2.spines['right'].set_visible(True)
        twin2.spines['right'].set_linewidth(1.5)
        twin2.spines['top'].set_visible(False)
        twin2.spines['left'].set_visible(False)
        twin2.spines['bottom'].set_visible(False)
        twin2.tick_params(axis='x', labelsize=13)
        twin2.tick_params(axis='y', labelsize=13)
        twin2.grid(False)

        # Offset the right spine of twin2.  The ticks and label have already been
        # placed on the right by twinx above.
        twin2.spines['right'].set_position(("axes", 1.07))

        # plot on panel a
        rows = np.where((ts[index_ts].dt.year >= int(p.split('_')[0])) & (ts[index_ts].dt.year <= int(p.split('_')[1])))[0]
        if p == '2020_2040':
            rows = np.where((ts[index_ts].dt.year >= int(p.split('_')[0])) & (ts[index_ts].dt.year < int(p.split('_')[1])))[0]
        else:
            rows = np.where((ts[index_ts].dt.year >= int(p.split('_')[0])) & (ts[index_ts].dt.year < int(p.split('_')[1])))[0]
        p1, = ax_A.plot(index_ts[rows], t_means[rows], c="#771913", lw=3, label="Temperature")
        p2, = twin1.plot(index_ts[rows], p_means[rows], c="#6094A5", lw=3, label="Precipitation")
        p3, = twin2.plot(index_ts[rows], melt_means[rows], c="#9680AF", lw=3, label="Glacier Melt")
        p3, = twin2.plot(index_ts[rows], t_means[rows], c="#771913", lw=3, label="Temperature", zorder=4)

        p1, = ax_A.plot(index_ts[rows], p_means[rows], c="#6094A5", lw=3, label="Precipitation", zorder=3)
        p2, = twin1.plot(index_ts[rows], melt_means[rows], c="#9680AF", lw=3, label="Glacier Melt", zorder=2)

        ax_A.set_xlim(index_ts[rows][0], index_ts[rows][-1])
        # ax_A.set_ylim(0, 15)
        # twin1.set_ylim(0, 30)
        # twin2.set_ylim(-20, 1750)
        ax_A.set_ylim(0, 500)
        #twin1.set_ylim(0, 30)
        twin1.set_ylim(-100, 5500)
        twin2.set_ylim(-20, 1750)

        #ax_A.set_xlabel("Distance", fontsize=15)
        # ax_A.get_xaxis().set_visible(False)
        # ax_A.set_ylabel(r"Temperature (C)", fontsize=15)
        # twin1.set_ylabel("Precipitation (mm)", fontsize=15)
        # twin2.set_ylabel("Glacier Melt (BCM)", fontsize=15)
        ax_A.set_ylabel(r"Precipitation (mm)", fontsize=15)
        twin1.set_ylabel("Glacier Melt (MCM)", fontsize=15)
        twin2.set_ylabel("Temperature (C)", fontsize=15)
        ax_A.set_title(f'{p.replace("_", "-")}', fontsize=16)

        ax_A.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())

        # legend for panel a
        tkw = dict(size=4, width=1.5)
        ax_A.tick_params(axis='y', colors=p1.get_color(), **tkw)
        twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        #ax_A.tick_params(axis='x', **tkw)
        ax_A.set_xticks([])
        legend = ax_A.legend(handles=[p1, p2, p3], fontsize=13, loc='upper center', ncol=3)
        legend = ax_A.legend(handles=[p1, p2], fontsize=13, loc='upper center', ncol=3)
        legend.get_frame().set_alpha(0)

        # lower panel: indices
        ax_B = fig.add_subplot(gs[2, 0])
        ax_B.patch.set_facecolor('0.9')
        ax_B.spines['top'].set_visible(False)
        ax_B.spines['right'].set_visible(False)
        ax_B.spines['bottom'].set_visible(False)
        ax_B.tick_params(axis='x', labelsize=13)
        ax_B.tick_params(axis='y', labelsize=13)
        ax_B.grid(False)

        # legend axis
        ax_leg = fig.add_subplot(gs[4, 0])
        ax_leg.set_xticks([])
        ax_leg.set_yticks([])
        for spine in ['top', 'bottom', 'left', 'right']:
            ax_leg.spines[spine].set_visible(False)

        # lower panel: drought indices
        ylims = [np.min([index1.iloc[:, z].min(), index2.iloc[:, z].min()]),
                 np.max([index1.iloc[:, z].max(), index2.iloc[:, z].max()])]

        # plot on lower panel
        # rows = np.where((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year >= int(p.split('_')[0])) &
        #                 ((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year <= int(p.split('_')[1]))))[0]
        rows = np.where((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year >= int(p.split('_')[0])) &
                        ((pd.to_datetime(index1.iloc[:, 0].values, format='%d-%m-%Y').year < int(p.split('_')[1]))))[0]
        ax_B.plot(index1.iloc[rows, z], color='black', label=drought_indices[0], lw=3, ls=":")
        ax_B.plot(index2.iloc[rows, z], color='#516E8C', label=drought_indices[0], lw=2.5)
        ax_B.set_ylabel('Drought Index (-)', fontsize=15)
        try:
            ax_B.set_xticks(np.arange(rows[0], rows[-1], 2*52),
                                  labels=np.arange(int(p.split("_")[0]), int(p.split("_")[1])+1, 2).astype(str))
        except:
            ax_B.set_xticks(np.arange(rows[0], rows[-1] + 2*52, 2 * 52),
                            labels=np.arange(int(p.split("_")[0]), int(p.split("_")[1]) + 1, 2).astype(str))
        ax_B.set_xlim(rows[0], rows[-1])
        ax_B.set_ylim(ylims)
        ax_B.axhline(-0, ls='--', label='Policy threshold', color='#D09972', lw=4)
        ax_B.annotate('k', [rows[-1]+10, 0], ha='center', va='center', weight='bold',
                            zorder=5, color='#D09972', fontsize=20, annotation_clip=False)
        for i in np.arange(rows[0], rows[-1], 0.5 * 52):
            ax.axvline(x=i, color='#B3BFC4', linestyle='-', alpha=0.8, lw=1)

        for i in np.arange(36 * 52 + 0, 36 * 52 + 10 * 52 + 1, 0.5 * 52):
            ax.axvline(x=i, color='#A498A3', linestyle='-', alpha=0.3, lw=1.5)

        # legend
        handles = [Line2D([0], [0], color=color, lw=2, ls=ls) for color, ls in zip(['black', '#516E8C', '#D09972'], [':', '-', '--'])]
        labels = drought_indices
        leg = ax_leg.legend(handles, labels, loc='center',
                            ncol=max(3, len(drought_indices)), frameon=False, fontsize=fontsize)
        leg.get_frame().set_alpha(0)  # Make the background transparent
        leg.get_frame().set_linewidth(0)

        fig.tight_layout()
        fig.show()

#%% Figure 4: Sample time series plots

drought_indices = ['SPI12', 'SRI6']  # which indicators to plot
climate_scenario = [5]  # pick climate scenario 0-35 from CMIP6
periods = ['2020_2040', '2080_2100']

### initialize figure for plotting
#fig, ax = plt.subplots(1, 1, figsize=(6, 4), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
#fig = plt.figure(figsize=(10, 8), constrained_layout=True)
fig = plt.figure(figsize=(15, 7), constrained_layout=True)
widths = [1]
heights = [1, 0.3, 1, 0.25, 0.1]
gs = fig.add_gridspec(ncols=1, nrows=5, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
fig.tight_layout(h_pad=0)

# name axes, based on objectives shown per subplot (here, num obj=3)
ax_periodA = fig.add_subplot(gs[0, 0])
#ax_periodA.patch.set_facecolor(color='#CCDADD')
ax_periodA.patch.set_facecolor('0.9')
ax_periodA.spines['top'].set_visible(False)
ax_periodA.spines['right'].set_visible(False)
ax_periodA.spines['bottom'].set_visible(False)
ax_periodA.tick_params(axis='x', labelsize=13)
ax_periodA.tick_params(axis='y', labelsize=13)
ax_periodA.grid(False)

ax_periodB = fig.add_subplot(gs[2, 0])
#ax_periodB.patch.set_facecolor(color='#E7D1D4')
ax_periodB.patch.set_facecolor('0.9')
ax_periodB.spines['top'].set_visible(False)
ax_periodB.spines['right'].set_visible(False)
ax_periodB.spines['bottom'].set_visible(False)
ax_periodB.tick_params(axis='x', labelsize=13)
ax_periodB.tick_params(axis='y', labelsize=13)
ax_periodB.grid(False)

ax_leg = fig.add_subplot(gs[4, 0])
ax_leg.set_xticks([])
ax_leg.set_yticks([])
for spine in ['top', 'bottom', 'left', 'right']:
    ax_leg.spines[spine].set_visible(False)

#for z in range(sri3CMIP6.columns.shape[0] -1): # for each climate scenario
for z in climate_scenario:

    index1 = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/TDP_scenarios_DS/{drought_indices[0]}.csv')
    index2 = pd.read_csv(f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/TDP_scenarios_DS/{drought_indices[1]}.csv')
    ylims = [np.min([index1.iloc[:, z].min(), index2.iloc[:, z].min()]),
             np.max([index1.iloc[:, z].max(), index2.iloc[:, z].max()])]
    # periodA: 2020-2040
    rows = np.where((pd.to_datetime(index1.iloc[:,0].values, format='%d-%m-%Y').year >= int(periods[0].split('_')[0])) &
                    ((pd.to_datetime(index1.iloc[:,0].values, format='%d-%m-%Y').year < int(periods[0].split('_')[1]))))[0]
    ax_periodA.plot(index1.iloc[rows, z], color='black', label=drought_indices[0], lw=3, ls=":")
    ax_periodA.plot(index2.iloc[rows, z], color='#516E8C', label=drought_indices[0], lw=2.5)
    ax_periodA.set_ylabel('Drought Index (-)', fontsize=15)
    ax_periodA.set_xticks(np.arange(rows[0], rows[-1], 2*52),
                          labels=np.arange(int(periods[0].split("_")[0]), int(periods[0].split("_")[1])+1, 2).astype(str))
    ax_periodA.set_xlim(rows[0], rows[-1])
    ax_periodA.set_ylim(ylims)
    ax_periodA.axhline(-0, ls='--', label='Policy threshold', color='#D09972', lw=4)
    ax_periodA.set_title(f'{periods[0].replace("_", "-")}', fontsize=16)
    for i in np.arange(rows[0], rows[-1], 0.5 * 52):
        ax.axvline(x=i, color='#B3BFC4', linestyle='-', alpha=0.8, lw=1)

    # period B
    rows = np.where((pd.to_datetime(index1.iloc[:,0].values, format='%d-%m-%Y').year >= 2078) &
                    ((pd.to_datetime(index1.iloc[:,0].values, format='%d-%m-%Y').year < 2098)))[0]
    ax_periodB.plot(index1.iloc[rows, z], color='black', label=drought_indices[0], lw=3, ls=':')
    ax_periodB.plot(index2.iloc[rows, z], color='#516E8C', label=drought_indices[0], lw=2.5)
    ax_periodB.set_ylabel('Drought Index (-)', fontsize=15)
    ax_periodB.set_xlabel('Time', fontsize=15)
    ax_periodB.set_xticks(np.arange(rows[0], rows[-1], 2*52),
                          labels=np.arange(int(periods[1].split("_")[0]), int(periods[1].split("_")[1]) + 1, 2).astype(
                              str))
    ax_periodB.set_xlim(rows[0], rows[-1])
    ax_periodB.set_ylim(ylims)
    ax_periodB.axhline(-0, ls='--', label='Policy threshold', color='#D09972', lw=4)
    ax_periodB.set_title(f'{periods[1].replace("_", "-")}', fontsize=16)
    for i in np.arange(rows[0], rows[-1], 0.5 * 52):
        ax.axvline(x=i, color='#B3BFC4', linestyle='-', alpha=0.8, lw=1)

    for i in np.arange(36*52+0, 36*52+10*52+1, 0.5*52):
        ax.axvline(x=i, color='#A498A3', linestyle='-', alpha=0.3, lw=1.5)

    handles = [Line2D([0], [0], color=color, lw=2) for color in ['black', '#516E8C']]
    labels = drought_indices
    leg = ax_leg.legend(handles, labels, loc='center',
                  ncol=max(3, len(drought_indices)), frameon=False, fontsize=fontsize)
    leg.get_frame().set_alpha(0)  # Make the background transparent
    leg.get_frame().set_linewidth(0)
    #plt.gca().patch.set_facecolor('0.9')

    fig.tight_layout()
    fig.show()

#%% Define functions and create dictionary to store and simulate sample policies for result plots

import platform  # helps identify directory locations on different types of OS
import sys

# pywr imports
from pywr.core import *
import tables
from pywr.parameters import *

from pywr.parameters._thresholds import StorageThresholdParameter, ParameterThresholdParameter
from pywr.recorders import TablesRecorder, DeficitFrequencyNodeRecorder, TotalDeficitNodeRecorder, MeanFlowNodeRecorder, \
    NumpyArrayParameterRecorder, NumpyArrayStorageRecorder, NumpyArrayNodeRecorder, RollingMeanFlowNodeRecorder, \
    AggregatedRecorder
# from MAIPO_searcher import *
from pywr.dataframe_tools import *

import os
print(os.chdir("/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR"))
os.chdir("/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR")

# set path to project root
# sys.path.append('..')
# os.chdir('..')
sys.path.append('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/dps_BORG/BorgMOEA_master/plugins/Python')

#from MAIPO_PYWR.dps_BORG.borg import *
try:
    from MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg import *  # Import borg wrapper
except:
    from dps_BORG.BorgMOEA_master.plugins.Python.borg import *

#from MAIPO_PYWR.MAIPO_parameters import *
try:
    from MAIPO_PYWR.MAIPO_parameters_DS_V2 import *
except:
    from MAIPO_parameters_DS_V2 import *
# from MAIPO_PYWR.dps_BORG.MAIPO_DPS import *

import importlib
# BorgMOEA = __import__("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\dps_BORG\\BorgMOEA_master")
# BorgMOEA = importlib.import_module("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\dps_BORG\\BorgMOEA_master")
# from MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg import borg as bg  # Import borg wrapper

num_k = 1  # number of levels in policy tree
num_DP = 6  # number of decision periods

def make_model_20yrA(contract_threshold_vals=-999999 * np.ones(num_DP), contract_action_vals=np.zeros(num_DP),
               demand_threshold_vals=[], demand_action_vals=[np.ones(12)], indicator="SRI3",
               drought_status_agg="drought_status_single_week",
               data_folder="TDP_scenarios_DS"):
    '''
    Purpose: Creates a Pywr model with the specified number and values for policy thresholds/actions. Intended for use with MOEAs.

    Args:
        threshold_vals: an array of policy thresholds for drought index
        action_vals: an array of policy actions corresponding to policy thresholds

    Returns:
        model: a Pywr Model object
    '''

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
    model.timestepper = Timestepper(
        start=pd.to_datetime('2020-03-12'),  # start
        end=pd.to_datetime('2040-02-16'),  # end
        delta=datetime.timedelta(7)  # interval
    )

    # SCENARIOS
    num_scenarios = 36
    Scenario(model, name="climate change", size=num_scenarios)

    # REQUIRED NODES FOR PARAMETERS
    # DP_index -- which development period we're in
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/{}.csv'.format(data_folder, indicator),  # 'data/SRI6.csv'
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/YESO.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/MAIPO.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/COLORADO.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/VOLCAN.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/LAGUNANEGRA.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/MAIPOEXTRA.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
            model.recorders["OctoberContractCost"]
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
    # model.check_graph()  # check the connectivity of the graph
    # model.check()  # check the validity of the model

    return model

def make_model_20yrB(contract_threshold_vals=-999999 * np.ones(num_DP), contract_action_vals=np.zeros(num_DP),
               demand_threshold_vals=[], demand_action_vals=[np.ones(12)], indicator="SRI3",
               drought_status_agg="drought_status_single_week",
               data_folder="TDP_scenarios_DS"):
    '''
    Purpose: Creates a Pywr model with the specified number and values for policy thresholds/actions. Intended for use with MOEAs.

    Args:
        threshold_vals: an array of policy thresholds for drought index
        action_vals: an array of policy actions corresponding to policy thresholds

    Returns:
        model: a Pywr Model object
    '''

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
    model.timestepper = Timestepper(
        start=pd.to_datetime('2078-12-22'),  # start
        end=pd.to_datetime('2098-11-27'),  # end
        delta=datetime.timedelta(7)  # interval
    )

    # SCENARIOS
    num_scenarios = 36
    Scenario(model, name="climate change", size=num_scenarios)

    # REQUIRED NODES FOR PARAMETERS
    # DP_index -- which development period we're in
    datestr = ["2075-01-03", "2079-12-28", "2084-12-21", "2089-12-15", "2094-12-09", "2099-12-03"]
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/{}.csv'.format(data_folder, indicator),  # 'data/SRI6.csv'
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/YESO.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/MAIPO.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/COLORADO.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/VOLCAN.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/LAGUNANEGRA.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/MAIPOEXTRA.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
        'url': '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/{}/Extra data.csv'.format(data_folder),
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
            model.recorders["OctoberContractCost"]
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
    # model.check_graph()  # check the connectivity of the graph
    # model.check()  # check the validity of the model

    return model


# dictionary of policies
drought_indices = ['SRI3', 'SRI6', 'SRI12', 'SPI3', 'SPI6', 'SPI12']
periods = ['2020_2040', '2080_2100']
nFunc = 2000  # number of function evaluations
disc =35
#seeds = np.arange(5) + 1  # for each seed
#seeds = [1]  # plot 1 seed for now


# load objective results and store in master dictionary
columns_axes = ['Total Cost', 'Urban Rel.', 'Agr. Rel.']  # for objs_dict
objs_dict = {}
threshs_dict = {}
acts_dict = {}

for p in periods:
    objs_dict[p] = {}
    threshs_dict[p] = {}
    acts_dict[p] = {}
    for d in drought_indices:
        objs_dict[p][d] = {}
        threshs_dict[p][d] = {}
        acts_dict[p][d] = {}

for p in periods:
    for j, d in enumerate(drought_indices):

        # Concatenate solutions across all seeds for this scenario
        all_objs = []
        all_threshs = []
        all_acts = []

        for i in seeds:

            file = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/outputs/{fol}/{d}_{p}_CMIP6_{date}_nFunc{nFunc}_disc{disc}/sets/Borg_DPS_PySedSim{i}.set'

            # formulate the table
            with open(file, 'r') as f:
               #data = pd.read_csv(file, skiprows=2)
               data = pd.read_csv(file, skiprows=2, skipfooter=1, header=None, engine='python').explode(0)
               data = pd.DataFrame(data.iloc[:,0].str.split(' ').explode())

            num_vars = np.sum(data.index == 0)
            num_rows = data.index.unique().shape[0]
            data['vars'] = np.tile(np.arange(num_vars), num_rows)

            data = data.reset_index().pivot(index='index', columns='vars')

            # objectives
            urban_rel = data.iloc[:, 4].values.astype('float')
            ag_rel = data.iloc[:, 5].values.astype('float')
            cost = data.iloc[:, 6].values.astype('float') * -1/1E6  # millions of dollars

            # thresholds
            contract_thresh = data.iloc[:, 0].values.astype('float')
            demand_thresh = data.iloc[:, 2].values.astype('float')

            # actions
            contract_action = data.iloc[:, 1].values.astype('float')
            demand_action = data.iloc[:, 3].values.astype('float')

            #% data syntax for plotting
            objs = pd.DataFrame(data=np.vstack([cost, urban_rel, ag_rel]).T, columns=['Total Cost', 'Urban Rel.', 'Agr. Rel.'])
            objs = objs.loc[:, columns_axes]

            threshs = pd.DataFrame(data=np.vstack([contract_thresh, demand_thresh]).T, columns=['Contracts', 'Demand Rest.'])

            acts = pd.DataFrame(data=np.vstack([contract_action, demand_action]).T, columns=['Contracts', 'Demand Rest.'])
            all_objs.append(objs)
            all_threshs.append(threshs)
            all_acts.append(acts)

        # Store pooled sets (all seeds) for this scenario
        objs_dict[p][d] = pd.concat(all_objs, ignore_index=True) if len(all_objs) else pd.DataFrame(columns=columns_axes)
        threshs_dict[p][d] = pd.concat(all_threshs, ignore_index=True) if len(all_threshs) else pd.DataFrame(columns=['Contracts', 'Demand Rest.'])
        acts_dict[p][d] = pd.concat(all_acts, ignore_index=True) if len(all_acts) else pd.DataFrame(columns=['Contracts', 'Demand Rest.'])



#% Simulate sample policies over different time periods
# drought_indices = ['SPI3', 'SRI6']
# policies_A = [0, 0]  # row index of sample policies in 2020-2040
# policies_B = [0, 0]  # row index of sample policies in 2020-2040
# climate_scenario = [5]  # number climate scenario index from 0-35 for CMIP6

urban_rel_dict = {}
ag_rel_dict = {}
total_cost_dict = {}
total_contracts_dict = {}
total_PT1_demand_restr_dict = {}
for p in periods:
    urban_rel_dict[p] = {}
    ag_rel_dict[p] = {}
    total_cost_dict[p] = {}
    total_contracts_dict[p] = {}
    total_PT1_demand_restr_dict[p] = {}
    for d in drought_indices:
        urban_rel_dict[p][d] = {}
        ag_rel_dict[p][d] = {}
        total_cost_dict[p][d] = {}
        total_contracts_dict[p][d] = {}
        total_PT1_demand_restr_dict[p][d] = {}


#%% Whether to rerun simulations of optimal policies -> use pickle to save results for plotting
run_sims = False

if run_sims == True:
    for i, p in enumerate(periods):  # for each time period
        for j, d in enumerate(drought_indices):  # for each drought index

            print(f'{p}, {d}')

            # initialize vectors to save performance of policies
            urban_rel = np.zeros([threshs_dict[p][d].shape[0], 36])
            ag_rel = np.zeros([threshs_dict[p][d].shape[0], 36])
            total_cost = np.zeros([threshs_dict[p][d].shape[0], 36])
            total_contracts = np.zeros([threshs_dict[p][d].shape[0], 36])
            total_PT1_demand_restr = np.zeros([threshs_dict[p][d].shape[0], 36])


            for k in range(threshs_dict[p][d].shape[0]):  # for each policy

                # load parameters for 2020-2040:
                contract_threshold_vals = threshs_dict[p][d]['Contracts'][k] * np.ones(num_DP)
                contract_action_vals = acts_dict[p][d]['Contracts'][k] * np.ones(num_DP)
                demand_threshold_vals = [threshs_dict[p][d]['Demand Rest.'][k] * np.ones(12)]
                demand_action_vals = [np.ones(12), acts_dict[p][d]['Demand Rest.'][k] * np.ones(12)]

                # make the model for the correct time period
                if p == '2020_2040':
                    m = make_model_20yrA(
                            contract_threshold_vals=contract_threshold_vals,
                            contract_action_vals=contract_action_vals,
                            demand_threshold_vals=demand_threshold_vals,
                            demand_action_vals=demand_action_vals,
                            indicator=d
                        )
                elif p == '2080_2100':
                    m = make_model_20yrB(
                        contract_threshold_vals=contract_threshold_vals,
                        contract_action_vals=contract_action_vals,
                        demand_threshold_vals=demand_threshold_vals,
                        demand_action_vals=demand_action_vals,
                        indicator=d
                    )

                    # set up tables recorder to record simulation outputs
                    # OUTPUT_DIR = f"/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/CMIP6_sims/2020_2040/{ind}/Policy_{policies_A[0]}"
                    # TablesRecorder(mA, OUTPUT_DIR + "/flows.h5", filter_kwds={"complib": "zlib", "complevel": 5},
                    #                parameters=["contract_value", "DP_index", "purchases_value", "Maipo_max_volume"])

                #simulate model
                m.run()

                # save simulation results in dictionary (1 aggregate value per simulation)
                #deficit_PT1_A = np.asarray(m.recorders['deficit PT1'].values())
                #deficit_Ag_A = np.asarray(m.recorders['deficit Ag'].values())

                urban_rel[k, :] = np.asarray(m.recorders['reliability_PT1'].values())
                ag_rel[k, :] = np.asarray(m.recorders['reliability_Ag'].values())
                total_cost[k, :] = np.asarray(m.recorders['TotalCost'].values())
                total_contracts[k, :] = np.asarray(m.recorders['Total Contracts Made'].values())
                PT1_demand = np.array(m.parameters["demanda_PT1"].get_all_values())
                PT1_demand_restr = np.array(m.parameters["demand_max_flow_PT1"].get_all_values())
                total_PT1_demand_restr[k, :] = PT1_demand - PT1_demand_restr  # how much demand restricted over period

            urban_rel_dict[p][d] = urban_rel
            ag_rel_dict[p][d] = ag_rel
            total_cost_dict[p][d] = total_cost
            total_contracts_dict[p][d] = total_contracts
            total_PT1_demand_restr_dict[p][d] = total_PT1_demand_restr

    # save dictionary results to avoid recalculation
    import pickle

    pkl_folder = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}'
    os.makedirs(pkl_folder, exist_ok=True)

    filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/urban_rel_dict.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(urban_rel_dict, file)

    filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/ag_rel_dict.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(ag_rel_dict, file)

    filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_cost_dict.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(total_cost_dict, file)

    filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_contracts_dict.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(total_contracts_dict, file)

    filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_PT1_demand_restr_dict.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(total_PT1_demand_restr_dict, file)


#%% FIGURE 5
# NEW from Sarah: Scatter plot comparing mean urban reliability of best policies at start vs. EOC by climate state
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter


fig = plt.figure(figsize=(10, 3), constrained_layout=True)
widths = [1, 0.1, 1]
heights = [1]
gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
fig.tight_layout(h_pad=0)

# 2020-2040
ax1 = fig.add_subplot(gs[0, 0])
# ax1.set_xticks([])
# ax1.set_yticks([])
#ax1.set_facecolor('0.9')
# for spine in ['top', 'bottom', 'left', 'right']:
#     ax1.spines[spine].set_visible(False)

# 2080-2100
ax2 = fig.add_subplot(gs[0, 2])
# ax2.set_xticks([])
# ax2.set_yticks([])
#ax2.set_facecolor('0.9')

## parameters for loading climate variables
path_clima = '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/'
models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
combinations = ["_".join(pair) for pair in product(models, experiments)]
PP = {el:pd.DataFrame() for el in combinations}  # dictionary for precip
TT = {el:pd.DataFrame() for el in combinations}  # dictionary for temp
SS = {el:pd.DataFrame() for el in combinations}  # dictionary for streamflow
# o = range(3)
o = 2  # change for each objective: total cost, urban rel, ag rel


for j in [o]: #range(3):  # for each objective

    import pickle

    # load simulation result dictionaries
    if j == 1:
        #filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/seed_{i}/urban_rel_dict.pkl'
        filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/urban_rel_dict.pkl'

    elif j == 2:
        #filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/seed_{i}/ag_rel_dict.pkl'
        filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/ag_rel_dict.pkl'
    elif j == 0:
        #filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/seed_{i}/total_cost_dict.pkl'
        filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_cost_dict.pkl'


    with open(filename, 'rb') as file:
        dic = pickle.load(file)

    # Dictionaries to store results
    global_best_dict = {}
    best_drought_index_dict = {}
    best_performance_dict = {}

    # Iterate over each period in the dictionary
    for p in dic.keys():
        # Initialize lists to store maximums and corresponding drought indices for the current period
        best_values = []
        best_indices = []
        best_performance_dict[p] = {}

        # Iterate over each drought index in the current period
        for d in dic[p].keys():
            # Get the dataframe for the current period and drought index
            df = dic[p][d]

            # Find the maximum value for each column and store in a list
            if j == 0:
                best_values.append(df.min(axis=0))
                best_performance_dict[p][d] = df.min(axis=0)
            else:
                best_values.append(df.max(axis=0))
                best_performance_dict[p][d] = df.max(axis=0)

        # Convert list of maxima to a numpy array
        best_values = np.array(best_values)

        # Find the global maximum for each column and its corresponding drought index
        if j == 0:
            global_best_dict[p] = np.min(best_values, axis=0)
            best_indices = np.argmin(best_values, axis=0)
        else:
            global_best_dict[p] = np.max(best_values, axis=0)
            best_indices = np.argmax(best_values, axis=0)

        # Map the index back to the actual drought index key
        drought_keys = list(dic[p].keys())
        best_drought_index_dict[p] = [drought_keys[i] for i in best_indices]

    # Convert the results to dataframes for better visualization (optional)
    global_best_df = pd.DataFrame(global_best_dict, index=[f"Column_{i + 1}" for i in range(36)])
    best_drought_df = pd.DataFrame(best_drought_index_dict, index=[f"Column_{i + 1}" for i in range(36)])

    # Display the global maxima and corresponding drought indices
    # print("Global Best Values:")
    # print(global_best_df)
    # print("\nDrought Indices with Global Best:")
    # print(best_drought_df)

    # get the maximum urban reliability if only thresholds updated from 2020-2040, not indicators themselves
    historical_best = []
    for i in range(36):  # for each climate change scenario
        if j == 0:
            #historical_best.append(dict['2080_2100'][best_drought_index_dict['2020_2040'][i]].min(axis=0)[i])
            historical_best.append(dic['2080_2100'][best_drought_index_dict['2020_2040'][i]].min(axis=0)[i])
        else:
            #historical_best.append(dict['2080_2100'][best_drought_index_dict['2020_2040'][i]].max(axis=0)[i])
            historical_best.append(dic['2080_2100'][best_drought_index_dict['2020_2040'][i]].max(axis=0)[i])


# load climate variables
for i, m in enumerate(models):
    for j, x in enumerate(experiments):

        # set sampling parameters
        T = 20  # number of years for smoothing/time steps
        # load the datafile and set its timestamp
        P = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')
        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')

        # indices to store 20-year average (i.e., starting 1999-12-27)
        index_ts = np.arange(len(ts))[::-1][0::52 * T][::-1][1::]
        p_means = P.iloc[:, 3].shift().rolling(T * 52, min_periods=T * 52).mean().values
        t_means = t.iloc[:, 3].shift().rolling(T * 52, min_periods=T * 52).mean().values

        # add 20-year average T and P to dictionaries
        PP[f"{m}_{x}"] = {"date": ts.to_numpy(), "mean": p_means}
        TT[f"{m}_{x}"] = {"date": ts.to_numpy(), "mean": t_means}

files = ['COLORADO', 'MAIPO', 'YESO', 'LAGUNANEGRA', 'VOLCAN', 'MAIPOEXTRA']
for i, c in enumerate(combinations):
    I_tot = pd.read_csv(f'data/TDP_scenarios/{files[0]}_2004Wk14-2099Wk13.csv', index_col=0)[c]
    for f in files[1::]:
        I_tot_plus = pd.read_csv(f'data/TDP_scenarios/{f}_2004Wk14-2099Wk13.csv', index_col=0)[c]
        I_tot = I_tot + I_tot_plus
    s_rolling = I_tot.shift().rolling(20*52, min_periods=5).mean()
    SS[f"{c}"] = {"date": s_rolling.index.to_numpy(), "mean": s_rolling.values}

# get the final climate state variables
final_T = np.zeros([1, 36])
final_P = np.zeros([1, 36])
final_S = np.zeros([1, 36])
for i, c in enumerate(combinations):
    final_T[0, i] = TT[f"{c}"]['mean'][-1]
    final_P[0, i] = PP[f"{c}"]['mean'][-1]
    final_S[0, i] = SS[f"{c}"]['mean'][-1]

# index of scenarios representing low and high final climate variables
index_lowT = np.where(final_T <= 8)[1]
index_highT = np.where(final_T >=10)[1]

index_lowP = np.where(final_P <= 9)[1]
index_highP = np.where(final_P >= 11)[1]

index_lowS = np.where(final_S <= 40)[1]
index_highS = np.where(final_S >= 55)[1]

#glacier melt
glac = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/CMIP6_Glacier_Melt_TDP.csv', index_col=1)[combinations]
glac_rolling = glac.shift().rolling(20*52, min_periods=5).mean() * 0.017

index_lowG = np.where(glac_rolling.iloc[-1, :] < 0.1)[0]
index_highG = np.where(glac_rolling.iloc[-1, :] > 5)[0]

# scatter plot of mean best simulated performance of policies conditioned on different indicators by climate partition

# low temperature
best_perf_lowT = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_lowT[i, j] = best_performance_dict[p][d][index_lowT].mean()

# high temp
best_perf_highT = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_highT[i, j] = best_performance_dict[p][d][index_highT].mean()

# low precip
best_perf_lowP = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_lowP[i, j] = best_performance_dict[p][d][index_lowP].mean()

# high precip
best_perf_highP = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_highP[i, j] = best_performance_dict[p][d][index_highP].mean()

# low streamflow
best_perf_lowS = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_lowS[i, j] = best_performance_dict[p][d][index_lowS].mean()

# high streamflow
best_perf_highS = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_highS[i, j] = best_performance_dict[p][d][index_highS].mean()

# low glacier
best_perf_lowG = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_lowG[i, j] = best_performance_dict[p][d][index_lowG].mean()

# high glacier
best_perf_highG = np.zeros((len(dic.keys()), len(dic[p].keys())))
for i, p in enumerate(dic.keys()):
    for j, d in enumerate(dic[p].keys()):
        best_perf_highG[i, j] = best_performance_dict[p][d][index_highG].mean()

# plot
colors = ['#994455', '#6699cc', '#004488', '#997700']#['r', 'g', 'b', 'chocolate']
labels = ['high T', 'high P', 'high S', 'high G', 'low T', 'low P', 'low S', 'low G']
alphas = [1, 0.5]
for i, p in enumerate(dic.keys()):
    if i == 0:
        ax = ax1
    elif i == 1:
        ax = ax2
    scatters = []
    for j in np.arange(len(dic[p].keys())):
        data = np.vstack([[best_perf_highT[i, j], best_perf_highP[i, j], best_perf_highS[i, j], best_perf_highG[i, j]],
                         [best_perf_lowT[i, j], best_perf_lowP[i, j], best_perf_lowS[i, j], best_perf_lowG[i, j]]])
        if o == 0: # total cost
            data = data/1E6 # millions
        # Plot each color separately
        for k, color in enumerate(colors):
            # Only label once for the first subplot + first drought index
            add_label = (i == 0 and j == 0)

            x_pos = j - 1 / 5 + k / 5
            sc_high = ax.scatter(x_pos, data[0, k], color=color, alpha=alphas[0],
                                 label=labels[k] if add_label else None)
            sc_low = ax.scatter(x_pos, data[1, k], color=color, alpha=alphas[1],
                                label=labels[k + 4] if add_label else None)

            # vertical connecting line (not in legend)
            ax.vlines(x_pos, data[1, k], data[0, k],
                color='black', alpha=0.8, linewidth=1, zorder=0
            )

    ax.set_title(p.replace('_', '-'))
    ax.set_xticklabels(dic[p].keys())

    ax.set_xticks(np.arange(len(dic[p].keys())))
    ax.set_xlim([-0.5, len(dic[p].keys()) - 0.25])
    if i==0:
        if o==0:
            ax.set_ylabel('Total Cost (M$)')
        if o==1:
            ax.set_ylabel('Urban Reliability')
            ax.legend(frameon=False, ncol=4)
        elif o==2:
            ax.set_ylabel('Agricultural Reliability')
    if i==1:
        ax.set_yticklabels([])

    # ylims
    if o == 0:
        ax.set_ylim([0, 10000])
    elif o == 1:
        ax.set_ylim([0.5, 1])
    elif o == 2:
        ax.set_ylim([0.2, 0.7])

fig.tight_layout()
fig.show()


#%% FIGURE 1 B-C
# Motivation plot V3! plot overlapping SRI and SPI -12 for conceptual comparison and streamflow, precip, and temp in panel above
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


# initialize figure
#fig = plt.figure(figsize=(10, 8), constrained_layout=True)
fig = plt.figure(figsize=(15, 8), constrained_layout=True)
widths = [1]
heights = [1, 0.1, 1, 0.25]
gs = fig.add_gridspec(ncols=1, nrows=4, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
fig.tight_layout(h_pad=0)

# format panels
ax_B = fig.add_subplot(gs[2, 0])
#ax_B.patch.set_facecolor('0.9')
# ax_B.spines['top'].set_visible(False)
# ax_B.spines['right'].set_visible(False)
# ax_B.spines['bottom'].set_visible(False)
ax_B.tick_params(axis='x', labelsize=13)
ax_B.tick_params(axis='y', labelsize=13)
ax_B.grid(False)

ax_A = fig.add_subplot(gs[0, 0])
#ax_A.patch.set_facecolor('0.9')
# ax_A.spines['top'].set_visible(False)
# ax_A.spines['right'].set_visible(False)
# ax_A.spines['bottom'].set_visible(False)
ax_A.tick_params(axis='x', labelsize=13)
ax_A.tick_params(axis='y', labelsize=13)
ax_A.grid(False)

# twin y axes for panel a
twin1 = ax_A.twinx()
twin2 = ax_A.twinx()

# format twin axes
twin1.spines['top'].set_visible(False)
twin1.spines['left'].set_visible(False)
twin1.spines['bottom'].set_visible(False)
twin1.tick_params(axis='x', labelsize=13)
twin1.tick_params(axis='y', labelsize=13)
twin1.grid(False)

twin2.spines['right'].set_visible(True)
twin2.spines['right'].set_linewidth(1.5)
twin2.spines['top'].set_visible(False)
twin2.spines['left'].set_visible(False)
twin2.spines['bottom'].set_visible(False)
twin2.tick_params(axis='x', labelsize=13)
twin2.tick_params(axis='y', labelsize=13)
twin2.grid(False)

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
#twin2.spines['right'].set_position(("axes", 1.1))
twin2.spines['right'].set_position(("axes", 1.07))

# parameters for loading climate variables
path_clima = '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/'
models = ['ACCESS-CM2']
experiments = ['ssp585']
combinations = ["_".join(pair) for pair in product(models, experiments)]
PP = {el:pd.DataFrame() for el in combinations}  # dictionary for precip
TT = {el:pd.DataFrame() for el in combinations}  # dictionary for temp

# load climate variables
for i, m in enumerate(models):
    for j, x in enumerate(experiments):

        # set sampling parameters
        T = 20  # number of years for smoothing/time steps
        # load the datafile and set its timestamp
        p = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{m}_{x}.csv')
        t = pd.read_csv(path_clima + f'UQM_weap_TempC_new/Tprom_semanal_{m}_{x}.csv')
        ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                            format='%Y%W-%w')

        # indices to store 20-year average (i.e., starting 1999-12-27)
        index_ts = np.arange(len(ts))[::-1][0::52 * T][::-1][1::]
        p_means = p.iloc[:, 3].shift().rolling(T * 52, min_periods=T * 52).mean().values
        t_means = t.iloc[:, 3].shift().rolling(T * 52, min_periods=T * 52).mean().values

        # add 20-year average T and P to dictionaries
        PP[f"{m}_{x}"] = {"date": ts.to_numpy(), "mean": p_means}
        TT[f"{m}_{x}"] = {"date": ts.to_numpy(), "mean": t_means}

files = ['COLORADO', 'MAIPO', 'YESO', 'LAGUNANEGRA', 'VOLCAN', 'MAIPOEXTRA']
I_tot = pd.read_csv(f'data/TDP_scenarios/{files[0]}_2004Wk14-2099Wk13.csv', index_col=0)[combinations[0]]
for f in files[1::]:
    I_tot_plus = pd.read_csv(f'data/TDP_scenarios/{f}_2004Wk14-2099Wk13.csv', index_col=0)[combinations[0]]
    I_tot = I_tot + I_tot_plus
s_rolling = I_tot.shift().rolling(20*52, min_periods=5).mean()

# glacier melt
glac = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/CMIP6_Glacier_Melt_TDP.csv', index_col=1)[combinations[0]]
glac_rolling = glac.shift().rolling(20*52, min_periods=5).mean() * 0.017

# plot climate variables on panel a
p1, = ax_A.plot(smooth(PP[combinations[0]]['mean'][-80*52:-1], 0.9), lw=3.5, color='#6699cc', label='Precipitation')
p2, = twin1.plot(smooth(TT[combinations[0]]['mean'][-80*52:-1], 0.9), lw=3.5, color='#994455', label='Temperature')
p3, = twin2.plot(smooth(s_rolling[-80*52:-1], 0.9), lw=3.5, color='#004488', label='Streamflow')
p4, = twin2.plot(smooth(glac_rolling[-80*52:-1], 0.9), lw=3.5, color='#997700', label='Glacier Melt')
# filll background indicating near term and end of century time periods
# 8DBECA

ylims = ax_A.get_ylim()
# ax_A.set_ylim([0, ylims[1]])
# ylims = twin1.get_ylim()
# twin1.set_ylim([0, ylims[1]])
# ylims = twin2.get_ylim()
# twin2.set_ylim([0, ylims[1]])

# fill near term and EOC time horizons
# ax_A.fill_between(range(len(SPI)), ylims[0]-10, ylims[1]+10, where=(SPI.index < 52*20), color='#8DBECA', alpha=0.3)
# ax_A.fill_between(range(len(SPI)), ylims[0]-10, ylims[1]+10, where=(SPI.index > 52*60) & (SPI.index <= 52*80), color='#D17779', alpha=0.3)

# ax_A.spines['top'].set_visible(False)
# ax_A.spines['right'].set_visible(False)
# ax_A.spines['bottom'].set_visible(False)
ax_A.set_ylim(ylims)
#ax.spines['left'].set_visible(False)

# legend = plt.legend(loc='lower left', fontsize=16, frameon=False,)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor('0.9')
#legend = ax_A.legend(handles=[p1, p2, p3], fontsize=13, loc='upper center', ncol=3)
legend = ax_A.legend(handles=[p1, p2, p3, p4], fontsize=15, ncol=1, loc='upper left', bbox_to_anchor=(0, 0.8))
legend.get_frame().set_alpha(0)
ax_A.set_xlim([0, 80*52])
twin1.set_xlim([0, 80*52])
twin2.set_xlim([0, 80*52])
ax_A.set_xticks([])

# ax_A.set_xlabel("Distance", fontsize=15)
# ax_A.get_xaxis().set_visible(False)
# ax_A.set_ylabel(r"Temperature (C)", fontsize=15)
# twin1.set_ylabel("Precipitation (mm)", fontsize=15)
# twin2.set_ylabel("Glacier Melt (BCM)", fontsize=15)
ax_A.set_ylabel(r"Precipitation (mm)", fontsize=18)
twin1.set_ylabel("Temperature (C)", fontsize=18)
twin2.set_ylabel("Volume (MCM)", fontsize=18)
# ax_A.set_title(f'{p.replace("_", "-")}', fontsize=16)

# ax_A.yaxis.label.set_color(p1.get_color())
# twin1.yaxis.label.set_color(p2.get_color())
# twin2.yaxis.label.set_color(p3.get_color())

# legend for panel a
# tkw = dict(size=4, width=1.5)
# ax_A.tick_params(axis='y', colors=p1.get_color(), **tkw)
# twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
# twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
# ax_A.tick_params(axis='x', **tkw)
ax_A.set_xticks([])
# ax_A.fill_between(range(len(SPI)), np.min(SPI[f'{mo}_{x}']) - 5, np.max(SPI[f'{mo}_{x}']) + 5,
#                   where=(SPI.index < 52 * 20), color='#8DBECA', alpha=0.3)
# ax_A.fill_between(range(len(SPI)), np.min(SPI[f'{mo}_{x}']) - 5, np.max(SPI[f'{mo}_{x}']) + 5,
#                   where=(SPI.index > 52 * 60) & (SPI.index <= 52 * 80), color='#D17779', alpha=0.3)

# plot indicators
for i, mo in enumerate(models):
    #fig = plt.figure(figsize=(6, 3.5))
    #fig = plt.figure(figsize=(10, 4))

    for j, x in enumerate(experiments):
        for id in [0, 1]:  # SRI, then SPI
            if id == 0:
                SPI = pd.read_csv('data/SRI12_CMIP6_2019-2099.csv')
            elif id == 1:
                SPI = pd.read_csv('data/SPI12_CMIP6_2019-2099.csv')

            smoothed = smooth(SPI[f'{mo}_{x}'], 0.9)
            if id == 0:
                #ax_B.plot(SPI['Timestamp'], smoothed, color='#4A7090', label='Runoff', linewidth=2.5) # label = 'SRI-12'
                sri = ax_B.plot(SPI['Timestamp'], smoothed, color='black', label='SRI', linewidth=2.5) # label = 'SRI-12'
            else:
                #ax_B.plot(SPI['Timestamp'], smoothed, color='black', label='Precipitation', linewidth=2.5) # label = 'SPI-12'
                spi = ax_B.plot(SPI['Timestamp'], smoothed, color='#6699cc', label='SPI', linewidth=2.5)  # label = 'SPI-12'

            #plt.plot(SPI['Timestamp'], smoothed, color='black')
            #plt.axhline(0, linestyle='--', color='black')
            #plt.axhline(-0.8, color='#ED7D31', linestyle='--', linewidth=3)
            ax_B.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            zero = np.zeros(SPI.shape[0])
            thresh = np.ones(SPI.shape[0]) * -1.5
            ax_B.set_xticks(np.array(range(0, len(SPI), 10*52)), np.array(range(2020, 2101, 10)), color='k', fontsize=16) #color='w')
            ax_B.set_yticks([0], [0], fontsize=16)
            #plt.xlim([41*52, 50.5*52])
            ax_B.set_xlim([0, 80 * 52])
            #plt.xticks([])

            k1 = ax_B.axhline(-1.5, color='#E39C6E', linestyle='--', linewidth=3.5, label='Threshold')
            k2 = ax_B.axhline(-2.25, color='#E39C6E', linestyle='--', linewidth=3.5, label='Drought threshold')
            k3 = ax_B.axhline(-3.25, color='#E39C6E', linestyle='--', linewidth=3.5, label='Drought threshold')

            # Shade red between runoff line and threshold
            if id == 0:  # runoff
                ax_B.fill_between(
                    SPI['Timestamp'],
                    smoothed,
                    k1.get_ydata()[0],
                    where=(np.array(smoothed) < k1.get_ydata()[0]),
                    color='#9BC5D2',
                    alpha=0.6
                )
                ax_B.fill_between(
                    SPI['Timestamp'],
                    smoothed,
                    k2.get_ydata()[0],
                    where=(np.array(smoothed) < k2.get_ydata()[0]),
                    color='#86A5AF',
                    alpha=0.6
                )
                ax_B.fill_between(
                    SPI['Timestamp'],
                    smoothed,
                    k3.get_ydata()[0],
                    where=(np.array(smoothed) < k3.get_ydata()[0]),
                    color='#758A92',
                    alpha=1
                )

            # Shade red between precipitation line and threshold
            if id == 1:  # precipitation
                ax_B.fill_between(
                    SPI['Timestamp'],
                    smoothed,
                    k1.get_ydata()[0],
                    where=(np.array(smoothed) < k1.get_ydata()[0]),
                    color='#9BC5D2',
                    alpha=0.6
                )
                ax_B.fill_between(
                    SPI['Timestamp'],
                    smoothed,
                    k2.get_ydata()[0],
                    where=(np.array(smoothed) < k2.get_ydata()[0]),
                    color='#86A5AF',
                    alpha=0.6
                )
                ax_B.fill_between(
                    SPI['Timestamp'],
                    smoothed,
                    k3.get_ydata()[0],
                    where=(np.array(smoothed) < k3.get_ydata()[0]),
                    color='#758A92',
                    alpha=1
                )

    #plt.title(f'Drought Indicator vs. Time', fontsize=20, fontweight=100)
    ax_B.set_xlabel('Time', fontsize=18, fontweight=2)
    ax_B.set_ylabel('SRI or SPI', fontsize=18, fontweight=2) # previouly Drought Indicator'

    # # filll background indicating near term and end of century time periods
    # ax_B.fill_between(range(len(SPI)), np.min(SPI[f'{mo}_{x}'])-5, np.max(SPI[f'{mo}_{x}'])+5, where=(SPI.index < 52*20), color='#8DBECA', alpha=0.3)
    # # ax_B.fill_between(range(len(SPI)), np.min(SPI[f'{mo}_{x}'])-5, np.max(SPI[f'{mo}_{x}'])+5, where=(SPI.index > 52*60) & (SPI.index <= 52*80), color='#D17779', alpha=0.3)

    #ax_B.patch.set_facecolor('0.9')
    # 8DBECA
    # plt.tight_layout()
    #plt.grid(True, color='w')
    ylims = ax_B.get_ylim()

    # shade near term and EOC
    # ax_B.fill_between(range(len(SPI)), np.min(SPI[f'{mo}_{x}'])-5, np.max(SPI[f'{mo}_{x}'])+5, where=(SPI.index < 52*20), color='#8DBECA', alpha=0.3)
    # ax_B.fill_between(range(len(SPI)), np.min(SPI[f'{mo}_{x}'])-5, np.max(SPI[f'{mo}_{x}'])+5, where=(SPI.index > 52*60) & (SPI.index <= 52*80), color='#D17779', alpha=0.3)

    #ax.spines['left'].set_visible(False)
    # --- Create custom legend handles ---
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Line handles for SRI and SPI
    sri_line = mlines.Line2D([], [], color='black', linewidth=2.5, label='SRI')
    spi_line = mlines.Line2D([], [], color='#56B4E9', linewidth=2.5, label='SPI')

    # Fill patches for actions (lightest to darkest)
    a1_patch = mpatches.Patch(color='#9BC5D2', alpha=0.6, label=r'$a_1$')
    a2_patch = mpatches.Patch(color='#86A5AF', alpha=0.6, label=r'$a_2$')
    a3_patch = mpatches.Patch(color='#758A92', alpha=1.0, label=r'$a_3$')

    # Combine all legend entries
    legend_handles = [sri_line, spi_line, k1, a1_patch, a2_patch, a3_patch]

    # Add legend
    legend = ax_B.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=16,
        frameon=False,
        ncol=2
    )

    legend.get_frame().set_alpha(None)
    #legend.get_frame().set_facecolor('0.9')
    ax_B.set_ylim([ylims[0]-1, ylims[1]])

    # Label k1, k2, k3 on the right end of each line
    for yval, label in [(k1.get_ydata()[0], r'$k_1$'), (k2.get_ydata()[0], r'$k_2$'), (k3.get_ydata()[0], r'$k_3$')]:
        ax_B.text(
            1.01, yval, label,
            color='#E39C6E',
            fontsize=20,
            fontweight='bold',
            transform=ax_B.get_yaxis_transform(),
            va='center'
        )

fig.tight_layout()
fig.show()

#%% FIGURE 6 A-B
# NEW from Sarah: CDF of actions simulating policies (contracts)
import pickle

# load the previously simulated optimal policies
filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_contracts_dict.pkl'
with open(filename, 'rb') as file:
    con = pickle.load(file)

# make the CDFs
# Define plotting parameters
periods = ['2020_2040', '2080_2100']
labels = ['SRI3', 'SRI6', 'SRI12', 'SPI3', 'SPI6', 'SPI12']
colors = ['chocolate', 'chocolate', 'chocolate', 'steelblue', 'steelblue', 'steelblue']
alphas = np.tile([0.35, 0.6, 1], 2)

# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for i, p in enumerate(periods):
    ax = axes[i]

    for j, d in enumerate(labels):
        if d in con[p]:
            # Get data (flatten in case it's a DataFrame)
            data = con[p][d]/1E3
            data = data[~np.isnan(data)]  # drop NaNs

            # Compute empirical CDF
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            print(f'{p}, {d}: {sorted_data[np.where((cdf - .50) == np.min(np.abs(cdf - .50)))[0][0]]}')


            ax.plot(sorted_data, cdf, color=colors[j], lw=1.8, label=d, alpha=alphas[j])

    # Formatting
    ax.set_title(p.replace('_', '-'), fontsize=13)
    ax.set_xlabel("Contracts (thousands)", fontsize=11)
    if i == 0:
        ax.set_ylabel("Cumulative Probability", fontsize=11)
        ax.legend(title="Drought Index", frameon=False)
    ax.grid(False, linestyle=':', alpha=0.5)
    ax.set_ylim(bottom=0, top=1.01)
    ax.set_xlim(left=0)

# make same xlims
right_lim = np.max([np.ceil(axes[0].get_xlim()[1]/100) * 100, np.ceil(axes[1].get_xlim()[1]/100) * 100])
axes[0].set_xlim(right=right_lim)
axes[1].set_xlim(right=right_lim)

fig.tight_layout()
plt.show()


#%% FIGURE 6 C-D
# NEW from Sarah: CDF of actions simulating policies (demand restriction)
import pickle

# load the previously simulated optimal policies
filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_PT1_demand_restr_dict.pkl'
with open(filename, 'rb') as file:
    con = pickle.load(file)

# make the CDFs
# Define plotting parameters
periods = ['2020_2040', '2080_2100']
labels = ['SRI3', 'SRI6', 'SRI12', 'SPI3', 'SPI6', 'SPI12']
colors = ['chocolate', 'chocolate', 'chocolate', 'steelblue', 'steelblue', 'steelblue']
alphas = np.tile([0.35, 0.6, 1], 2)

# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for i, p in enumerate(periods):
    ax = axes[i]

    for j, d in enumerate(labels):
        if d in con[p]:
            # Get data (flatten in case it's a DataFrame)
            data = con[p][d]  # m3/s
            data = data[~np.isnan(data)]  # drop NaNs
            #data = data * 604800  # convert to m3 over simulation period
            data = data * 604800 / 1E6  # convert to Mm3 over simulation period

            # Compute empirical CDF
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            ax.plot(sorted_data, cdf, color=colors[j], lw=1.8, label=d, alpha=alphas[j])

    # Formatting
    ax.set_title(p.replace('_', '-'), fontsize=13)
    ax.set_xlabel("Demand Restricted (Mm$^3$)", fontsize=11)
    if i == 0:
        ax.set_ylabel("Cumulative Probability", fontsize=11)
        #ax.legend(title="Drought Index", frameon=False)
    ax.grid(False, linestyle=':', alpha=0.5)
    ax.set_ylim(bottom=0.85, top=1)
    #ax.set_ylim(bottom=0, top=1.01)
    ax.set_xlim(left=0)

# make same xlims
right_lim = np.max([np.ceil(axes[0].get_xlim()[1]), np.ceil(axes[1].get_xlim()[1])])
axes[0].set_xlim(right=6.02)
axes[1].set_xlim(right=6.02)

fig.tight_layout()
plt.show()



#%%  FIGURE 3
# NEW FROM SARAH: Time Series Optimal Policy Comparison with bar plot of actions


drought_indices = ['SRI3', 'SRI6', 'SRI12',
                   'SPI3', 'SPI6', 'SPI12']
periods = ['2020_2040', '2080_2100']
nFunc = 2000  # number of function evaluations
disc = 35
o = 1  # objective: 0 - total cost, 1 - urban rel, 2 - ag rel

# scenario index for time series to plot
scenario_index = 31 #6  # index of sample scenario to simulate

colors = ['steelblue', 'firebrick']
fontsize = 13


# load objective results and store in master dictionary
columns_axes = ['Total Cost', 'Urban Rel.', 'Agr. Rel.']  # for objs_dict
objs_dict = {}
threshs_dict = {}
acts_dict = {}

for p in periods:
    objs_dict[p] = {}
    threshs_dict[p] = {}
    acts_dict[p] = {}
    for d in drought_indices:
        objs_dict[p][d] = {}
        threshs_dict[p][d] = {}
        acts_dict[p][d] = {}

for p in periods:
    for j, d in enumerate(drought_indices):

        # Concatenate solutions across all seeds for this scenario
        all_objs = []
        all_threshs = []
        all_acts = []

        for i in seeds:

            file = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/outputs/{fol}/{d}_{p}_CMIP6_{date}_nFunc{nFunc}_disc{disc}/sets/Borg_DPS_PySedSim{i}.set'

            # formulate the table
            with open(file, 'r') as f:
               #data = pd.read_csv(file, skiprows=2)
               data = pd.read_csv(file, skiprows=2, skipfooter=1, header=None, engine='python').explode(0)
               data = pd.DataFrame(data.iloc[:,0].str.split(' ').explode())

            num_vars = np.sum(data.index == 0)
            num_rows = data.index.unique().shape[0]
            data['vars'] = np.tile(np.arange(num_vars), num_rows)

            data = data.reset_index().pivot(index='index', columns='vars')

            # objectives
            urban_rel = data.iloc[:, 4].values.astype('float')
            ag_rel = data.iloc[:, 5].values.astype('float')
            cost = data.iloc[:, 6].values.astype('float') * -1/1E6  # millions of dollars

            # thresholds
            contract_thresh = data.iloc[:, 0].values.astype('float')
            demand_thresh = data.iloc[:, 2].values.astype('float')

            # actions
            contract_action = data.iloc[:, 1].values.astype('float')
            demand_action = data.iloc[:, 3].values.astype('float')

            #% data syntax for plotting
            objs = pd.DataFrame(data=np.vstack([cost, urban_rel, ag_rel]).T, columns=['Total Cost', 'Urban Rel.', 'Agr. Rel.'])
            objs = objs.loc[:, columns_axes]

            threshs = pd.DataFrame(data=np.vstack([contract_thresh, demand_thresh]).T, columns=['Contracts', 'Demand Rest.'])

            acts = pd.DataFrame(data=np.vstack([contract_action, demand_action]).T, columns=['Contracts', 'Demand Rest.'])
            all_objs.append(objs)
            all_threshs.append(threshs)
            all_acts.append(acts)

        # Store pooled sets (all seeds) for this scenario
        objs_dict[p][d] = pd.concat(all_objs, ignore_index=True) if len(all_objs) else pd.DataFrame(columns=columns_axes)
        threshs_dict[p][d] = pd.concat(all_threshs, ignore_index=True) if len(all_threshs) else pd.DataFrame(columns=['Contracts', 'Demand Rest.'])
        acts_dict[p][d] = pd.concat(all_acts, ignore_index=True) if len(all_acts) else pd.DataFrame(columns=['Contracts', 'Demand Rest.'])



for j in [o]: #range(3):  # for each objective

    import pickle

    # load simulation result dictionaries
    if j == 1:
        filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/urban_rel_dict.pkl'
    elif j == 2:
        filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/ag_rel_dict.pkl'
    elif j == 0:
        filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_cost_dict.pkl'


    with open(filename, 'rb') as file:
        dic = pickle.load(file)

    # Dictionaries to store results
    global_best_dict = {}
    best_drought_index_dict = {}
    best_performance_dict = {}
    best_policy_index_dict = {}

    # Iterate over each period in the dictionary
    for p in dic.keys():
        # Initialize lists to store maximums and corresponding drought indices for the current period
        best_values = []
        best_indices = []
        best_performance_dict[p] = {}
        best_policy_index_dict[p] = {}

        # Iterate over each drought index in the current period
        for d in dic[p].keys():
            # Get the dataframe for the current period and drought index
            df = dic[p][d]

            # Find the maximum value for each column and store in a list
            if j == 0:
                best_values.append(df.min(axis=0))
                best_performance_dict[p][d] = df.min(axis=0)
                best_policy_index_dict[p][d] = df.argmin(axis=0)
            else:
                best_values.append(df.max(axis=0))
                best_performance_dict[p][d] = df.max(axis=0)
                best_policy_index_dict[p][d] = df.argmax(axis=0)

        # Convert list of maxima to a numpy array
        best_values = np.array(best_values)

        # Find the global maximum for each column and its corresponding drought index
        if j == 0:
            global_best_dict[p] = np.min(best_values, axis=0)
            best_indices = np.argmin(best_values, axis=0)
        else:
            global_best_dict[p] = np.max(best_values, axis=0)
            best_indices = np.argmax(best_values, axis=0)

        # Map the index back to the actual drought index key
        drought_keys = list(dic[p].keys())
        best_drought_index_dict[p] = [drought_keys[i] for i in best_indices]

    # Convert the results to dataframes for better visualization (optional)
    global_best_df = pd.DataFrame(global_best_dict, index=[f"Column_{i + 1}" for i in range(36)])
    best_drought_df = pd.DataFrame(best_drought_index_dict, index=[f"Column_{i + 1}" for i in range(36)])

    # get the maximum urban reliability if only thresholds updated from 2020-2040, not indicators themselves
    historical_best = []
    for i in range(36):  # for each climate change scenario
        if j == 0:
            historical_best.append(dic['2080_2100'][best_drought_index_dict['2020_2040'][i]].min(axis=0)[i])
        else:
            historical_best.append(dic['2080_2100'][best_drought_index_dict['2020_2040'][i]].max(axis=0)[i])

# load climate time series data
## parameters for loading climate variables
path_clima = '/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/'
models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
combinations = ["_".join(pair) for pair in product(models, experiments)]

# load climate variables
# load the datafile and set its timestamp
pp = pd.read_csv(path_clima + f'UQM_weap_TempC_new/PP_semanal_{combinations[scenario_index]}.csv').iloc[:,2::].mean(axis=1)
ts = pd.to_datetime(t['year'].astype(str) + t['week'].astype(str).str.zfill(2) + '-1',
                    format='%Y%W-%w')
pp.index = ts
glac = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/data/CMIP6_Glacier_Melt_TDP.csv',
                   index_col=1)[combinations[scenario_index]]
glac.index = pd.to_datetime(glac.index)


import pickle

# load the previously simulated optimal policies
# contracts
filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_contracts_dict.pkl'
with open(filename, 'rb') as file:
    con = pickle.load(file)

# demand restrictions
filename = f'/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR/plots/{fol}/{date}_disc{disc}/total_PT1_demand_restr_dict.pkl'
with open(filename, 'rb') as file:
    dem_restr = pickle.load(file)

# intialize figure
fig = plt.figure(figsize=(10, 12), constrained_layout=True)
widths = [1, 0.2, 1]
heights = [1, 0.3, 1, 0.3, 1, 0.3, 1, 0.3, 1]
gs = fig.add_gridspec(ncols=3, nrows=9, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
fig.tight_layout(h_pad=0)

# name of indices yielding best performance for scenario of choice
d_bests = best_drought_df.iloc[scenario_index, :].values

ax_climA = fig.add_subplot(gs[0, 0:3])
ax_climA.set_ylabel('Precipitation (mm)')
twin_climA = ax_climA.twinx()
twin_climA.set_ylabel(r'Glacier Melt (Mm$^3$)')
ax_climA.set_title(f"{periods[0].replace('_', '-')}")
ax_indxA = fig.add_subplot(gs[2, 0:3])
ax_indxA.set_ylabel(f'{d_bests[0][0:3]} or {d_bests[1][0:3]} (-)')
ax_indxA.set_title(f"{periods[0].replace('_', '-')}")

ax_climB = fig.add_subplot(gs[4, 0:3])
twin_climB = ax_climB.twinx()
twin_climB.set_ylabel(r'Glacier Melt (Mm$^3$)')
ax_climB.set_ylabel('Precipitation (mm)')
ax_climB.set_title(f"{periods[1].replace('_', '-')}")
ax_indxB = fig.add_subplot(gs[6, 0:3])
ax_indxB.set_ylabel('SPI or SRI (-)')
ax_indxB.set_title(f"{periods[1].replace('_', '-')}")


ax1 = fig.add_subplot(gs[8, 0])
ax1.set_ylabel('Contracts (thousands)')

ax2 = fig.add_subplot(gs[8, 2])
ax2.set_ylabel(r'Demand Restricted (Mm$^3$)')

# load drought indicator time series
d1 = pd.read_csv(f'data/{d_bests[0]}_CMIP6_2019-2099.csv')[combinations[scenario_index]]
d1.index = ts[-len(d1)::]
d2 = pd.read_csv(f'data/{d_bests[1]}_CMIP6_2019-2099.csv')[combinations[scenario_index]]
d2.index = ts[-len(d2)::]
ds = pd.concat([d1, d2], axis=1)

d_colors = []
clim_colors = []

for j, d_best in enumerate(d_bests):

    for i, p in enumerate(periods):

        # plot climate time series
        if j == 0:
            if i==0:
                ax_climA.plot(pp[(pp.index.year >= int(p.split('_')[0])) & (pp.index.year < int(p.split('_')[1]))],
                              label='Precipitation', color='#6699cc')
                twin_climA.plot(glac[(glac.index.year >= int(p.split('_')[0])) & (glac.index.year < int(p.split('_')[1]))],
                                label='Glacier Melt', color='#997700')
            else:
                ax_climB.plot(pp[(pp.index.year >= int(p.split('_')[0])) & (pp.index.year < int(p.split('_')[1]))], color='#6699cc')
                twin_climB.plot(
                    glac[(glac.index.year >= int(p.split('_')[0])) & (glac.index.year < int(p.split('_')[1]))], color='#997700')


        # extract policy for desired scenario, time period, and indicator
        p_best_index = best_policy_index_dict[p][d_best][scenario_index]  # index of best performing policy for objective
        a_best = acts_dict[p][d_best].iloc[p_best_index, :]
        k_best = threshs_dict[p][d_best].iloc[p_best_index, :]

        # plot drought indicator series
        if i == 0:
            if j==0:
                ax_indxA.plot(d1[(d1.index.year >= int(p.split('_')[0])) & (d1.index.year < int(p.split('_')[1]))],
                              label=f'{d_best}', color='#6699cc')
                for thresh in k_best:
                    ax_indxA.axhline(thresh, ls='--', color='orange', lw=3, label=f'{d_best} Thresholds')
            else:
                ax_indxA.plot(d2[(d2.index.year >= int(p.split('_')[0])) & (d2.index.year < int(p.split('_')[1]))],
                              label=f'{d_best}', color='black')
                for thresh in k_best:
                    ax_indxA.axhline(thresh, ls=':', color='orange', lw=2, label=f'{d_best} Thresholds')
        elif i == 1:
            if j==0:
                ax_indxB.plot(d1[(d1.index.year >= int(p.split('_')[0])) & (d1.index.year < int(p.split('_')[1]))],
                              label=f'{d_best}', color='#6699cc')
                for thresh in k_best:
                    ax_indxB.axhline(thresh, ls='--', color='orange', lw=3)
            else:
                ax_indxB.plot(d2[(d2.index.year >= int(p.split('_')[0])) & (d2.index.year < int(p.split('_')[1]))],
                              label=f'{d_best}', color='black')
                for thresh in k_best:
                    ax_indxB.axhline(thresh, ls=':', color='orange', lw=2)

        # get simulated actions for sample sceanrio and policy
        con_sim = con[p][d_best][p_best_index, scenario_index] / 1E3  # contracts in thousands
        dem_restr_sim = dem_restr[p][d_best][p_best_index, scenario_index] * 604800 / 1E6  # convert to Mm3 over simulation period

        # add to bar plot of actions
        if j==0:
            if i==0:
                ax1.bar(-0.15, con_sim, color='steelblue', width=0.25)
                ax2.bar(-0.15, dem_restr_sim, color='steelblue', width=0.25)
            else:
                ax1.bar(0.15, con_sim, color='firebrick', width=0.25)
                ax2.bar(0.15, dem_restr_sim, color='firebrick', width=0.25)

        elif j==1:
            if i==0:
                ax1.bar(0.5, con_sim, color='steelblue', width=0.25, label=f"{p.replace('_', '-')}")
                ax2.bar(0.5, dem_restr_sim, color='steelblue', width=0.25)
            else:
                ax1.bar(0.8, con_sim, color='firebrick', width=0.25, label=f"{p.replace('_', '-')}")
                ax2.bar(0.8, dem_restr_sim, color='firebrick', width=0.25)

# for x limits on first plots
ts_limsA = ts = d2.index[(d2.index.year >= int(periods[0].split('_')[0])) & (d2.index.year < int(periods[0].split('_')[1]))][[0, -1]]
ts_limsB = ts = d2.index[(d2.index.year >= int(periods[1].split('_')[0])) & (d2.index.year < int(periods[1].split('_')[1]))][[0, -1]]

ax1.legend(frameon=False, loc='upper left')
ax1.set_xticks([0, 0.65])
ax1.set_xticklabels(d_bests)
ax1.set_xlim([-.35, 1])
ax1.set_ylim([0, 1900])
ax2.set_xticks([0, 0.65])
ax2.set_xticklabels(d_bests)
ax2.set_xlim([-.35, 1])

ax_climA.set_ylim([0, 400])
ax_climA.legend([ax_climA.get_legend_handles_labels()[0][0], twin_climA.get_legend_handles_labels()[0][0]],
    [ax_climA.get_legend_handles_labels()[1][0], twin_climA.get_legend_handles_labels()[1][0]],
                frameon=False, ncol=1, loc='upper right')
ax_climA.set_xlim(ts_limsA)
twin_climA.set_ylim([-5, 6500])
twin_climA.set_xlim(ts_limsA)

ax_climB.set_ylim([0, 400])
ax_climB.set_xlim(ts_limsB)
twin_climB.set_ylim([-5, 6500])
twin_climB.set_xlim(ts_limsB)

ax_indxA.set_ylim([-6.5, 3])
ax_indxA.set_xlim(ts_limsA)
ax_indxA.legend([ax_indxA.get_legend_handles_labels()[0][i] for i in [0, 3, 1, 4]],
    [ax_indxA.get_legend_handles_labels()[1][i] for i in [0, 3, 1, 4]],
                frameon=False, ncol=2, loc='lower right')
ax_indxB.set_ylim([-6.5, 3])
ax_indxB.set_xlim(ts_limsB)

fig.tight_layout()
fig.show()

