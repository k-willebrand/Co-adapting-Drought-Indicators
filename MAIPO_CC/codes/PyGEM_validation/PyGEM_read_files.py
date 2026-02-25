import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt

# === USER INPUT ===
# Option 1: Single file
netcdf_file = '/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/codes/PyGEM_validation/PyGEM_data/R17_glac_runoff_fixed_monthly_1set_2000_2100-ssp126-Batch-1-1000.nc'

# Option 2: Multiple files (e.g., for a time series)
# Set to None if not using
netcdf_folder = '/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/codes/PyGEM_validation/PyGEM_data'

# define glacier variables and SSP climate scenarios to plot
variables = ['glacier_runoff']  # 'glacier_volume' or 'glacier_runoff' or 'glacier_mass'
#scenarios = ['126', '585']  # '126' or '245' or '370' or '585'
scenarios = ['126', '245', '370', '585']

#file_pattern = 'R17_glac_runoff_fixed_monthly_1set_2000_2100-ssp*.nc'  # or e.g., 'monthly_*.nc'
#file_pattern = 'R17_glac_runoff_fixed_monthly_1set_2000_2100-ssp585*.nc'  # focus on first batch group first

# === LOAD DATA ===

def open_single_file(filepath):
    ds = xr.open_dataset(filepath)
    print("Dataset summary:")
    print(ds)
    return ds

def open_multiple_files(folder, variable, scenario):
    import glob

    if variable == 'glacier_runoff':
        file_paths = sorted(glob.glob(os.path.join(folder, f'R17_glac_runoff_fixed_monthly_1set_2000_2100-ssp{scenario}*.nc')))
        ds = xr.open_mfdataset(file_paths[0])

        # initialize output vector
        glacier_melt = np.array(ds.glac_runoff_fixed_monthly.values)
        out = glacier_melt.sum(axis=1)  # initialize

        # populate output vector
        for f in file_paths[1::]:
            ds = xr.open_mfdataset(f)  #, combine='by_coords')
            glacier_melt = np.array(ds.glac_runoff_fixed_monthly.values)
            melt_total = glacier_melt.sum(axis=1)
            # print(melt_total.shape)
            out = out + melt_total

        # convert units to MCM/month
        out = out / 1E6

    elif variable == 'glacier_mass':
        file_paths = sorted(glob.glob(os.path.join(folder, f'R17_glac_mass_annual_50sets_2000_2100-ssp{scenario}.nc')))
        ds = xr.open_mfdataset(file_paths[0])

        # initialize output vector
        glacier_mass = np.array(ds.glac_mass_annual.values)  # kg
        mass_total = glacier_mass.sum(axis=1)

        # return mass in kg
        out = mass_total  # kg

    elif variable == 'glacier_volume':
        file_paths = sorted(glob.glob(os.path.join(folder, f'R17_glac_mass_annual_50sets_2000_2100-ssp{scenario}.nc')))
        ds = xr.open_mfdataset(file_paths[0])

        # initialize output vector
        glacier_mass = np.array(ds.glac_mass_annual.values)  # kg
        mass_total = glacier_mass.sum(axis=1)

        # convert kg to Mm3
        #n = 0.019405960290035287  # previous infiltration scaling parameter for melt calibration
        density = 900  # kg/m3
        out = mass_total / density / 1E6  # Mm3

    return out

# === MAIN ===

for var in variables:
    for sc in scenarios:
        if netcdf_folder:
            # ds = open_multiple_files(netcdf_folder, file_pattern)
            out = open_multiple_files(netcdf_folder, var, sc)
        else:
            ds = open_single_file(netcdf_file)

        # === OPTIONAL: View data ===
        #print("\nVariables in the dataset:")
        #print(list(ds.data_vars))

        # # Close the dataset after use (for open_dataset)
        # if not isinstance(ds, xr.core.dataset.Dataset):
        #     ds.close()

        # === PLOT OUTPUT ===

        # time series plot
        if var == 'glacier_runoff':  # monthly
            plt.figure()
            plt.plot(out.T)
            plt.xlabel('Month')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Runoff (MCM/month)')
            plt.title(f'Total Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.show()

        if var == 'glacier_runoff':  # annual
            plt.figure()
            plt.plot(out.reshape(out.shape[0], int(out.shape[1]/12), 12).sum(axis=2).T)
            plt.xlabel('Year')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Runoff (MCM/year)')
            plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.show()

        elif var == 'glacier_mass':
            plt.figure()
            plt.plot(out.T, label='PyGEM')
            plt.xlabel('Year')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Mass (kg)')
            plt.title(f'Total Glacier Mass vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.show()

        elif var == 'glacier_volume':
            plt.figure()
            plt.plot(out.T, label='PyGEM')
            plt.xlabel('Year')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Volume (Mm3)')
            plt.title(f'Total Glacier Volume vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.show()

        #%% Comparison with SSC approach
        import pandas as pd

        if var == 'glacier_volume':
            plt.figure()

            # PyGEM
            lines1 = plt.plot(out[:, 20::].T, label='PyGEM', color='steelblue')

            # SSC (copy over results)
            volume_annual = pd.read_csv('/Users/keaniw/Documents/Research/Chile Project/Maipo_CC/codes/PyGEM_validation/CMIP6_Annual_Glacier_Volume_agg_TDP.csv').to_numpy()
            lines2 = plt.plot(volume_annual[41::, 3::4], label='SSC', color='chocolate')  # only SSP5-8.5 scenario

            plt.xlabel('Year')
            plt.ylabel('Total Glacier Volume (Mm3)')
            plt.ylim(bottom=0)
            plt.title(f'Total Glacier Volume vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(handles=[lines1[0], lines2[0], lines3[0]])
            plt.show()

        if var == 'glacier_runoff':

            ## LINEAR TREND ANALYSIS PLOT ##
            plt.figure()

            # PyGEM
            plt.plot(out.reshape(out.shape[0], int(out.shape[1]/12), 12).sum(axis=2)[:, 20::].T, label='PyGEM',
                              color='steelblue', alpha=0.2)
            # PyGEM with overlapping GCMs
            data = out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[-5::, 20::].T
            lines1 = plt.plot(data, label='PyGEM', color='steelblue')

            # Calculate coefficients for a linear trendline (degree=1)
            coefficients = np.polyfit(np.tile(np.arange(data.shape[0]), data.shape[1]),
                                      data.reshape(-1), 1)
            print(coefficients)
            t1 = np.poly1d(coefficients)
            plt.plot(np.arange(data.shape[0]), t1(np.arange(data.shape[0])), color='black', linestyle='--', label='Trendline Y1')

            # SSC
            melt_weekly = pd.read_csv('../../CMIP6_Glacier_Melt_TDP.csv')
            # [1::4] for 2.6, [2::4] for 3.5, [3::4] for 7.0, [4::4] for 8.5
            if sc=='585':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 1::4]  # only SSP5-8.5 scenario
            elif sc=='370':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 2::4]  # only SSP5-8.5 scenario
            elif sc == '245':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 3::4]  # only SSP5-8.5 scenario
            elif sc == '126':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 4::4]  # only SSP5-8.5 scenario

            plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual, label='SSC', color='chocolate',
                              alpha=0.2)
            # Highlight GCMs:INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM'
            lines2 = plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual.iloc[:, [2, 3, 6, 7, 8]], label='SSC', color='chocolate')

            # Calculate coefficients for a linear trendline (degree=1)
            coefficients = np.polyfit(np.tile(np.arange(melt_annual.shape[0]), melt_annual.iloc[:, [2, 3, 6, 7, 8]].shape[1]),
                                      melt_annual.iloc[:, [2, 3, 6, 7, 8]].values.reshape(-1), 1)
            print(coefficients)
            t1 = np.poly1d(coefficients)
            lines3 = plt.plot(np.arange(data.shape[0]), t1(np.arange(data.shape[0])), color='black', linestyle='--', label='Linear trendline')


            plt.xlabel('Year')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Runoff (MCM/year)')
            #plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(handles=[lines1[0], lines2[0], lines3[0]], frameon=False)
            plt.show()



            # 20-YEAR ANALYSIS BOX PLOT
            plt.figure()

            # PyGEM
            # plt.plot(out.reshape(out.shape[0], int(out.shape[1]/12), 12).sum(axis=2)[:, 20::].T, label='PyGEM',
            #                   color='steelblue', alpha=0.2)
            # PyGEM with overlapping GCMs
            data = out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[-5::, 20::].T
            bplot_pygem = plt.boxplot(x=[data[0:20, :].reshape(-1),
                                       data[-20::, :].reshape(-1)],
                                    positions=[1.1, 2.1], patch_artist=True, showmeans=True, label='PyGEM')
            # fill with colors
            for patch in bplot_pygem['boxes']:
                patch.set_facecolor('steelblue')
            for patch in bplot_pygem['means']:
                patch.set_color('black')
            for patch in bplot_pygem['medians']:
                patch.set_color('black')

            # print statistics
            print('PyGEM')
            print('2020-40:')
            print(f'mean: {data[0:20, :].reshape(-1).mean()}')
            print(f'var: {data[0:20, :].reshape(-1).var()}')
            print(f'std: {data[0:20, :].reshape(-1).std()}')
            print('\n2080-00:')
            print(f'mean: {data[-20::, :].reshape(-1).mean()}')
            print(f'var: {data[-20::, :].reshape(-1).var()}')
            print(f'std: {data[-20::, :].reshape(-1).std()}')


            # SSC
            melt_weekly = pd.read_csv('../../CMIP6_Glacier_Melt_TDP.csv')
            # [1::4] for 2.6, [2::4] for 3.5, [3::4] for 7.0, [4::4] for 8.5
            if sc=='585':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 1::4]  # only SSP5-8.5 scenario
            elif sc=='370':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 2::4]  # only SSP5-8.5 scenario
            elif sc == '245':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 3::4]  # only SSP5-8.5 scenario
            elif sc == '126':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 4::4]  # only SSP5-8.5 scenario

            # plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual, label='SSC', color='chocolate',
            #                   alpha=0.2)
            # Highlight GCMs:INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM'
            melt_annual_gcms = melt_annual.iloc[:, [2, 3, 6, 7, 8]]
            bplot_ssc = plt.boxplot(x=[melt_annual_gcms.iloc[0:20, :].values.reshape(-1), melt_annual_gcms.iloc[-20::, :].values.reshape(-1)],
                        positions=[0.9, 1.9], patch_artist=True, showmeans=True, label='SSC')
            # fill with colors
            for patch in bplot_ssc['boxes']:
                patch.set_facecolor('chocolate')
            for patch in bplot_ssc['medians']:
                patch.set_color('black')
            for patch in bplot_ssc['means']:
                patch.set_color('black')

            # print statistics
            print('SSC')
            print('2020-40:')
            print(f'mean: {melt_annual_gcms.iloc[0:20, :].values.mean()}')
            print(f'var: {melt_annual_gcms.iloc[0:20, :].values.var()}')
            print(f'std: {melt_annual_gcms.iloc[0:20, :].values.std()}')
            print('\n2080-00:')
            print(f'mean: {melt_annual_gcms.iloc[-20::, :].values.mean()}')
            print(f'var: {melt_annual_gcms.iloc[-20::, :].values.var()}')
            print(f'std: {melt_annual_gcms.iloc[-20::, :].values.std()}')


            plt.xticks([1, 2], ['2020-2040', '2080-2100'])
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Runoff (MCM/year)')
            #plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(frameon=False)
            plt.show()

            # MEAN AND VARIANCE ANALYSIS PLOT
            plt.figure()
            w = 5  # rolling window

            # PyGEM
            # plt.plot(out.reshape(out.shape[0], int(out.shape[1]/12), 12).sum(axis=2)[:, 20::].T, label='PyGEM',
            #                   color='steelblue', alpha=0.2)
            # PyGEM with overlapping GCMs
            data = out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[-5::, 20::].T
            lines1 = plt.plot(data, label='PyGEM', color='steelblue', alpha=0.2)

            # plot rolling mean and std
            plt.plot(pd.Series(data.mean(axis=1)).rolling(w).mean(), color='steelblue', ls='--')
            plt.plot(pd.Series(data.mean(axis=1)).rolling(w).std() + pd.Series(data.mean(axis=1)).rolling(w).mean(),
                     color='steelblue', ls=':')
            plt.plot(-1 * pd.Series(data.mean(axis=1)).rolling(w).std() + pd.Series(data.mean(axis=1)).rolling(w).mean(),
                     color='steelblue', ls=':')

            # SSC
            melt_weekly = pd.read_csv('../../CMIP6_Glacier_Melt_TDP.csv')
            # [1::4] for 2.6, [2::4] for 3.5, [3::4] for 7.0, [4::4] for 8.5
            if sc=='585':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 1::4]  # only SSP5-8.5 scenario
            elif sc=='370':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 2::4]  # only SSP5-8.5 scenario
            elif sc == '245':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 3::4]  # only SSP5-8.5 scenario
            elif sc == '126':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 4::4]  # only SSP5-8.5 scenario

            # plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual, label='SSC', color='chocolate',
            #                   alpha=0.2)
            # Highlight GCMs:INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM'
            lines2 = plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual.iloc[:, [2, 3, 6, 7, 8]], label='SSC',
                              color='chocolate', alpha=0.2)

            # plot rolling mean and std
            plt.plot(melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values,
                    color='chocolate', ls='--')
            plt.plot(melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).std().values
                     + melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values,
                    color='chocolate', ls=':')
            plt.plot(
                -1 * melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).std().values
                + melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values,
                color='chocolate', ls=':')

            plt.xlabel('Year')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Runoff (MCM/year)')
            #plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(handles=[lines1[0], lines2[0], lines3[0]], frameon=False)
            plt.show()

            # *ABSOLUTE CHANGE* IN MEAN AND VARIANCE ANALYSIS PLOT
            plt.figure()
            w = 5  # rolling window

            # PyGEM
            # plt.plot(out.reshape(out.shape[0], int(out.shape[1]/12), 12).sum(axis=2)[:, 20::].T, label='PyGEM',
            #                   color='steelblue', alpha=0.2)
            # PyGEM with overlapping GCMs
            data = out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[-5::, 20::].T
            lines1 = plt.plot(data - pd.Series(data.mean(axis=1)).rolling(w).mean()[4], label='PyGEM', color='steelblue', alpha=0.1)

            # plot rolling mean and std
            plt.plot(- pd.Series(data.mean(axis=1)).rolling(w).mean()[4] + pd.Series(data.mean(axis=1)).rolling(w).mean(),
                     color='steelblue', ls='-')
            plt.plot(- pd.Series(data.mean(axis=1)).rolling(w).mean()[4] + pd.Series(data.mean(axis=1)).rolling(w).mean()
                     - pd.Series(data.mean(axis=1)).rolling(w).std(),
                     color='steelblue', ls=':')
            plt.plot(- pd.Series(data.mean(axis=1)).rolling(w).mean()[4] + pd.Series(data.mean(axis=1)).rolling(w).mean() +
                     pd.Series(data.mean(axis=1)).rolling(w).std(),
                     color='steelblue', ls=':')

            # SSC
            melt_weekly = pd.read_csv('../../CMIP6_Glacier_Melt_TDP.csv')
            # [1::4] for 2.6, [2::4] for 3.5, [3::4] for 7.0, [4::4] for 8.5
            if sc=='585':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 1::4]  # only SSP5-8.5 scenario
            elif sc=='370':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 2::4]  # only SSP5-8.5 scenario
            elif sc == '245':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 3::4]  # only SSP5-8.5 scenario
            elif sc == '126':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 4::4]  # only SSP5-8.5 scenario

            # plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual, label='SSC', color='chocolate',
            #                   alpha=0.2)
            # Highlight GCMs:INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM'
            lines2 = plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual.iloc[:, [2, 3, 6, 7, 8]] - melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4]
                              , label='SSC',
                              color='chocolate', alpha=0.1)

            # plot rolling mean and std
            plt.plot(- melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4] +
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values,
                    color='chocolate', ls='-')
            plt.plot(- melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4] +
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values
                    + melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).std().values,
                    color='chocolate', ls=':')
            plt.plot(- melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4] +
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values
                    -1 * melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).std().values,
                    color='chocolate', ls=':')

            plt.xlabel('Year')
            #plt.ylim(bottom=0)
            lines4 = plt.plot(0, 0, color='black', linestyle='--', label='20-year mean')
            plt.ylabel('Absolute Change in Total Glacier Runoff (MCM/year)')
            plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(handles=[lines1[0], lines2[0], lines4[0]], frameon=False)
            plt.tight_layout()
            plt.show()

            # *PERCENT CHANGE* IN MEAN AND VARIANCE ANALYSIS PLOT
            plt.figure()
            w = 5  # rolling window

            # PyGEM
            # plt.plot(out.reshape(out.shape[0], int(out.shape[1]/12), 12).sum(axis=2)[:, 20::].T, label='PyGEM',
            #                   color='steelblue', alpha=0.2)
            # PyGEM with overlapping GCMs
            data = out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[-5::, 20::].T
            lines1 = plt.plot((data - pd.Series(data.mean(axis=1)).rolling(w).mean()[4])/pd.Series(data.mean(axis=1)).rolling(w).mean()[4], label='PyGEM', color='steelblue', alpha=0.1)

            # plot rolling mean and std
            plt.plot((- pd.Series(data.mean(axis=1)).rolling(w).mean()[4] + pd.Series(data.mean(axis=1)).rolling(w).mean())/
                     pd.Series(data.mean(axis=1)).rolling(w).mean()[4],
                     color='steelblue', ls='-')
            plt.plot((- pd.Series(data.mean(axis=1)).rolling(w).mean()[4] + pd.Series(data.mean(axis=1)).rolling(w).mean()
                     - pd.Series(data.mean(axis=1)).rolling(w).std()) / pd.Series(data.mean(axis=1)).rolling(w).mean()[4],
                     color='steelblue', ls=':')
            plt.plot((- pd.Series(data.mean(axis=1)).rolling(w).mean()[4] + pd.Series(data.mean(axis=1)).rolling(w).mean() +
                     pd.Series(data.mean(axis=1)).rolling(w).std()) / pd.Series(data.mean(axis=1)).rolling(w).mean()[4],
                     color='steelblue', ls=':')

            # SSC
            melt_weekly = pd.read_csv('../../CMIP6_Glacier_Melt_TDP.csv')
            # [1::4] for 2.6, [2::4] for 3.5, [3::4] for 7.0, [4::4] for 8.5
            if sc=='585':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 1::4]  # only SSP5-8.5 scenario
            elif sc=='370':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 2::4]  # only SSP5-8.5 scenario
            elif sc == '245':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 3::4]  # only SSP5-8.5 scenario
            elif sc == '126':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 4::4]  # only SSP5-8.5 scenario

            # plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual, label='SSC', color='chocolate',
            #                   alpha=0.2)
            # Highlight GCMs:INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM'
            lines2 = plt.plot(np.arange(0, melt_annual.shape[0]), (melt_annual.iloc[:, [2, 3, 6, 7, 8]] - melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4])/melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4]
                              , label='SSC',
                              color='chocolate', alpha=0.1)

            # plot rolling mean and std
            plt.plot((- melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4] +
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values) /
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4],
                    color='chocolate', ls='-')
            plt.plot((- melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4] +
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values
                    + melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).std().values) /
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4],
                    color='chocolate', ls=':')
            plt.plot((- melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4] +
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values
                    -1 * melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).std().values) /
                     melt_annual.iloc[:, [2, 3, 6, 7, 8]].mean(axis=1).rolling(w).mean().values[4],
                    color='chocolate', ls=':')

            plt.xlabel('Year')
            #plt.ylim(bottom=0)
            lines4 = plt.plot(0, 0, color='black', linestyle='--', label='20-year mean')
            plt.ylabel('Percent Change in Total Glacier Runoff (%)')
            plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(handles=[lines1[0], lines2[0], lines4[0]], frameon=False)
            plt.tight_layout()
            plt.show()

            ## GCM RANK ORDER PLOT ##
            plt.figure()
            colors = ['firebrick', 'darkorange', 'seagreen', 'steelblue', 'purple']  # for each GCM

            # PyGEM
            lines1 = plt.plot(out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[:, 20::].T,
                              label='PyGEM',
                              color='gray', alpha=0.2, linestyle='--')
            # PyGEM with overlapping GCMs
            data = out.reshape(out.shape[0], int(out.shape[1] / 12), 12).sum(axis=2)[-5::, 20::].T
            for i, c in enumerate(colors):
                plt.plot(data[:, i], label='PyGEM GCMS', color=c, linestyle='--')

            # SSC
            melt_weekly = pd.read_csv('../../CMIP6_Glacier_Melt_TDP.csv')
            # [1::4] for 2.6, [2::4] for 3.5, [3::4] for 7.0, [4::4] for 8.5
            if sc == '585':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 1::4]  # only SSP5-8.5 scenario
            elif sc == '370':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 2::4]  # only SSP5-8.5 scenario
            elif sc == '245':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 3::4]  # only SSP5-8.5 scenario
            elif sc == '126':
                melt_annual = melt_weekly.groupby(pd.to_datetime(melt_weekly['Date']).dt.year).sum(
                    numeric_only=True).iloc[41::, 4::4]  # only SSP5-8.5 scenario

            lines2 = plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual, label='SSC', color='gray',
                              alpha=0.2)
            # Highlight GCMs:INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM'
            gcms = [2, 3, 6, 7, 8]
            glines = []
            for i, c in enumerate(colors):
                g = plt.plot(np.arange(0, melt_annual.shape[0]), melt_annual.iloc[:, gcms[i]],
                                     color=c, label=melt_annual.iloc[:, gcms[i]].name[:-7])
                glines.append(g[0])

            # additional legend entries
            g = plt.plot([0], [0], color='black', label='PyGEM', ls='--')
            glines.append(g[0])
            g = plt.plot([0], [0], color='black', label='SSC', ls='-')
            glines.append(g[0])

            plt.xlabel('Year')
            plt.ylim(bottom=0)
            plt.ylabel('Total Glacier Runoff (MCM/year)')
            #plt.title(f'Total Annual Glacier Runoff vs. Time for SSP{sc[0]}-{sc[1]}.{sc[2]} scenario')
            plt.legend(handles=glines, ncol=3, frameon=False)
            plt.show()

