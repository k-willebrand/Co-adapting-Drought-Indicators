import xarray as xr
import pandas as pd

models = ['ACCESS-CM2']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # note: 'historical' experiment exists
variables = ['pr', 'tasmax', 'tasmin']

for m in models:
    for x in experiments:
        for v in variables:
            # load the data file
            data_hist = xr.open_dataset(f'../Climate_series/ARCLIM_nc/{v}_day_{m}_historical_r1i1p1f1_gn_19700101-20141231_v20191108.nc')
            data_proj = xr.open_dataset(f'../Climate_series/ARCLIM_nc/{v}_day_{m}_{x}_r1i1p1f1_gn_20150101-21001231_v20191108.nc')

            # convert to dataframe
            df_hist = data_hist.to_dataframe().reset_index()
            df_proj = data_proj.to_dataframe().reset_index()

            # concatenate historical and projected time period
            df = pd.concat([df_hist, df_proj])

            # save to csv
            df.to_csv(f'../Climate_series/ARCLIM/{v}_{m}_{x}_Agua_1970-2100_day.csv')

#%% Reading shapefile
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

# load the catchment dataframe
catch = gpd.read_file('./shp/Cuencas/Catchment_MAIPO_WEAP.shp')

models = ['ACCESS-CM2']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # note: 'historical' experiment exists
variables = ['pr', 'tasmax', 'tasmin']

for m in models:
    for ex in experiments:
        for v in variables:

            # load grid climate data and convert to GeoDataFrame
            df = pd.read_csv(f'./Climate_series/ARCLIM/{v}_{m}_{ex}_Agua_1970-2100_day.csv')

            # preallocate dataframe
            df_interp = pd.DataFrame(columns=np.append('time', catch['Catchment'].values))

            dates = df['time'].unique()
            for i, d in enumerate(dates):

                df_interp['time'][i] = d

                # subset by date
                df_d = df[df['time'] == d]

                # Set up interpolation function
                f = interp2d(df_d['lat'].values, df_d['lon'].values, df_d[f'{v}'].values, kind='linear')

                # interpolate current variable for points in Maipo Catchment
                df_interp.iloc[i, 1::] = f(catch['Lat'].values, catch['Lon'].values)

            df_interp["time"] = pd.to_datetime(df_interp["time"])
            times = pd.to_datetime(df_interp["time"])
            df_interp['Year'] = times.year
            df_interp['Mes'] = times.month
            df_interp['day'] = times.day


