# Description:
# This pre-processing script does the following:
# 1. References a sample ARCLIM_reference file from Sebastian Aedo to determine the necessary output fields
# 2. Regrids the available GCM projections to be consistent with the Maipo Basin shapefile (e.g., ’wgs_84')
# 3. Extracts the GCM projections for Maipo Basin polygons

#%% Import necessary packages

# general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# geospatial packages
import netCDF4 as nc
import xarray as xr
import geopandas as gpd
import shapely as sh
import os

# set path
os.chdir('/Users/keaniw/Documents/Research/Chile Project/Maipo_CC')

#%% Reference a sample ARCLIM_reference file from Sebastian Aedo to determine the necessary output fields

# Sebastian Aedo's reference shapefile geometry ids
ARCLIM_out = pd.read_csv('./Climate_series/ARCLIM_reference/tasmin_GFDL-CM3_Agua_1970-2069_day.csv')
shape_ids = ARCLIM_out.columns[3::].values  # polygon names
print(shape_ids.size)  # number of geometry references in Sebastian's file

# Provided Maipo Basin shapefile geometry ids
maipo = gpd.read_file('./shp/Cuencas/Catchment_MAIPO_WEAP.shp')
maipo_ids = maipo['Catchment'].values  # number of geometry references in shapefile
sort_maipo_ids = maipo_ids.sort()

# make figure
plt.figure()
plt.scatter(maipo['Lon'], maipo['Lat'], color='steelblue')
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.title('Maipo Basin Shapefile from Sebastian Aedo')
plt.show()

# Provided Maipo Basin shapefile geometry ids
weap = gpd.read_file('./shp/Exported_WEAP_Catchments/mygeodata_merged.shp')
weap_ids = weap['Name'].values  # number of geometry references in shapefile
sort_weap_ids = weap_ids.sort()

# make figure
plt.figure()
plt.scatter(weap.geometry.x, weap.geometry.y, color='steelblue')
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.title('Maipo Basin Shapefile from WEAP')
plt.show()

#%% Find shapefile catchment name differences

# check differences between given shape file and Sebastian's file
s = set(maipo_ids)
diff = [x for x in shape_ids if x not in s]
print(diff)  # geometries in Sebastian's file missing from shapefile

# check differences between exported weap shapefile and Sebastian's file
s = set(weap_ids)
diff = [x for x in shape_ids if x not in s]
print(diff)  # geometries in Sebastian's file missing from shapefile

#%% Compare grid locations to check overlap

# GCM projections
models = ['ACCESS-CM2']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # note: 'historical' experiment exists
variables = ['pr', 'tasmax', 'tasmin']

# load GCM projection data
for m in models:
    for x in experiments:
        for v in variables:
            # load the data file
            data_hist = xr.open_dataset(f'./Climate_series/ARCLIM_nc/{v}_day_{m}_historical_r1i1p1f1_gn_19700101-20141231_v20191108.nc')
            data_proj = xr.open_dataset(f'./Climate_series/ARCLIM_nc/{v}_day_{m}_{x}_r1i1p1f1_gn_20150101-21001231_v20191108.nc')

            # convert to dataframe
            df_hist = data_hist.to_dataframe().reset_index()
            df_proj = data_proj.to_dataframe().reset_index()

            # concatenate historical and projected time period
            df = pd.concat([df_hist, df_proj])



