import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import glob
from rtree import index
from multiprocessing import Pool


def extract_data(nc_file, shp_file, variable_name):
    # Load NetCDF file
    ds = xr.open_dataset(nc_file)

    # Load shapefile
    gdf = gpd.read_file(shp_file)

    # Extract centroids from shapefile
    centroids = gdf.geometry.centroid

    # Extract time variable from NetCDF
    time_var = ds.time

    # Create an empty DataFrame to store data for each catchment
    data = pd.DataFrame(index=time_var, columns=gdf.index)

    # Loop through each time step
    for i, t in enumerate(time_var):
        # Extract data for each catchment at each time step
        for j, centroid in enumerate(centroids):
            # Find the nearest grid cell in the NetCDF file
            lat = centroid.y
            lon = centroid.x
            lat_idx = np.abs(ds.lat - lat).argmin()
            lon_idx = np.abs(ds.lon - lon).argmin()

            # Extract data for the nearest grid cell
            data.iloc[i, j] = ds[variable_name].isel(lat=lat_idx, lon=lon_idx)

    # Close the NetCDF dataset
    ds.close()

    return data


def extract_data2(nc_file, shp_file, variable_name):
    # Load NetCDF file
    ds = xr.open_dataset(nc_file)

    # Load shapefile
    gdf = gpd.read_file(shp_file)
    if gdf.geometry.centroid.crs.is_projected is True:
        gdf = gdf.to_crs(epsg=4326)  # to standard projection wsg_84

    # Extract centroids from shapefile
    centroids = gdf.geometry.centroid

    # Extract latitude and longitude values from NetCDF
    lat_values = ds.lat.values
    lon_values = ds.lon.values

    # Find the nearest latitude and longitude indices for each centroid
    lat_indices = np.abs(lat_values[:, np.newaxis] - centroids.y.values).argmin(axis=0)
    lon_indices = np.abs(lon_values[:, np.newaxis] - centroids.x.values).argmin(axis=0)

    # Extract time variable from NetCDF
    time_var = ds.time

    # Create an empty DataFrame to store data for each catchment
    data = np.empty((len(time_var), len(gdf)))

    # Loop through each time step
    for i, t in enumerate(time_var):
        # Extract data for each catchment at each time step
        data[i, :] = ds[variable_name].isel(lat=lat_indices, lon=lon_indices)

    # Close the NetCDF dataset
    ds.close()

    # Convert extracted data to DataFrame
    data_df = pd.DataFrame(data, index=time_var, columns=gdf.index)

    return data_df


def extract_data3(nc_file, shp_file, variable_name):
    # Load NetCDF file
    ds = xr.open_dataset(nc_file)

    # Load shapefile and correct for any projections
    gdf = gpd.read_file(shp_file)
    if gdf.geometry.centroid.crs.is_projected is True:
        gdf = gdf.to_crs(epsg=4326)  # to lat lon

    # Extract centroids of catchments from shapefile
    #centroids = gdf.geometry.centroid

    # Extract coordinates from shapefile
    gdf_lats, gdf_lons = gdf.Lat.values, gdf.Lon.values

    # Extract latitude and longitude values from NetCDF
    lat_values = ds.lat.values
    lon_values = ds.lon.values

    # Find the nearest latitude and longitude indices for each centroid
    #lat_indices = np.abs(lat_values[:, np.newaxis] - centroids.y.values).argmin(axis=0)
    #lon_indices = np.abs(lon_values[:, np.newaxis] - centroids.x.values).argmin(axis=0)
    lat_indices = np.abs(lat_values[:, np.newaxis] - gdf_lats).argmin(axis=0)
    lon_indices = np.abs(lon_values[:, np.newaxis] - gdf_lons).argmin(axis=0)


    # Extract data for all time steps
    #data = ds[variable_name].isel(lat=lat_indices, lon=lon_indices).values
    data = ds[variable_name][:, xr.DataArray(lat_indices), xr.DataArray(lon_indices)].values

    # Close the NetCDF dataset
    ds.close()

    # Convert extracted data to DataFrame
    data_df = pd.DataFrame(data, index=ds.time.values, columns=gdf.Catchment.values)

    return data_df


# vectorized approach using centroids
def extract_data4(nc_file, shp_file, variable_name):
    # Load NetCDF file
    ds = xr.open_dataset(nc_file)
    try:
        ds.indexes['time'].day
    except:
        ds.indexes['time'].to_datetimeindex()

    # Load shapefile
    gdf = gpd.read_file(shp_file)

    # Extract centroids of catchments from shapefile
    if gdf.geometry.centroid.crs.is_projected is True:
        centroids = gdf.geometry.centroid.to_crs(epsg=4326)  # to lat lon
    else:
        centroids = gdf.geometry.centroid

    # Extract latitude and longitude values from NetCDF
    lat_values = ds.lat.values
    lon_values = ds.lon.values - 360

    # Find the nearest latitude and longitude indices for each centroid
    lat_indices = np.abs(lat_values[:, np.newaxis] - centroids.y.values).argmin(axis=0)
    lon_indices = np.abs(lon_values[:, np.newaxis] - centroids.x.values).argmin(axis=0)

    # Extract data for all time steps
    if variable_name == 'pr':  # convert to mm/d
        data = 86400 * ds[variable_name][:, xr.DataArray(lat_indices), xr.DataArray(lon_indices)].values
    else:
        data = -273.15 + ds[variable_name][:, xr.DataArray(lat_indices), xr.DataArray(lon_indices)].values

    # Convert extracted data to DataFrame
    #data_df = pd.DataFrame(data, index=ds.time.values, columns=gdf.Catchment.values)  # alt: columns=gdf.Micro.values
    #data_df = pd.DataFrame(data, index=ds.indexes['time'], columns=gdf.Catchment.values)  # alt: columns=gdf.Micro.values
    data_df = pd.DataFrame(data, index=ds.indexes['time'], columns=gdf.Name.values)

    # add Year, Mes, day columns from datetime
    data_df.insert(0, 'day', ds.indexes['time'].day)
    data_df.insert(0, 'Mes', ds.indexes['time'].month)
    data_df.insert(0, 'Year', ds.indexes['time'].year)
    data_df.reset_index(drop=True, inplace=True)

    # Close the NetCDF dataset
    ds.close()

    return data_df


#%% Usage

# GCM projections
models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # note: 'historical' experiment exists
variables = ['pr', 'tasmin']  # ['pr', 'tasmax', 'tasmin']

# load GCM projection data
for m in models:
    for x in experiments:
        for v in variables:

            # specify files for extraction
            #nc_file_hist = f'../Climate_series/ARCLIM_nc/{v}_day_{m}_historical_r1i1p1f1_gn_19700101-20141231_v20191108.nc'
            #nc_file_proj = f'../Climate_series/ARCLIM_nc/{v}_day_{m}_{x}_r1i1p1f1_gn_20150101-21001231_v20191108.nc'
            nc_file_hist = glob.glob(f'../Climate_series/ARCLIM_nc/{v}_day_{m}_historical_*')[0]
            nc_file_proj = glob.glob(f'../Climate_series/ARCLIM_nc/{v}_day_{m}_{x}_*')[0]

            shp_file = '../shp/Cuencas/Catchment_MAIPO_WEAP.shp'  # using Sebastian Aedo's reference file
            shp_file = '../shp/Exported_WEAP_Catchments/mygeodata_merged.shp'
            #shp_file = '../shp/PEGH_Maipo_Cordillera_2021-vCCG_CC/SIG/Subcuencas Cordillea.shp'  # using WEAP model reference file directly

            # extract GCM data for shapefile catchments
            data_hist = extract_data4(nc_file_hist, shp_file, v)
            data_proj = extract_data4(nc_file_proj, shp_file, v)

            # concatenate historical and projected time period
            data = pd.concat([data_hist, data_proj])

            # add Year, Mes, day columns from datetime
            #data.insert(0, 'day', data.index.day)
            #data.insert(0, 'Mes', data.index.month)
            #data.insert(0, 'Year', data.index.year)
            #data.reset_index(drop=True, inplace=True)

            # save extracted data to file
            data.to_csv(f'../Climate_series/ARCLIM/{v}_{m}_{x}_Agua_1970-2100_day.csv')
            print(f'{v}_{m}_{x} complete.')
