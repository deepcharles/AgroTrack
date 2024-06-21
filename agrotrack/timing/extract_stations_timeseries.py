import xarray as xr
import pandas as pd

def extract_stations_timeseries(lst_ir,lst_nir,st_info,year):
    '''
    extract four different LST time series for each pixel, namely (original LST, natural pixel LST, delta LST and Delta LST with nan values removed and store them in a single xarray dataset 
    
    Parameters:
    
    lst_ir = the lst of the original pixel 
    
    lst_nir: the lst of the natural pixel assigned to the original pixel
    
    st_info: [Pandas DataFrame] a data frame with three columns {'name','lat','lon'}
    
    year: the year for which the timeseries will be extracted
    
    
    Return:
    
    st_data: [Xarray dataset] the lst of the natural pixel assigned to the original pixel
    
    '''
    lat = st_info['lat']
    lon = st_info['lon']
    delta_lst = lst_ir-lst_nir
    delta_lst_nonan = delta_lst.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
    lst_ir_st = lst_ir.sel(lat = lat, lon = lon, method = 'nearest').sel(time = year)
    lst_nir_st = lst_nir.sel(lat = lat, lon = lon, method = 'nearest').sel(time = year)
    delta_lst_st = delta_lst.sel(lat = lat, lon = lon, method = 'nearest').sel(time = year)
    delta_lst_nonan_st = delta_lst_nonan.sel(lat = lat, lon = lon, method = 'nearest').sel(time = year)

    lst_ir_st = lst_ir_st.rename('lst_ir_st')
    lst_nir_st = lst_nir_st.rename('lst_nir_st')
    delta_lst_st = delta_lst_st.rename('delta_lst_st')
    delta_lst_nonan_st = delta_lst_nonan_st.rename('delta_lst_nonan_st')
    st_data = xr.combine_by_coords([lst_ir_st,lst_nir_st,delta_lst_st,delta_lst_nonan_st])
    return st_data