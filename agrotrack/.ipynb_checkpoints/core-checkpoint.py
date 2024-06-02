import os
import glob
import xarray as xr
from datetime import datetime
import os
import numpy as np
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import ruptures as rpt 


# clip a dataset function with a bbox
def clip_around_point(ds,bounding_box): 
    return ds.sel(lat = slice(bounding_box[1], bounding_box[3]), lon = slice(bounding_box[0], bounding_box[2]))

# Reading multiple dataset using Xarray open_mfdataset, adding time dimension and concatenating the data
def create_lai_datacube(years, bounding_box, save_dir, to_nc=True):
 
    # creating a filelist
    subdir = '/discover/nobackup/projects/lis/LS_PARAMETERS/MODIS'
    product_name= 'MCD15A3H.061'
    if isinstance(years, list):
        file_list = []
        for year in years:
            file_list += sorted(glob.glob(os.path.join(subdir,product_name,str(year),'MCD15A3H.061_20*.nc4')))
    else:
        file_list = sorted(glob.glob(os.path.join(subdir,product_name,str(years),'MCD15A3H.061_20*.nc4')))
    


    # Extracting time stamp in datetime format from the name of the MODIS files
    def extract_date_from_filename(files):
        return [datetime.strptime(file.split('/')[-1].split('_')[1].split('.')[0],'%Y%j') for file in files]

    # Preprocessing function that is executed on each nc file which slice the data around the "point_of_interest" and add a time dimension 
    def preprocess_func(ds):
        return clip_around_point(ds,bounding_box).expand_dims(time = [datetime.now()])
    ds1 = xr.open_dataset(file_list[0])
    varname  = set(ds1.keys())
    var_to_drop = list(varname-{'Lai_500m'})
    output =  xr.open_mfdataset(file_list,
                                    #chunks={'lat': 48, 'lon': 48, 'time': 10}, # this takes time maybe because the number of chunks are too many
                                    preprocess = preprocess_func, # this will apply to each of the nc files in the list
                                    drop_variables= var_to_drop,
                                    concat_dim='time',
                                    combine='nested',
                                    parallel=True)
    output['time'] = extract_date_from_filename(file_list) # assigning the dates extracted from the file name to the time dimension
     
        
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f'rm -r {save_dir}'
            subprocess.Popen(bash_cmd.split())
        output.to_netcdf(path = save_dir,mode='w', format='NETCDF4', engine='netcdf4')
    return output


# Reading multiple LST files
def create_lst_datacube(years,bounding_box,save_dir,am_pm = 'am', to_nc=True):
 
    # creating a filelist
    if am_pm == 'am':
        subdir = '/discover/nobackup/projects/lis/LS_PARAMETERS/MODIS'
        product_name = 'MOD11A1.061'
    elif am_pm == 'pm':
        subdir = '/css/modis/Collection6.1/L3/Analysis_Ready'
        product_name = 'MYD11A1.061'
    if isinstance(years, list):
        file_list = []
        for year in years:
            file_list += sorted(glob.glob(os.path.join(subdir,product_name,str(year),f'{product_name}_*.nc4')))
    else:
        file_list = sorted(glob.glob(os.path.join(subdir,product_name,str(years),f'{product_name}_*.nc4')))

    # Extracting time stamp in datetime format from the name of the MODIS files
    def extract_date_from_filename(files):
        return [datetime.strptime(file.split('/')[-1].split('_')[1].split('.')[0],'%Y%j') for file in files]

    # Preprocessing function that is executed on each nc file which slice the data around the "point_of_interest" and add a time dimension 
    def preprocess_func(ds):
        return clip_around_point(ds,bounding_box).expand_dims(time = [datetime.now()])
    ds1 = xr.open_dataset(file_list[0])
    varname  = set(ds1.keys())
    var_to_drop = list(varname-{'LST_Day_1km'})
    output =  xr.open_mfdataset(file_list,
                                    #chunks={'lat': 48, 'lon': 48, 'time': 10}, # this takes time maybe because the number of chunks are too many
                                    preprocess = preprocess_func, # this will apply to each of the nc files in the list
                                    drop_variables= var_to_drop,
                                    concat_dim='time',
                                    combine='nested',
                                    parallel=True)
    output['time'] = extract_date_from_filename(file_list) # assigning the dates extracted from the file name to the time dimension
    
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f'rm -r {save_dir}'
            subprocess.Popen(bash_cmd.split())
        output.to_netcdf(path = save_dir,mode='w', format='NETCDF4', engine='netcdf4')
    return output




# Reading multiple SM files
def create_sm_datacube(years,bounding_box,save_dir, to_nc=True):

    subdir = '/discover/nobackup/projects/lis/RS_DATA/SMAP/SPL3SMP_E.005/'
    
    # Extracting time stamp in datetime format from the name of the MODIS files
    def extract_date(filename):
        return datetime.strptime(filename.split('/')[-1],'%Y.%m.%d')

    # reading all the files and concating them
    if isinstance(years, list):
        files = []
        for year in years:
            files += sorted(glob.glob(os.path.join(subdir,f'{year}.*')))
    else:
        files = sorted(glob.glob(os.path.join(subdir,f'{years}.*')))
    filenames = []
    date = []
    for file in files:
        filename,*other = glob.glob(file +'/*.h5')
        filenames.append(filename)
        date.append(extract_date(file)) 
        
    def preprocess_func(ds):
        # reading lat and lon from the file
        lat = ds['latitude']
        lon = ds['longitude']

        lat = xr.where(lat>-9999, lat, np.nan)
        lat,ind = np.unique(lat, return_index=True)
        lat = lat[np.argsort(ind)]
        lat = lat[~np.isnan(lat)] # remove the nan from the array
        # lat = np.flipud(lat) # lat is upside down

        lon = xr.where(lon>-9999, lon, np.nan)
        lon = np.unique(lon)
        lon = lon[~np.isnan(lon)] # remove the nan from the array

        # changing the dim name and adding coordinates
        ds['phony_dim_0'] = ('phony_dim_0',lat)
        ds['phony_dim_1'] = ('phony_dim_1',lon)
        ds = ds.rename({'phony_dim_0': 'lat','phony_dim_1': 'lon'})
        ds = ds.reindex(lat=ds.lat[::-1])
        ds.expand_dims(time = [datetime.now()]) # adding time dimension
        return clip_around_point(ds,bounding_box)

    ds1 = xr.open_dataset(filenames[1],engine='h5netcdf',phony_dims='sort',group="/Soil_Moisture_Retrieval_Data_AM")
    varsToKeep = ['latitude','longitude','soil_moisture','tb_h_corrected','surface_temperature']
    varsToDrop = list(set(list(ds1.variables))-set(varsToKeep))

    output = xr.open_mfdataset(filenames,
                            group = 'Soil_Moisture_Retrieval_Data_AM',
                            phony_dims='access', # this line a name for the unlabeled dimensions
                            chunks={'lat': 100, 'lon': 100, 'time': 10}, # this takes time maybe because the number of chunks are too many
                            preprocess = preprocess_func, # this will apply to each of the nc files in the list
                            drop_variables = varsToDrop,
                            concat_dim='time',
                            combine='nested',
                            parallel=True,
                            engine='h5netcdf')
    output['time'] = date # assigning the dates extracted from the file name to the time dimension
    output = output.drop_vars(['latitude','longitude'])
    
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f'rm -r {save_dir}'
            subprocess.Popen(bash_cmd.split())
        output.to_netcdf(path = save_dir,mode='w', format='NETCDF4', engine='netcdf4')
    return output

# Reading multiple SM files
def create_dem(bounding_box,save_dir=None, to_nc=False):
    subdir = '/discover/nobackup/projects/lis/LS_PARAMETERS/topo_parms/MERIT'
    import rasterio
    import math
    # find the filename based on the bounding box
    if bounding_box[0]>0:
        e_w = 'e'
    else:
        e_w = 'w'

    if bounding_box[1]>0:
        n_s = 'n'
    else:
        n_s = 's'
    ew = abs(int(math.floor(bounding_box[0] / 30.0)) * 30)
    ns = abs(int(math.ceil(bounding_box[1] / 30.0)) * 30)
    filename = "merit_%s%03d%s%02d.dem" % (e_w,ew,n_s,ns)
    
    # Open the dem file
    with rasterio.open(os.path.join(subdir,filename)) as src:
        # Read the HDR data
        dem = src.read(1)
        dem[dem<=0] = np.nan # removing the negative and zero values
    
    extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
    lat = np.linspace(extent[2],extent[3],dem.shape[0])
    lon = np.linspace(extent[0],extent[1],dem.shape[1]) 
    
    ds = xr.Dataset(
        {
        "dem": (["lat", "lon"], np.flipud(dem))
        },
        coords= 
        {
        "lat" : lat,
        "lon" : lon
        })
    ds = ds.sel(lat = slice(bounding_box[1],bounding_box[3]),lon= slice(bounding_box[0],bounding_box[2])).dem
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f'rm -r {save_dir}'
            subprocess.Popen(bash_cmd.split())
            ds.to_netcdf(path = save_dir,mode='w', format='NETCDF4', engine='netcdf4')
    return ds


def extract_natural_land_cover_lst(bounding_box,lc,lst, max_radius = 15, add_plot = False, plot_time = None, save_dir=None, max_elev_diff = 100, to_nc=False):
    non_irrigated_lc_type = [8,9,10,16] # 8 Woody Savannas, 9 Savannas, 10 Grasslands, 16 Barren or Sparsely Vegetated
    buffer_zone = 1
    masks = []
    LSTs = []
    base_masks = []
    base_LSTs = []
    base_radiuses = []
    i = 0
    
    lst = lst.sel(lat = slice(bounding_box[1],bounding_box[3]),lon= slice(bounding_box[0],bounding_box[2]))
    lc = lc.interp(lat = lst.lat, lon = lst.lon, method = 'nearest')
    dem = create_dem(bounding_box)
    dem = dem.interp(lat = lst.lat, lon = lst.lon, method = 'linear').interpolate_na(dim="lat", method="linear", fill_value="extrapolate").interpolate_na(dim="lon", method="linear", fill_value="extrapolate") # filling the first col and row after interpolation
        

    while buffer_zone<max_radius+1:
        # just considering the outer ring in each iteration
        outer_ring = [(dx,dy) for dx in range(-buffer_zone,buffer_zone+1) for dy in range(-buffer_zone,buffer_zone+1) if abs(dx)==buffer_zone or abs(dy)==buffer_zone]
        for dx,dy in outer_ring:
            lc_mask = lc.LC_Type1.shift(lon = dx,lat = dy).isin(non_irrigated_lc_type)
            dem_shift = dem.shift(lon = dx,lat = dy)
            dem_mask = abs(dem_shift-dem)<max_elev_diff
            mask = np.logical_and(lc_mask,dem_mask).astype(int)
            LST = lst.LST_Day_1km.shift(lon = dx,lat = dy).where(mask)*.02
            masks.append(mask)
            LSTs.append(LST)
        LST_comb = xr.concat(LSTs,"new_dim").mean(dim = 'new_dim',skipna=True)
        mask_comb = xr.concat(masks,"new_dim").sum(dim = 'new_dim')

        if i<1: # for 3x3 kernel
            base_mask = mask_comb
            base_LST = LST_comb
            base_radius  = xr.where(base_mask>0,1,np.nan) # np.logical_and(np.isnan(base_LST), base_mask<1),np.nan,1
        else: # for 5,7,9
            base_LST = xr.where(np.logical_and(mask_comb>0 , base_mask<1),LST_comb,base_LST)
            base_radius = xr.where(np.logical_and(mask_comb>0 , base_mask<1),i,base_radius)
            base_mask = xr.where(np.logical_and(mask_comb>0 , base_mask<1),mask_comb,base_mask)
        base_masks.append(base_mask)
        base_LSTs.append(base_LST)
        base_radiuses.append(base_radius)  
        
        i +=1
                    
        print(f'kernel size = {i*2+1}x{i*2+1}')
        if not np.isnan(base_radius).any():
            break
        buffer_zone+=1
        masks = []
        LSTs = []
        del LST_comb,mask_comb
        
    baseLST_comb = xr.concat(base_LSTs,"radius")
    basemask_comb = xr.concat(base_masks,"radius") 
    baseradius_comb = xr.concat(base_radiuses,"radius")

    baseradius_comb['radius'] = baseradius_comb.radius+1
    basemask_comb['radius'] = basemask_comb.radius+1
    baseLST_comb['radius'] = baseLST_comb.radius+1
    
    if add_plot == True:
        radius_label = [f'nir within {k} km' for k in range(1,i+1)]
        
        from matplotlib.colors import ListedColormap
        cmap = plt.cm.rainbow  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(int(k)) for k in np.round(np.linspace(0,cmap.N,i))]
        cmaplist[0] = (.5, .5, .5, 1.0)
        cmap = ListedColormap(cmaplist)
        if i<6:
            col_num = i;
        else:
            col_num = int(np.ceil(i/2))
        baseradius_comb.plot(x="lon", y="lat", col="radius", col_wrap=col_num,figsize = [col_num*4,10],cmap = cmap,levels = [*range(1,len(baseradius_comb['radius'])+1)],cbar_kwargs = {'label':'Radius of Search'}) 
        baseLST_comb.sel(time = plot_time).plot(x="lon", y="lat", col="radius", col_wrap=col_num,figsize = [col_num*4,10],cmap = 'Oranges')
    base_LST = base_LST.rename('lst_nir')
    base_radius = base_radius.rename('searchRadius')
    lst_natural_lc = xr.combine_by_coords([base_LST, base_radius])
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f'rm -r {save_dir}'
            subprocess.Popen(bash_cmd.split())
        lst_natural_lc.to_netcdf(path = save_dir,mode='w', format='NETCDF4', engine='netcdf4')
    return lst_natural_lc

def irrigation_mapping_with_deltaLST(delta_lst,lc,thereshold = -2, add_plot = True):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    # Create a list of colors from red to blue
    colors = [(0.8, 0.8, 0.8), (0, 0, 1)]
    mpl.rcParams["font.size"] = 16
    # Create a colormap object
    cmap = LinearSegmentedColormap.from_list('red_blue', colors)
    # fig,ax = plt.subplots(1,3,figsize = (28,8))
    # fig.autofmt_xdate()
    neg_day_count = (delta_lst<thereshold).groupby('time.year').sum(dim = 'time')
    
    water_mask = lc.LC_Type1.isin([17]) # 17 is water in modis IGBP land cover
    crop_mask = lc.LC_Type1.isin([12]) # 12 is cropland in modis IGBP land cover
    mask = np.logical_and(neg_day_count>neg_day_count.quantile(0.8),crop_mask)
    neg_day_count = neg_day_count.where(crop_mask)
    if add_plot == True:
        plot = neg_day_count.plot(x="lon", y="lat", col="year",figsize = (20,8),cmap=cmap,cbar_kwargs = {'label':'number of irrigated days in cropland area'},robust=True)

        for axes in plot.axes.flat: # rotating the xtick label
            axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    return neg_day_count

def irrigation_season_mapping(delta_lst,lc,start_time , end_time, model = 'l2',add_plot = True):
    import ruptures as rpt
    def bin_seg_st(diff_lst_dataCube,binseg_model):
        algo = rpt.Binseg(model=binseg_model).fit(diff_lst_dataCube)
        my_bkps = algo.predict(n_bkps=2)
        # bp0 = diff_lst_dataCube['time'][my_bkps[0]]
        # bp1 = diff_lst_dataCube['time'][my_bkps[1]]
        return my_bkps[0]
    def bin_seg_end(diff_lst_dataCube,binseg_model):
        algo = rpt.Binseg(model=binseg_model).fit(diff_lst_dataCube)
        my_bkps = algo.predict(n_bkps=2)
        # bp0 = diff_lst_dataCube['time'][my_bkps[0]]
        # bp1 = diff_lst_dataCube['time'][my_bkps[1]]
        return my_bkps[1]

    water_mask = lc.LC_Type1.isin([17]) # 17 is water in modis IGBP land cover
    crop_mask = lc.LC_Type1.isin([12]) # 12 is cropland in modis IGBP land cover
    delta_lst_nonan = delta_lst.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
    
    ir_season_start = xr.apply_ufunc(
    bin_seg_st,
    delta_lst_nonan.sel(time= slice(start_time, end_time)),# now arguments in the order expected by bin_seg
    'l2',  # as above
    input_core_dims=[["time"], []],  # our function expects to receive a 1D vector along 'time' dim so time will be the input core dim
    vectorize=True,  # loop over non-core dims
    exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be set!
    )
    
    ir_season_end = xr.apply_ufunc(
    bin_seg_end,
    delta_lst_nonan.sel(time= slice(start_time, end_time)),# now arguments in the order expected by bin_seg
    'l2',  # as above
    input_core_dims=[["time"], []],  # our function expects to receive a 1D vector along 'time' dim so time will be the input core dim
    vectorize=True,  # loop over non-core dims
    exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be set!
    )
    
    ir_season_duration = ir_season_end-ir_season_start
    if add_plot == True:
        import matplotlib as mpl
        mpl.rcParams["font.size"] = 14
        import cmocean
        fig,(ax1,ax2, ax3) = plt.subplots(1,3,figsize = (28,8))
        fig.autofmt_xdate()
        delta_lst_nonan.sel(time = '2021-07-11').plot(ax = ax1, x = 'lon',vmin = -10, vmax = 10, cmap = cmocean.cm.balance, cbar_kwargs={"label": "Delta LST ($K^{0}$)"})
        ax1.set_title('Delta LST')
        ax2.set_title('Irrigation season duration (days)')

        C = np.array([
            [0,.4,0],      #  1 Evergreen Needleleaf Forest
            [0,.4,.2],      #! 2 Evergreen Broadleaf Forest    
            [.2,.8,.2],     #  3 Deciduous Needleleaf Forest
            [.2,.8,.4],     #  4 Deciduous Broadleaf Forest
            [.2,.6,.2],     #  5 Mixed Forests
            [.3,.7,0],      #  6 Closed Shrublands
            [.82,.41,.12],     #  7 Open Shurblands
            [.74,.71,.41],       #  8 Woody Savannas
            [1,.84,.0],     #  9 Savannas
            [0,1,0],        #  10 Grasslands
            [0,1,1],        #! 11 Permanant Wetlands
            [1,1,0],      #  12 Croplands
            [1,0,0],     #  13 Urban and Built-up
            [.7,.9,.3],      #! 14 Cropland/Natual Vegation Mosaic
            [1,1,1],        #! 15 Snow and Ice
            [.914,.914,.7], #  16 Barren or Sparsely Vegetated
            [.5,.7,1]])        #  17 Water (like oceans)
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        cmap = ListedColormap(C)
        lc_labels =  ['Evergreen Needleleaf Forest',
                      'Evergreen Broadleaf Forest',
                      'Deciduous Needleleaf Forest',
                      'Deciduous Broadleaf Forest',
                      'Mixed Forests',
                      'Closed Shrublands',
                      'Open Shrublands',
                      'Woody Savannas',
                      'Savannas',
                      'Grasslands',
                      'Permanent Wetlands',
                      'Croplands',
                      'Urban and Built-Up',
                      'Cropland/Natural Vegetation Mosaic',
                      'Snow and Ice',
                      'Barren or Sparsely Vegetated',
                      'Water']
        plot = lc.LC_Type1.plot(x= 'lon', ax = ax3,levels = [*range(1,19)],cmap = cmap,cbar_kwargs={"label": "Land Cover Type (MODIS IGBP)"})
        ax3.set_title('LC northern california')
        cbar = plot.colorbar
        cbar.set_ticks(np.array([*range(1,18)])+0.5)
        cbar.set_ticklabels(lc_labels)
        ir_season_duration.where(np.logical_and(~water_mask,crop_mask)).where(ir_season_duration<200).plot(x ='lon',cmap = cmocean.cm.deep,vmin = 60, ax = ax2, cbar_kwargs={"label": "Duration of irrigation season (days)"})

    return ir_season_start, ir_season_end, ir_season_duration

def extract_stations_timeseries(lst_ir,lst_nir,st_info,year):
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
def irrigation_season_timing(st_data,st_info,model = 'l2',add_plot = True):
    stid = st_info['canal']
    algo = rpt.Binseg(model=model).fit(st_data.delta_lst_nonan_st.values)
    my_bkps = algo.predict(n_bkps=2)
    
    ir_season_bkps = my_bkps

    bp0 = st_data.delta_lst_nonan_st['time'][my_bkps[0]]
    bp1 = st_data.delta_lst_nonan_st['time'][my_bkps[1]]

    # show results
    if add_plot == True:    
        mpl.rcParams["font.size"] = 14
        fig,ax1 = plt.subplots(figsize = [14,8])
        label0 = bp0.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp0.values, color='g', label=f'start: {label0}',linestyle='--')
        label1 = bp1.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp1.values, color='r', label=f'end: {label1}',linestyle='--')
        plt.axvspan(bp0.values, bp1.values, color='g', alpha=0.1)
        ax1.set_title('')
        
        st_data.lst_ir_st.plot(ax = ax1, x = 'time',label = 'Irrigated',color = 'green',linewidth=2.0)
        st_data.lst_nir_st.plot(ax = ax1, x = 'time',label = 'non-irrigated',color = 'orange',linewidth=2.0)
        plt.ylabel('LST ($K^{0}$)')
        plt.xlabel('')
        ax2 = plt.twinx(ax1)
        st_data.delta_lst_st.plot(ax = ax2, x = 'time',label = 'diff',color = 'red',linewidth=2.0)
        plt.ylabel('LST difference ($K^{0}$)')
        ax1.set_title('')
        plt.title(f'{stid}')
        fig.legend(loc = 'upper center', ncol = 5)
    return bp0,bp1
        
def irrigation_event_timing(st_data, st_info, year, df_binary, df_insitu_irrigation, model = 'l2', penalty = 1, min_seg_size = 3, segmentation_method = 'mean', add_plot = True): 
    stid = st_info['canal']

    bp0,bp1 = irrigation_season_timing(st_data, st_info, model = 'l2', add_plot = False)
    diff_lst_ir_season = st_data.delta_lst_nonan_st.sel(time=slice(bp0,bp1)) # slicing the time series for the irrigation period
    algo = rpt.KernelCPD(kernel="linear", min_size=min_seg_size).fit(diff_lst_ir_season.values)
    my_bkps = algo.predict(pen=penalty)
                                   
    diff_lst_ir_arr = diff_lst_ir_season.values
    # diff_sm_ir_arr = diff_sm_ir_season.values

    date_ir = diff_lst_ir_season['time'].values
    final_bps = []
                                   
    for bp in my_bkps[:-1]:
        if bp+1<len(diff_lst_ir_arr)-1:

            # lst0 = diff_lst_ir_arr[bp]
            # lst1 = diff_lst_ir_arr[bp+1]
            # if lst1-lst0>0:
            #     bp = bp+1
            #     if diff_lst_ir_arr[bp]>diff_lst_ir_arr[bp+1]:
            #         final_bps.append(bp)
            # elif lst1-lst0<0:
            #     final_bps.append(bp)  
            final_bps.append(bp)

    bps = date_ir[final_bps]
    bps_diff_lst = diff_lst_ir_arr[final_bps]
    lst_ir_ir_season = st_data.lst_ir_st.sel(time=slice(bp0,bp1))
    lst_ir_arr = lst_ir_ir_season.values
    final_bps.append(len(lst_ir_arr))

    # calculatiing mean delta lst for each segment
    quantile_threshold = 0.7
    bp_start = 0
    diff_lst_segs = []
    for bp in final_bps:
        diff_lst_seg = diff_lst_ir_season.isel(time = slice(bp_start,bp)).mean(dim = 'time')
        diff_lst_segs.append(diff_lst_seg)
        bp_start = bp
    diff_lst_seg_mean = xr.concat(diff_lst_segs,'time')
    if add_plot == True:
        diff_lst_seg_mean.plot(x = 'time',figsize = [18,10],linestyle='dashed',color='g', linewidth=2, marker='o',markersize=10)
    seg_irr_flag = (diff_lst_seg_mean<diff_lst_seg_mean.quantile(quantile_threshold)).astype(int)
               
    # calculating mean gradient within each segment
    datetime = diff_lst_ir_season["time"]
    diff_lst_ir_season["time"] = diff_lst_ir_season["time"].dt.dayofyear
    bp_start = 0
    lst_seg_grads = []
    for bp in final_bps:
        lst_seg_grad = diff_lst_ir_season.isel(time = slice(bp_start,bp)).differentiate("time"). mean(dim = 'time')
        lst_seg_grads.append(lst_seg_grad)
        bp_start = bp
    lst_seg_grad_mean = xr.concat(lst_seg_grads,'time')
    if add_plot == True:
        lst_seg_grad_mean.plot(x = 'time',figsize = [18,10],linestyle='dashed',color='r', linewidth=2, marker='o',markersize=10)
    # seg_irr_flag = (diff_lst_seg_mean<diff_lst_seg_mean.quantile(quantile_threshold)).astype(int)
    diff_lst_ir_season["time"] = datetime
                                   
    delta_lst_seg_mean_grad_combined = xr.merge([diff_lst_seg_mean.rename("segments mean"),lst_seg_grad_mean.rename("segments mean gradient")])

    seg_mean_grad = delta_lst_seg_mean_grad_combined.to_dataframe()
    print('number of samples: ', len(seg_mean_grad))                               
    # filling the gap between the break points (which segment is irrigated and wich one is not)
    if segmentation_method == 'mean':
        irr_segs = np.zeros_like(diff_lst_ir_season)
        bp_start = 0
        for i,bp in enumerate(final_bps):
            irr_segs[bp_start: bp]= seg_irr_flag[i]
            bp_start = bp

        irr_segs
        irr_seg = xr.zeros_like(diff_lst_ir_season)
        irr_seg = irr_seg.rename("detected segments")
        irr_seg.values = irr_segs
        df_detected_seg = irr_seg.to_dataframe()  
    elif segmentation_method == 'trend':
        irr_segs = np.zeros_like(diff_lst_ir_season)
        bp_start = 0
        prev_seg_mean = 0
        for i,bp in enumerate(final_bps):
            if (diff_lst_seg_mean[i]<prev_seg_mean):
                irr_segs[bp_start: bp]= 1
            elif (diff_lst_seg_mean[i]-prev_seg_mean)<diff_lst_seg_mean.std(dim = 'time') and diff_lst_seg_mean[i]<diff_lst_seg_mean.quantile(quantile_threshold): # if the segment mean has increased but not significantly it can be and irrigation episode
                irr_segs[bp_start: bp]= 1
            bp_start = bp
            prev_seg_mean =diff_lst_seg_mean[i]

        irr_segs
        irr_seg = xr.zeros_like(diff_lst_ir_season)
        irr_seg = irr_seg.rename("detected segments")
        irr_seg.values = irr_segs
        df_detected_seg = irr_seg.to_dataframe()
    elif segmentation_method == 'kmean': 
        from sklearn.cluster import KMeans
        features = seg_mean_grad[["segments mean","segments mean gradient"]].values
        kmeans= KMeans(n_clusters=3,random_state=0).fit(features)
        kmean_flags = kmeans.labels_
        centeroid = kmeans.cluster_centers_

        # we are working under the assumption that the irrigated segments having lower mean Delta LST
        irrigated_idx = np.argmin(centeroid[:,0])
        kmean_flags = (kmean_flags == irrigated_idx).astype(int)

        irr_segs = np.zeros_like(diff_lst_ir_season)
        bp_start = 0
        for i,bp in enumerate(final_bps):
            irr_segs[bp_start: bp]= kmean_flags[i]
            bp_start = bp

        irr_segs
        irr_seg = xr.zeros_like(diff_lst_ir_season)
        irr_seg = irr_seg.rename("detected segments")
        irr_seg.values = irr_segs
            
        df_detected_seg = irr_seg.to_dataframe()
    if add_plot == True:
        diff_lst_ir_season.plot(x = 'time',figsize = (15,8))
        label0 = bp0.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp0.values, color='g', label=f'start: {label0}',linestyle='--')
        label1 = bp1.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp1.values, color='r', label=f'end: {label1}',linestyle='--')
        plt.axvspan(bp0.values, bp1.values, color='g', alpha=0.1)
        # plt.plot(bps,bps_diff_lst,'ro',label = 'potential irrigation application based on diff LST break points')
        plt.plot(bps,bps_diff_lst,'ro',label = 'potential irrigation application based on diff LST break points')
        # plt.plot(bps_diff_sm[bps_diff_sm>0].time, diff_lst_ir_season.sel(time = bps_diff_sm[bps_diff_sm>0].time),'kx',label = 'potential irrigation application based on both LST break points and (+) SM diff (IR-nIR)', markersize=10)
        plt.legend(loc= 'lower center', ncol =2)
        # ax.set_title('')
        plt.title(f'station code: {stid}')

    df_bps = pd.DataFrame(np.zeros([366,1]),columns = ['Break points'])
    df_bps['date'] = pd.date_range(start=f'{year}-01-01', periods=366,freq='D')
    df_bps.set_index('date',inplace=True)
    df_bps.loc[bps] = 1#*df_insitu_irrigation[stid[ID]].max()
    bps = pd.to_datetime(bps)

    df_insitu_irrigation.index = pd.to_datetime(df_insitu_irrigation.index)
    df_binary.index = pd.to_datetime(df_binary.index)

    df_insitu_elec  = df_insitu_irrigation[df_insitu_irrigation.index.year == int(year)] # only year 2020
    df_insitu_bin  = df_binary[df_binary.index.year == int(year)] # only year 2020
    df_bps = df_bps[df_bps.index.isin(df_insitu_bin.index)] # filtering df_bps based on df_insitu_2020
    # df_bps_sm = df_bps_sm[df_bps_sm.index.isin(df_insitu_bin.index)] # filtering df_bps based on df_insitu_2020

    # df_combo_bin_bps = pd.concat([df_insitu_bin[stid],df_bps,df_bps_sm], axis=1) # ,df_bps_sm
    df_combo_bin_bps = pd.concat([df_insitu_bin[stid],df_bps,df_detected_seg['detected segments']], axis=1) # ,df_bps_sm

    df_combo_bin_bps.index = df_combo_bin_bps.index.strftime('%y-%b-%d')
    df_combo_bin_bps = df_combo_bin_bps.rename(columns={f"{stid}": f"observed segments ({stid})"})
    
    if add_plot == True:
        plt.subplots(figsize = [24,4])
        colors = ((.85, .85, 0.85), (0, 0, 0))
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
        ax = sns.heatmap(df_combo_bin_bps.T,cmap = cmap,cbar_kws={'label': 'irrigation (0=No, 1=Yes)'})
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25,0.75])
        colorbar.set_ticklabels(['0', '1'])
        plt.title('Irrigation episodes vs detected irrigation events')
        plt.ylabel('observed vs detected')
                                   
        df_combo_elec_bps = pd.concat([df_insitu_elec[stid],df_bps*df_insitu_irrigation[stid].quantile(.95)/2,df_detected_seg['detected segments']*df_insitu_irrigation[stid].quantile(.95)/2], axis=1)
        df_combo_elec_bps.index = df_combo_elec_bps.index.strftime('%y-%b-%d')
        plt.subplots(figsize = [24,4])
        sns.heatmap(df_combo_elec_bps.T,cmap = sns.color_palette("rocket_r", as_cmap=True),cbar_kws={'label': 'Electricity consumption (${m^3/day}$)'},vmax = df_insitu_irrigation[stid].quantile(.98))
        plt.title('Electricity consumption vs detected irrigation events')
        plt.ylabel('Fields')

        df_combo_elec_bin = pd.concat([df_insitu_elec[stid],(df_insitu_irrigation[stid]*df_detected_seg['detected segments'].T).T.replace(0,np.nan)], axis=1)
        axes = df_combo_elec_bin.plot.line(figsize = [18,6],style=['c-','k*-'],linewidth = 3,fontsize = 16,grid = True)
        axes.legend(["water release", "detected irrigation episodes"]);

    # Creating the confusion matirx and other classification metrics
    irr_true = df_combo_bin_bps[f'observed segments ({stid})'].fillna(0)
    irr_pred = df_combo_bin_bps['detected segments'].fillna(0)
    from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score, f1_score,ConfusionMatrixDisplay
    cm = confusion_matrix(irr_true,irr_pred)
    # Create a confusion matrix display object


    # Performance score
    acc = accuracy_score(irr_true,irr_pred)*100
    prec = precision_score(irr_true,irr_pred)*100
    recall = recall_score(irr_true,irr_pred)*100
    f1 = f1_score(irr_true,irr_pred)*100
    performance = [acc,prec,recall,f1]
    if add_plot == True: 
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # Plot the confusion matrix
        disp.plot()
        plt.show()
        print('accuracy: {:.2f}%'.format(acc))
        print('precision: {:.2f}%'.format(prec))
        print('recall: {:.2f}%'.format(recall))
        print('f1_score: {:.2f}%'.format(f1))
    return performance