import os
import glob
import xarray as xr
from datetime import datetime
import subprocess

# clip a dataset function with a bbox
def clip_around_point(ds,bounding_box): 
    return ds.sel(lat = slice(bounding_box[1], bounding_box[3]), lon = slice(bounding_box[0], bounding_box[2]))

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
