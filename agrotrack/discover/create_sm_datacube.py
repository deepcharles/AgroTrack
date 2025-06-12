import glob
import os
import subprocess
from datetime import datetime

import numpy as np
import xarray as xr


# clip a dataset function with a bbox
def clip_around_point(ds, bounding_box):
    return ds.sel(
        lat=slice(bounding_box[1], bounding_box[3]),
        lon=slice(bounding_box[0], bounding_box[2]),
    )


# creates SMAP Enhanced soil moisture data cube from h5 file on discover
def create_sm_datacube(years, bounding_box, save_dir, to_nc=True):
    """creates SMAP Enhanced soil moisture data cube from h5 file on discover"""

    subdir = "/discover/nobackup/projects/lis/RS_DATA/SMAP/SPL3SMP_E.005/"

    # Extracting time stamp in datetime format from the name of the MODIS files
    def extract_date(filename):
        return datetime.strptime(filename.split("/")[-1], "%Y.%m.%d")

    # reading all the files and concating them
    if isinstance(years, list):
        files = []
        for year in years:
            files += sorted(glob.glob(os.path.join(subdir, f"{year}.*")))
    else:
        files = sorted(glob.glob(os.path.join(subdir, f"{years}.*")))
    filenames = []
    date = []
    for file in files:
        filename, *other = glob.glob(file + "/*.h5")
        filenames.append(filename)
        date.append(extract_date(file))

    def preprocess_func(ds):
        # reading lat and lon from the file
        lat = ds["latitude"]
        lon = ds["longitude"]

        lat = xr.where(lat > -9999, lat, np.nan)
        lat, ind = np.unique(lat, return_index=True)
        lat = lat[np.argsort(ind)]
        lat = lat[~np.isnan(lat)]  # remove the nan from the array
        # lat = np.flipud(lat) # lat is upside down

        lon = xr.where(lon > -9999, lon, np.nan)
        lon = np.unique(lon)
        lon = lon[~np.isnan(lon)]  # remove the nan from the array

        # changing the dim name and adding coordinates
        ds["phony_dim_0"] = ("phony_dim_0", lat)
        ds["phony_dim_1"] = ("phony_dim_1", lon)
        ds = ds.rename({"phony_dim_0": "lat", "phony_dim_1": "lon"})
        ds = ds.reindex(lat=ds.lat[::-1])
        ds.expand_dims(time=[datetime.now()])  # adding time dimension
        return clip_around_point(ds, bounding_box)

    ds1 = xr.open_dataset(
        filenames[1],
        engine="h5netcdf",
        phony_dims="sort",
        group="/Soil_Moisture_Retrieval_Data_AM",
    )
    varsToKeep = [
        "latitude",
        "longitude",
        "soil_moisture",
        "tb_h_corrected",
        "surface_temperature",
    ]
    varsToDrop = list(set(list(ds1.variables)) - set(varsToKeep))

    output = xr.open_mfdataset(
        filenames,
        group="Soil_Moisture_Retrieval_Data_AM",
        phony_dims="access",  # this line a name for the unlabeled dimensions
        chunks={
            "lat": 100,
            "lon": 100,
            "time": 10,
        },  # this takes time maybe because the number of chunks are too many
        preprocess=preprocess_func,  # this will apply to each of the nc files in the list
        drop_variables=varsToDrop,
        concat_dim="time",
        combine="nested",
        parallel=True,
        engine="h5netcdf",
    )
    output[
        "time"
    ] = date  # assigning the dates extracted from the file name to the time dimension
    output = output.drop_vars(["latitude", "longitude"])

    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f"rm -r {save_dir}"
            subprocess.Popen(bash_cmd.split())
        output.to_netcdf(path=save_dir, mode="w", format="NETCDF4", engine="netcdf4")
    return output
