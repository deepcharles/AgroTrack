import math
import os
import subprocess
from datetime import datetime

import numpy as np
import rasterio
import xarray as xr


# Reading DEM files
def create_dem(bounding_box, save_dir=None, to_nc=False):
    """Create DEM from MERIT data by mosaicing the tiles"""

    subdir = "/discover/nobackup/projects/lis/LS_PARAMETERS/topo_parms/MERIT"

    # find the filename based on the bounding box
    if bounding_box[0] > 0:
        e_w = "e"
    else:
        e_w = "w"

    if bounding_box[1] > 0:
        n_s = "n"
    else:
        n_s = "s"
    ew = abs(int(math.floor(bounding_box[0] / 30.0)) * 30)
    ns = abs(int(math.ceil(bounding_box[1] / 30.0)) * 30)
    filename = "merit_%s%03d%s%02d.dem" % (e_w, ew, n_s, ns)

    # Open the dem file
    with rasterio.open(os.path.join(subdir, filename)) as src:
        # Read the HDR data
        dem = src.read(1)
        dem[dem <= 0] = np.nan  # removing the negative and zero values

    extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
    lat = np.linspace(extent[2], extent[3], dem.shape[0])
    lon = np.linspace(extent[0], extent[1], dem.shape[1])

    ds = xr.Dataset(
        {"dem": (["lat", "lon"], np.flipud(dem))}, coords={"lat": lat, "lon": lon}
    )
    ds = ds.sel(
        lat=slice(bounding_box[1], bounding_box[3]),
        lon=slice(bounding_box[0], bounding_box[2]),
    ).dem
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f"rm -r {save_dir}"
            subprocess.Popen(bash_cmd.split())
            ds.to_netcdf(path=save_dir, mode="w", format="NETCDF4", engine="netcdf4")
    return ds
