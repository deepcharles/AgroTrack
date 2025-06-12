import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def extract_natural_land_cover_lst(
    bounding_box,
    lc,
    lst,
    max_radius=15,
    max_elev_diff=100,
    add_plot=False,
    plot_time=None,
    to_nc=False,
    save_dir=None,
):
    """
    Process each pixel to find and average the land surface temperature (LST) from surrounding natural land cover pixels.

    This function evaluates each pixel in the provided area. For every pixel, it identifies neighboring pixels
    classified as natural land cover, starting from immediate neighbors and expanding the search radius until
    at least one natural pixel is found or the maximum search radius is reached. It then calculates the average
    LST of all identified natural land cover pixels and assigns this value to the original pixel.

    :param bounding_box: Defines the area of interest as [lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat]
    :type bounding_box: list
    :param lc: MODIS annual land cover map dataset
    :type lc: xarray.Dataset
    :param lst: MODIS LST dataset
    :type lst: xarray.Dataset
    :param max_radius: Maximum radius (in km) to search for natural pixels
    :type max_radius: float
    :param max_elev_diff: Maximum valid elevation difference between the original and natural land cover pixels
    :type max_elev_diff: float
    :param add_plot: If True, generates a plot showing the search radius for each pixel and the resulting natural pixels LST map, defaults to False
    :type add_plot: bool, optional
    :param plot_time: The date to create the plot for, if add_plot is True
    :type plot_time: datetime or str, optional
    :param to_nc: If True, saves the output as a NetCDF file, defaults to False
    :type to_nc: bool, optional
    :param save_dir: Directory to save the output NetCDF file, if to_nc is True
    :type save_dir: str, optional

    :return: Dataset containing the LST of natural pixels assigned to the original pixels
    :rtype: xarray.Dataset

    :raises ValueError: If max_radius is not a positive number
    :raises IOError: If unable to save NetCDF file when to_nc is True

    .. note::
       - The function iteratively searches for natural pixels, expanding the search radius up to max_radius.
       - Elevation differences between original and natural pixels are considered, limited by max_elev_diff.
       - If add_plot is True, the function visualizes the search process and final LST assignment.

    .. warning::
       Large bounding boxes or small max_radius values may result in long processing times.

    """
    from agrotrack import create_dem

    non_irrigated_lc_type = [
        8,
        9,
        10,
        16,
    ]  # 8 Woody Savannas, 9 Savannas, 10 Grasslands, 16 Barren or Sparsely Vegetated
    buffer_zone = 1
    masks = []
    LSTs = []
    base_masks = []
    base_LSTs = []
    base_radiuses = []
    i = 0

    lst = lst.sel(
        lat=slice(bounding_box[1], bounding_box[3]),
        lon=slice(bounding_box[0], bounding_box[2]),
    )
    lc = lc.interp(lat=lst.lat, lon=lst.lon, method="nearest")
    dem = create_dem(bounding_box)
    dem = (
        dem.interp(lat=lst.lat, lon=lst.lon, method="linear")
        .interpolate_na(dim="lat", method="linear", fill_value="extrapolate")
        .interpolate_na(dim="lon", method="linear", fill_value="extrapolate")
    )  # filling the first col and row after interpolation

    while buffer_zone < max_radius + 1:
        # just considering the outer ring in each iteration
        outer_ring = [
            (dx, dy)
            for dx in range(-buffer_zone, buffer_zone + 1)
            for dy in range(-buffer_zone, buffer_zone + 1)
            if abs(dx) == buffer_zone or abs(dy) == buffer_zone
        ]
        for dx, dy in outer_ring:
            lc_mask = lc.LC_Type1.shift(lon=dx, lat=dy).isin(non_irrigated_lc_type)
            dem_shift = dem.shift(lon=dx, lat=dy)
            dem_mask = abs(dem_shift - dem) < max_elev_diff
            mask = np.logical_and(lc_mask, dem_mask).astype(int)
            LST = lst.LST_Day_1km.shift(lon=dx, lat=dy).where(mask) * 0.02
            masks.append(mask)
            LSTs.append(LST)
        LST_comb = xr.concat(LSTs, "new_dim").mean(dim="new_dim", skipna=True)
        mask_comb = xr.concat(masks, "new_dim").sum(dim="new_dim")

        if i < 1:  # for 3x3 kernel
            base_mask = mask_comb
            base_LST = LST_comb
            base_radius = xr.where(
                base_mask > 0, 1, np.nan
            )  # np.logical_and(np.isnan(base_LST), base_mask<1),np.nan,1
        else:  # for 5,7,9
            base_LST = xr.where(
                np.logical_and(mask_comb > 0, base_mask < 1), LST_comb, base_LST
            )
            base_radius = xr.where(
                np.logical_and(mask_comb > 0, base_mask < 1), i, base_radius
            )
            base_mask = xr.where(
                np.logical_and(mask_comb > 0, base_mask < 1), mask_comb, base_mask
            )
        base_masks.append(base_mask)
        base_LSTs.append(base_LST)
        base_radiuses.append(base_radius)

        i += 1

        print(f"kernel size = {i*2+1}x{i*2+1}")
        if not np.isnan(base_radius).any():
            break
        buffer_zone += 1
        masks = []
        LSTs = []
        del LST_comb, mask_comb

    baseLST_comb = xr.concat(base_LSTs, "radius")
    basemask_comb = xr.concat(base_masks, "radius")
    baseradius_comb = xr.concat(base_radiuses, "radius")

    baseradius_comb["radius"] = baseradius_comb.radius + 1
    basemask_comb["radius"] = basemask_comb.radius + 1
    baseLST_comb["radius"] = baseLST_comb.radius + 1

    if add_plot == True:
        radius_label = [f"nir within {k} km" for k in range(1, i + 1)]

        from matplotlib.colors import ListedColormap

        cmap = plt.cm.rainbow  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(int(k)) for k in np.round(np.linspace(0, cmap.N, i))]
        cmaplist[0] = (0.5, 0.5, 0.5, 1.0)
        cmap = ListedColormap(cmaplist)
        if i < 6:
            col_num = i
        else:
            col_num = int(np.ceil(i / 2))
        baseradius_comb.plot(
            x="lon",
            y="lat",
            col="radius",
            col_wrap=col_num,
            figsize=[col_num * 4, 10],
            cmap=cmap,
            levels=[*range(1, len(baseradius_comb["radius"]) + 1)],
            cbar_kwargs={"label": "Radius of Search"},
        )
        baseLST_comb.sel(time=plot_time).plot(
            x="lon",
            y="lat",
            col="radius",
            col_wrap=col_num,
            figsize=[col_num * 4, 10],
            cmap="Oranges",
        )
    base_LST = base_LST.rename("lst_nir")
    base_radius = base_radius.rename("searchRadius")
    lst_natural_lc = xr.combine_by_coords([base_LST, base_radius])
    if to_nc == True:
        if os.path.exists(save_dir):
            bash_cmd = f"rm -r {save_dir}"
            subprocess.Popen(bash_cmd.split())
        lst_natural_lc.to_netcdf(
            path=save_dir, mode="w", format="NETCDF4", engine="netcdf4"
        )
    return lst_natural_lc
