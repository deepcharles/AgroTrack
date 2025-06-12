import os

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def irrigation_season_timing_array(delta_lst, lc, year, model="l2", add_plot=True):
    """
    create an array of start, end and duration of irrigation season for each pixel by analyzing delta LST signal with binary change point detection algorithm

    Parameters:

    delta_lst: [Xarray dataset] Original pixel lst - natural pixel lst
    lc: [Xarray dataset] MODIS annual land cover map xarray dataset
    year: [20xx] year for which we want to extract the irrigation season information
    model: cost function usded for detection of break points, default 'l2' (This cost function detects mean-shifts in a signal) other options: "l2"  # "l1", "rbf", "linear", "normal", "ar"
    add_plot: [True or False] option to add a plot that shows the map of the irrigation season duration along with the landcover map

    Return:

    ir_season_start, ir_season_end, ir_season_duration: [Xarray dataset] returns irrigation season attribute including start, end and duration
    """

    def bin_seg_st(diff_lst_dataCube, binseg_model):
        algo = rpt.Binseg(model=binseg_model).fit(diff_lst_dataCube)
        my_bkps = algo.predict(n_bkps=2)
        # bp0 = diff_lst_dataCube['time'][my_bkps[0]]
        # bp1 = diff_lst_dataCube['time'][my_bkps[1]]
        return my_bkps[0]

    def bin_seg_end(diff_lst_dataCube, binseg_model):
        algo = rpt.Binseg(model=binseg_model).fit(diff_lst_dataCube)
        my_bkps = algo.predict(n_bkps=2)
        # bp0 = diff_lst_dataCube['time'][my_bkps[0]]
        # bp1 = diff_lst_dataCube['time'][my_bkps[1]]
        return my_bkps[1]

    water_mask = lc.LC_Type1.isin([17])  # 17 is water in modis IGBP land cover
    crop_mask = lc.LC_Type1.isin([12])  # 12 is cropland in modis IGBP land cover
    delta_lst_nonan = delta_lst.interpolate_na(
        dim="time", method="linear", fill_value="extrapolate"
    )

    ir_season_start = xr.apply_ufunc(
        bin_seg_st,
        delta_lst_nonan.sel(
            time=year
        ),  # now arguments in the order expected by bin_seg
        "l2",  # as above
        input_core_dims=[
            ["time"],
            [],
        ],  # our function expects to receive a 1D vector along 'time' dim so time will be the input core dim
        vectorize=True,  # loop over non-core dims
        exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be set!
    )

    ir_season_end = xr.apply_ufunc(
        bin_seg_end,
        delta_lst_nonan.sel(
            time=year
        ),  # now arguments in the order expected by bin_seg
        "l2",  # as above
        input_core_dims=[
            ["time"],
            [],
        ],  # our function expects to receive a 1D vector along 'time' dim so time will be the input core dim
        vectorize=True,  # loop over non-core dims
        exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be set!
    )

    ir_season_duration = ir_season_end - ir_season_start
    if add_plot == True:
        mpl.rcParams["font.size"] = 14

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 22))
        fig.autofmt_xdate()
        ir_season_duration.where(np.logical_and(~water_mask, crop_mask)).where(
            ir_season_duration < 200
        ).plot(
            x="lon",
            cmap=cmocean.cm.deep,
            vmin=60,
            ax=ax1,
            cbar_kwargs={"label": "Duration of irrigation season (days)"},
        )
        ax1.set_title("Irrigation season duration (days)")

        C = np.array(
            [
                [0, 0.4, 0],  #  1 Evergreen Needleleaf Forest
                [0, 0.4, 0.2],  #! 2 Evergreen Broadleaf Forest
                [0.2, 0.8, 0.2],  #  3 Deciduous Needleleaf Forest
                [0.2, 0.8, 0.4],  #  4 Deciduous Broadleaf Forest
                [0.2, 0.6, 0.2],  #  5 Mixed Forests
                [0.3, 0.7, 0],  #  6 Closed Shrublands
                [0.82, 0.41, 0.12],  #  7 Open Shurblands
                [0.74, 0.71, 0.41],  #  8 Woody Savannas
                [1, 0.84, 0.0],  #  9 Savannas
                [0, 1, 0],  #  10 Grasslands
                [0, 1, 1],  #! 11 Permanant Wetlands
                [1, 1, 0],  #  12 Croplands
                [1, 0, 0],  #  13 Urban and Built-up
                [0.7, 0.9, 0.3],  #! 14 Cropland/Natual Vegation Mosaic
                [1, 1, 1],  #! 15 Snow and Ice
                [0.914, 0.914, 0.7],  #  16 Barren or Sparsely Vegetated
                [0.5, 0.7, 1],
            ]
        )  #  17 Water (like oceans)

        cmap = ListedColormap(C)
        lc_labels = [
            "Evergreen Needleleaf Forest",
            "Evergreen Broadleaf Forest",
            "Deciduous Needleleaf Forest",
            "Deciduous Broadleaf Forest",
            "Mixed Forests",
            "Closed Shrublands",
            "Open Shrublands",
            "Woody Savannas",
            "Savannas",
            "Grasslands",
            "Permanent Wetlands",
            "Croplands",
            "Urban and Built-Up",
            "Cropland/Natural Vegetation Mosaic",
            "Snow and Ice",
            "Barren or Sparsely Vegetated",
            "Water",
        ]
        plot = lc.LC_Type1.plot(
            x="lon",
            ax=ax2,
            levels=[*range(1, 19)],
            cmap=cmap,
            cbar_kwargs={"label": "Land Cover Type (MODIS IGBP)"},
        )
        ax2.set_title("LC northern california")
        cbar = plot.colorbar
        cbar.set_ticks(np.array([*range(1, 18)]) + 0.5)
        cbar.set_ticklabels(lc_labels)

    return ir_season_start, ir_season_end, ir_season_duration
