import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def irrigation_mapping_with_deltaLST(delta_lst, lc, thereshold=-2, add_plot=True):
    """Use deltaLST to xarray dataset to map the irrigated area by putting a threshold on the drop in the temperature caused by irrigation

    Parameters:

    delta_lst: [Xarray dataset] Original pixel lst - natural pixel lst
    lc: [Xarray dataset] MODIS annual land cover map xarray dataset
    thereshold: [negative number] the threshold for temprature reduction to be considered as an irrigation event
    add_plot: [True or False] option to add a plot that shows the number of irrigated days (days with deltaLST<threshold) for each year separatly

    Return:

    neg_day_count: [Xarray dataset] number of irrigation day at each pixel
    """

    # Create a list of colors from red to blue
    colors = [(0.8, 0.8, 0.8), (0, 0, 1)]
    mpl.rcParams["font.size"] = 16
    # Create a colormap object
    cmap = LinearSegmentedColormap.from_list("red_blue", colors)
    # fig,ax = plt.subplots(1,3,figsize = (28,8))
    # fig.autofmt_xdate()
    neg_day_count = (delta_lst < thereshold).groupby("time.year").sum(dim="time")

    water_mask = lc.LC_Type1.isin([17])  # 17 is water in modis IGBP land cover
    crop_mask = lc.LC_Type1.isin([12])  # 12 is cropland in modis IGBP land cover
    mask = np.logical_and(neg_day_count > neg_day_count.quantile(0.8), crop_mask)
    neg_day_count = neg_day_count.where(crop_mask)
    if add_plot == True:
        plot = neg_day_count.plot(
            x="lon",
            y="lat",
            col="year",
            figsize=(20, 8),
            cmap=cmap,
            cbar_kwargs={"label": "number of irrigated days in cropland area"},
            robust=True,
        )

        for axes in plot.axes.flat:  # rotating the xtick label
            axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    return neg_day_count
