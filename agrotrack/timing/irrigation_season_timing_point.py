import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt
import xarray as xr


def irrigation_season_timing_point(st_data, st_info, model="l2", add_plot=True):
    """
    Determine the start and end of the irrigation season in the pixel using binary change point detection algorithm from ruptures package to perform fast signal segmentation.

    :param st_data: The LST of the natural pixel assigned to the original pixel
    :type st_data: Xarray dataset

    :param st_info: A data frame with three columns {'name', 'lat', 'lon'}
    :type st_info: Pandas DataFrame

    :param model: Cost function used for detection of break points. Default is 'l2' (This cost function detects mean-shifts in a signal). Other options: "l1", "rbf", "linear", "normal", "ar"
    :type model: str, optional

    :param add_plot: Option to add a plot that shows the LST of original pixel, natural pixel assigned to the original pixel and the delta LST + two lines indicating the start and the end of the irrigation season
    :type add_plot: bool, optional

    :returns: Start (bp0) and the end (bp1) of the irrigation season
    :rtype: tuple(date, date)

    """
    stid = st_info["name"]
    algo = rpt.Binseg(model=model).fit(st_data.delta_lst_nonan_st.values)
    my_bkps = algo.predict(n_bkps=2)

    ir_season_bkps = my_bkps

    bp0 = st_data.delta_lst_nonan_st["time"][my_bkps[0]]
    bp1 = st_data.delta_lst_nonan_st["time"][my_bkps[1]]

    # show results
    if add_plot == True:
        mpl.rcParams["font.size"] = 14
        fig, ax1 = plt.subplots(figsize=[14, 8])
        label0 = bp0.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp0.values, color="g", label=f"start: {label0}", linestyle="--")
        label1 = bp1.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp1.values, color="r", label=f"end: {label1}", linestyle="--")
        plt.axvspan(bp0.values, bp1.values, color="g", alpha=0.1)
        ax1.set_title("")

        st_data.lst_ir_st.plot(
            ax=ax1, x="time", label="Irrigated", color="green", linewidth=2.0
        )
        st_data.lst_nir_st.plot(
            ax=ax1, x="time", label="non-irrigated", color="orange", linewidth=2.0
        )
        plt.ylabel("LST ($K^{0}$)")
        plt.xlabel("")
        ax2 = plt.twinx(ax1)
        st_data.delta_lst_st.plot(
            ax=ax2, x="time", label="diff", color="red", linewidth=2.0
        )
        plt.ylabel("LST difference ($K^{0}$)")
        ax1.set_title("")
        plt.title(f"{stid}")
        fig.legend(loc="upper center", ncol=5)
    return bp0, bp1
