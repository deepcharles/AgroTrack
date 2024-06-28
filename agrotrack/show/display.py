import numpy as np
from itertools import cycle
from scipy import stats

def display(signal, true_chg_pts, computed_chg_pts=None, computed_chg_pts_color="k", 
            computed_chg_pts_linewidth=3, computed_chg_pts_linestyle="--", 
            computed_chg_pts_alpha=1.0, show_piecewise_linear=True, **kwargs):
    """
    Display a signal and the change points provided in alternating colors.
    If another set of change points is provided, they are displayed with dashed vertical lines.
    A piecewise linear line is added by fitting a line to each segment.

    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.
        computed_chg_pts_color (str, optional): color of the lines indicating the computed_chg_pts.
        computed_chg_pts_linewidth (int, optional): linewidth of the lines indicating the computed_chg_pts.
        computed_chg_pts_linestyle (str, optional): linestyle of the lines indicating the computed_chg_pts.
        computed_chg_pts_alpha (float, optional): alpha of the lines indicating the computed_chg_pts.
        show_piecewise_linear (bool, optional): whether to show the piecewise linear line.
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.

    Returns:
        tuple: (figure, axarr) with a matplotlib.figure.Figure object and an array of Axes objects.
    """
    def pairwise(iterable):
        # pairwise('ABCDEFG') → AB BC CD DE EF FG
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield a, b
            a = b
        
    import matplotlib.pyplot as plt


    if type(signal) != np.ndarray:
        signal = signal.values
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    matplotlib_options = {
        "figsize": (10, 5 * n_features),
    }
    matplotlib_options.update(kwargs)

    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    if n_features == 1:
        axarr = [axarr]
        
    COLOR_CYCLE = ["#42f4b0", "#f44174"]
    for axe, sig in zip(axarr, signal.T):
        color_cycle = cycle(COLOR_CYCLE)
        axe.plot(range(n_samples), sig, alpha=0.7,color='black', linewidth=2)

        bkps = [0] + sorted(true_chg_pts) + [n_samples]
        alpha = 0.2

        for (start, end), col in zip(pairwise(bkps), color_cycle):
            if start!=end:
                axe.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=alpha)

                if show_piecewise_linear:
                    x = np.arange(start, end)
                    y = sig[start:end]
                    slope, intercept, _, _, _ = stats.linregress(x, y)
                    fitted_line = slope * x + intercept
                    axe.plot(x, fitted_line, color='red', linewidth=2, linestyle='-')

        if computed_chg_pts is not None:
            for bkp in computed_chg_pts:
                if bkp != 0 and bkp < n_samples:
                    axe.axvline(
                        x=bkp - 0.5,
                        color=computed_chg_pts_color,
                        linewidth=computed_chg_pts_linewidth,
                        linestyle=computed_chg_pts_linestyle,
                        alpha=computed_chg_pts_alpha,
                    )

    fig.tight_layout()
    return fig, axarr