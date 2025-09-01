import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"  # \usepackage{siunitx}"
)

FONTSIZE = 18
FONTSIZE_LEGEND = 18
FONTSIZE_TICK = 18

COLORS = {
    "part": "#000000",
    "reco": "#E69F00",
    "Transformer": "#B1AFD4",
    "Transf": "#B1AFD4",
    "L-GATr": "#419108",
    # "#F0E442",
    # "#0072B2",
    # "#D55E00",
    # "#CC79A7",
}


def plot_loss(file, losses, lr=None, labels=None, logy=True):
    labels = [None for _ in range(len(losses))] if labels is None else labels
    if len(losses[1]) == 0:  # catch no-validations case
        losses = [losses[0]]
        labels = [labels[0]]
    iterations = range(1, len(losses[0]) + 1)
    fig, ax = plt.subplots()
    for i, loss, label in zip(range(len(losses)), losses, labels):
        if len(loss) == len(iterations):
            its = iterations
        else:
            frac = len(losses[0]) / len(loss)
            its = np.arange(1, len(loss) + 1) * frac
        ax.plot(its, loss, label=label)

    if logy:
        ax.set_yscale("log")
    if lr is not None:
        axright = ax.twinx()
        axright.plot(iterations, lr, label="learning rate", color="crimson")
        axright.set_ylabel("Learning rate", fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel("Loss", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_metric(file, metrics, metric_label, labels=None, logy=False):
    labels = [None for _ in range(len(metrics))] if labels is None else labels
    iterations = range(1, len(metrics[0]) + 1)
    fig, ax = plt.subplots()
    for i, metric, label in zip(range(len(metrics)), metrics, labels):
        if len(metric) == len(iterations):
            its = iterations
        else:
            frac = len(metrics[0]) / len(metric)
            its = np.arange(1, len(metric) + 1) * frac
        ax.plot(its, metric, label=label)

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(metric_label, fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper left")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_ratio_histogram(
    data,
    reference_key,
    xlabel,
    bins_range,
    logy=False,
    n_bins=40,
    error_range=None,
    error_ticks=[0.95, 1.0, 1.05],
    file=None,
    legend_loc="upper right",
    no_ratio_keys=[],
    n_int_per_bin=1,
    xrange=None,
    yrange=None,
):
    """
    Plotting code used for all 1d distributions

    Parameters:
    file: str | None
        Path to the output file, None to show()
    data: dict
        Contains keys with np.ndarray of shape (nevents)
        Keys are used as labels
    reference_key: str
        Key of the data to use as the reference for ratio
    xlabel: str
    xrange: tuple with 2 floats
    logy: bool
    n_bins: int
    error_range: tuple with 2 floats
    error_ticks: tuple with 3 floats
    """
    # construct labels and colors
    no_ratio_keys.append(reference_key)

    labels = data.keys()
    if bins_range.dtype in [torch.int32, torch.int64]:
        bins = np.arange(bins_range[0] - 0.5, bins_range[1] + 0.5, n_int_per_bin)
    else:
        bins = np.linspace(bins_range[0], bins_range[1], n_bins)

    # construct histograms
    hists = {}
    for key in labels:
        y, _ = np.histogram(data[key], bins=bins, range=bins_range)
        hists[key] = y
    hist_errors = {key: np.sqrt(y) for key, y in hists.items()}

    integrals = {key: np.sum((bins[1:] - bins[:-1]) * y) for key, y in hists.items()}
    scales = {
        key: 1 / integral if integral != 0.0 else 1.0
        for key, integral in integrals.items()
    }

    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00},
    )

    for key in data.keys():
        y = hists[key]
        y_err = hist_errors[key]
        scale = scales[key]
        label = key
        color = COLORS.get(key, "#000000")

        axs[0].step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y + y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y - y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].fill_between(
            bins,
            dup_last(y - y_err) * scale,
            dup_last(y + y_err) * scale,
            facecolor=color,
            alpha=0.3,
            step="post",
        )

        # vertical lines at lower and upper bin edges
        left, right = bins[0], bins[-1]

        # lower bound
        axs[0].vlines(
            left,
            0,
            y[0] * scale,
            colors=color,
            linewidth=1.0,
        )

        # upper bound
        axs[0].vlines(
            right,
            0,
            y[-1] * scale,
            colors=color,
            linewidth=1.0,
        )

        # if label == reference_key:
        #     # axs[0].fill_between(
        #     #     bins,
        #     #     dup_last(y) * scale,
        #     #     0.0 * dup_last(y),
        #     #     facecolor=color,
        #     #     alpha=0.1,
        #     #     step="post",
        #     # )
        #     continue

        if label in no_ratio_keys:
            continue

        ratio = (y * scale) / (hists[reference_key] * scales[reference_key])
        ratio_err = np.sqrt(
            (y_err / y) ** 2 + (hist_errors[reference_key] / hists[reference_key]) ** 2
        )
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.0
        ratio_err[ratio_isnan] = 0.0

        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
        axs[1].step(
            bins,
            dup_last(ratio + ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].step(
            bins,
            dup_last(ratio - ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].fill_between(
            bins,
            dup_last(ratio - ratio_err),
            dup_last(ratio + ratio_err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )

    if isinstance(legend_loc, dict):
        axs[0].legend(
            **legend_loc, frameon=False, fontsize=FONTSIZE_LEGEND, handlelength=1.0
        )
    elif legend_loc is not None:
        axs[0].legend(
            loc=legend_loc, frameon=False, fontsize=FONTSIZE_LEGEND, handlelength=1.0
        )

    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    if logy:
        axs[0].set_yscale("log")

    if yrange is None:
        _, ymax = axs[0].get_ylim()
        axs[0].set_ylim(0.0, ymax)
    else:
        axs[0].set_ylim(yrange)

    if xrange is not None:
        axs[0].set_xlim(xrange)
    else:
        axs[0].set_xlim(bins_range)

    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    plt.xlabel(
        r"${%s}$" % xlabel,
        fontsize=FONTSIZE,
    )

    axs[1].set_ylabel(r"$\frac{\text{Model}}{\text{Truth}}$", fontsize=FONTSIZE)
    axs[1].set_yticks(error_ticks)
    if error_range is None:
        error_range = [
            1.0 - (1.0 - error_ticks[0]) * 2,
            1.0 + (error_ticks[2] - 1.0) * 2,
        ]
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=error_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=error_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=error_ticks[2], c="black", ls="dotted", lw=0.5)

    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    if isinstance(file, PdfPages):
        file.savefig(fig, bbox_inches="tight")
    elif file is not None:
        with PdfPages(file) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_2d_histogram(
    file,
    x1,
    y1,
    x2,
    y2,
    x1_label,
    y1_label,
    range,
    first_label="Truth",
    second_label="Model",
    x2_label=None,
    y2_label=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    if isinstance(range, tuple):
        xrange = range[0]
        yrange = range[1]
    else:
        xrange = range
        yrange = range
    if xrange.dtype in [torch.int32, torch.int64]:
        step = (xrange[1] - xrange[0]) // 60 + 1
        xbins = np.arange(xrange[0], xrange[1] + 1, step)
    else:
        xbins = np.linspace(xrange[0], xrange[1], 50)
    if yrange.dtype in [torch.int32, torch.int64]:
        step = (yrange[1] - yrange[0]) // 60 + 1
        ybins = np.arange(yrange[0], yrange[1] + 1, step)
    else:
        ybins = np.linspace(yrange[0], yrange[1], 50)
    bins = (
        xbins,
        ybins,
    )
    hist = ax1.hist2d(x1, y1, bins=bins, norm=mcolors.LogNorm(), rasterized=True)
    vmin, vmax = hist[-1].get_clim()
    ax1.set_xlabel(
        r"${%s}$" % x1_label,
        fontsize=FONTSIZE,
    )
    ax1.set_ylabel(
        r"${%s}$" % y1_label,
        fontsize=FONTSIZE,
    )
    ax1.set_title(first_label, fontsize=FONTSIZE)
    ax1.grid(False)

    ax2.hist2d(
        x2, y2, bins=bins, norm=mcolors.LogNorm(vmax=vmax, vmin=vmin), rasterized=True
    )
    ax2.set_xlabel(
        r"${%s}$" % (x2_label if x2_label is not None else x1_label),
        fontsize=FONTSIZE,
    )
    ax2.set_ylabel(
        r"${%s}$" % (y2_label if y2_label is not None else y1_label),
        fontsize=FONTSIZE,
    )
    ax2.set_title(second_label, fontsize=FONTSIZE)
    ax2.grid(False)

    if file is None:
        plt.show()
    else:
        plt.savefig(file, bbox_inches="tight", format="pdf")
        plt.close()
