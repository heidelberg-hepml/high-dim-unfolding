import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"
)

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, title, plot_dict):
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        plot_loss(
            file,
            [plot_dict["train_loss"], plot_dict["val_loss"]],
            plot_dict["train_lr"],
            labels=["train loss", "val loss"],
            logy=True,
        )

    if cfg.evaluate:
        file = f"{plot_path}/params_histograms.pdf"
        plot_param_histograms(
            file,
            [
                plot_dict["results_train"],
                plot_dict["results_test"],
                plot_dict["results_val"],
            ],
            ["train", "test", "val"],
        )


def plot_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
):
    hists = []
    for dat in data:
        hist, bins = np.histogram(dat, bins=bins, range=xrange)
        hists.append(hist)
    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
    for i, hist, scale, label, color in zip(
        range(len(hists)), hists, scales, labels, colors
    ):
        axs[0].step(
            bins,
            dup_last(hist) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        if i == 0:
            axs[0].fill_between(
                bins,
                dup_last(hist) * scale,
                0.0 * dup_last(hist),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = np.divide(
            hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
        )  # sets denominator=0 terms to 0
        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)

    if logx:
        axs[0].set_xscale("log")

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[0].text(
        0.04,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_yticks(ratio_ticks)
    axs[1].set_ylim(ratio_range)
    axs[1].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_param_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
):
    params_labels = ["k", "theta", "coeff"]
    hists = []
    for params in data:
        hists_mix = []
        for mixture_component in range(params.shape[1]):
            hist_k, _ = np.histogram(
                params[:, mixture_component, 0], bins=bins, range=xrange
            )
            hist_theta, _ = np.histogram(
                params[:, mixture_component, 1], bins=bins, range=xrange
            )
            hist_coeff, _ = np.histogram(
                params[:, mixture_component, 2], bins=bins, range=xrange
            )
            hists_mix.append([hist_k, hist_theta, hist_coeff])
        hists.append(hists_mix)

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    for i, hists_mix in enumerate(hists):
        for i_mix, hists_param in enumerate(hists_mix):
            for j, hist in enumerate(hists_param):
                axs[j, i].step(
                    bins, hist, linewidth=1.0, where="post", color=colors[i_mix]
                )
            axs[j, i].set_xlabel(f"dataset {labels[i]}, component {params_labels[j]}")

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()
