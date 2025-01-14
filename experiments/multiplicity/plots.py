import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss
from experiments.multiplicity.distributions import GammaMixture

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
                plot_dict["results_train"]["params"].numpy(),
                plot_dict["results_val"]["params"].numpy(),
                plot_dict["results_test"]["params"].numpy(),
            ],
            ["train", "val", "test"],
        )

        file = f"{plot_path}/multiplicity_histograms.pdf"
        plot_mult_histograms(
            file,
            [
                plot_dict["results_train"]["samples"].numpy(),
                plot_dict["results_val"]["samples"].numpy(),
                plot_dict["results_test"]["samples"].numpy(),
            ],
            ["train", "val", "test"],
        )

        file = f"{plot_path}/distributions.pdf"
        plot_distributions(
            file,
            [
                (plot_dict["results_train"]["params"][:cfg.plotting.n_distributions],plot_dict["results_train"]["samples"][:cfg.plotting.n_distributions,1].numpy()),
                (plot_dict["results_val"]["params"][:cfg.plotting.n_distributions],plot_dict["results_val"]["samples"][:cfg.plotting.n_distributions,1].numpy()),
                (plot_dict["results_test"]["params"][:cfg.plotting.n_distributions],plot_dict["results_test"]["samples"][:cfg.plotting.n_distributions,1].numpy()),
            ],
            ["train", "val", "test"],
            title=title,
            x_max=cfg.data.max_num_particles,
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
    bins_list = []
    fill = True
    for params in data:
        hists_mix = []
        for mixture_component in range(params.shape[1]):
            hist_k, bins_k = np.histogram(
                params[:, mixture_component, 0], bins=bins, range=xrange
            )
            hist_theta, bins_theta = np.histogram(
                params[:, mixture_component, 1], bins=bins, range=xrange
            )
            hist_coeff, bins_coeff = np.histogram(
                params[:, mixture_component, 2], bins=bins, range=xrange
            )
            hists_mix.append([hist_k, hist_theta, hist_coeff])
            if fill:
                bins_list.append(bins_k)
                bins_list.append(bins_theta)
                bins_list.append(bins_coeff)
                fill = False
        hists.append(hists_mix)

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    dup_last = lambda a: np.append(a, a[-1])

    for i, hists_mix in enumerate(hists):
        for i_mix, hists_param in enumerate(hists_mix):
            for j, hist in enumerate(hists_param):
                axs[j, i].step(
                    bins_list[j], dup_last(hist), label=f"{i_mix+1}", linewidth=1.0, where="post", color=colors[i_mix]
                )
                axs[j, i].legend(
                    loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND
                )
                axs[j, i].set_title(f"dataset {labels[i]}, component {params_labels[j]}", fontsize=FONTSIZE)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

def plot_mult_histograms(
    file,
    data,
    labels,
):
    bins_max = int(max([np.max(dat).item() for dat in data]))
    bins = np.arange(1, bins_max+1)
    hists = []
    for dat in data:
        hist_sample, _ = np.histogram(dat[:, 0], bins=bins)
        hist_target, _ = np.histogram(dat[:, 1], bins=bins)
        hists.append([hist_sample/dat.shape[0], hist_target/dat.shape[0]])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    dup_last = lambda a: np.append(a, a[-1])
    for i, label in enumerate(labels):
        axs[i].step(bins, dup_last(hists[i][0]), label='sample', linewidth=1.0, where="post")
        axs[i].step(bins, dup_last(hists[i][1]), label='target', linewidth=1.0, where="post")
        axs[i].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
        axs[i].set_title(f'dataset {label}', fontsize=FONTSIZE)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

def plot_distributions(
    file, data, labels, title, x_max
):
    x = torch.linspace(0, x_max, 1000).reshape(-1, 1).repeat(1, len(data[0][0]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, dat, label in zip(range(3), data, labels):
        params, target = dat
        dist = GammaMixture(params)
        pdf = dist.log_prob(x).exp().detach().numpy()
        for j in range(len(params)):
            axs[i].plot(x[:,j], pdf[:,j])
        axs[i].set_title(f'dataset {label}', fontsize=FONTSIZE)
    
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()



