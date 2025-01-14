import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np


class smoothCELoss(nn.Module):
    def __init__(self, max_num_particles, spread=0.5):
        super().__init__()
        self.spread = spread
        self.max_num_particles = max_num_particles

    def forward(self, dist, targets):
        # plot_dists(dist, len(targets))
        logprobs = dist.log_prob(
            torch.arange(1, self.max_num_particles + 1, device=targets.device)
            .unsqueeze(1)
            .repeat(1, len(targets))
        ).transpose(0, 1)
        weights = torch.zeros_like(logprobs, device=targets.device)
        for i in range(len(targets)):
            weights[i, targets[i] - 1] = 1.0
        kernel = (
            torch.distributions.Normal(0, self.spread)
            .log_prob(torch.arange(-2, 2 + 1e-5, device=targets.device))
            .exp()
        )
        kernel /= kernel.sum()
        weights = torch.nn.functional.conv1d(
            weights.unsqueeze(1), kernel.view(1, 1, -1), bias=None, groups=1, padding=2
        ).squeeze(1)
        return torch.mean(torch.sum(-logprobs * weights, dim=-1))


def plot_dists(mixture, N_MIX):
    x = torch.linspace(0, 10, 200).view(-1, 1).repeat(1, N_MIX)
    pdfs = mixture.log_prob(x).exp().detach().numpy()
    cdfs = mixture.cdf(x).detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    for i in range(5):
        ax1.plot(x[:, i], pdfs[:, i], label=f"mixture {i}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Gamma Distribution PDFs")
    # ax1.legend()
    ax1.grid(True)

    for i in range(5):
        ax2.plot(x[:, i], cdfs[:, i], label=f"mixture {i}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Gamma Distribution CDFs")
    # ax2.legend()
    ax2.grid(True)

    fig.savefig("plot_dists.pdf", format="pdf", bbox_inches="tight")
