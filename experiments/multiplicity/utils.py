import torch
import torch.nn as nn


class smoothCELoss(nn.Module):
    def __init__(self, max_num_particles, spread=0.5):
        super().__init__()
        self.spread = spread
        self.max_num_particles = max_num_particles

    def forward(self, dist, targets):
        logprobs = dist.log_prob(
            torch.arange(self.max_num_particles).unsqueeze(0).repeat(len(targets), 1)
        )
        print("loss", logprobs.shape, targets.shape)
        weights = torch.zeros_like(logprobs)
        for i in range(len(targets)):
            weights[i, targets[i]] = 1.0
        kernel = (
            torch.distributions.Normal(0, self.spread)
            .log_prob(torch.arange(-2, 2 + 1e-5))
            .exp()
        )
        kernel /= kernel.sum()
        weights = torch.nn.functional.conv1d(
            weights.unsqueeze(1), kernel.view(1, 1, -1), bias=None, groups=1, padding=2
        ).squeeze(1)
        return torch.sum(-logprobs * weights, dim=-1)
