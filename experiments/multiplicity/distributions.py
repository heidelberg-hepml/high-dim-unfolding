import torch
import torch.distributions as D
import einops


def cross_entropy(distribution, target):
    assert target.shape == distribution.batch_shape
    return -distribution.log_prob(target)


def smooth_cross_entropy(distribution, target, max_num_particles, smoothness=1.0):
    assert target.shape == distribution.batch_shape
    bins = (
        torch.arange(1, max_num_particles + 1, device=target.device)
        .unsqueeze(1)
        .repeat(1, len(target))
    )
    logprobs = distribution.log_prob(bins).transpose(0, 1)

    weights = torch.zeros_like(logprobs, device=target.device)
    for i in range(len(target)):
        weights[i, target[i] - 1] = 1.0

    kernel_width = torch.ceil(3 * smoothness)
    kernel = (
        torch.distributions.Normal(0, smoothness)
        .log_prob(
            torch.arange(
                -kernel_width // 2, kernel_width // 2 + 1, device=target.device
            )
        )
        .exp()
    )
    kernel /= kernel.sum()
    weights = torch.nn.functional.conv1d(
        weights.unsqueeze(1),  # add C=1 to ensure shape N,C,L
        kernel.view(1, 1, -1),
        bias=None,
        padding=kernel_width // 2,
    ).squeeze(1)
    return torch.sum(-logprobs * weights, dim=-1)


class GammaMixture(D.MixtureSameFamily):
    def __init__(self, params):
        if len(params.shape) == 2:
            params = einops.rearrange(
                params, "... (n_mix n_params) -> ... n_mix n_params", n_params=3
            )
        self.params = params
        mix = D.Categorical(params[..., 2])
        gammas = D.Gamma(params[..., 0], params[..., 1])
        super().__init__(mix, gammas)

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        return torch.round(samples)


class CategoricalDistribution(D.Categorical):
    def __init__(self, logits):
        super().__init__(logits=logits)
        self.params = logits


class GaussianMixture(D.MixtureSameFamily):
    def __init__(self, params):
        if len(params.shape) == 2:
            params = einops.rearrange(
                params, "... (n_mix n_params) -> ... n_mix n_params", n_params=3
            )
        self.params = params
        mix = D.Categorical(params[..., 2])
        gammas = D.Normal(params[..., 0], params[..., 1])
        super().__init__(mix, gammas)

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        return torch.round(samples)
