import torch
import torch.distributions as D
import einops
import inspect

from experiments.utils import EPS2


def cross_entropy(distribution, target):
    assert (
        target.shape == distribution.batch_shape
    ), f"Target shape {target.shape} does not match distribution batch shape {distribution.batch_shape}"
    return -distribution.log_prob(target).mean()


class IntegerMixture(D.MixtureSameFamily):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_prob(self, x):
        prob_mass = self.cdf(x + 0.5) - self.cdf(x - 0.5)
        return torch.log(prob_mass + EPS2)

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        return torch.round(samples)


class GammaMixture(IntegerMixture):
    def __init__(self, params):
        self.params = params
        mix = D.Categorical(probs=params[..., 2])
        gammas = D.Gamma(params[..., 0], params[..., 1])
        super().__init__(mix, gammas)

    # workaround as Gamma.cdf has no autograd yet
    def log_prob(self, x):
        return D.MixtureSameFamily.log_prob(self, x)


class GaussianMixture(IntegerMixture):
    def __init__(self, params):
        if len(params.shape) == 2:
            params = einops.rearrange(
                params, "... (n_mix n_params) -> ... n_mix n_params", n_params=3
            )
        self.params = params
        mix = D.Categorical(probs=params[..., 2])
        gaussians = D.Normal(params[..., 0], params[..., 1])
        super().__init__(mix, gaussians)


class RangedCategorical(D.Categorical):
    def __init__(self, low, high, probs):
        self.low = low
        self.high = high
        assert probs.shape[-1] == (
            high - low + 1
        ), f"Expected probs shape {probs.shape[-1]} to match range {low} to {high}"
        super().__init__(probs=probs)

    def sample(self, *args, **kwargs):
        idx = super().sample(*args, **kwargs)
        return self.low + idx

    def log_prob(self, value):
        idx = value - self.low
        return super().log_prob(idx)


def ranged_categorical(low, high):
    return lambda probs: RangedCategorical(low, high, probs)


def process_params(params, dist):
    if inspect.isfunction(dist):  # ranged_categorical lambda
        params = torch.clamp(params, min=-10, max=5)
        params = torch.exp(params)
        return params
    if issubclass(dist, IntegerMixture) and len(params.shape) == 2:
        params = einops.rearrange(
            params, "... (n_mix n_params) -> ... n_mix n_params", n_params=3
        )
    if dist is GaussianMixture:  # keep mean possibly negative
        mean = params[..., 0:1]
        rest = params[..., 1:]
        rest = torch.clamp(rest, min=-10, max=5)
        rest = torch.exp(rest)
        params = torch.cat([mean, rest], dim=-1)
    else:
        params = torch.clamp(params, min=-10, max=5)
        params = torch.exp(params)

    return params
