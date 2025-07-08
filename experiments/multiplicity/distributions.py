import torch
import torch.distributions as D
import einops


def cross_entropy(distribution, target):
    assert target.shape == distribution.batch_shape
    return -distribution.log_prob(target)


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


def process_params(params):
    params = torch.clamp(params, min=-10, max=5)  # avoid inf and 0
    params = torch.exp(params)  # ensure positive params
    return params
