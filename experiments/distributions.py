import torch
import torch.distributions as D
import math
import einops

import experiments.coordinates as c


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


class CategoricalDistribution(D.Categorical):
    def __init__(self, logits):
        # logits = logits/logits.sum(-1, keepdim=True)
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


class BaseDistribution(torch.nn.Module):
    """
    Abstract base distribution
    All child classes work in fourmomenta space,
    i.e. they generate fourmomenta and return log_prob in fourmomenta space
    """

    def sample(self, shape, device, dtype, generator=None, **kwargs):
        raise NotImplementedError

    def log_prob(self, x, **kwargs):
        raise NotImplementedError


class OnShellDistribution(BaseDistribution):
    """
    Implement fixed mass sampling
    """

    def __init__(
        self,
        mass,
        pt_min,
        units,
    ):
        super().__init__()
        self.mass = torch.tensor(mass)
        self.pt_min = pt_min
        self.units = units

    def sample(self, shape, device, dtype, generator=None):
        fourmomenta = self.propose(
            shape, device=device, dtype=c.DTYPE, generator=generator
        )
        mass = (
            self.mass.to(device, dtype=c.DTYPE).expand(fourmomenta.shape[:-1])
            / self.units
        )
        fourmomenta[..., 0] = torch.sqrt(
            mass**2 + torch.sum(fourmomenta[..., 1:] ** 2, dim=-1)
        )
        return fourmomenta.to(dtype=dtype)

    def propose(self, shape, device, dtype, generator=None):
        raise NotImplementedError


class NaivePPP(OnShellDistribution):
    """Base distribution 1: 3-momentum from standard normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.PPPLogM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        fourmomenta = self.coordinates.x_to_fourmomenta(eps)
        return fourmomenta

    def log_prob(self, fourmomenta):
        ppplogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ppplogm2)
        log_prob[..., 3] = 0.0  # fixed mass does not contribute
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ppplogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardPPP(OnShellDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.StandardPPPLogM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        fourmomenta = self.coordinates.x_to_fourmomenta(eps)
        return fourmomenta

    def log_prob(self, fourmomenta):
        ppplogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ppplogm2)
        log_prob[..., 3] = 0.0  # fixed mass does not contribute
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ppplogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardLogPtPhiEta(OnShellDistribution):
    """Base distribution 4: phi uniform; eta, log(pt) and log(mass) from fitted normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.StandardLogPtPhiEtaLogM2(self.pt_min, self.units)

    def propose(self, shape, device, dtype, generator=None):
        # sample (logpt, phi, eta)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        # inverse standadization
        eps = self.coordinates.transforms[-1].inverse(eps)

        # sample phi uniformly
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )

        for t in self.coordinates.transforms[:-1][::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        logptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(logptphietalogm2)
        log_prob[..., 1] = -math.log(
            2 * math.pi
        )  # normalization factor for uniform phi distribution: 1/(2 pi)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(logptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardPtPhiEta(OnShellDistribution):
    """Base distribution 4: phi uniform; eta, pt from fitted normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.StandardPtPhiEtaLogM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        eps = self.coordinates.transforms[-1].inverse(eps)
        # sample phi uniformly
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )

        for t in self.coordinates.transforms[:-1][::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        ptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ptphietalogm2)
        log_prob[..., 1] = -math.log(
            2 * math.pi
        )  # normalization factor for uniform phi distribution: 1/(2 pi)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class LogPtPhiEta(OnShellDistribution):
    """Base distribution 4: phi uniform; eta, log(pt) from normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.LogPtPhiEtaLogM2(self.pt_min, self.units)

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        # sample phi uniformly
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )

        for t in self.coordinates.transforms[::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        logptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(logptphietalogm2)
        log_prob[..., 1] = -math.log(
            2 * math.pi
        )  # normalization factor for uniform phi distribution: 1/(2 pi)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(logptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class PtPhiEta(OnShellDistribution):
    """Base distribution 4: phi uniform; eta, pt from normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.PtPhiEtaLogM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        # sample phi uniformly
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )

        for t in self.coordinates.transforms[::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        ptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ptphietalogm2)
        log_prob[..., 1] = -math.log(
            2 * math.pi
        )  # normalization factor for uniform phi distribution: 1/(2 pi)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class JetScaledPtEtaPhi(BaseDistribution):

    def __init__(self, *args, scaling=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling
        self.coordinates = c.JetScaledPtPhiEtaM2()

    def propose(self, shape, device, dtype, generator=None):
        # sample (logpt, phi, eta)
        eps = (
            torch.randn(shape, device=device, dtype=dtype, generator=generator)
            * self.scaling
        )

        for t in self.coordinates.transforms[::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        ptphietam2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ptphietam2)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ptphietam2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardJetScaledLogPtEtaPhi(BaseDistribution):

    def __init__(self, *args, scaling=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling
        self.coordinates = c.StandardJetScaledLogPtPhiEtaLogM2(
            pt_min=self.pt_min, units=self.units, fixed_dims=[3], scaling=scaling
        )

    def propose(self, shape, device, dtype, generator=None):
        # sample (logpt, phi, eta)
        eps = (
            torch.randn(shape, device=device, dtype=dtype, generator=generator)
            * self.scaling
        )

        for t in self.coordinates.transforms[::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        logptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(logptphietalogm2, std=self.scaling)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(logptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


def log_prob_normal(z, mean=0.0, std=1.0):
    std_term = torch.log(std) if type(std) == torch.Tensor else math.log(std)
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std_term
