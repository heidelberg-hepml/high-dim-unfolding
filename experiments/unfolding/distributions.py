import torch
import math

import experiments.unfolding.coordinates as c


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
        onshell_mass,
        pt_min,
        units,
    ):
        super().__init__()
        self.onshell_mass = torch.tensor(onshell_mass)
        self.pt_min = pt_min
        self.units = units

    def sample(self, shape, device, dtype, generator=None):
        fourmomenta = self.propose(
            shape, device=device, dtype=c.DTYPE, generator=generator
        )
        onshell_mass = (
            self.onshell_mass.to(device, dtype=c.DTYPE).expand(fourmomenta.shape[:-1])
            / self.units
        )
        fourmomenta[..., 0] = torch.sqrt(
            onshell_mass**2 + torch.sum(fourmomenta[..., 1:] ** 2, dim=-1)
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


def log_prob_normal(z, mean=0.0, std=1.0):
    std_term = torch.log(std) if type(std) == torch.Tensor else math.log(std)
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std_term
