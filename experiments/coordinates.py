import torch
import experiments.transforms as tr

DTYPE = torch.float64


class BaseCoordinates(torch.nn.Module):
    """
    Class that implements transformations
    from fourmomenta to an abstract set of variables
    Heavily uses functionality from Transforms classes
    """

    def __init__(self):
        super().__init__()
        self.contains_phi = False
        self.contains_mass = False
        self.transforms = []

    def init_fit(self, fourmomenta, mask, **kwargs):
        x = fourmomenta.clone()
        assert torch.isfinite(x).all()
        for transform in self.transforms[:-1]:
            x[mask] = transform.forward(x[mask], **kwargs)
            assert torch.isfinite(
                x
            ).all(), (
                f"Transform {transform.__class__.__name__} produced non-finite values."
            )
        self.transforms[-1].init_fit(x, mask=mask, **kwargs)

    def fourmomenta_to_x(self, a_in, **kwargs):
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        for transform in self.transforms:
            a = transform.forward(a, **kwargs)
        return a.to(dtype=a_in.dtype)

    def x_to_fourmomenta(self, a_in, **kwargs):
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            a = transform.inverse(a, **kwargs)
        return a.to(dtype=a_in.dtype)

    def velocity_fourmomenta_to_x(self, v_in, a_in, **kwargs):
        assert torch.isfinite(a_in).all() and torch.isfinite(v_in).all()
        v, a = v_in.to(dtype=DTYPE), a_in.to(dtype=DTYPE)
        for transform in self.transforms:
            b = transform.forward(a, **kwargs)
            v = transform.velocity_forward(v, a, b, **kwargs)
            a = b
        return v.to(dtype=v_in.dtype), a.to(dtype=a_in.dtype)

    def velocity_x_to_fourmomenta(self, v_in, a_in, **kwargs):
        assert torch.isfinite(a_in).all() and torch.isfinite(v_in).all()
        v, a = v_in.to(dtype=DTYPE), a_in.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            b = transform.inverse(a, **kwargs)
            v = transform.velocity_inverse(v, a, b, **kwargs)
            a = b
        return v.to(dtype=v_in.dtype), a.to(dtype=a_in.dtype)

    def logdetjac_fourmomenta_to_x(self, a_in, **kwargs):
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        b = self.transforms[0].forward(a, **kwargs)
        logdetjac = -self.transforms[0].logdetjac_forward(a, b, **kwargs)
        a = b
        for transform in self.transforms[1:]:
            b = transform.forward(a, **kwargs)
            logdetjac -= transform.logdetjac_forward(a, b, **kwargs)
            a = b
        return logdetjac.to(dtype=a_in.dtype), a.to(dtype=a_in.dtype)

    def logdetjac_x_to_fourmomenta(self, a_in, **kwargs):
        # logdetjac = log|da/db| = -log|db/da| with a=x, b=fourmomenta
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        b = self.transforms[-1].inverse(a, **kwargs)
        logdetjac = -self.transforms[-1].logdetjac_inverse(a, b, **kwargs)
        a = b
        for transform in self.transforms[::-1][1:]:
            b = transform.inverse(a, **kwargs)
            logdetjac -= transform.logdetjac_inverse(a, b, **kwargs)
            a = b
        return logdetjac.to(dtype=a_in.dtype), a.to(dtype=a_in.dtype)


class Fourmomenta(BaseCoordinates):
    # (E, px, py, pz)
    # this class effectively does nothing,
    # because fourmomenta are already the baseline representation
    def __init__(self, **kwargs):
        super().__init__()
        self.transforms = [tr.EmptyTransform()]


class PPPM2(BaseCoordinates):
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [tr.EPPP_to_PPPM2()]


class StandardPPPM2(BaseCoordinates):
    # fitted (px, py, pz, m^2)
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.StandardNormal([3]),
        ]


class EPhiPtPz(BaseCoordinates):
    # (E, phi, pt, pz)
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_EPhiPtPz()]


class PtPhiEtaE(BaseCoordinates):
    # (pt, phi, eta, E)
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_PtPhiEtaE()]


class PtPhiEtaM2(BaseCoordinates):
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
        ]


class StandardPtPhiEtaM2(BaseCoordinates):
    def __init__(self, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.StandardNormal(fixed_dims=fixed_dims, scaling=scaling),
        ]


class StandardJetScaledPtPhiEtaM2(BaseCoordinates):
    def __init__(self, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.PtPhiEtaM2_to_JetScale(),
            tr.StandardNormal(fixed_dims=fixed_dims, scaling=scaling),
        ]


class PPPLogM2(BaseCoordinates):
    # (px, py, pz, log(m^2))
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
        ]


class StandardPPPLogM2(BaseCoordinates):
    # fitted (px, py, pz, log(m^2))
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
            tr.StandardNormal([3]),
        ]


class LogPtPhiEtaE(BaseCoordinates):
    # (log(pt), phi, eta, E)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_PtPhiEtaE(), tr.Pt_to_LogPt(pt_min)]


class PtPhiEtaLogM2(BaseCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
        ]


class StandardPtPhiEtaLogM2(BaseCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
            tr.StandardNormal(fixed_dims=fixed_dims, scaling=scaling),
        ]


class LogPtPhiEtaM2(BaseCoordinates):
    # (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min),
        ]


class LogPtPhiEtaM2(BaseCoordinates):
    # (log(pt), phi, eta, log(m^2))
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min),
        ]


class LogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.M2_to_LogM2(),
        ]


class StandardLogPtPhiEtaLogM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2))
    def __init__(self, pt_min, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.M2_to_LogM2(),
            tr.StandardNormal([1] + fixed_dims, scaling),
        ]


class StandardLogPtPhiEtaM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.StandardNormal([1] + fixed_dims, scaling),
        ]


class IndividualStandardLogPtPhiEtaLogM2(BaseCoordinates):
    # Position fitted (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.M2_to_LogM2(),
            tr.IndividualNormal([1] + fixed_dims, scaling),
        ]


class JetScaledPtPhiEtaM2(BaseCoordinates):
    # (pt/pt_jet, phi-phi_jet, eta-eta_jet, m^2)
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_phi = False
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.PtPhiEtaM2_to_JetScale(),
        ]


class JetScaledLogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt)-log(pt_jet), phi-phi_jet, eta-eta_jet, log(m^2) - log(m^2_jet)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = False
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
        ]


class StandardJetScaledLogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt)-log(pt_jet), phi-phi_jet, eta-eta_jet, log(m^2) - log(m^2_jet)
    def __init__(self, pt_min, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = False
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
            tr.StandardNormal(fixed_dims, scaling),
        ]


class IndividualStandardJetScaledLogPtPhiEtaLogM2(BaseCoordinates):
    # (pt/pt_jet, phi-phi_jet, eta-eta_jet, log(m^2))
    def __init__(self, pt_min, fixed_dims=[3], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = False
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
            tr.Pt_to_LogPt(pt_min),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
            tr.IndividualNormal(fixed_dims, scaling),
        ]


ptphietam2 = PtPhiEtaM2()


def fourmomenta_to_jetmomenta(fourmomenta):
    """
    Convert four-momenta (E, px, py, pz) to jet momenta (pt, phi, eta, m^2).
    """
    return ptphietam2.fourmomenta_to_x(fourmomenta)


def jetmomenta_to_fourmomenta(jetmomenta):
    """
    Convert jet momenta (pt, phi, eta, m^2) to four-momenta (E, px, py, pz).
    """
    return ptphietam2.x_to_fourmomenta(jetmomenta)
