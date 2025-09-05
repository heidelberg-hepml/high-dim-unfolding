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
        assert torch.isfinite(a_in).all()
        assert torch.isfinite(v_in).all()
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


class StandardFourmomenta(BaseCoordinates):
    # (E, px, py, pz)
    def __init__(self, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.transforms = [
            tr.StandardNormal(fixed_dims=fixed_dims, scaling=scaling),
        ]


class PPPM2(BaseCoordinates):
    def __init__(self, **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_ClampedM2(),
        ]


class StandardPPPM2(BaseCoordinates):
    # fitted (px, py, pz, m^2)
    def __init__(self, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_ClampedM2(),
            tr.StandardNormal(fixed_dims=fixed_dims, scaling=scaling),
        ]


class EPhiPtPz(BaseCoordinates):
    # (E, phi, pt, pz)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [
            tr.EPPP_to_EPhiPtPz(),
            tr.Pt_to_ClampedPt(pt_min, pt_pos=2),
        ]


class StandardEPhiPtPz(BaseCoordinates):
    # (E, phi, pt, pz)
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [
            tr.EPPP_to_EPhiPtPz(),
            tr.Pt_to_ClampedPt(pt_min, pt_pos=2),
            tr.StandardNormal(
                fixed_dims=fixed_dims, scaling=scaling, contains_uniform_phi=True
            ),
        ]


class PtPhiEtaE(BaseCoordinates):
    # (pt, phi, eta, E)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.Pt_to_ClampedPt(pt_min),
        ]


class PtPhiEtaM2(BaseCoordinates):
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
        ]


class StandardPtPhiEtaM2(BaseCoordinates):
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.StandardNormal(
                fixed_dims=fixed_dims, scaling=scaling, contains_uniform_phi=True
            ),
        ]


class StandardJetScaledPtPhiEtaM2(BaseCoordinates):
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
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
            tr.M2_to_ClampedM2(),
            tr.M2_to_LogM2(),
        ]


class StandardPPPLogM2(BaseCoordinates):
    # fitted (px, py, pz, log(m^2))
    def __init__(self, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_ClampedM2(),
            tr.M2_to_LogM2(),
            tr.StandardNormal(fixed_dims=fixed_dims, scaling=scaling),
        ]


class LogPtPhiEtaE(BaseCoordinates):
    # (log(pt), phi, eta, E)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
        ]


class PtPhiEtaLogM2(BaseCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.M2_to_LogM2(),
        ]


class StandardPtPhiEtaLogM2(BaseCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.M2_to_LogM2(),
            tr.StandardNormal(
                fixed_dims=fixed_dims, scaling=scaling, contains_uniform_phi=True
            ),
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
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
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
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.M2_to_LogM2(),
        ]


class StandardLogPtPhiEtaLogM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2))
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.M2_to_LogM2(),
            tr.StandardNormal(fixed_dims, scaling, contains_uniform_phi=True),
        ]


class StandardAsinhPtPhiEtaLogM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2))
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_AsinhPt(),
            tr.M2_to_LogM2(),
            tr.StandardNormal(fixed_dims, scaling, contains_uniform_phi=True),
        ]


class StandardLogPtPhiEtaAsinhM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2))
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.M2_to_AsinhM2(),
            tr.StandardNormal(fixed_dims, scaling, contains_uniform_phi=True),
        ]


class StandardLogPtPhiEtaM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.StandardNormal(fixed_dims, scaling, contains_uniform_phi=True),
        ]


class IndividualStandardLogPtPhiEtaLogM2(BaseCoordinates):
    # Position fitted (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.M2_to_LogM2(),
            tr.IndividualNormal(fixed_dims, scaling),
        ]


class JetScaledPtPhiEtaM2(BaseCoordinates):
    # (pt, phi-phi_jet, eta-eta_jet, m^2)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.PtPhiEtaM2_to_JetScale(),
        ]


class JetScaledLogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt)-log(pt_jet), phi-phi_jet, eta-eta_jet, log(m^2) - log(m^2_jet)
    def __init__(self, pt_min, **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.M2_to_LogM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
        ]


class StandardJetScaledLogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt)-log(pt_jet), phi-phi_jet, eta-eta_jet, log(m^2) - log(m^2_jet)
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.M2_to_LogM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
            tr.StandardNormal(fixed_dims, scaling),
        ]


class StandardJetScaledLogPtPhiEtaM2(BaseCoordinates):
    # (log(pt), phi-phi_jet, eta-eta_jet, m^2
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
            tr.StandardNormal(fixed_dims, scaling),
        ]


class IndividualStandardJetScaledLogPtPhiEtaLogM2(BaseCoordinates):
    # (pt/pt_jet, phi-phi_jet, eta-eta_jet, log(m^2))
    def __init__(self, pt_min, fixed_dims=[], scaling=torch.ones(1, 4), **kwargs):
        super().__init__()
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_ClampedM2(),
            tr.M2_to_LogM2(),
            tr.Pt_to_ClampedPt(pt_min),
            tr.Pt_to_LogPt(),
            tr.LogPtPhiEtaLogM2_to_JetScale(),
            tr.IndividualNormal(fixed_dims, scaling),
        ]


ptphietam2 = PtPhiEtaM2(pt_min=0.0)


def fourmomenta_to_jetmomenta(fourmomenta):
    """
    Convert fourmomenta (E, px, py, pz) to jet momenta (pt, phi, eta, m^2).
    """
    in_dtype = fourmomenta.dtype
    output = fourmomenta.to(dtype=DTYPE)
    return ptphietam2.fourmomenta_to_x(output).to(in_dtype)


def jetmomenta_to_fourmomenta(jetmomenta):
    """
    Convert jet momenta (pt, phi, eta, m^2) to fourmomenta (E, px, py, pz).
    """
    in_dtype = jetmomenta.dtype
    output = jetmomenta.to(dtype=DTYPE)
    return ptphietam2.x_to_fourmomenta(output).to(in_dtype)
