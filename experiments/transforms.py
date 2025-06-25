import torch
from torch import nn

from experiments.utils import (
    unpack_last,
    EPS1,
    EPS2,
    CUTOFF,
    get_pt,
    get_phi,
    get_eta,
    ensure_angle,
)


class BaseTransform(nn.Module):
    """
    Abstract class for transformations between two coordinate systems
    For CFM, we need forward and inverse transformations,
    the corresponding jacobians for the RCFM aspect
    and log(det(jacobian)) when we want to extract probabilities from the CFM
    """

    def forward(self, x, **kwargs):
        y = self._forward(x, **kwargs)
        assert torch.isfinite(y).all(), self.__class__.__name__
        return y

    def inverse(self, x, **kwargs):
        y = self._inverse(x, **kwargs)
        assert torch.isfinite(y).all(), self.__class__.__name__
        return y

    def velocity_forward(self, v_x, x, y, **kwargs):
        # v_y = dy/dx * v_x
        jac = self._jac_forward(x, y, **kwargs)
        v_y = torch.einsum("...ij,...j->...i", jac, v_x)
        assert torch.isfinite(v_y).all(), self.__class__.__name__
        return v_y

    def velocity_inverse(self, v_y, y, x, **kwargs):
        # v_x = dx/dy * v_y
        jac = self._jac_inverse(y, x, **kwargs)
        v_x = torch.einsum("...ij,...j->...i", jac, v_y)
        assert torch.isfinite(v_x).all(), self.__class__.__name__
        return v_x

    def logdetjac_forward(self, x, y, **kwargs):
        # log(det(J))
        # J = dy/dx
        logdetjac = torch.log(self._detjac_forward(x, y, **kwargs).abs() + EPS2).sum(
            dim=-1, keepdims=True
        )
        assert torch.isfinite(logdetjac).all(), self.__class__.__name__
        return logdetjac

    def logdetjac_inverse(self, y, x, **kwargs):
        # log(det(J^-1)) = log(1/det(J)) = -log(det(J))
        # J = dy/dx
        logdetjac = -torch.log(self._detjac_forward(x, y, **kwargs).abs() + EPS2).sum(
            dim=-1, keepdims=True
        )
        assert torch.isfinite(logdetjac).all(), self.__class__.__name__
        return logdetjac

    def _forward(self, x, **kwargs):
        raise NotImplementedError

    def _inverse(self, x, **kwargs):
        raise NotImplementedError

    def _jac_forward(self, x, y, **kwargs):
        raise NotImplementedError

    def _jac_inverse(self, y, x, **kwargs):
        raise NotImplementedError

    def _detjac_forward(self, x, y, **kwargs):
        raise NotImplementedError

    def init_fit(self, xs, **kwargs):
        # currently only needed for StandardNormal()
        # default: do nothing
        pass

    def init_unit(self, xs, **kwargs):
        # for debugging and tests
        pass


class EmptyTransform(BaseTransform):
    # empty transform
    # needed for formal reasons
    def forward(self, x, **kwargs):
        return x

    def inverse(self, x, **kwargs):
        return x

    def velocity_forward(self, v, x, y, **kwargs):
        return v

    def velocity_inverse(self, v, y, x, **kwargs):
        return v

    def logdetjac_forward(self, x, y, **kwargs):
        return torch.zeros_like(x[..., 0])

    def logdetjac_inverse(self, x, y, **kwargs):
        return torch.zeros_like(x[..., 0])


class EPPP_to_PPPM2(BaseTransform):
    def _forward(self, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        m2 = E**2 - (px**2 + py**2 + pz**2)
        m2 = torch.abs(m2)

        return torch.stack((px, py, pz, m2), dim=-1)

    def _inverse(self, pppm2, **kwargs):
        px, py, pz, m2 = unpack_last(pppm2)
        m2 = torch.abs(m2)
        E = torch.sqrt(m2 + (px**2 + py**2 + pz**2))
        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, pppm2, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        # jac_ij = dpppm2_i / deppp_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((zero, zero, zero, 2 * E), dim=-1)
        jac_px = torch.stack((one, zero, zero, -2 * px), dim=-1)
        jac_py = torch.stack((zero, one, zero, -2 * py), dim=-1)
        jac_pz = torch.stack((zero, zero, one, -2 * pz), dim=-1)
        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, pppm2, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        # jac_ij = deppp_i / dpppm2_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_px = torch.stack((px / E, one, zero, zero), dim=-1)
        jac_py = torch.stack((py / E, zero, one, zero), dim=-1)
        jac_pz = torch.stack((pz / E, zero, zero, one), dim=-1)
        jac_m2 = torch.stack((1 / (2 * E), zero, zero, zero), dim=-1)
        return torch.stack((jac_px, jac_py, jac_pz, jac_m2), dim=-1)

    def _detjac_forward(self, eppp, pppm2, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        return 2 * E


class EPPP_to_EPhiPtPz(BaseTransform):
    def _forward(self, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        pt = get_pt(eppp)
        phi = get_phi(eppp)
        return torch.stack((E, phi, pt, pz), dim=-1)

    def _inverse(self, ephiptpz, **kwargs):
        E, phi, pt, pz = unpack_last(ephiptpz)

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, ephiptpz, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        E, phi, pt, pz = unpack_last(ephiptpz)

        # jac_ij = dephiptpz_i / dfourmomenta_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((one, zero, zero, zero), dim=-1)
        jac_px = torch.stack(
            (zero, -torch.sin(phi) / pt, torch.cos(phi), zero),
            dim=-1,
        )
        jac_py = torch.stack(
            (zero, torch.cos(phi) / pt, torch.sin(phi), zero),
            dim=-1,
        )
        jac_pz = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, ephiptpz, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        E, phi, pt, pz = unpack_last(ephiptpz)

        # jac_ij = dfourmomenta_i / dephiptpz_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((one, zero, zero, zero), dim=-1)
        jac_phi = torch.stack(
            (zero, -pt * torch.sin(phi), pt * torch.cos(phi), zero), dim=-1
        )
        jac_pt = torch.stack((zero, torch.cos(phi), torch.sin(phi), zero), dim=-1)
        jac_pz = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_E, jac_phi, jac_pt, jac_pz), dim=-1)

    def _detjac_forward(self, eppp, ephiptpz, **kwargs):
        E, phi, pt, pz = unpack_last(ephiptpz)

        # det (dephiptpz / dfourmomenta)
        return 1 / pt


class EPPP_to_PtPhiEtaE(BaseTransform):
    def _forward(self, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        pt = get_pt(eppp)
        phi = get_phi(eppp)
        eta = get_eta(eppp)
        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        assert torch.isfinite(eta).all()

        return torch.stack((pt, phi, eta, E), dim=-1)

    def _inverse(self, ptphietae, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)

        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, ptphietae, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # jac_ij = dptphietae_i / dfourmomenta_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((zero, zero, zero, one), dim=-1)
        jac_px = torch.stack(
            (
                torch.cos(phi),
                -torch.sin(phi) / pt,
                -torch.cos(phi) * torch.tanh(eta) / pt,
                zero,
            ),
            dim=-1,
        )
        jac_py = torch.stack(
            (
                torch.sin(phi),
                torch.cos(phi) / pt,
                -torch.sin(phi) * torch.tanh(eta) / pt,
                zero,
            ),
            dim=-1,
        )
        jac_pz = torch.stack((zero, zero, 1 / (pt * torch.cosh(eta)), zero), dim=-1)

        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, ptphietae, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # jac_ij = dfourmomenta_i / dptphietae_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack(
            (zero, torch.cos(phi), torch.sin(phi), torch.sinh(eta)), dim=-1
        )
        jac_phi = torch.stack(
            (zero, -pt * torch.sin(phi), pt * torch.cos(phi), zero), dim=-1
        )
        jac_eta = torch.stack((zero, zero, zero, pt * torch.cosh(eta)), dim=-1)
        jac_E = torch.stack((one, zero, zero, zero), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_E), dim=-1)

    def _detjac_forward(self, eppp, ptphietae, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # det (dptphietae / dfourmomenta)
        return 1 / (pt**2 * torch.cosh(eta))


class PtPhiEtaE_to_PtPhiEtaM2(BaseTransform):
    def _forward(self, ptphietae, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)

        p_abs = pt * torch.cosh(eta)
        m2 = E**2 - p_abs**2
        m2 = torch.abs(m2)
        return torch.stack((pt, phi, eta, m2), dim=-1)

    def _inverse(self, ptphietam2, **kwargs):
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        m2 = torch.abs(m2)
        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        p_abs = pt * torch.cosh(eta)
        E = torch.sqrt(m2 + p_abs**2)

        return torch.stack((pt, phi, eta, E), dim=-1)

    def _jac_forward(self, ptphietae, ptphietam2, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        # jac_ij = dptphietam2_i / dptphietae_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack((one, zero, zero, -2 * pt * torch.cosh(eta) ** 2), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack((zero, zero, one, -(pt**2) * torch.sinh(2 * eta)), dim=-1)
        jac_E = torch.stack((zero, zero, zero, 2 * E), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_E), dim=-1)

    def _jac_inverse(self, ptphietam2, ptphietae, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        # jac_ij = dptphietae_i / dptphietam2_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack((one, zero, zero, pt * torch.cosh(eta) ** 2 / E), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack(
            (zero, zero, one, pt**2 * torch.sinh(2 * eta) / (2 * E)), dim=-1
        )
        jac_m2 = torch.stack((zero, zero, zero, 1 / (2 * E)), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _detjac_forward(self, ptphietae, ptphietam2, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)
        return 2 * E


class M2_to_LogM2(BaseTransform):
    def _forward(self, xm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)
        m2 = m2.clamp(min=EPS2)
        logm2 = torch.log(m2 + EPS1)
        return torch.stack((x1, x2, x3, logm2), dim=-1)

    def _inverse(self, xlogm2, **kwargs):
        x1, x2, x3, logm2 = unpack_last(xlogm2)
        m2 = logm2.clamp(max=CUTOFF).exp() - EPS1
        return torch.stack((x1, x2, x3, m2), dim=-1)

    def _jac_forward(self, xm2, logxm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)

        # jac_ij = dxlogm2_i / dxm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, 1 / (m2 + EPS1 + EPS2)), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_m2), dim=-1)

    def _jac_inverse(self, logxm2, xm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)

        # jac_ij = dxm2_i / dxlogm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_logm2 = torch.stack((zero, zero, zero, m2 + EPS1), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_logm2), dim=-1)

    def _detjac_forward(self, xm2, logxm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)
        return 1 / (m2 + EPS1 + EPS2)


class Pt_to_LogPt(BaseTransform):
    def __init__(self, pt_min):
        super().__init__()
        self.pt_min = torch.tensor(pt_min)

    def get_dpt(self, pt):
        return torch.clamp(pt - self.pt_min.to(pt.device), min=EPS2)

    def _forward(self, ptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)
        dpt = self.get_dpt(pt)
        logpt = torch.log(dpt + EPS1)
        return torch.stack((logpt, x1, x2, x3), dim=-1)

    def _inverse(self, logptx, **kwargs):
        logpt, x1, x2, x3 = unpack_last(logptx)
        pt = logpt.clamp(max=CUTOFF).exp() + self.pt_min.to(logpt.device) - EPS1
        return torch.stack((pt, x1, x2, x3), dim=-1)

    def _jac_forward(self, ptx, logptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)

        # jac_ij = dlogptx_i / dptx_j
        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        dpt = self.get_dpt(pt)
        jac_pt = torch.stack(
            (
                1 / (dpt + EPS1 + EPS2),
                zero,
                zero,
                zero,
            ),
            dim=-1,
        )
        jac_x1 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, zero, one), dim=-1)
        return torch.stack((jac_pt, jac_x1, jac_x2, jac_x3), dim=-1)

    def _jac_inverse(self, logptx, ptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)

        # jac_ij = dptx_i / dlogptx_j
        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        dpt = self.get_dpt(pt)
        jac_logpt = torch.stack((dpt + EPS1, zero, zero, zero), dim=-1)
        jac_x1 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, zero, one), dim=-1)
        return torch.stack((jac_logpt, jac_x1, jac_x2, jac_x3), dim=-1)

    def _detjac_forward(self, ptx, logptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)
        dpt = self.get_dpt(pt)
        return 1 / (dpt + EPS1 + EPS2)


class StandardNormal(BaseTransform):
    # standardize to unit normal distribution
    # particle- and process-wise mean and std are determined by initial_fit
    # note: this transform will always come last in the self.transforms list of a coordinates class
    def __init__(self, dims_fixed=[], scaling=torch.ones(1, 4)):
        super().__init__()
        self.dims_fixed = dims_fixed
        self.mean = torch.zeros(1, 4)
        self.std = torch.ones(1, 4)
        self.scaling = scaling

    def init_fit(self, x, mask, **kwargs):
        self.mean = torch.mean(x[mask], dim=0, keepdim=True)
        self.std = torch.std(x[mask], dim=0, keepdim=True) / self.scaling.to(
            x.device, dtype=x.dtype
        )
        # self.mean[:, self.dims_fixed] = 0
        self.std[:, self.dims_fixed] = 1

    def _forward(self, x, **kwargs):
        xunit = (x - self.mean.to(x.device, dtype=x.dtype)) / self.std.to(
            x.device, dtype=x.dtype
        )
        return xunit

    def _inverse(self, xunit, **kwargs):
        x = xunit * self.std.to(xunit.device, dtype=xunit.dtype) + self.mean.to(
            xunit.device, dtype=xunit.dtype
        )
        return x

    def _jac_forward(self, x, xunit, **kwargs):
        jac = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
        std = self.std.unsqueeze(0).to(x.device, dtype=x.dtype)
        jac[..., torch.arange(4), torch.arange(4)] = 1 / std
        return jac

    def _jac_inverse(self, xunit, x, **kwargs):
        jac = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
        std = self.std.unsqueeze(0).to(x.device, dtype=x.dtype)
        jac[..., torch.arange(4), torch.arange(4)] = std
        return jac

    def _detjac_forward(self, x, xunit, **kwargs):
        detjac = 1 / torch.prod(self.std, dim=-1)
        detjac = detjac.unsqueeze(0).expand(x.shape[0], x.shape[1])
        return detjac


class PtPhiEtaM2_to_JetScale(BaseTransform):
    def _forward(self, ptphietam2, jet, **kwargs):
        pt, phi, eta, m2 = unpack_last(ptphietam2)
        jet_pt, jet_phi, jet_eta, _ = unpack_last(jet)

        pt = pt / jet_pt
        phi = phi - jet_phi
        phi = ensure_angle(phi)
        eta = eta - jet_eta

        return torch.stack((pt, phi, eta, m2), dim=-1)

    def _inverse(self, y, jet, **kwargs):
        pt, phi, eta, m2 = unpack_last(y)
        jet_pt, jet_phi, jet_eta, _ = unpack_last(jet)

        pt = pt * jet_pt
        phi = phi + jet_phi
        phi = ensure_angle(phi)
        eta = eta + jet_eta

        return torch.stack((pt, phi, eta, m2), dim=-1)

    def _jac_forward(self, ptphietam2, y, jet, **kwargs):
        jet_pt = jet[:, 0]

        zero, one = torch.zeros_like(jet_pt), torch.ones_like(jet_pt)

        jac_pt = torch.stack((one / jet_pt, zero, zero, zero), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _jac_inverse(self, ptphietam2, y, jet, **kwargs):

        jet_pt = jet[:, 0]

        zero, one = torch.zeros_like(jet_pt), torch.ones_like(jet_pt)

        jac_pt = torch.stack((jet_pt, zero, zero, zero), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _detjac_forward(self, ptphietam2, y, jet, **kwargs):
        jet_pt = jet[:, 0]
        return 1 / jet_pt


class LogPtPhiEtaLogM2_to_JetScale(BaseTransform):
    def _forward(self, logptphietalogm2, jet, **kwargs):
        logpt, phi, eta, logm2 = unpack_last(logptphietalogm2)
        jet_pt, jet_phi, jet_eta, jet_m2 = unpack_last(jet)

        # pt = pt / jet_pt
        logpt = logpt - torch.log(jet_pt + EPS1)
        phi = phi - jet_phi
        phi = ensure_angle(phi)
        eta = eta - jet_eta
        logm2 = logm2 - torch.log(jet_m2 + EPS1)

        return torch.stack((logpt, phi, eta, logm2), dim=-1)

    def _inverse(self, y, jet, **kwargs):
        logpt, phi, eta, logm2 = unpack_last(y)
        jet_pt, jet_phi, jet_eta, jet_m2 = unpack_last(jet)

        logpt = logpt + torch.log(jet_pt + EPS1)
        phi = phi + jet_phi
        phi = ensure_angle(phi)
        eta = eta + jet_eta
        logm2 = logm2 + torch.log(jet_m2 + EPS1)

        return torch.stack((logpt, phi, eta, logm2), dim=-1)

    def _jac_forward(self, logptphietalogm2, y, jet, **kwargs):

        zero, one = torch.zeros_like(logptphietalogm2[..., 0]), torch.ones_like(
            logptphietalogm2[..., 0]
        )

        jac_pt = torch.stack((one, zero, zero, zero), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _jac_inverse(self, logptphietalogm2, y, jet, **kwargs):
        zero, one = torch.zeros_like(logptphietalogm2[..., 0]), torch.ones_like(
            logptphietalogm2[..., 0]
        )

        jac_pt = torch.stack((one, zero, zero, zero), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _detjac_forward(self, logptphietalogm2, y, jet, **kwargs):
        return torch.ones_like(logptphietalogm2[..., 0])


class IndividualNormal(BaseTransform):

    def __init__(self, dims_fixed=[], scaling=torch.ones(1, 4)):
        super().__init__()
        self.dims_fixed = dims_fixed
        self.scaling = scaling

    def init_fit(self, x, mask, **kwargs):
        mask = mask.unsqueeze(-1)
        self.mean = (x * mask).sum(dim=0) / mask.sum(dim=0)
        self.mean[-20:] = torch.mean(
            self.mean[-20:][~torch.isnan(self.mean[-20:]).any(dim=-1)], dim=0
        )
        self.std = torch.sqrt(
            ((x * mask - self.mean * mask) ** 2).sum(dim=0) / mask.sum(dim=0)
        ) / self.scaling.to(x.device, dtype=x.dtype)
        self.std[-20:] = torch.mean(
            self.std[-20:][~torch.isnan(self.std[-20:]).any(dim=-1)], dim=0
        )
        self.std[..., self.dims_fixed] = 1.0
        self.std[torch.abs(self.std) <= 1e-3] = 1.0

    def _forward(self, x, ptr, **kwargs):
        idx = torch.arange(
            x.shape[0], device=ptr.device, dtype=torch.int64
        ) - torch.repeat_interleave(ptr[:-1], ptr.diff(), dim=0)
        return (x - self.mean.to(x.device)[idx]) / self.std.to(x.device)[idx]

    def _inverse(self, xunit, ptr, **kwargs):
        idx = torch.arange(
            xunit.shape[0], device=ptr.device, dtype=torch.int64
        ) - torch.repeat_interleave(ptr[:-1], ptr.diff(), dim=0)
        return xunit * self.std.to(xunit.device)[idx] + self.mean.to(xunit.device)[idx]

    def _jac_forward(self, x, xunit, ptr, **kwargs):
        idx = torch.arange(
            x.shape[0], device=ptr.device, dtype=torch.int64
        ) - torch.repeat_interleave(ptr[:-1], ptr.diff(), dim=0)
        jac = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)[idx].unsqueeze(0)
        jac[..., torch.arange(4), torch.arange(4)] = 1 / std
        return jac

    def _jac_inverse(self, xunit, x, ptr, **kwargs):
        idx = torch.arange(
            x.shape[0], device=ptr.device, dtype=torch.int64
        ) - torch.repeat_interleave(ptr[:-1], ptr.diff(), dim=0)
        jac = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)[idx].unsqueeze(0)
        jac[..., torch.arange(4), torch.arange(4)] = std
        return jac
