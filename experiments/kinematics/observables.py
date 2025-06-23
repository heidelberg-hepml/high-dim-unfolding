import torch
import numpy as np
from fastjet_contribs import (
    compute_nsubjettiness,
    apply_soft_drop,
)

from experiments.utils import get_ptr_from_batch, ensure_angle
from experiments.coordinates import fourmomenta_to_jetmomenta


def create_partial_jet(start, end, filter=None, units=10.0):
    assert end > start or end == -1, "End index must be greater than start index"

    def form_partial_jet(constituents, batch_idx, other_batch_idx):

        batch_ptr = get_ptr_from_batch(batch_idx)
        jets = []
        for n in range(len(batch_ptr) - 1):
            event_size = batch_ptr[n + 1] - batch_ptr[n]
            if isinstance(start, float):
                start_idx = int(start * event_size)
            else:
                start_idx = start
            if isinstance(end, float):
                end_idx = int(end * event_size)
            elif end == -1:
                end_idx = event_size
            else:
                end_idx = end
            if start_idx < event_size:
                if end_idx >= event_size:
                    if filter is not None:
                        true_jet = constituents[batch_ptr[n] : batch_ptr[n + 1]].sum(
                            dim=0
                        )
                        true_jet_pt = fourmomenta_to_jetmomenta(true_jet)[..., 0]
                        if true_jet_pt < filter[0] or true_jet_pt > filter[1]:
                            continue
                    jet = constituents[batch_ptr[n] + start_idx : batch_ptr[n + 1]].sum(
                        dim=0
                    )
                else:
                    jet = constituents[
                        batch_ptr[n] + start_idx : batch_ptr[n] + end_idx
                    ].sum(dim=0)
                jets.append(jet)
        return torch.stack(jets) * units

    return form_partial_jet


def compute_angles(start1, end1, start2, end2, angle_type="R", filter=None):
    assert start1 < end1, "start1 must be less than end1"
    assert start2 < end2, "start2 must be less than end2"
    assert end1 <= start2, "end1 must be less than or equal to start2"

    def compute_angle(constituents, batch_idx, other_batch_idx):

        batch_ptr = get_ptr_from_batch(batch_idx)
        angles = []
        for n in range(len(batch_ptr) - 1):
            event_size = batch_ptr[n + 1] - batch_ptr[n]
            if isinstance(start1, float):
                start_idx1 = int(start1 * event_size)
                end_idx1 = int(end1 * event_size)
                start_idx2 = int(start2 * event_size)
                end_idx2 = int(end2 * event_size)
            else:
                start_idx1 = start1
                end_idx1 = end1
                start_idx2 = start2
                end_idx2 = end2
            if end_idx2 > event_size:
                continue

            jet1 = constituents[
                batch_ptr[n] + start_idx1 : batch_ptr[n] + end_idx1
            ].sum(dim=0)
            jet2 = constituents[
                batch_ptr[n] + start_idx2 : batch_ptr[n] + end_idx2
            ].sum(dim=0)
            jet1 = fourmomenta_to_jetmomenta(jet1)
            jet2 = fourmomenta_to_jetmomenta(jet2)
            if angle_type == "R":
                d = torch.sqrt(
                    ensure_angle((jet1[..., 1] - jet2[..., 1])) ** 2
                    + (jet1[..., 2] - jet2[..., 2]) ** 2
                ).item()
            elif angle_type == "phi":
                d = ensure_angle(jet1[..., 1] - jet2[..., 1]).item()
            elif angle_type == "eta":
                d = (jet1[..., 2] - jet2[..., 2]).item()
            angles.append(d)
        return torch.tensor(angles).view(-1, 1)

    return compute_angle


def select_pt(i, bound=None, filter=None):
    if bound == None:
        bound = i
    assert bound >= i, "bound must be greater than i"

    # create a function that returns the i-th highest pt constituent
    # if the jet has less than bound constituents, it is not selected
    def ith_pt(constituents, batch_idx, other_batch_idx):
        idx = []
        batch_ptr = get_ptr_from_batch(batch_idx)
        other_batch_ptr = get_ptr_from_batch(other_batch_idx)
        for n in range(len(batch_ptr) - 1):
            if bound < batch_ptr[n + 1] - batch_ptr[n]:
                if bound < other_batch_ptr[n + 1] - other_batch_ptr[n]:
                    if filter is not None:
                        true_jet = constituents[batch_ptr[n] : batch_ptr[n + 1]].sum(
                            dim=0
                        )
                        true_jet_pt = fourmomenta_to_jetmomenta(true_jet)[..., 0]
                        if true_jet_pt < filter[0] or true_jet_pt > filter[1]:
                            continue
                    idx.append(batch_ptr[n] + i)
        selected_constituents = constituents[idx]
        return selected_constituents

    return ith_pt


def dimass(i, j):
    def dimass_ij(constituents, batch_idx, other_batch_idx):
        batch_ptr = get_ptr_from_batch(batch_idx)
        other_batch_ptr = get_ptr_from_batch(other_batch_idx)
        dimass = []
        for n in range(len(batch_ptr) - 1):
            if batch_ptr[n + 1] - batch_ptr[n] == 3:
                dijet = constituents[batch_ptr[n] + i] + constituents[batch_ptr[n] + j]
                dimass.append(torch.sqrt(dijet[0] ** 2 - (dijet[1:] ** 2).sum(dim=-1)))
        return torch.stack(dimass)

    return dimass_ij


def deltaR(i, j):
    def deltaR_ij(constituents, batch_idx, other_batch_idx):
        batch_ptr = get_ptr_from_batch(batch_idx)
        other_batch_ptr = get_ptr_from_batch(other_batch_idx)
        deltaR = []
        for n in range(len(batch_ptr) - 1):
            if batch_ptr[n + 1] - batch_ptr[n] == 3:
                jet_i = fourmomenta_to_jetmomenta(constituents[batch_ptr[n] + i])
                jet_j = fourmomenta_to_jetmomenta(constituents[batch_ptr[n] + j])
                dR2 = (
                    ensure_angle(jet_i[..., 1] - jet_j[..., 1]) ** 2
                    + (jet_i[..., 2] - jet_j[..., 2]) ** 2
                )
                deltaR.append(torch.sqrt(dR2))
        return torch.stack(deltaR)

    return deltaR_ij


def tau1(constituents, batch_idx, other_batch_idx):
    constituents = np.array(constituents.detach().cpu())
    batch_ptr = get_ptr_from_batch(batch_idx)
    taus = []
    for i in range(len(batch_ptr) - 1):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        tau = compute_nsubjettiness(
            event[..., [1, 2, 3, 0]], N=1, beta=1.0, R0=R0, axis_mode=3
        )
        taus.append(tau)
    return torch.tensor(taus)


def tau2(constituents, batch_idx, other_batch_idx):
    constituents = np.array(constituents.detach().cpu())
    batch_ptr = get_ptr_from_batch(batch_idx)
    taus = []
    for i in range(len(batch_ptr) - 1):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        tau = compute_nsubjettiness(
            event[..., [1, 2, 3, 0]], N=2, beta=1.0, R0=R0, axis_mode=3
        )
        taus.append(tau)
    return torch.tensor(taus)


def sd_mass(constituents, batch_idx, other_batch_idx):
    constituents = np.array(constituents.detach().cpu())
    batch_ptr = get_ptr_from_batch(batch_idx)
    log_rhos = []
    for i in range(len(batch_ptr) - 1):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        sd_fourm = np.array(
            apply_soft_drop(event[..., [1, 2, 3, 0]], R0=R0SoftDrop, beta=0.0, zcut=0.1)
        )
        mass2 = sd_fourm[3] ** 2 - np.sum(sd_fourm[..., :3] ** 2)
        pt2 = np.sum(np.sum(event[..., 1:3], axis=0) ** 2)
        log_rho = np.log(mass2 / pt2)
        log_rhos.append(log_rho)
    return torch.tensor(log_rhos)


def compute_zg(constituents, batch_idx, other_batch_idx):
    constituents = np.array(constituents.detach().cpu())
    batch_ptr = get_ptr_from_batch(batch_idx)
    zgs = []
    for i in range(len(batch_ptr) - 1):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        zg = apply_soft_drop(
            event[..., [1, 2, 3, 0]], R0=R0SoftDrop, beta=0.0, zcut=0.1
        )[-1]
        zgs.append(zg)
    return torch.tensor(zgs)


def jet_mass(constituents, batch_idx, other_batch_idx):
    batch_ptr = get_ptr_from_batch(batch_idx)
    other_batch_ptr = get_ptr_from_batch(other_batch_idx)
    jet_masses = []
    for n in range(len(batch_ptr) - 1):
        jet = constituents[batch_ptr[n] : batch_ptr[n + 1]].sum(dim=0)
        mass2 = jet[0] ** 2 - (jet[1:] ** 2).sum(dim=-1)
        jet_masses.append(torch.sqrt(mass2))
    return torch.stack(jet_masses)
