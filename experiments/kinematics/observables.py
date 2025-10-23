from experiments.logger import LOGGER
import tqdm

try:
    from fastjet_contribs import (
        compute_nsubjettiness,
    )

    NSUB_AVAIL = True
except ImportError:
    LOGGER.info("compute_nsubjettiness is not available.")
    NSUB_AVAIL = False
try:
    from fastjet_contribs import apply_soft_drop

    SOFTDROP_AVAIL = True
except ImportError:
    LOGGER.info("apply_soft_drop is not available.")
    SOFTDROP_AVAIL = False

import torch
import numpy as np

from experiments.utils import (
    get_ptr_from_batch,
    fix_mass,
)

R0 = None
R0SoftDrop = None
MASS = 0.0


def tau(
    constituents,
    batch_idx,
    other_batch_idx=None,
    N=1,
    beta=1.0,
    R0=R0,
    axis_mode=3,
    **kwargs
):
    constituents = fix_mass(constituents, MASS).detach().cpu().numpy()
    batch_ptr = get_ptr_from_batch(batch_idx).detach().cpu().numpy()
    taus = []
    axis_modes = {"onepass_kt": 2}
    for i in tqdm.tqdm(range(len(batch_ptr) - 1)):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        tau = compute_nsubjettiness(
            jet=event[..., [1, 2, 3, 0]],
            N=N,
            beta=beta,
            R0=R0,
            axis_mode=axis_mode,
        )
        taus.append(tau)
    return torch.tensor(taus)


def sd_mass(constituents, batch_idx, other_batch_idx=None, R0=R0SoftDrop, **kwargs):
    constituents = fix_mass(constituents, MASS).detach().cpu().numpy()
    batch_ptr = get_ptr_from_batch(batch_idx).detach().cpu().numpy()
    log_rhos = []
    for i in tqdm.tqdm(range(len(batch_ptr) - 1)):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        sd_fourm = np.array(
            apply_soft_drop(event[..., [1, 2, 3, 0]], R0=R0, beta=0.0, zcut=0.1)
        )
        mass2 = sd_fourm[3] ** 2 - np.sum(sd_fourm[..., :3] ** 2)
        pt2 = np.sum(np.sum(event[..., 1:3], axis=0) ** 2)
        log_rho = np.log(np.clip(mass2 / pt2, a_min=1e-10, a_max=None))
        log_rhos.append(log_rho)
    return torch.tensor(log_rhos)


def compute_zg(constituents, batch_idx, other_batch_idx=None, R0=R0SoftDrop, **kwargs):
    constituents = fix_mass(constituents, MASS).detach().cpu().numpy()
    batch_ptr = get_ptr_from_batch(batch_idx).detach().cpu().numpy()
    zgs = []
    for i in tqdm.tqdm(range(len(batch_ptr) - 1)):
        event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
        zg = apply_soft_drop(event[..., [1, 2, 3, 0]], R0=R0, beta=0.0, zcut=0.1)[-1]
        zgs.append(zg)
    return torch.tensor(zgs)


def calculate_eec(batch, ptr):
    zs_all = []
    ws_all = []

    for i in tqdm.tqdm(range(ptr.shape[0] - 1)):
        particles = batch[ptr[i] : ptr[i + 1]]
        p = particles[:, 1:]
        E = (p**2).sum(dim=1).sqrt()
        pt = (p[:, :-1] ** 2).sum(dim=1).sqrt()
        total_pt = pt.sum()
        # pairwise cosθ
        dot = p @ p.T
        denom = E[:, None] * E[None, :]
        cos_theta = dot / denom
        z = (1 - cos_theta) / 2

        # pairwise weights
        w = 2 * (pt[:, None] * pt[None, :]) / total_pt**2

        # keep upper triangle
        i_idx, j_idx = torch.triu_indices(z.shape[0], z.shape[0])
        zs_all.append(z[i_idx, j_idx])
        ws_all.append(w[i_idx, j_idx])

    return torch.stack((torch.cat(zs_all), torch.cat(ws_all)), dim=1)
