import math
import torch
from lgatr.interface import embed_vector, get_spurions

from experiments.utils import get_batch_from_ptr
from experiments.coordinates import jetmomenta_to_fourmomenta
from experiments.logger import LOGGER


def embed_data_into_ga(fourmomenta, scalars, ptr, ga_cfg=None):
    """
    Embed data into geometric algebra representation
    We use torch_geometric sparse representations to be more memory efficient
    Note that we do not embed the label, because it is handled elsewhere

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    scalars: torch.tensor of shape (n_particles, n_features)
        Optional scalar features, n_features=0 is possible
    ptr: torch.tensor of shape (batchsize+1)
        Indices of the first particle for each jet
        Also includes the first index after the batch ends
    ga_cfg: settings for embedding

    Returns
    -------
    embedding: dict
        Embedded data
        Includes keys for multivectors, scalars and ptr
    """
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)
    new_ptr = ptr.clone()

    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    if ga_cfg is not None:
        spurions = get_spurions(
            ga_cfg.beam_spurion,
            ga_cfg.add_time_spurion,
            ga_cfg.beam_mirror,
            fourmomenta.device,
            fourmomenta.dtype,
        )
        n_spurions = spurions.shape[0]
        if getattr(ga_cfg, "spurion_channels", False):
            spurions = spurions.unsqueeze(0).repeat(multivectors.shape[0], 1, 1)
            multivectors = torch.cat([multivectors, spurions], dim=1)
            mask = torch.zeros(
                multivectors.shape[0],
                dtype=torch.bool,
                device=multivectors.device,
            )

        else:
            spurion_idxs = torch.stack(
                [ptr[:-1] + i for i in range(n_spurions)], dim=0
            ) + n_spurions * torch.arange(batchsize, device=ptr.device)

            spurion_idxs = spurion_idxs.permute(1, 0).flatten()
            insert_spurion = torch.zeros(
                multivectors.shape[0] + n_spurions * batchsize,
                dtype=torch.bool,
                device=multivectors.device,
            )
            insert_spurion[spurion_idxs] = True
            multivectors_buffer = multivectors.clone()
            multivectors = torch.empty(
                insert_spurion.shape[0],
                *multivectors.shape[1:],
                dtype=multivectors.dtype,
                device=multivectors.device,
            )
            multivectors[~insert_spurion] = multivectors_buffer
            multivectors[insert_spurion] = spurions.repeat(batchsize, 1).unsqueeze(-2)

            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                multivectors.shape[0],
                scalars_buffer.shape[1],
                dtype=scalars.dtype,
                device=scalars.device,
            )
            scalars[~insert_spurion] = scalars_buffer
            if hasattr(ga_cfg, "spurion_channels"):
                scalars = torch.cat(
                    [scalars, insert_spurion.to(scalars.dtype).unsqueeze(-1)], dim=-1
                )
            new_ptr[1:] = new_ptr[1:] + (arange + 1) * n_spurions

            mask = insert_spurion
    else:
        mask = torch.zeros(
            multivectors.shape[0],
            dtype=torch.bool,
            device=multivectors.device,
        )

    batch = get_batch_from_ptr(new_ptr)

    return multivectors, scalars, batch, mask


def add_jet_to_sequence(batch):
    new_batch = batch.clone()

    batchsize = len(new_batch.x_gen_ptr) - 1
    arange = torch.arange(batchsize, device=new_batch.x_gen.device)

    gen_jets_idx = new_batch.x_gen_ptr[:-1] + arange
    det_jets_idx = new_batch.x_det_ptr[:-1] + arange

    insert_gen_jets = torch.zeros(
        new_batch.x_gen.shape[0] + batchsize,
        dtype=torch.bool,
        device=new_batch.x_gen.device,
    )
    insert_gen_jets[gen_jets_idx] = True
    x_gen = torch.empty(
        insert_gen_jets.shape[0],
        *new_batch.x_gen.shape[1:],
        dtype=new_batch.x_gen.dtype,
        device=new_batch.x_gen.device,
    )
    x_gen[~insert_gen_jets] = new_batch.x_gen

    x_gen[insert_gen_jets] = new_batch.jet_gen
    scalars_gen = torch.zeros(
        x_gen.shape[0],
        new_batch.scalars_gen.shape[1],
        dtype=new_batch.scalars_gen.dtype,
        device=new_batch.scalars_gen.device,
    )
    scalars_gen[~insert_gen_jets] = new_batch.scalars_gen
    scalars_gen = torch.cat(
        [scalars_gen, insert_gen_jets.unsqueeze(-1).to(scalars_gen.dtype)], dim=-1
    )

    insert_det_jets = torch.zeros(
        new_batch.x_det.shape[0] + batchsize,
        dtype=torch.bool,
        device=new_batch.x_det.device,
    )
    insert_det_jets[det_jets_idx] = True
    x_det = torch.empty(
        insert_det_jets.shape[0],
        *new_batch.x_det.shape[1:],
        dtype=new_batch.x_det.dtype,
        device=new_batch.x_det.device,
    )
    x_det[~insert_det_jets] = new_batch.x_det
    x_det[insert_det_jets] = new_batch.jet_det

    scalars_det = torch.zeros(
        x_det.shape[0],
        new_batch.scalars_det.shape[1],
        dtype=new_batch.scalars_det.dtype,
        device=new_batch.scalars_det.device,
    )
    scalars_det[~insert_det_jets] = new_batch.scalars_det
    scalars_det = torch.cat(
        [scalars_det, insert_det_jets.unsqueeze(-1).to(scalars_det.dtype)], dim=-1
    )

    new_batch.x_gen = x_gen
    new_batch.scalars_gen = scalars_gen
    new_batch.x_gen_ptr[1:] = new_batch.x_gen_ptr[1:] + (arange + 1)
    new_batch.x_gen_batch = get_batch_from_ptr(new_batch.x_gen_ptr)

    new_batch.x_det = x_det
    new_batch.scalars_det = scalars_det
    new_batch.x_det_ptr[1:] = new_batch.x_det_ptr[1:] + (arange + 1)
    new_batch.x_det_batch = get_batch_from_ptr(new_batch.x_det_ptr)

    return new_batch, ~insert_gen_jets


import torch


def add_jet_det_and_stop_to_x_gen(batch):
    """
    Modify only x_gen by:
      1. Prepending jet_det at the start of each sequence.
      2. Appending a sampled stop token at the end of each sequence.

    Args:
        batch: original batch object
    """
    new_batch = batch.clone()
    device = new_batch.x_gen.device
    batchsize = len(new_batch.x_gen_ptr) - 1
    arange = torch.arange(batchsize, device=device)

    extra_tokens = 2 * batchsize
    x_gen = torch.empty(
        new_batch.x_gen.shape[0] + extra_tokens,
        *new_batch.x_gen.shape[1:],
        dtype=new_batch.x_gen.dtype,
        device=device,
    )
    scalars_gen = torch.zeros(
        x_gen.shape[0],
        new_batch.scalars_gen.shape[1] + 2,  # +2 channels
        dtype=new_batch.scalars_gen.dtype,
        device=device,
    )

    starts = new_batch.x_gen_ptr[:-1] + 2 * arange  # start insert positions
    ends = new_batch.x_gen_ptr[1:] + 2 * arange  # end insert positions

    insert_start = torch.zeros(x_gen.shape[0], dtype=torch.bool, device=device)
    insert_start[starts] = True
    insert_end = torch.zeros(x_gen.shape[0], dtype=torch.bool, device=device)
    insert_end[ends] = True
    insert_mask = insert_start | insert_end

    x_gen[~insert_mask] = new_batch.x_gen
    scalars_gen[~insert_mask, :-2] = new_batch.scalars_gen

    # Insert jet_det at starts
    x_gen[starts] = new_batch.jet_det

    # Insert sampled stop token at ends
    stop_tokens = sample_stop_tokens(batchsize, device=device, dtype=x_gen.dtype)
    x_gen[ends] = stop_tokens

    # Add channels indicating inserted token
    scalars_gen[:, -2] = insert_start.to(scalars_gen.dtype)
    scalars_gen[:, -1] = insert_end.to(scalars_gen.dtype)

    # Update batch
    new_batch.x_gen = x_gen
    new_batch.scalars_gen = scalars_gen
    # shift pointers: +2 per sequence
    new_batch.x_gen_ptr[1:] = new_batch.x_gen_ptr[1:] + 2 * arange + 2
    new_batch.x_gen_batch = get_batch_from_ptr(new_batch.x_gen_ptr)

    return new_batch, ~insert_mask


def sample_stop_tokens(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    mu: float = 5.0,
    sigma: float = 0.08,
    kappa: float = 200.0,
    theta0: float = 0.0,
) -> torch.Tensor:
    # θ ~ von Mises
    u1 = torch.rand(n, device=device)
    u2 = torch.rand(n, device=device)
    a = 1.0 + math.sqrt(1.0 + 4.0 * kappa**2)
    b = (a - math.sqrt(2 * a)) / (2 * kappa)
    r = (1 + b**2) / (2 * b)
    thetas = torch.zeros(n, device=device)
    for i in range(n):
        while True:
            u = torch.rand(3, device=device)
            z = torch.cos(math.pi * u[0])
            f = (1 + r * z) / (r + z)
            c = kappa * (r - f)
            if u[1] < c * (2 - c) or u[1] <= c * torch.exp(1 - c):
                break
        thetas[i] = (torch.sign(u[2] - 0.5) * torch.acos(f) + theta0) % (2 * math.pi)
    # z1,z2,z3 ~ Normal(mu, sigma)
    z = mu + sigma * torch.randn(n, 3, device=device)
    return torch.cat([thetas[:, None], z], dim=1).to(dtype)


def stop_threshold_fn(
    token,
    mu=torch.tensor([5.0, 5.0, 5.0]),
    sigma=0.08,
    theta0=0.0,
    kappa=200.0,
    tau=10.0,
):
    # token: shape [4]  (theta, z1, z2, z3)
    theta = token[0]
    z = token[1:]
    # circular difference wrapped to [-pi,pi]
    dtheta = ((theta - theta0 + math.pi) % (2 * math.pi)) - math.pi
    sigma_theta = 1.0 / math.sqrt(kappa)  # von Mises -> approx Normal
    d2 = (dtheta**2) / (sigma_theta**2) + ((z - mu.to(z.device)) ** 2).sum() / (
        sigma**2
    )
    return d2 < tau
