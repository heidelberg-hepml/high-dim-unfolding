import torch

from lgatr.interface import embed_vector

import torch

from lgatr.interface import embed_vector, get_spurions

from experiments.utils import get_batch_from_ptr


def embed_data_into_ga(fourmomenta, scalars, ptr, cfg_data):
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
    cfg_data: settings for embedding

    Returns
    -------
    embedding: dict
        Embedded data
        Includes keys for multivectors, scalars and ptr
    """
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)

    # embed fourmomenta into multivectors
    if cfg_data.units is not None:
        fourmomenta /= cfg_data.units
    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    # beam reference
    spurions = get_spurions(
        cfg_data.beam_spurion,
        cfg_data.add_time_spurion,
        cfg_data.beam_mirror,
        fourmomenta.device,
        fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]
    if cfg_data.beam_token and n_spurions > 0:
        # prepend spurions to the token list (within each block)
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
            scalars.shape[1],
            dtype=scalars.dtype,
            device=scalars.device,
        )
        scalars[~insert_spurion] = scalars_buffer
        ptr[1:] = ptr[1:] + (arange + 1) * n_spurions
    else:
        # append spurion to multivector channels
        spurions = spurions.unsqueeze(0).repeat(multivectors.shape[0], 1, 1)
        multivectors = torch.cat((multivectors, spurions), dim=-2)

    batch = get_batch_from_ptr(ptr)

    # return dict
    embedding = {
        "mv": multivectors,
        "s": scalars,
        "batch": batch,
    }
    return embedding


def event_to_GA_with_spurions(fourmomenta, scalars, spurions):

    multivectors = embed_vector(fourmomenta)

    spurions = spurions.to(device=fourmomenta.device, dtype=fourmomenta.dtype)
    spurions_scalars = torch.zeros(
        (spurions.shape[0], *scalars.shape[1:]),
        device=fourmomenta.device,
        dtype=fourmomenta.dtype,
    )

    multivectors = torch.cat((multivectors, spurions), dim=0).unsqueeze(-2)
    scalars = torch.cat((scalars, spurions_scalars), dim=0)

    return multivectors, scalars
