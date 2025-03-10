import torch
from torch.nn.functional import one_hot
from torch_geometric.utils import scatter

from experiments.unfolding.utils import get_batch_from_ptr, get_pt, get_phi, get_eta
from experiments.logger import LOGGER
from gatr.interface import embed_vector, embed_spurions


def embed_into_ga_with_spurions(fourmomenta, scalars, batch_ptr, cfg_data):
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
        Includes keys for multivectors, scalars, and ptr
    """
    ptr = batch_ptr.clone()
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)

    # add extra scalar channels
    if cfg_data.add_scalar_features:
        log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
        log_energy = fourmomenta[..., 0].unsqueeze(-1).log()

        batch = get_batch_from_ptr(ptr)
        jet = scatter(fourmomenta, index=batch, dim=0, reduce="sum").index_select(
            0, batch
        )
        log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
        log_energy_rel = (fourmomenta[..., 0].log() - jet[..., 0].log()).unsqueeze(-1)
        phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
        dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
        eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
        deta = -(eta_4 - eta_jet).unsqueeze(-1)
        dr = torch.sqrt(dphi**2 + deta**2)
        scalar_features = [
            log_pt,
            log_energy,
            log_pt_rel,
            log_energy_rel,
            dphi,
            deta,
            dr,
        ]
        for i, feature in enumerate(scalar_features):
            mean, factor = cfg_data.scalar_features_preprocessing[i]
            scalar_features[i] = (feature - mean) * factor
        scalars = torch.cat(
            (scalars, *scalar_features),
            dim=-1,
        )

    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    # beam reference
    spurions = embed_spurions(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        cfg_data.add_xzplane,
        cfg_data.add_yzplane,
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

    return multivectors, scalars, batch
