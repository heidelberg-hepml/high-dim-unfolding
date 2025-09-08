import torch
from lgatr.interface import embed_vector, get_spurions

from experiments.utils import get_batch_from_ptr
from experiments.coordinates import jetmomenta_to_fourmomenta


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


def add_jet_to_sequence(batch, jet_to_fourmomenta=False):
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
    if jet_to_fourmomenta:
        x_gen[insert_gen_jets] = jetmomenta_to_fourmomenta(new_batch.jet_gen)
    else:
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
    if jet_to_fourmomenta:
        x_det[insert_det_jets] = jetmomenta_to_fourmomenta(new_batch.jet_det)
    else:
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
