import torch

from lgatr.interface import embed_vector


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
