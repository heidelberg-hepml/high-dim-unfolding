import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.aggr import MeanAggregation

from xformers.ops.fmha import BlockDiagonalMask
from gatr.interface import embed_vector, extract_scalar


def xformers_sa_mask(batch, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device)
    return mask


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net, force_xformers=False):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers
        self.aggregation = MeanAggregation()

    def forward(self, batch, ptr):
        mask = xformers_sa_mask(ptr, materialize=not self.force_xformers)
        outputs = self.net(batch.unsqueeze(0), attention_mask=mask)
        outputs = self.aggregation(outputs, ptr).squeeze(0)
        return outputs
