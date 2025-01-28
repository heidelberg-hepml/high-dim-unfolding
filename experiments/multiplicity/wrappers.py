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


class MultiplicityConditionalTransformerWrapper(nn.Module):
    def __init__(self, net, force_xformers=False):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers
        self.aggregation = MeanAggregation()

    def forward(self, batch, ptr):
        mask = xformers_sa_mask(ptr, materialize=not self.force_xformers)
        outputs = self.net(
            x=batch.unsqueeze(0),
            condition=batch.unsqueeze(0),
            attention_mask=mask,
            condition_attention_mask=mask,
            crossattention_mask=mask,
        )
        outputs = self.aggregation(outputs, ptr).squeeze(0)
        return outputs


class MultiplicityGATrWrapper(nn.Module):
    """
    L-GATr for multiplicity
    """

    def __init__(
        self,
        net,
        mean_aggregation=True,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.aggregation = MeanAggregation() if mean_aggregation else None
        self.force_xformers = force_xformers

    def forward(self, embedding):
        multivector = embedding["mv"].unsqueeze(0)
        scalars = embedding["s"].unsqueeze(0)

        mask = xformers_sa_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        params = self.extract_from_ga(
            multivector_outputs,
            scalar_outputs,
            embedding["batch"],
            embedding["is_global"],
        )

        return params

    def extract_from_ga(self, multivector, scalars, batch):
        outputs = extract_scalar(multivector)[0, :, :, 0]
        if self.aggregation is not None:
            params = self.aggregation(outputs, index=batch)
        else:
            raise NotImplementedError
        return params


class MultiplicityConditionalGATrWrapper(nn.Module):
    """
    L-GATr for multiplicity
    """

    def __init__(
        self,
        net,
        mean_aggregation=True,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.aggregation = MeanAggregation() if mean_aggregation else None
        self.force_xformers = force_xformers

    def forward(self, embedding):
        input_multivector = embedding["mv"].unsqueeze(0)
        input_scalars = embedding["s"].unsqueeze(0)
        condition_multivector = embedding["mv"].unsqueeze(0)
        condition_scalars = embedding["s"].unsqueeze(0)

        mask = xformers_sa_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivectors=input_multivector,
            multivectors_condition=condition_multivector,
            scalars=input_scalars,
            scalars_condition=condition_scalars,
            attention_mask=mask,
            attention_mask_condition=mask,
            crossattention_mask=mask,
        )
        params = self.extract_from_ga(
            multivector_outputs,
            scalar_outputs,
            embedding["batch"],
            embedding["is_global"],
        )

        return params

    def extract_from_ga(self, multivector, scalars, batch):
        outputs = extract_scalar(multivector)[0, :, :, 0]
        if self.aggregation is not None:
            params = self.aggregation(outputs, index=batch)
        else:
            raise NotImplementedError
        return params
