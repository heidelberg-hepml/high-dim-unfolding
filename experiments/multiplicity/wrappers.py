from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from lgatr.interface import extract_scalar
from experiments.utils import xformers_mask


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net, force_xformers=False):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers
        self.aggregation = MeanAggregation()

    def forward(self, batch, batch_idx):
        mask = xformers_mask(batch_idx, materialize=not self.force_xformers)
        outputs = self.net(batch.unsqueeze(0), attn_bias=mask)
        outputs = self.aggregation(outputs, batch_idx).squeeze(0)
        return outputs


class MultiplicityLGATrWrapper(nn.Module):
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

        mask = xformers_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attn_bias=mask
        )
        params = self.extract_from_ga(
            multivector_outputs,
            scalar_outputs,
            embedding["batch"],
        )

        return params

    def extract_from_ga(self, multivector, scalars, batch_idx):
        outputs = extract_scalar(multivector)[0, :, :, 0]
        params = self.aggregation(outputs, index=batch_idx)
        return params
