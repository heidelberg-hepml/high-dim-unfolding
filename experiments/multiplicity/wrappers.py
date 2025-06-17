import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from lgatr.interface import extract_scalar
from experiments.utils import xformers_mask
from experiments.embedding import embed_data_into_ga


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net, force_xformers=False):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers
        self.aggregation = MeanAggregation()

    def forward(self, batch):
        input = torch.cat([batch.x_det, batch.scalars_det], dim=-1)

        mask = xformers_mask(batch.x_det_batch, materialize=not self.force_xformers)

        outputs = self.net(input.unsqueeze(0), attention_mask=mask)
        outputs = self.aggregation(outputs, batch.x_det_batch).squeeze(0)

        return outputs


class MultiplicityLGATrWrapper(nn.Module):
    """
    L-GATr for multiplicity
    """

    def __init__(
        self,
        net,
        GA_config,
        mean_aggregation=True,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.aggregation = MeanAggregation() if mean_aggregation else None
        self.ga_cfg = GA_config
        self.force_xformers = force_xformers

    def forward(self, batch):
        embedding = embed_data_into_ga(
            batch.x_det,
            batch.scalars_det,
            batch.x_det_ptr,
            self.ga_cfg,
        )
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
