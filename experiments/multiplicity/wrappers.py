import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from lgatr.interface import extract_scalar
from experiments.utils import xformers_mask
from experiments.embedding import embed_data_into_ga


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.use_xformers = torch.cuda.is_available()
        self.aggregation = MeanAggregation()

    def forward(self, batch):
        input = torch.cat([batch.x_det, batch.scalars_det], dim=-1)

        mask = xformers_mask(batch.x_det_batch, materialize=not self.use_xformers)

        attn_kwargs = {"attn_bias" if self.use_xformers else "attn_mask": mask}

        outputs = self.net(input.unsqueeze(0), **attn_kwargs)
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
    ):
        super().__init__()
        self.net = net
        self.aggregation = MeanAggregation()
        self.use_xformers = torch.cuda.is_available()
        self.ga_cfg = GA_config

    def forward(self, batch):
        mv, s, batch_idx, _ = embed_data_into_ga(
            batch.x_det,
            batch.scalars_det,
            batch.x_det_ptr,
            self.ga_cfg,
        )
        multivector = mv.unsqueeze(0)
        scalars = s.unsqueeze(0)

        mask = xformers_mask(batch_idx, materialize=not self.use_xformers)
        attn_kwargs = {"attn_bias" if self.use_xformers else "attn_mask": mask}
        multivector_outputs, _ = self.net(multivector, scalars=scalars, **attn_kwargs)
        outputs = extract_scalar(multivector_outputs)[0, :, :, 0]
        params = self.aggregation(outputs, index=batch_idx)

        return params
