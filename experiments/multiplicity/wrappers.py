import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from lgatr.interface import extract_scalar
from experiments.logger import LOGGER
from experiments.utils import xformers_mask
from experiments.embedding import embed_data_into_ga
from experiments.multiplicity.distributions import (
    process_params,
    cross_entropy,
    GaussianMixture,
    GammaMixture,
    ranged_categorical,
)


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net, distribution, range=None):
        super().__init__()
        self.net = net
        self.use_xformers = torch.cuda.is_available()
        self.aggregation = MeanAggregation()
        if distribution == "GaussianMixture":
            self.distribution = GaussianMixture
        elif distribution == "GammaMixture":
            self.distribution = GammaMixture
        elif distribution == "Categorical":
            self.distribution = ranged_categorical(*range)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def forward(self, batch):
        input = torch.cat([batch.x_det, batch.scalars_det], dim=-1)

        mask = xformers_mask(batch.x_det_batch, materialize=not self.use_xformers)
        attn_kwargs = {"attn_bias" if self.use_xformers else "attn_mask": mask}

        outputs = self.net(input.unsqueeze(0), **attn_kwargs)
        outputs = self.aggregation(outputs, batch.x_det_batch).squeeze(0)

        return outputs

    def batch_loss(self, batch, diff=False):

        output = self.forward(batch)
        params = process_params(output, self.distribution)
        predicted_dist = self.distribution(params)  # batch of mixtures

        label = batch.x_gen_ptr.diff()
        det_mult = batch.x_det_ptr.diff()

        if diff:
            loss = cross_entropy(predicted_dist, label - det_mult)
        else:
            loss = cross_entropy(predicted_dist, label)

        assert torch.isfinite(loss).all()
        return loss

    def sample(self, batch, range, diff=False):
        output = self.forward(batch)
        params = process_params(output, self.distribution)
        predicted_dist = self.distribution(params)  # batch of mixtures

        label = batch.x_gen_ptr.diff()
        det_mult = batch.x_det_ptr.diff()

        if diff:
            nll = cross_entropy(predicted_dist, label - det_mult).mean()
            sample = det_mult + predicted_dist.sample()
        else:
            nll = cross_entropy(predicted_dist, label).mean()
            sample = predicted_dist.sample()

        sample = torch.clamp(sample, *range)

        sample_tensor = torch.stack([sample, label, det_mult], dim=1)

        return (
            sample_tensor.cpu().detach(),
            params.cpu().detach(),
            nll.cpu().detach(),
        )


class MultiplicityLGATrWrapper(MultiplicityTransformerWrapper):
    """
    L-GATr for multiplicity
    """

    def __init__(self, GA_config, **kwargs):
        super().__init__(**kwargs)
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
