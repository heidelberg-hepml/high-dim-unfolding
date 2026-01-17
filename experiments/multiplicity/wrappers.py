import torch
from lgatr.interface import extract_scalar
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

import experiments.coordinates as c
from experiments.embedding import add_jet_to_sequence, embed_data_into_ga
from experiments.misc import get_attention_mask
from experiments.multiplicity.distributions import (
    GammaMixture,
    GaussianMixture,
    cross_entropy,
    process_params,
    ranged_categorical,
)


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net, distribution, wrapper_cfg, range=None):
        super().__init__()
        self.net = net
        self.wrapper_cfg = wrapper_cfg
        self.use_xformers = torch.cuda.is_available()
        self.aggregation = MeanAggregation()
        if distribution == "GaussianMixture":
            self.distribution = GaussianMixture
        elif distribution == "GammaMixture":
            self.distribution = GammaMixture
        elif distribution == "Categorical":
            assert range is not None, "Range must be provided for Categorical distribution"
            self.distribution = ranged_categorical(*range)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def init_coordinates(self):
        self.const_coordinates = getattr(c, self.wrapper_cfg.const_coordinates)(
            **self.wrapper_cfg.const_coordinates_options
        )
        self.condition_const_coordinates = getattr(c, self.wrapper_cfg.const_coordinates)(
            **self.wrapper_cfg.const_coordinates_options
        )
        self.jet_coordinates = getattr(c, self.wrapper_cfg.jet_coordinates)(
            **self.wrapper_cfg.jet_coordinates_options
        )
        self.condition_jet_coordinates = getattr(c, self.wrapper_cfg.jet_coordinates)(
            **self.wrapper_cfg.jet_coordinates_options
        )
        if self.wrapper_cfg.transforms_float64:
            self.const_coordinates.to(torch.float64)
            self.condition_const_coordinates.to(torch.float64)
            self.jet_coordinates.to(torch.float64)
            self.condition_jet_coordinates.to(torch.float64)

    def forward(self, batch):
        if self.wrapper_cfg.add_jet:
            new_batch, _, _ = add_jet_to_sequence(batch)
        else:
            new_batch = batch
        input = torch.cat([new_batch.x_det, new_batch.scalars_det], dim=-1)

        mask = get_attention_mask(new_batch.x_det_batch, attention_backend='xformers', dtype=input.dtype)

        outputs = self.net(input.unsqueeze(0), **mask)
        outputs = self.aggregation(outputs, new_batch.x_det_batch).squeeze(0)

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

    def __init__(self, GA_config, scalar_inputs, **kwargs):
        super().__init__(**kwargs)
        self.ga_cfg = GA_config
        self.scalar_inputs = scalar_inputs

    def forward(self, batch):
        if self.wrapper_cfg.add_jet:
            new_batch, _, det_const_mask = add_jet_to_sequence(batch)
        else:
            new_batch = batch.clone()
            det_const_mask = torch.ones(
                new_batch.x_det.shape[0],
                dtype=torch.bool,
                device=new_batch.x_det.device,
            )

        det_jets = self.jet_coordinates.x_to_fourmomenta(batch.jet_det)
        ext_det_jets = torch.repeat_interleave(det_jets, batch.x_det_ptr.diff(), dim=0)

        fourmomenta = torch.zeros_like(new_batch.x_det)
        fourmomenta[det_const_mask] = self.const_coordinates.x_to_fourmomenta(
            new_batch.x_det[det_const_mask],
            jet=ext_det_jets,
            ptr=new_batch.x_det_ptr,
        )

        if self.wrapper_cfg.add_jet:
            fourmomenta[~det_const_mask] = det_jets

        if len(self.scalar_inputs) > 0:
            scalars = torch.cat(
                [new_batch.scalars_det, new_batch.x_det[:, self.scalar_inputs]], dim=-1
            )
        else:
            scalars = new_batch.scalars_det

        mv, s, batch_idx, _ = embed_data_into_ga(
            new_batch.x_det,
            scalars,
            new_batch.x_det_ptr,
            self.ga_cfg,
        )
        multivector = mv.unsqueeze(0)
        scalars = s.unsqueeze(0)

        mask = get_attention_mask(batch_idx, attention_backend='xformers', dtype=multivector.dtype)
        multivector_outputs, _ = self.net(multivector, scalars=scalars, **mask)
        outputs = extract_scalar(multivector_outputs)[0, :, :, 0]
        params = self.aggregation(outputs, index=batch_idx)

        return params
