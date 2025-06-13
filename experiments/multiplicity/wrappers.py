import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from gatr.interface import extract_scalar
from experiments.utils import xformers_mask
from experiments.logger import LOGGER


class MultiplicityTransformerWrapper(nn.Module):
    def __init__(self, net, force_xformers=False):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers
        self.aggregation = MeanAggregation()

    def forward(self, batch, batch_idx):
        mask = xformers_mask(batch_idx, materialize=not self.force_xformers)
        outputs = self.net(batch.unsqueeze(0), attention_mask=mask)
        outputs = self.aggregation(outputs, batch_idx).squeeze(0)
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

        mask = xformers_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
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


#############################
# For testing purposes only #
#############################


class MultiplicityConditionalTransformerWrapper(nn.Module):
    def __init__(self, net, force_xformers=False):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers
        self.aggregation = MeanAggregation()

    def forward(self, batch, ptr):
        mask = xformers_mask(ptr, materialize=not self.force_xformers)
        outputs = self.net(
            x=batch.unsqueeze(0),
            condition=batch.unsqueeze(0),
            attention_mask=mask,
            attention_mask_condition=mask,
            crossattention_mask=mask,
        )
        outputs = self.aggregation(outputs, ptr).squeeze(0)
        return outputs


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
        multivectors_condition = embedding["mv"].unsqueeze(0)
        scalars_condition = embedding["s"].unsqueeze(0)

        LOGGER.info(f"input_multivector : {input_multivector.shape}")
        LOGGER.info(f"input_scalars : {input_scalars.shape}")
        LOGGER.info(f"condition_multivector : {multivectors_condition.shape}")
        LOGGER.info(f"condition_scalars : {scalars_condition.shape}")

        mask = xformers_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivectors=input_multivector,
            multivectors_condition=multivectors_condition,
            scalars=input_scalars,
            scalars_condition=scalars_condition,
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

    def extract_from_ga(self, multivector, scalars, batch_idx, is_global):
        outputs = extract_scalar(multivector)[0, :, :, 0]
        if self.aggregation is not None:
            params = self.aggregation(outputs, index=batch_idx)
        else:
            raise NotImplementedError
        return params
