from abc import ABCMeta
from typing import Union

import torch
import torch.autograd
from torch import Tensor
from torch import nn
from torch.distributions.utils import lazy_property

from torchrua import C
from torchrua import D
from torchrua import P


class StructuredDistribution(object, metaclass=ABCMeta):
    def __init__(self, emissions: Union[C, D, P]) -> None:
        super(StructuredDistribution, self).__init__()
        self.emissions = emissions

    def log_scores(self, targets: Union[C, D, P]) -> Tensor:
        raise NotImplementedError

    def log_probs(self, targets: Union[C, D, P]) -> Tensor:
        return self.log_scores(targets=targets) - self.log_partitions

    @lazy_property
    def log_partitions(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def marginals(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.log_partitions, self.emissions.data, torch.ones_like(self.log_partitions),
            create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True,

        )
        return grad

    @lazy_property
    def max(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def argmax(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.max, self.emissions.data, torch.ones_like(self.max),
            create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True,
        )
        return grad


class StructuredDecoder(nn.Module):
    def __init__(self, *, num_targets: int) -> None:
        super(StructuredDecoder, self).__init__()
        self.num_targets = num_targets

    def reset_parameters(self) -> None:
        pass

    def extra_repr(self) -> str:
        return f'num_targets={self.num_targets}'

    def forward(self, emissions: Union[C, D, P]) -> StructuredDistribution:
        raise NotImplementedError
