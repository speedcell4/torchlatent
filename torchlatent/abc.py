from abc import ABCMeta
from typing import Union

import torch
import torch.autograd
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence

from torchrua import CattedSequence

Sequence = Union[CattedSequence, PackedSequence]


class DistributionABC(Distribution, metaclass=ABCMeta):
    emissions: Tensor

    def log_scores(self, targets: Sequence) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def log_partitions(self) -> Tensor:
        raise NotImplementedError

    def log_prob(self, targets: Sequence) -> Tensor:
        return self.log_scores(targets=targets) - self.log_partitions

    @lazy_property
    def max(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def argmax(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.max, self.emissions, torch.ones_like(self.max),
            create_graph=False, only_inputs=True, allow_unused=True,
        )
        return grad

    @lazy_property
    def marginals(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.log_partitions, self.emissions, torch.ones_like(self.log_partitions),
            create_graph=True, only_inputs=True, allow_unused=True,

        )
        return grad

    @lazy_property
    def entropy(self) -> Tensor:
        raise NotImplementedError
