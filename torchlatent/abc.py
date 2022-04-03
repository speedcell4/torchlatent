from abc import ABCMeta

import torch.autograd
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.utils import lazy_property

from torchlatent.types import Sequence

__all__ = [
    'DistributionABC',
]


class DistributionABC(Distribution, metaclass=ABCMeta):
    scores: Tensor

    def log_scores(self, value: Sequence) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def log_partitions(self) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Sequence) -> Tensor:
        return self.log_scores(value=value) - self.log_partitions

    @lazy_property
    def max(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def argmax(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.max, self.scores, torch.ones_like(self.max),
            create_graph=False, only_inputs=True, allow_unused=False,
        )
        return grad

    @lazy_property
    def marginals(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def entropy(self) -> Tensor:
        raise NotImplementedError
