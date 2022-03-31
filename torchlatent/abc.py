from abc import ABCMeta

from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.utils import lazy_property

from torchlatent.types import Sequence

__all__ = [
    'DistributionABC',
]


class DistributionABC(Distribution, metaclass=ABCMeta):
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
        raise NotImplementedError

    @lazy_property
    def log_marginals(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def entropy(self) -> Tensor:
        raise NotImplementedError
