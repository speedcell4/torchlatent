from abc import ABCMeta
from typing import Any

from torch import Tensor
from torch import distributions
from torch.distributions.utils import lazy_property


class LatentDistribution(distributions.Distribution, metaclass=ABCMeta):
    def log_prob(self, target: Any) -> Tensor:
        return self.log_scores(target=target) - self.log_partitions

    def log_scores(self, target: Any) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def log_partitions(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def marginals(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def argmax(self) -> Any:
        raise NotImplementedError
