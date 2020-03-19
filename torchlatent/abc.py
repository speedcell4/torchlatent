from abc import ABCMeta
from typing import Any

import torch
from torch import Tensor
from torch import autograd
from torch import distributions
from torch.distributions.utils import lazy_property


class LatentDistribution(distributions.Distribution, metaclass=ABCMeta):
    def __init__(self, log_potentials: Tensor):
        super(LatentDistribution, self).__init__()
        self.log_potentials = log_potentials

    def log_prob(self, target: Any) -> Tensor:
        return self.log_scores(target=target) - self.log_partitions

    def log_scores(self, target: Any) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def log_partitions(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def marginals(self) -> Tensor:
        marginals, = autograd.grad(
            self.log_partitions, self.log_potentials, torch.ones_like(self.log_partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        return marginals

    @lazy_property
    def argmax(self) -> Any:
        raise NotImplementedError
