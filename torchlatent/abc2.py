from abc import ABCMeta
from typing import Union

import torch
import torch.autograd
from torch import Tensor
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
    def argmax(self) -> Union[C, D, P]:
        grad, = torch.autograd.grad(
            self.max, self.emissions.data, torch.ones_like(self.max),
            create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True,
        )
        return self.emissions._replace(data=grad.argmax(dim=-1))
