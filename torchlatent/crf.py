from typing import Tuple, Type, Union

import torch
from torch import nn, Tensor
from torch.distributions.utils import lazy_property
from torch.nn import init
from torchrua import C, D, P

from torchlatent.abc import StructuredDecoder, StructuredDistribution
from torchlatent.semiring import Log, Max, Semiring

T = Tuple[Tensor, Tensor, Tensor]


def crf_scores(emissions: Union[C, D, P], targets: Union[C, D, P], transitions: T, semiring: Type[Semiring]) -> Tensor:
    transitions, head_transitions, last_transitions = transitions

    targets = _, token_sizes = targets.cat()
    head_transitions = targets.head().rua(head_transitions)
    last_transitions = targets.last().rua(last_transitions)
    transitions = targets.data.roll(1).rua(transitions, targets)

    emissions, _ = emissions.idx().cat().rua(emissions, targets)
    emissions = semiring.segment_prod(emissions, sizes=token_sizes)

    token_sizes = torch.stack([torch.ones_like(token_sizes), token_sizes - 1], dim=-1)
    transitions = semiring.segment_prod(transitions, sizes=token_sizes.view(-1))[1::2]

    return semiring.mul(
        semiring.mul(head_transitions, last_transitions),
        semiring.mul(emissions, transitions),
    )


def crf_partitions(emissions: Union[C, D, P], transitions: T, semiring: Type[Semiring]) -> Tensor:
    transitions, head_transitions, last_transitions = transitions

    emissions = emissions.pack()
    last_indices = emissions.idx().last()
    emissions, batch_sizes, _, _ = emissions

    _, *batch_sizes = sections = batch_sizes.detach().cpu().tolist()
    emission, *emissions = torch.split(emissions, sections, dim=0)

    charts = [semiring.mul(head_transitions, emission)]
    for emission, batch_size in zip(emissions, batch_sizes):
        charts.append(semiring.mul(
            semiring.bmm(charts[-1][:batch_size], transitions),
            emission,
        ))

    emission = torch.cat(charts, dim=0)[last_indices]
    return semiring.sum(semiring.mul(emission, last_transitions), dim=-1)


class CrfDistribution(StructuredDistribution):
    def __init__(self, emissions: Union[C, D, P], transitions: T) -> None:
        super(CrfDistribution, self).__init__(emissions=emissions)
        self.transitions = transitions

    def log_scores(self, targets: Union[C, D, P]) -> Tensor:
        return crf_scores(
            emissions=self.emissions, targets=targets,
            transitions=self.transitions,
            semiring=Log,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return crf_partitions(
            emissions=self.emissions,
            transitions=self.transitions,
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return crf_partitions(
            emissions=self.emissions,
            transitions=self.transitions,
            semiring=Max,
        )

    @lazy_property
    def argmax(self) -> Union[C, D, P]:
        argmax = super(CrfDistribution, self).argmax.argmax(dim=-1)
        return self.emissions._replace(data=argmax)


class CrfDecoder(StructuredDecoder):
    def __init__(self, *, num_targets: int) -> None:
        super(CrfDecoder, self).__init__(num_targets=num_targets)

        self.transitions = nn.Parameter(torch.empty((num_targets, num_targets)))
        self.head_transitions = nn.Parameter(torch.empty((num_targets,)))
        self.last_transitions = nn.Parameter(torch.empty((num_targets,)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.transitions)
        init.zeros_(self.head_transitions)
        init.zeros_(self.last_transitions)

    def forward(self, emissions: Union[C, D, P]) -> CrfDistribution:
        return CrfDistribution(
            emissions=emissions,
            transitions=(
                self.transitions,
                self.head_transitions,
                self.last_transitions,
            ),
        )
