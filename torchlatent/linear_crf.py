from typing import Tuple
from typing import Type

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torchrua import CattedSequence
from torchrua import last_packed_indices
from torchrua import pack_catted_sequence
from torchrua import pack_padded_sequence

from torchlatent.abc2 import Sequence
from torchlatent.abc2 import StructuredDistribution
from torchlatent.semiring import Log
from torchlatent.semiring import Max
from torchlatent.semiring import Semiring

Transitions = Tuple[Tensor, Tensor, Tensor]


def crf_partitions(emissions: Sequence, transitions: Transitions, semiring: Type[Semiring]) -> Tensor:
    if isinstance(emissions, CattedSequence):
        emissions = pack_catted_sequence(emissions)
    elif isinstance(emissions, tuple) and len(emissions) == 2:
        emissions = pack_padded_sequence(*emissions, batch_first=True)

    assert isinstance(emissions, PackedSequence)

    transitions, head_transitions, last_transitions = transitions
    emissions, batch_sizes, _, unsorted_indices = emissions

    last_indices = last_packed_indices(
        batch_sizes=batch_sizes,
        unsorted_indices=unsorted_indices,
        device=emissions.device,
    )

    _, *batch_sizes = sections = batch_sizes.detach().cpu().tolist()
    emission, *emissions = torch.split(emissions, sections, dim=0)

    chunks = [semiring.mul(head_transitions, emission)]
    for emission, batch_size in zip(emissions, batch_sizes):
        chunks.append(semiring.mul(
            semiring.bmm(chunks[-1][:batch_size], transitions),
            emission,
        ))

    emission = torch.cat(chunks, dim=0)[last_indices]
    return semiring.sum(semiring.mul(emission, last_transitions), dim=-1)


class CrfDistribution(StructuredDistribution):
    def __init__(self, emissions: Sequence, transitions: Transitions) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.transitions = transitions

    def log_scores(self, targets: Sequence) -> Tensor:
        raise NotImplementedError

    def log_partitions(self) -> Tensor:
        return crf_partitions(
            emissions=self.emissions,
            transitions=self.transitions,
            semiring=Log,
        )

    def max(self) -> Tensor:
        return crf_partitions(
            emissions=self.emissions,
            transitions=self.transitions,
            semiring=Max,
        )


class CrfDecoder(nn.Module):
    def __init__(self, num_targets: int) -> None:
        super(CrfDecoder, self).__init__()

        self.num_targets = num_targets

        self.transitions = nn.Parameter(torch.empty((num_targets, num_targets)))
        self.head_transitions = nn.Parameter(torch.empty((num_targets,)))
        self.last_transitions = nn.Parameter(torch.empty((num_targets,)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.transitions)
        init.zeros_(self.head_transitions)
        init.zeros_(self.last_transitions)

    def extra_repr(self) -> str:
        return f'num_targets={self.num_targets}'

    def forward(self, emissions: Sequence) -> CrfDistribution:
        return CrfDistribution(
            emissions=emissions,
            transitions=(
                self.transitions,
                self.head_transitions,
                self.last_transitions,
            ),
        )
