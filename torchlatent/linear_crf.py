from typing import Tuple
from typing import Type
from typing import Union

import torch
from torch import Tensor

from torchlatent.semiring import Semiring
from torchrua import C
from torchrua import D
from torchrua import P

T = Tuple[Tensor, Tensor, Tensor]


def crf_scores(emissions: Union[C, D, P], targets: Union[C, D, P], transitions: T, semiring: Type[Semiring]) -> Tensor:
    transitions, head_transitions, last_transitions = transitions

    targets = targets.cat()
    head_transitions = head_transitions[targets.head().data]
    last_transitions = last_transitions[targets.last().data]
    transitions = transitions[targets.roll(1).data, targets.data]

    emissions, _ = emissions.idx().cat().rua(emissions, targets)
    emissions = semiring.segment_prod(emissions, sizes=targets.token_sizes)

    token_sizes = torch.stack([torch.ones_like(targets.token_sizes), targets.token_sizes - 1], dim=-1)
    transitions = semiring.segment_prod(transitions, sizes=token_sizes.view(-1))[1::2]

    return semiring.mul(
        semiring.mul(head_transitions, last_transitions),
        semiring.mul(emissions, transitions),
    )

# def crf_partitions(emissions: Sequence, transitions: Transitions, semiring: Type[Semiring]) -> Tensor:
#     if isinstance(emissions, CattedSequence):
#         emissions = pack_catted_sequence(emissions)
#     elif isinstance(emissions, tuple) and len(emissions) == 2:
#         emissions = pack_padded_sequence(*emissions, batch_first=True)
#
#     assert isinstance(emissions, PackedSequence)
#
#     emissions, batch_sizes, _, unsorted_indices = emissions
#     transitions, head_transitions, last_transitions = transitions
#
#     last_indices = last_packed_indices(
#         batch_sizes=batch_sizes,
#         unsorted_indices=unsorted_indices,
#         device=emissions.device,
#     )
#
#     _, *batch_sizes = sections = batch_sizes.detach().cpu().tolist()
#     emission, *emissions = torch.split(emissions, sections, dim=0)
#
#     charts = [semiring.mul(head_transitions, emission)]
#     for emission, batch_size in zip(emissions, batch_sizes):
#         charts.append(semiring.mul(
#             semiring.bmm(charts[-1][:batch_size], transitions),
#             emission,
#         ))
#
#     emission = torch.cat(charts, dim=0)[last_indices]
#     return semiring.sum(semiring.mul(emission, last_transitions), dim=-1)
#
#
# class CrfDistribution(StructuredDistribution):
#     def __init__(self, emissions: Sequence, transitions: Transitions) -> None:
#         super(CrfDistribution, self).__init__()
#         self.emissions = emissions
#         self.transitions = transitions
#
#     def log_scores(self, targets: Sequence) -> Tensor:
#         return crf_scores(
#             emissions=self.emissions,
#             targets=targets,
#             transitions=self.transitions,
#             semiring=Log,
#         )
#
#     @lazy_property
#     def log_partitions(self) -> Tensor:
#         return crf_partitions(
#             emissions=self.emissions,
#             transitions=self.transitions,
#             semiring=Log,
#         )
#
#     @lazy_property
#     def max(self) -> Tensor:
#         return crf_partitions(
#             emissions=self.emissions,
#             transitions=self.transitions,
#             semiring=Max,
#         )
#
#
# class CrfDecoder(nn.Module):
#     def __init__(self, *, num_targets: int) -> None:
#         super(CrfDecoder, self).__init__()
#
#         self.num_targets = num_targets
#
#         self.transitions = nn.Parameter(torch.empty((num_targets, num_targets)))
#         self.head_transitions = nn.Parameter(torch.empty((num_targets,)))
#         self.last_transitions = nn.Parameter(torch.empty((num_targets,)))
#
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         init.zeros_(self.transitions)
#         init.zeros_(self.head_transitions)
#         init.zeros_(self.last_transitions)
#
#     def extra_repr(self) -> str:
#         return f'num_targets={self.num_targets}'
#
#     def forward(self, emissions: Sequence) -> CrfDistribution:
#         return CrfDistribution(
#             emissions=emissions,
#             transitions=(
#                 self.transitions,
#                 self.head_transitions,
#                 self.last_transitions,
#             ),
#         )
