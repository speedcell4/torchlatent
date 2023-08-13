from typing import Tuple
from typing import Type
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import CattedSequence
from torchrua import last_packed_indices
from torchrua import pack_catted_sequence
from torchrua import pack_padded_sequence

from torchlatent.semiring import Semiring

Sequence = Union[CattedSequence, PackedSequence, Tuple[Tensor, Tensor]]
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
