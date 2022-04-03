from typing import Sequence
from typing import Tuple
from typing import Type

import torch
import torchcrf
from torch import Tensor
from torch import nn
from torch.distributions.utils import lazy_property
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import reduce_catted_indices, reduce_packed_indices
from torchrua import roll_catted_indices, CattedSequence, head_catted_indices, last_catted_indices, head_packed_indices, \
    last_packed_indices, accumulate_sizes, ReductionIndices, pack_sequence, pad_sequence, pad_packed_indices

from torchlatent.abc import DistributionABC
from torchlatent.semiring import segment_catted_indices, segment_packed_indices, Semiring, Log, Max

CrfIndices = ReductionIndices


@torch.no_grad()
def broadcast_catted_shapes(sequence: CattedSequence, transitions: Tuple[Tensor, Tensor, Tensor]):
    sequence, token_sizes = sequence
    transitions, head_transitions, last_transitions = transitions

    t1, c1, *_ = sequence.size()
    h1, = token_sizes.size()

    t2, c2, _, _ = transitions.size()
    h3, c3, _ = head_transitions.size()
    h4, c4, _ = last_transitions.size()

    return torch.broadcast_shapes((t1, c1, h1), (t2, c2, 1), (1, c3, h3), (1, c4, h4))


@torch.no_grad()
def broadcast_packed_shapes(sequence: PackedSequence, transitions: Tuple[Tensor, Tensor, Tensor]):
    sequence, batch_sizes, _, _ = sequence
    transitions, head_transitions, last_transitions = transitions

    t1, c1, *_ = sequence.size()
    h1 = batch_sizes[0].item()

    t2, c2, _, _ = transitions.size()
    h3, c3, _ = head_transitions.size()
    h4, c4, _ = last_transitions.size()

    return torch.broadcast_shapes((t1, c1, h1), (t2, c2, 1), (1, c3, h3), (1, c4, h4))


@torch.no_grad()
def crf_segment_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    curr, _, token_sizes = segment_catted_indices(token_sizes=token_sizes, device=device)

    prev = roll_catted_indices(token_sizes=token_sizes, shifts=1, device=device)
    head = head_catted_indices(token_sizes=token_sizes, device=device)
    last = last_catted_indices(token_sizes=token_sizes, device=device)
    return prev, curr, torch.arange(token_sizes.size()[0], device=device), head, last, token_sizes


@torch.no_grad()
def crf_segment_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor, device: Device):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        else:
            device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    unsorted_indices = unsorted_indices.to(device=device)

    curr, _, token_sizes = segment_packed_indices(
        batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=device,
    )
    prev = roll_catted_indices(token_sizes=token_sizes, shifts=1, device=device)
    head = head_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=device)
    last = last_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=device)
    return curr[prev], curr, unsorted_indices, head, last, token_sizes


def crf_segment_reduce(emissions: Sequence, targets: Sequence,
                       transitions: Tuple[Tensor, Tensor, Tensor], semiring: Type[Semiring]) -> Tensor:
    if isinstance(emissions, CattedSequence):
        emissions, token_sizes = emissions
        prev, curr, unsorted_indices, head, last, sizes = crf_segment_catted_indices(
            token_sizes=token_sizes, device=emissions.device,
        )
    elif isinstance(emissions, PackedSequence):
        emissions, batch_sizes, _, unsorted_indices = emissions
        prev, curr, unsorted_indices, head, last, sizes = crf_segment_packed_indices(
            batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=emissions.device,
        )
    else:
        raise NotImplementedError

    if isinstance(targets, CattedSequence):
        t, c, h = broadcast_catted_shapes(targets, transitions=transitions)
        targets, _ = targets
    elif isinstance(targets, PackedSequence):
        t, c, h = broadcast_packed_shapes(targets, transitions=transitions)
        targets, _, _, _ = targets
    else:
        raise NotImplementedError

    transitions, head_transitions, last_transitions = transitions
    emissions = emissions.expand((t, c, -1))
    targets = targets.expand((t, c))
    transitions = transitions.expand((t, c, -1, -1))
    head_transitions = head_transitions.expand((h, c, -1))
    last_transitions = last_transitions.expand((h, c, -1))

    c = torch.arange(c, device=emissions.device)

    emissions = emissions[curr[:, None], c[None, :], targets[curr]]
    transitions = transitions[curr[:, None], c[None, :], targets[prev], targets[curr]]
    transitions[accumulate_sizes(sizes=sizes)] = semiring.one
    head_transitions = head_transitions[unsorted_indices[:, None], c[None, :], targets[head]]
    last_transitions = last_transitions[unsorted_indices[:, None], c[None, :], targets[last]]

    emissions = semiring.segment_prod(semiring.mul(emissions, transitions), sizes=sizes)
    return semiring.mul(emissions, semiring.mul(head_transitions, last_transitions))


def crf_partition(emissions: Sequence, indices: ReductionIndices,
                  transitions: Tuple[Tensor, Tensor, Tensor], semiring: Type[Semiring]):
    if isinstance(emissions, CattedSequence):
        t, c, h = broadcast_catted_shapes(emissions, transitions=transitions)
        emissions, token_sizes = emissions
        prev, curr, unsorted_indices, head, last, sizes = crf_segment_catted_indices(
            token_sizes=token_sizes, device=emissions.device,
        )
    elif isinstance(emissions, PackedSequence):
        t, c, h = broadcast_packed_shapes(emissions, transitions=transitions)
        emissions, batch_sizes, _, unsorted_indices = emissions
        prev, curr, unsorted_indices, head, last, sizes = crf_segment_packed_indices(
            batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=emissions.device,
        )
    else:
        raise NotImplementedError

    transitions, head_transitions, last_transitions = transitions
    emissions = emissions.expand((t, c, -1))
    transitions = transitions.expand((t, c, -1, -1))
    head_transitions = head_transitions.expand((h, c, -1))
    last_transitions = last_transitions.expand((h, c, -1))

    c = torch.arange(c, device=emissions.device)

    transitions = semiring.mul(emissions[:, :, None, :], transitions)
    transitions[head] = semiring.eye_like(transitions)[None, None, :, :]

    head_emissions = emissions[head[:, None], c[None, :], None, :]
    head_transitions = head_transitions[unsorted_indices[:, None], c[None, :], None, :]
    last_transitions = last_transitions[unsorted_indices[:, None], c[None, :], :, None]

    scores = semiring.mul(head_emissions, head_transitions)
    scores = semiring.bmm(scores, semiring.reduce(transitions, indices=indices))
    scores = semiring.bmm(scores, last_transitions)

    return scores[..., 0, 0]


class CrfDistribution(DistributionABC):
    def __init__(self, log_potentials: Sequence, indices: CrfIndices,
                 transitions: Tuple[Tensor, Tensor, Tensor]) -> None:
        super(CrfDistribution, self).__init__(validate_args=False)

        self.log_potentials = log_potentials
        self.indices = indices
        self.transitions = transitions

    def log_scores(self, targets: Sequence) -> Tensor:
        return crf_segment_reduce(
            emissions=self.log_potentials,
            targets=targets,
            transitions=self.transitions,
            semiring=Log,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return crf_partition(
            emissions=self.log_potentials,
            indices=self.indices,
            transitions=self.transitions,
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return crf_partition(
            emissions=self.log_potentials,
            indices=self.indices,
            transitions=self.transitions,
            semiring=Max,
        )

    @lazy_property
    def entropy(self) -> Tensor:
        raise NotImplementedError


class CrfDecoder(nn.Module):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoder, self).__init__()

        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

        self.transitions = nn.Parameter(torch.empty((1, num_conjugates, num_tags, num_tags)))
        self.head_transitions = nn.Parameter(torch.empty((1, num_conjugates, num_tags)))
        self.last_transitions = nn.Parameter(torch.empty((1, num_conjugates, num_tags)))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        init.zeros_(self.transitions)
        init.zeros_(self.head_transitions)
        init.zeros_(self.last_transitions)

    def forward(self, emissions: Sequence, indices: CrfIndices = None) -> CrfDistribution:
        if indices is None:
            if isinstance(emissions, CattedSequence):
                indices = reduce_catted_indices(
                    token_sizes=emissions.token_sizes,
                    device=emissions.data.device,
                )
            elif isinstance(emissions, PackedSequence):

                indices = reduce_packed_indices(
                    batch_sizes=emissions.batch_sizes,
                    unsorted_indices=emissions.unsorted_indices,
                    device=emissions.data.device,
                )
            else:
                raise NotImplementedError

        return CrfDistribution(
            log_potentials=emissions,
            indices=indices,
            transitions=(self.transitions, self.head_transitions, self.last_transitions),
        )


if __name__ == '__main__':
    num_tags = 3

    decoder1 = CrfDecoder(num_tags)
    decoder2 = torchcrf.CRF(num_tags, batch_first=False)

    decoder1.transitions.data = decoder2.transitions[None, None, :, :]
    decoder1.head_transitions.data = decoder2.start_transitions[None, None, :]
    decoder1.last_transitions.data = decoder2.end_transitions[None, None, :]

    sequence = [
        torch.randn((5, num_tags), requires_grad=True),
        torch.randn((2, num_tags), requires_grad=True),
        torch.randn((3, num_tags), requires_grad=True),
    ]

    token_sizes = torch.tensor([5, 2, 3])

    e1 = pack_sequence([s[:, None, :] for s in sequence])

    e2, _ = pad_sequence(sequence, batch_first=False)
    size, ptr, _ = pad_packed_indices(
        e1.batch_sizes, False, e1.sorted_indices, e1.unsorted_indices
    )
    mask = torch.zeros(size, dtype=torch.bool)
    mask[ptr] = True

    dist = decoder1.forward(e1)
    lhs = dist.log_partitions[:, 0]
    rhs = decoder2._compute_normalizer(e2, mask)
    print(f'lhs => {lhs}')
    print(f'rhs => {rhs}')

    print(torch.allclose(lhs, rhs))
