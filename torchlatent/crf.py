from functools import singledispatch
from typing import NamedTuple, Union
from typing import Tuple
from typing import Type

import torch
from torch import Tensor
from torch import nn
from torch.distributions.utils import lazy_property
from torch.nn import init
from torch.types import Device

from torchlatent.abc import DistributionABC
from torchlatent.semiring import Semiring, Log, Max
from torchrua import ReductionIndices, accumulate_sizes
from torchrua import head_catted_indices, last_catted_indices, reduce_catted_indices
from torchrua import head_packed_indices, last_packed_indices, reduce_packed_indices
from torchrua import roll_catted_indices, cat_packed_indices, CattedSequence, PackedSequence

Sequence = Union[CattedSequence, PackedSequence]


class CrfIndices(NamedTuple):
    head: Tensor
    last: Tensor
    prev: Tensor
    curr: Tensor
    token_sizes: Tensor
    unsorted_indices: Tensor
    indices: ReductionIndices


@singledispatch
def broadcast_shapes(sequence: Sequence, transitions: Tuple[Tensor, Tensor, Tensor]) -> Sequence:
    raise TypeError(f'type {type(sequence)} is not supported')


@broadcast_shapes.register
def broadcast_catted_shapes(sequence: CattedSequence, transitions: Tuple[Tensor, Tensor, Tensor]):
    sequence, token_sizes = sequence
    transitions, head_transitions, last_transitions = transitions

    t1, c1, *_ = sequence.size()
    h1, = token_sizes.size()

    t2, c2, _, _ = transitions.size()
    h3, c3, _ = head_transitions.size()
    h4, c4, _ = last_transitions.size()

    return torch.broadcast_shapes((t1, c1, h1), (t2, c2, 1), (1, c3, h3), (1, c4, h4))


@broadcast_shapes.register
def broadcast_packed_shapes(sequence: PackedSequence, transitions: Tuple[Tensor, Tensor, Tensor]):
    sequence, batch_sizes, _, _ = sequence
    transitions, head_transitions, last_transitions = transitions

    t1, c1, *_ = sequence.size()
    h1 = batch_sizes[0].item()

    t2, c2, _, _ = transitions.size()
    h3, c3, _ = head_transitions.size()
    h4, c4, _ = last_transitions.size()

    return torch.broadcast_shapes((t1, c1, h1), (t2, c2, 1), (1, c3, h3), (1, c4, h4))


@singledispatch
def crf_scores_indices(sequence: Sequence, device: Device = None):
    raise TypeError(f'type {type(sequence)} is not supported')


@crf_scores_indices.register
def crf_scores_catted_indices(sequence: CattedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    token_sizes = sequence.token_sizes.to(device=device)
    curr = torch.arange(token_sizes.sum().item(), device=device)
    unsorted_indices = torch.arange(token_sizes.size()[0], device=device)

    prev = roll_catted_indices(token_sizes=token_sizes, device=device, shifts=1)
    head = head_catted_indices(token_sizes=token_sizes, device=device)
    last = last_catted_indices(token_sizes=token_sizes, device=device)
    return head, last, prev, curr, token_sizes, unsorted_indices


@crf_scores_indices.register
def crf_scores_packed_indices(sequence: PackedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    unsorted_indices = sequence.unsorted_indices.to(device=device)
    curr, token_sizes = cat_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=device)

    prev = roll_catted_indices(token_sizes=token_sizes, device=device, shifts=1)
    head = head_packed_indices(batch_sizes=batch_sizes, device=device, unsorted_indices=unsorted_indices)
    last = last_packed_indices(batch_sizes=batch_sizes, device=device, unsorted_indices=unsorted_indices)
    return head, last, curr[prev], curr, token_sizes, unsorted_indices


def crf_scores(sequence: Sequence, emissions: Tensor,
               transitions: Tuple[Tensor, Tensor, Tensor], semiring: Type[Semiring]) -> Tensor:
    head, last, prev, curr, token_sizes, unsorted_indices = crf_scores_indices(sequence)

    sequence, *_ = sequence
    transitions, head_transitions, last_transitions = transitions
    c = torch.arange(transitions.size()[1], device=emissions.device)

    emissions = emissions[curr[:, None], c[None, :], sequence[curr]]
    transitions = transitions[curr[:, None], c[None, :], sequence[prev], sequence[curr]]
    transitions[accumulate_sizes(sizes=token_sizes)] = semiring.one
    head_transitions = head_transitions[unsorted_indices[:, None], c[None, :], sequence[head]]
    last_transitions = last_transitions[unsorted_indices[:, None], c[None, :], sequence[last]]

    emissions = semiring.segment_prod(semiring.mul(emissions, transitions), sizes=token_sizes)
    return semiring.mul(emissions, semiring.mul(head_transitions, last_transitions))


@torch.no_grad()
def crf_indices(emissions: Sequence) -> CrfIndices:
    head, last, prev, curr, token_sizes, unsorted_indices = crf_scores_indices(emissions)
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
        raise KeyError(f'type {type(emissions)} is not supported')

    return CrfIndices(
        head=head, last=last,
        prev=prev, curr=curr,
        token_sizes=token_sizes,
        unsorted_indices=unsorted_indices,
        indices=indices,
    )


def crf_partition(emissions: Tensor, transitions: Tuple[Tensor, Tensor, Tensor],
                  indices: CrfIndices, semiring: Type[Semiring]):
    head, _, _, _, _, unsorted_indices, indices = indices

    transitions, head_transitions, last_transitions = transitions
    c = torch.arange(transitions.size()[1], device=emissions.device)

    transitions = semiring.mul(emissions[:, :, None, :], transitions)
    transitions[head] = semiring.eye_like(transitions)[None, None, :, :]

    head_transitions = head_transitions[unsorted_indices[:, None], c[None, :], None, :]
    last_transitions = last_transitions[unsorted_indices[:, None], c[None, :], :, None]

    scores = semiring.mul(emissions[head[:, None], c[None, :], None, :], head_transitions)
    scores = semiring.bmm(scores, semiring.reduce(transitions, indices=indices))
    scores = semiring.bmm(scores, last_transitions)

    return scores[..., 0, 0]


class CrfDistribution(DistributionABC):
    def __init__(self, emissions: Tensor, transitions: Tuple[Tensor, Tensor, Tensor], indices: CrfIndices) -> None:
        super(CrfDistribution, self).__init__(validate_args=False)

        self.emissions = emissions
        self.indices = indices
        self.transitions = transitions

    def log_scores(self, targets: Sequence) -> Tensor:
        return crf_scores(
            emissions=self.emissions,
            sequence=targets,
            transitions=self.transitions,
            semiring=Log,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return crf_partition(
            emissions=self.emissions,
            transitions=self.transitions,
            indices=self.indices,
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return crf_partition(
            emissions=self.emissions,
            transitions=self.transitions,
            indices=self.indices,
            semiring=Max,
        )

    @lazy_property
    def argmax(self) -> Tensor:
        return super(CrfDistribution, self).argmax.argmax(dim=-1)

    @lazy_property
    def entropy(self) -> Tensor:
        tensor = (self.marginals * self.marginals.log()).sum(dim=-1)
        return -Log.segment_prod(
            tensor=tensor[self.indices.curr],
            sizes=self.indices.token_sizes,
        )


class CrfDecoderABC(nn.Module):
    def reset_parameters(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def forward_parameters(self, emissions: Sequence):
        raise NotImplementedError


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoder, self).__init__()

        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

        self.transitions = nn.Parameter(torch.empty((1, num_conjugates, num_tags, num_tags)))
        self.head_transitions = nn.Parameter(torch.empty((1, num_conjugates, num_tags)))
        self.last_transitions = nn.Parameter(torch.empty((1, num_conjugates, num_tags)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.transitions)
        init.zeros_(self.head_transitions)
        init.zeros_(self.last_transitions)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])

    def forward_parameters(self, emissions: Sequence):
        transitions = (self.transitions, self.head_transitions, self.last_transitions)
        t, c, h = broadcast_shapes(emissions, transitions=transitions)

        emissions = emissions.data.expand((t, c, -1))
        transitions = self.transitions.expand((t, c, -1, -1))
        head_transitions = self.head_transitions.expand((h, c, -1))
        last_transitions = self.last_transitions.expand((h, c, -1))

        return emissions, (transitions, head_transitions, last_transitions)

    def forward(self, emissions: Sequence, indices: CrfIndices = None) -> CrfDistribution:
        if indices is None:
            indices = crf_indices(emissions=emissions)

        emissions, transitions = self.forward_parameters(emissions=emissions)

        return CrfDistribution(emissions=emissions, transitions=transitions, indices=indices)

    def fit(self, emissions: Sequence, targets: Sequence, indices: CrfIndices = None) -> Tensor:
        dist = self.forward(emissions=emissions, indices=indices)
        return dist.log_prob(targets=targets).neg()

    def decode(self, emissions: Sequence, indices: CrfIndices = None) -> Sequence:
        dist = self.forward(emissions=emissions, indices=indices)
        return emissions._replace(data=dist.argmax)
