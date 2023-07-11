from functools import singledispatch
from typing import NamedTuple
from typing import Tuple
from typing import Type
from typing import Union

import torch
from torch import nn
from torch import Tensor
from torch.distributions.utils import lazy_property
from torch.nn import functional as F
from torch.nn import init
from torch.types import Device
from torchrua import accumulate_sizes
from torchrua import CattedSequence
from torchrua import minor_sizes_to_ptr
from torchrua import PackedSequence
from torchrua import reduce_catted_indices
from torchrua import reduce_packed_indices
from torchrua import ReductionIndices
from torchrua import RuaSequential

from torchlatent.abc import DistributionABC
from torchlatent.nn.classifier import Classifier
from torchlatent.semiring import Log
from torchlatent.semiring import Max
from torchlatent.semiring import Semiring

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
    acc_token_sizes = token_sizes.cumsum(dim=0)

    index = torch.arange(token_sizes.sum().item(), device=device)
    unsorted_indices = torch.arange(token_sizes.size()[0], device=device)

    return F.pad(acc_token_sizes, [1, -1]), acc_token_sizes - 1, index - 1, index, token_sizes, unsorted_indices


@crf_scores_indices.register
def crf_scores_packed_indices(sequence: PackedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    unsorted_indices = sequence.unsorted_indices.to(device=device)
    acc_batch_sizes = F.pad(batch_sizes.cumsum(dim=0), [2, -1])

    batch_ptr, token_ptr, token_sizes = minor_sizes_to_ptr(
        sizes=batch_sizes, minor_ptr=unsorted_indices,
    )
    prev = acc_batch_sizes[token_ptr + 0] + batch_ptr
    curr = acc_batch_sizes[token_ptr + 1] + batch_ptr
    last = acc_batch_sizes[token_sizes] + unsorted_indices

    return unsorted_indices, last, prev, curr, token_sizes, unsorted_indices


def crf_scores(sequence: Sequence, emissions: Tensor, transitions: Tuple[Tensor, Tensor, Tensor],
               semiring: Type[Semiring]) -> Tensor:
    head, last, prev, curr, token_sizes, unsorted_indices = crf_scores_indices(sequence)

    transitions, head_transitions, last_transitions = transitions
    c = torch.arange(transitions.size()[1], device=emissions.device)

    emissions = emissions[curr[:, None], c[None, :], sequence.data[curr]]
    transitions = transitions[curr[:, None], c[None, :], sequence.data[prev], sequence.data[curr]]
    transitions[accumulate_sizes(sizes=token_sizes)] = semiring.one
    head_transitions = head_transitions[unsorted_indices[:, None], c[None, :], sequence.data[head]]
    last_transitions = last_transitions[unsorted_indices[:, None], c[None, :], sequence.data[last]]

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


def crf_partitions(emissions: Tensor, transitions: Tuple[Tensor, Tensor, Tensor],
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
        return crf_partitions(
            emissions=self.emissions,
            transitions=self.transitions,
            indices=self.indices,
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return crf_partitions(
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


class CrfLayerABC(nn.Module):
    def reset_parameters(self) -> None:
        raise NotImplementedError

    def forward_parameters(self, emissions: Sequence):
        raise NotImplementedError


class CrfLayer(CrfLayerABC):
    def __init__(self, num_targets: int, num_conjugates: int = 1) -> None:
        super(CrfLayer, self).__init__()

        self.num_targets = num_targets
        self.num_conjugates = num_conjugates

        self.transitions = nn.Parameter(torch.empty((1, num_conjugates, num_targets, num_targets)))
        self.head_transitions = nn.Parameter(torch.empty((1, num_conjugates, num_targets)))
        self.last_transitions = nn.Parameter(torch.empty((1, num_conjugates, num_targets)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.transitions)
        init.zeros_(self.head_transitions)
        init.zeros_(self.last_transitions)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_targets={self.num_targets}',
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
        dist: CrfDistribution = self.forward(emissions=emissions, indices=indices)
        return dist.log_partitions - dist.log_scores(targets=targets)

    def decode(self, emissions: Sequence, indices: CrfIndices = None) -> Sequence:
        dist: CrfDistribution = self.forward(emissions=emissions, indices=indices)
        return emissions._replace(data=dist.argmax)


class CrfDecoder(nn.Module):
    def __init__(self, in_features: int, num_targets: int, num_conjugates: int, dropout: float) -> None:
        super(CrfDecoder, self).__init__()

        self.in_features = in_features
        self.num_targets = num_targets
        self.num_conjugates = num_conjugates
        num_conjugates = max(1, num_conjugates)

        self.classifier = RuaSequential(
            nn.Dropout(dropout),
            Classifier(
                num_conjugates=num_conjugates,
                in_features=in_features,
                out_features=num_targets,
                bias=False,
            )
        )

        self.crf = CrfLayer(
            num_targets=num_targets,
            num_conjugates=num_conjugates,
        )

    def forward(self, sequence: Sequence) -> CrfDistribution:
        if self.num_conjugates == 0:
            sequence = sequence._replace(data=sequence.data[..., None, :])

        emissions = self.classifier(sequence)
        return self.crf(emissions)

    def fit(self, sequence: Sequence, targets: Sequence) -> Tensor:
        dist: CrfDistribution = self(sequence=sequence)
        loss = dist.log_partitions - dist.log_scores(targets=targets)

        if self.num_conjugates == 0:
            loss = loss[..., 0]
        return loss

    def decode(self, sequence: Sequence) -> Sequence:
        dist: CrfDistribution = self(sequence=sequence)
        argmax = dist.argmax

        if self.num_conjugates == 0:
            argmax = argmax[..., 0]
        return sequence._replace(data=argmax)
