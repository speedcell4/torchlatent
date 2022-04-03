from typing import Type

import torch
from torch import Tensor, autograd
from torch.distributions.utils import lazy_property
from torchrua import CattedSequence
from torchrua import ReductionIndices, head_catted_indices

from torchlatent.crf2 import crf_segment_reduce, crf_partition
from torchlatent.semiring import Semiring, Log, Max

__all__ = [
    'compute_catted_sequence_scores',
    'compute_catted_sequence_partitions',
    'CattedCrfDistribution',
]


def compute_catted_sequence_scores(semiring: Type[Semiring]):
    def _compute_catted_sequence_scores(
            emissions: CattedSequence, tags: CattedSequence,
            transitions: Tensor, head_transitions: Tensor, last_transitions: Tensor) -> Tensor:
        return crf_segment_reduce(
            emissions=emissions, targets=tags,
            transitions=(transitions, head_transitions, last_transitions),
            semiring=semiring,
        )

    return _compute_catted_sequence_scores


def compute_catted_sequence_partitions(semiring: Type[Semiring]):
    def _compute_catted_sequence_partitions(
            emissions: CattedSequence, indices: ReductionIndices,
            transitions: Tensor, head_transitions: Tensor, last_transitions: Tensor, eye: Tensor) -> Tensor:
        return crf_partition(
            emissions=emissions, indices=indices,
            transitions=(transitions, head_transitions, last_transitions),
            semiring=semiring,
        )

    return _compute_catted_sequence_partitions


class CattedCrfDistribution(object):
    def __init__(self, emissions: CattedSequence, indices: ReductionIndices,
                 transitions: Tensor, head_transitions: Tensor, last_transitions: Tensor) -> None:
        super(CattedCrfDistribution, self).__init__()
        self.emissions = emissions
        self.indices = indices

        self.transitions = transitions
        self.head_transitions = head_transitions
        self.last_transitions = last_transitions

    def semiring_scores(self, semiring: Type[Semiring], tags: CattedSequence) -> Tensor:
        return compute_catted_sequence_scores(semiring=semiring)(
            emissions=self.emissions, tags=tags,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            last_transitions=self.last_transitions,
        )

    def semiring_partitions(self, semiring: Type[Semiring]) -> Tensor:
        return compute_catted_sequence_partitions(semiring=semiring)(
            emissions=self.emissions, indices=self.indices,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            last_transitions=self.last_transitions,
            eye=semiring.eye_like(self.transitions),
        )

    def log_prob(self, tags: CattedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: CattedSequence) -> Tensor:
        return self.semiring_scores(semiring=Log, tags=tags)

    @lazy_property
    def log_partitions(self) -> Tensor:
        return self.semiring_partitions(semiring=Log)

    @lazy_property
    def marginals(self) -> Tensor:
        log_partitions = self.log_partitions
        grad, = autograd.grad(
            log_partitions, self.emissions.data, torch.ones_like(log_partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        return grad

    @lazy_property
    def argmax(self) -> CattedSequence:
        max_partitions = self.semiring_partitions(semiring=Max)

        grad, = torch.autograd.grad(
            max_partitions, self.emissions.data, torch.ones_like(max_partitions),
            retain_graph=False, create_graph=False, allow_unused=False,
        )
        return CattedSequence(
            data=grad.argmax(dim=-1),
            token_sizes=self.emissions.token_sizes,
        )
