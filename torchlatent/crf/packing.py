from typing import Type

import torch
from torch import Tensor, autograd
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence
from torchrua import head_packed_indices, ReductionIndices
from torchrua import roll_packed_sequence, head_packed_sequence, last_packed_sequence, major_sizes_to_ptr

from torchlatent.semiring import Semiring, Log, Max

__all__ = [
    'compute_packed_sequence_scores',
    'compute_packed_sequence_partitions',
    'PackedCrfDistribution',
]


def compute_packed_sequence_scores(semiring: Type[Semiring]):
    def _compute_packed_sequence_scores(
            emissions: PackedSequence, tags: PackedSequence,
            transitions: Tensor, head_transitions: Tensor, last_transitions: Tensor) -> Tensor:
        device = transitions.device

        emission_scores = emissions.data.gather(dim=-1, index=tags.data[..., None])[..., 0]  # [t, c]

        h = emissions.batch_sizes[0].item()
        t = torch.arange(transitions.size()[0], device=device)  # [t]
        c = torch.arange(transitions.size()[1], device=device)  # [c]

        x, y = roll_packed_sequence(tags, shifts=1).data, tags.data  # [t, c]
        head = head_packed_sequence(tags, unsort=False)  # [h, c]
        last = last_packed_sequence(tags, unsort=False)  # [h, c]

        transition_scores = transitions[t[:, None], c[None, :], x, y]  # [t, c]
        transition_head_scores = head_transitions[t[:h, None], c[None, :], head]  # [h, c]
        transition_last_scores = last_transitions[t[:h, None], c[None, :], last]  # [h, c]

        indices = head_packed_indices(tags.batch_sizes)
        transition_scores[indices] = transition_head_scores  # [h, c]

        batch_ptr, _ = major_sizes_to_ptr(sizes=emissions.batch_sizes)
        scores = semiring.mul(emission_scores, transition_scores)
        scores = semiring.scatter_mul(scores, index=batch_ptr)

        scores = semiring.mul(scores, transition_last_scores)

        if emissions.unsorted_indices is not None:
            scores = scores[emissions.unsorted_indices]

        return scores

    return _compute_packed_sequence_scores


def compute_packed_sequence_partitions(semiring: Type[Semiring]):
    def _compute_packed_sequence_partitions(
            emissions: PackedSequence, indices: ReductionIndices,
            transitions: Tensor, head_transitions: Tensor, last_transitions: Tensor, eye: Tensor) -> Tensor:
        h = emissions.batch_sizes[0].item()
        t = torch.arange(transitions.size()[0], device=transitions.device)  # [t]
        c = torch.arange(transitions.size()[1], device=transitions.device)  # [c]

        emission_scores = semiring.mul(transitions, emissions.data[..., None, :])  # [t, c, n, n]
        emission_scores[:h] = eye[None, None, :, :]
        emission_scores = semiring.reduce(tensor=emission_scores, indices=indices)

        emission_head_scores = emissions.data[:h, :, None, :]
        transition_head_scores = head_transitions[t[:h, None], c[None, :], None, :]
        transition_last_scores = last_transitions[t[:h, None], c[None, :], :, None]

        scores = semiring.mul(transition_head_scores, emission_head_scores)
        scores = semiring.bmm(scores, emission_scores)
        scores = semiring.bmm(scores, transition_last_scores)[..., 0, 0]

        if emissions.unsorted_indices is not None:
            scores = scores[emissions.unsorted_indices]
        return scores

    return _compute_packed_sequence_partitions


class PackedCrfDistribution(object):
    def __init__(self, emissions: PackedSequence, indices: ReductionIndices,
                 transitions: Tensor, head_transitions: Tensor, last_transitions: Tensor) -> None:
        super(PackedCrfDistribution, self).__init__()
        self.emissions = emissions
        self.indices = indices

        self.transitions = transitions
        self.head_transitions = head_transitions
        self.last_transitions = last_transitions

    def semiring_scores(self, semiring: Type[Semiring], tags: PackedSequence) -> Tensor:
        return compute_packed_sequence_scores(semiring=semiring)(
            emissions=self.emissions, tags=tags,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            last_transitions=self.last_transitions,
        )

    def semiring_partitions(self, semiring: Type[Semiring]) -> Tensor:
        return compute_packed_sequence_partitions(semiring=semiring)(
            emissions=self.emissions, indices=self.indices,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            last_transitions=self.last_transitions,
            eye=semiring.eye_like(self.transitions),
        )

    def log_prob(self, tags: PackedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: PackedSequence) -> Tensor:
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
    def argmax(self) -> PackedSequence:
        max_partitions = self.semiring_partitions(semiring=Max)

        grad, = torch.autograd.grad(
            max_partitions, self.emissions.data, torch.ones_like(max_partitions),
            retain_graph=False, create_graph=False, allow_unused=False,
        )
        return PackedSequence(
            data=grad.argmax(dim=-1),
            batch_sizes=self.emissions.batch_sizes,
            sorted_indices=self.emissions.sorted_indices,
            unsorted_indices=self.emissions.unsorted_indices,
        )
