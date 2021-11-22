from typing import Type

import torch
from torch import Tensor, autograd
from torch.distributions.utils import lazy_property
from torchrua import CattedSequence
from torchrua import roll_catted_sequence, head_catted_sequence, last_catted_sequence, batch_sizes_to_ptr, \
    TreeReduceIndices

from torchlatent.semiring import Semiring, Log, Max

__all__ = [
    'compute_catted_sequence_scores',
    'compute_catted_sequence_partitions',
    'CattedCrfDistribution',
]


def compute_catted_sequence_scores(semiring: Type[Semiring]):
    def _compute_catted_sequence_scores(
            emissions: CattedSequence, tags: CattedSequence,
            transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor) -> Tensor:
        device = transitions.device

        emission_scores = emissions.data.gather(dim=-1, index=tags.data[..., None])[..., 0]  # [t, c]

        h = emissions.token_sizes.size()[0]
        t = torch.arange(transitions.size()[0], device=device)  # [t]
        c = torch.arange(transitions.size()[1], device=device)  # [c]

        x, y = roll_catted_sequence(tags, shifts=1).data, tags.data  # [t, c]
        head = head_catted_sequence(tags)  # [h, c]
        tail = last_catted_sequence(tags)  # [h, c]

        transition_scores = transitions[t[:, None], c[None, :], x, y]  # [t, c]
        transition_head_scores = head_transitions[t[:h, None], c[None, :], head]  # [h, c]
        transition_tail_scores = tail_transitions[t[:h, None], c[None, :], tail]  # [h, c]

        transition_scores[:h] = transition_head_scores  # [h, c]

        _, batch_ptr, _ = batch_sizes_to_ptr(batch_sizes=emissions.token_sizes)
        scores = semiring.mul(emission_scores, transition_scores)
        scores = semiring.scatter_mul(scores, index=batch_ptr)
        scores = semiring.mul(scores, transition_tail_scores)

        return scores

    return _compute_catted_sequence_scores


def compute_catted_sequence_partitions(semiring: Type[Semiring]):
    def _compute_catted_sequence_partitions(
            emissions: CattedSequence, indices: TreeReduceIndices,
            transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor, eye: Tensor) -> Tensor:
        h = emissions.token_sizes.size()[0]
        t = torch.arange(transitions.size()[0], device=transitions.device)  # [t]
        c = torch.arange(transitions.size()[1], device=transitions.device)  # [c]

        emission_scores = semiring.mul(transitions, emissions.data[..., None, :])  # [t, c, n, n]
        emission_scores[:h] = eye[None, None, :, :]
        emission_scores = semiring.reduce(tensor=emission_scores, indices=indices)

        emission_head_scores = emissions.data[:h, :, None, :]
        transition_head_scores = head_transitions[t[:h, None], c[None, :], None, :]
        transition_tail_scores = tail_transitions[t[:h, None], c[None, :], :, None]

        scores = semiring.mul(transition_head_scores, emission_head_scores)
        scores = semiring.bmm(scores, emission_scores)
        scores = semiring.bmm(scores, transition_tail_scores)[..., 0, 0]

        return scores

    return _compute_catted_sequence_partitions


class CattedCrfDistribution(object):
    def __init__(self, emissions: CattedSequence, indices: TreeReduceIndices,
                 transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor) -> None:
        super(CattedCrfDistribution, self).__init__()
        self.emissions = emissions
        self.indices = indices

        self.transitions = transitions
        self.head_transitions = head_transitions
        self.tail_transitions = tail_transitions

    def semiring_scores(self, semiring: Type[Semiring], tags: CattedSequence) -> Tensor:
        return compute_catted_sequence_scores(semiring=semiring)(
            emissions=self.emissions, tags=tags,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            tail_transitions=self.tail_transitions,
        )

    def semiring_partitions(self, semiring: Type[Semiring]) -> Tensor:
        return compute_catted_sequence_partitions(semiring=semiring)(
            emissions=self.emissions, indices=self.indices,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            tail_transitions=self.tail_transitions,
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