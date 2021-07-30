from abc import ABCMeta
from typing import Optional, Type, Tuple

import torch
from torch import Tensor
from torch import nn, autograd
from torch.distributions.utils import lazy_property
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torchrua import TreeReduceIndices, tree_reduce_packed_indices
from torchrua import select_head, select_last, roll_packed_sequence, pad_packed_sequence

from torchlatent.semiring import Semiring, Log, Max

__all__ = [
    'compute_scores',
    'compute_partitions',
    'CrfDistribution',
    'CrfDecoderABC', 'CrfDecoder',
]


def compute_scores(semiring: Type[Semiring]):
    def _compute_scores(
            emissions: PackedSequence, tags: PackedSequence,
            transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor) -> Tensor:
        device = transitions.device

        emission_scores = emissions.data.gather(dim=-1, index=tags.data[..., None])[..., 0]  # [t, c]

        h = emissions.batch_sizes[0].item()
        t = torch.arange(transitions.size()[0], device=device)  # [t]
        c = torch.arange(transitions.size()[1], device=device)  # [c]

        x, y = roll_packed_sequence(tags, shifts=1).data, tags.data  # [t, c]
        head = select_head(tags, unsort=False)  # [h, c]
        tail = select_last(tags, unsort=False)  # [h, c]

        transition_scores = transitions[t[:, None], c[None, :], x, y]  # [t, c]
        transition_head_scores = head_transitions[t[:h, None], c[None, :], head]  # [h, c]
        transition_tail_scores = tail_transitions[t[:h, None], c[None, :], tail]  # [h, c]

        transition_scores[:h] = transition_head_scores  # [h, c]
        scores, _ = pad_packed_sequence(
            PackedSequence(
                data=semiring.mul(emission_scores, transition_scores),
                batch_sizes=emissions.batch_sizes,
                sorted_indices=None,
                unsorted_indices=None,
            ),
            batch_first=False,
        )
        scores = semiring.prod(scores, dim=0)
        scores = semiring.mul(scores, transition_tail_scores)

        if emissions.unsorted_indices is not None:
            scores = scores[emissions.unsorted_indices]

        return scores

    return _compute_scores


def compute_partitions(semiring: Type[Semiring]):
    def _compute_partitions(
            emissions: PackedSequence, indices: TreeReduceIndices,
            transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor, eye: Tensor) -> Tensor:
        h = emissions.batch_sizes[0].item()
        t = torch.arange(transitions.size()[0], device=transitions.device)  # [t]
        c = torch.arange(transitions.size()[1], device=transitions.device)  # [c]

        scores = semiring.mul(transitions, emissions.data[..., None, :])  # [t, c, n, n]
        scores[:h] = eye[None, None, :, :]
        scores = semiring.reduce(tensor=scores, indices=indices)

        emission_head_scores = emissions.data[:h, :, None, :]
        transition_head_scores = head_transitions[t[:h, None], c[None, :], None, :]
        transition_tail_scores = tail_transitions[t[:h, None], c[None, :], :, None]

        scores = semiring.bmm(
            semiring.bmm(semiring.mul(transition_head_scores, emission_head_scores), scores),
            transition_tail_scores,
        )[..., 0, 0]

        if emissions.unsorted_indices is not None:
            scores = scores[emissions.unsorted_indices]
        return scores

    return _compute_partitions


class CrfDistribution(object):
    def __init__(self, emissions: PackedSequence, indices: TreeReduceIndices,
                 transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.indices = indices

        self.transitions = transitions
        self.head_transitions = head_transitions
        self.tail_transitions = tail_transitions

    def semiring_scores(self, semiring: Type[Semiring], tags: PackedSequence) -> Tensor:
        return compute_scores(semiring=semiring)(
            emissions=self.emissions, tags=tags,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            tail_transitions=self.tail_transitions,
        )

    def semiring_partitions(self, semiring: Type[Semiring]) -> Tensor:
        return compute_partitions(semiring=semiring)(
            emissions=self.emissions, indices=self.indices,
            transitions=self.transitions,
            head_transitions=self.head_transitions,
            tail_transitions=self.tail_transitions,
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


class CrfDecoderABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_tags: int, num_conjugates: int):
        super(CrfDecoderABC, self).__init__()

        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

    def reset_parameters(self) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])

    def compile_indices(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                        indices: Optional[TreeReduceIndices] = None, **kwargs):
        assert emissions.data.dim() == 3, f'{emissions.data.dim()} != {3}'
        if tags is not None:
            assert tags.data.dim() == 2, f'{tags.data.dim()} != {2}'

        if indices is None:
            batch_sizes = emissions.batch_sizes.to(device=emissions.data.device)
            indices = tree_reduce_packed_indices(batch_sizes=batch_sizes)

        return indices

    def obtain_parameters(self, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return self.transitions, self.head_transitions, self.tail_transitions

    def forward(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                indices: Optional[TreeReduceIndices] = None, **kwargs):
        indices = self.compile_indices(emissions=emissions, tags=tags, indices=indices)
        transitions, head_transitions, tail_transitions = self.obtain_parameters(
            emissions=emissions, tags=tags, indices=indices,
        )

        dist = CrfDistribution(
            emissions=emissions, indices=indices,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=tail_transitions,
        )

        return dist, tags

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            indices: Optional[TreeReduceIndices] = None, **kwargs) -> Tensor:
        dist, tags = self(emissions=emissions, tags=tags, instr=indices, **kwargs)

        return dist.log_prob(tags=tags)

    def decode(self, emissions: PackedSequence,
               indices: Optional[TreeReduceIndices] = None, **kwargs) -> PackedSequence:
        dist, _ = self(emissions=emissions, tags=None, instr=indices, **kwargs)
        return dist.argmax

    def marginals(self, emissions: PackedSequence,
                  indices: Optional[TreeReduceIndices] = None, **kwargs) -> Tensor:
        dist, _ = self(emissions=emissions, tags=None, instr=indices, **kwargs)
        return dist.marginals


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoder, self).__init__(num_tags=num_tags, num_conjugates=num_conjugates)

        self.transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags, self.num_tags)),
            requires_grad=True,
        )
        self.head_transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags)),
            requires_grad=True,
        )
        self.tail_transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags)),
            requires_grad=True,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        bound = 0.01
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.head_transitions, -bound, +bound)
        init.uniform_(self.tail_transitions, -bound, +bound)
