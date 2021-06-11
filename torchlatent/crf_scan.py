from abc import ABCMeta
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torchrua import reversed_indices, select_last

from torchlatent.crf import compute_log_scores
from torchlatent.instr import BatchedInstr
from torchlatent.semiring import log, max
from torchlatent.utils import broadcast_packed_sequences


def scan_scores(semiring):
    def _scan_scores(emissions: PackedSequence, indices: Tensor,
                     transitions: Tensor, head_transitions: Tensor) -> Tensor:
        """

        Args:
            emissions: [t1, c1, n]
            indices: [t1]
            transitions: [t2, c2, n, n]
            head_transitions: [t2, c2, n]

        Returns:
            [t, c, n]
        """

        emissions, _, transitions, head_transitions, _, (t, c, n, h) = broadcast_packed_sequences(
            emissions=emissions, tags=None,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=head_transitions,
        )

        data = torch.empty(
            (t, c, 1, emissions.data.size()[-1]),
            dtype=emissions.data.dtype, device=emissions.data.device, requires_grad=False)
        data[indices[:h]] = head_transitions[:, :, None, :]

        start, end = 0, h
        for h in emissions.batch_sizes.detach().cpu().tolist()[1:]:
            last_start, last_end, start, end = start, start + h, end, end + h
            data[indices[start:end]] = semiring.bmm(
                semiring.mul(
                    data[indices[last_start:last_end]],
                    emissions.data[indices[last_start:last_end], :, None],
                ),
                transitions[indices[start:end]],
            )

        return data[..., 0, :]

    return _scan_scores


scan_log_scores = scan_scores(log)


def compute_marginals(semiring, scan_semi_scores):
    def _compute_marginals(emissions: PackedSequence, transitions: Tensor,
                           head_transitions: Tensor, tail_transitions: Tensor) -> Tensor:
        alpha = scan_semi_scores(
            emissions._replace(data=emissions.data),
            torch.arange(emissions.data.size(0)),
            transitions,
            head_transitions,
        )

        beta = scan_semi_scores(
            emissions._replace(data=emissions.data),
            reversed_indices(emissions),
            transitions.transpose(-2, -1),
            tail_transitions,
        )

        return semiring.prod(torch.stack([
            alpha, beta, emissions.data
        ], dim=-1), dim=-1)

    return _compute_marginals


compute_log_marginals = compute_marginals(log, scan_log_scores)


def scan_partitions(semiring):
    def _scan_partitions(emissions: PackedSequence, transitions: Tensor,
                         head_transitions: Tensor, tail_transitions: Tensor) -> Tensor:
        """

        Args:
            emissions: [t1, c1, n]
            transitions: [t2, c2, n, n]
            head_transitions: [t2, c2, n]
            tail_transitions: [t2, c2, n]

        Returns:
            [t, c, n]
        """

        h = emissions.batch_sizes[0].item()

        scores = semiring.mul(emissions.data[:, :, None, :], transitions)
        data = torch.empty_like(scores[:, :, :1, :], requires_grad=False)

        index = torch.arange(data.size()[0], dtype=torch.long, device=data.device)
        data[index[:h]] = semiring.mul(
            emissions.data[index[:h], :, None, :],
            head_transitions[:, :, None, :],
        )

        start, end = 0, h
        for h in emissions.batch_sizes.detach().cpu().tolist()[1:]:
            last_start, last_end, start, end = start, start + h, end, end + h
            prev_index, curr_index = index[start:end], index[last_start:last_end]
            data[prev_index] = semiring.bmm(data[curr_index], scores[prev_index])

        data = select_last(emissions._replace(data=data), unsort=False)
        data = semiring.bmm(data, tail_transitions[..., None])[..., 0, 0]

        if emissions.unsorted_indices is not None:
            data = data[emissions.unsorted_indices]
        return data

    return _scan_partitions


scan_log_partitions = scan_partitions(log)
scan_max_partitions = scan_partitions(max)


class CrfDecoderScanABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_tags: int, num_conjugates: int) -> None:
        super(CrfDecoderScanABC, self).__init__()

        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

    def reset_parameters(self, bound: float = 0.01) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])

    @staticmethod
    def _validate(emissions: PackedSequence, tags: Optional[PackedSequence], instr: Optional[BatchedInstr]):
        return emissions, tags, None

    def obtain_parameters(self, *args, **kwargs):
        return self.transitions, self.head_transitions, self.tail_transitions

    def forward(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                instr: Optional[BatchedInstr] = None):
        emissions, tags, instr = self._validate(emissions=emissions, tags=tags, instr=instr)
        transitions, head_transitions, tail_transitions = self.obtain_parameters(
            emissions=emissions, tags=tags, instr=instr)

        return (emissions, tags, instr), (transitions, head_transitions, tail_transitions)

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            instr: Optional[BatchedInstr] = None, reduction: str = 'none') -> Tensor:
        (emissions, tags, instr), (transitions, head_transitions, tail_transitions) = self(
            emissions=emissions, tags=tags, instr=instr)

        log_scores = compute_log_scores(
            emissions=emissions, tags=tags,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=tail_transitions,
        )
        log_partitions = scan_log_partitions(
            emissions=emissions,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=tail_transitions,
        )
        log_prob = log_scores - log_partitions

        if reduction == 'none':
            return log_prob
        if reduction == 'sum':
            return log_prob.sum()
        if reduction == 'mean':
            return log_prob.mean()
        raise NotImplementedError(f'{reduction} is not supported')

    def decode(self, emissions: PackedSequence, instr: Optional[BatchedInstr] = None) -> PackedSequence:
        (emissions, _, instr), (transitions, head_transitions, tail_transitions) = self(
            emissions=emissions, tags=None, instr=instr)

        max_partitions = scan_max_partitions(
            emissions=emissions,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=tail_transitions,
        )
        predictions, = torch.autograd.grad(
            max_partitions, emissions.data, torch.ones_like(max_partitions),
            create_graph=False, allow_unused=False, only_inputs=True,
        )

        return emissions._replace(data=predictions.argmax(dim=-1))

    def marginals(self, emissions: PackedSequence, instr: Optional[BatchedInstr] = None) -> Tensor:
        (emissions, _, instr), (transitions, head_transitions, tail_transitions) = self(
            emissions=emissions, tags=None, instr=instr)

        scores = compute_log_marginals(
            emissions=emissions,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=tail_transitions,
        )

        return scores.exp() / scores.exp().sum(dim=-1, keepdim=True)


class CrfDecoderScan(CrfDecoderScanABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoderScan, self).__init__(num_tags=num_tags, num_conjugates=num_conjugates)

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
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.head_transitions, -bound, +bound)
        init.uniform_(self.tail_transitions, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])
