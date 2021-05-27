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
                     transitions: Tensor, start_transitions: Tensor) -> Tensor:
        """

        Args:
            emissions: [t1, c1, n]
            indices: [t1]
            transitions: [t2, c2, n, n]
            start_transitions: [t2, c2, n]

        Returns:
            [t, c, n]
        """

        emissions, _, transitions, start_transitions, _, (t, c, n, h) = broadcast_packed_sequences(
            emissions=emissions, tags=None,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=start_transitions,
        )

        data = torch.empty(
            (t, c, 1, emissions.data.size()[-1]),
            dtype=emissions.data.dtype, device=emissions.data.device, requires_grad=False)
        data[indices[:h]] = start_transitions[:, :, None, :]

        start, end = 0, h
        for h in emissions.batch_sizes.detach().cpu().tolist()[1:]:
            last_start, last_end, start, end = start, start + h, end, end + h
            data[indices[start:end]] = semiring.bmm(
                semiring.mul(
                    data[indices[last_start:last_end]],
                    emissions.data[indices[last_start:last_end], :, None],
                ),
                transitions[:h],
            )

        return data[..., 0, :]

    return _scan_scores


scan_log_scores = scan_scores(log)


def compute_marginals(semiring, scan_semi_scores):
    def _compute_marginals(emissions: PackedSequence, transitions: Tensor,
                           start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
        alpha = scan_semi_scores(
            emissions._replace(data=emissions.data),
            torch.arange(emissions.data.size(0)),
            transitions,
            start_transitions,
        )

        beta = scan_semi_scores(
            emissions._replace(data=emissions.data),
            reversed_indices(emissions),
            transitions.transpose(-2, -1),
            end_transitions,
        )

        return semiring.prod(torch.stack([
            alpha, beta, emissions.data
        ], dim=-1), dim=-1)

    return _compute_marginals


compute_log_marginals = compute_marginals(log, scan_log_scores)


def scan_partitions(semiring):
    def _scan_partitions(emissions: PackedSequence, transitions: Tensor,
                         start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
        """

        Args:
            emissions: [t1, c1, n]
            indices: [t1]
            transitions: [t2, c2, n, n]
            start_transitions: [t2, c2, n]
            end_transitions: [t2, c2, n]

        Returns:
            [t, c, n]
        """

        emissions, _, transitions, start_transitions, _, (t, c, n, h) = broadcast_packed_sequences(
            emissions=emissions, tags=None,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=start_transitions,
        )

        data = torch.empty(
            (t, c, 1, emissions.data.size()[-1]),
            dtype=emissions.data.dtype, device=emissions.data.device, requires_grad=False)
        indices = torch.arange(data.size()[0], dtype=torch.long, device=data.device)

        data[indices[:h]] = semiring.mul(
            start_transitions[:, :, None, :],
            emissions.data[:h, :, None, :],
        )

        start, end = 0, h
        for h in emissions.batch_sizes.detach().cpu().tolist()[1:]:
            last_start, last_end, start, end = start, start + h, end, end + h
            data[indices[start:end]] = semiring.bmm(
                data[indices[last_start:last_end]],
                semiring.mul(
                    transitions[:h],
                    emissions.data[indices[start:end], :, None, :],
                ),
            )

        data = select_last(emissions._replace(data=data), unsort=True)
        ans = semiring.bmm(data, end_transitions[..., None])
        return ans[..., 0, 0]

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
        return self.transitions, self.start_transitions, self.end_transitions

    def forward(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                instr: Optional[BatchedInstr] = None):
        emissions, tags, instr = self._validate(emissions=emissions, tags=tags, instr=instr)
        transitions, start_transitions, end_transitions = self.obtain_parameters(
            emissions=emissions, tags=tags, instr=instr)

        return (emissions, tags, instr), (transitions, start_transitions, end_transitions)

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            instr: Optional[BatchedInstr] = None, reduction: str = 'none') -> Tensor:
        (emissions, tags, instr), (transitions, start_transitions, end_transitions) = self(
            emissions=emissions, tags=tags, instr=instr)

        log_scores = compute_log_scores(
            emissions=emissions, tags=tags,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )
        log_partitions = scan_log_partitions(
            emissions=emissions,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
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
        (emissions, _, instr), (transitions, start_transitions, end_transitions) = self(
            emissions=emissions, tags=None, instr=instr)

        max_partitions = scan_max_partitions(
            emissions=emissions,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )
        predictions, = torch.autograd.grad(
            max_partitions, emissions.data, torch.ones_like(max_partitions),
            create_graph=False, allow_unused=False, only_inputs=True,
        )

        return emissions._replace(data=predictions.argmax(dim=-1))

    def marginals(self, emissions: PackedSequence, instr: Optional[BatchedInstr] = None) -> Tensor:
        (emissions, _, instr), (transitions, start_transitions, end_transitions) = self(
            emissions=emissions, tags=None, instr=instr)

        scores = compute_log_marginals(
            emissions=emissions,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

        return scores.exp() / scores.exp().sum(dim=-1, keepdim=True)


class CrfDecoderScan(CrfDecoderScanABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoderScan, self).__init__(num_tags=num_tags, num_conjugates=num_conjugates)

        self.transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags, self.num_tags)),
            requires_grad=True,
        )
        self.start_transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags)),
            requires_grad=True,
        )
        self.end_transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags)),
            requires_grad=True,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.start_transitions, -bound, +bound)
        init.uniform_(self.end_transitions, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])
