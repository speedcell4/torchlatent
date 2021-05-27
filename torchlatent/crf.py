from abc import ABCMeta
from typing import Optional

import torch
from torch import Tensor
from torch import nn, autograd, distributions
from torch.distributions.utils import lazy_property
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torchrua import packed_sequence_to_lengths
from torchrua import roll_packed_sequence
from torchrua import select_head, select_last, batch_sizes_to_ptr

from torchlatent.instr import BatchedInstr, build_crf_batched_instr
from torchlatent.semiring import log, max
from torchlatent.utils import broadcast_packed_sequences


def compute_log_scores(
        emissions: PackedSequence, tags: PackedSequence,
        transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    emissions, tags, transitions, start_transitions, end_transitions, (t, c, n, h) = broadcast_packed_sequences(
        emissions=emissions, tags=tags,
        transitions=transitions,
        start_transitions=start_transitions,
        end_transitions=end_transitions,
    )

    device = transitions.device
    batch_ptr, _, _ = batch_sizes_to_ptr(
        batch_sizes=emissions.batch_sizes.to(device=device),
        sorted_indices=None,
        unsorted_indices=None,
        total_length=None, device=device,
    )  # [t]

    tidx = torch.arange(t, device=device)  # [t]
    cidx = torch.arange(c, device=device)  # [c]
    head = select_head(tags, unsort=False)  # [h, c]
    tail = select_last(tags, unsort=False)  # [h, c]

    src = roll_packed_sequence(tags, offset=1).data  # [t, c]
    dst = tags.data  # [t, c]

    scores = emissions.data.gather(dim=-1, index=tags.data[..., None])[..., 0]  # [t, c]

    sorted_transitions = transitions[tidx[:, None], cidx[None, :], src, dst]  # [t, c]
    sorted_transitions[:h] = start_transitions[tidx[:h, None], cidx[None, :], head]  # [b, c]

    scores = log.mul(scores, sorted_transitions)
    scores = torch.scatter_add(
        end_transitions[tidx[:h, None], cidx[None, :], tail],
        index=batch_ptr[:, None].expand((t, c)),
        dim=0, src=scores,
    )

    if emissions.unsorted_indices is not None:
        scores = scores[emissions.unsorted_indices]
    return scores


def compute_partitions(semiring):
    def _compute_partitions_fn(
            emissions: PackedSequence, instr: BatchedInstr,
            transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> Tensor:
        emissions, _, transitions, start_transitions, end_transitions, (t, c, n, h) = broadcast_packed_sequences(
            emissions=emissions, tags=None,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

        tidx = torch.arange(t, device=transitions.device)  # [t]
        cidx = torch.arange(c, device=transitions.device)  # [c]
        hidx = tidx if emissions.unsorted_indices is None else emissions.unsorted_indices  # [h]

        scores = log.mul(transitions, emissions.data[..., None, :])  # [t, c, n, n]
        scores[:h] = unit[None, None, :, :]
        scores = semiring.tree_reduce(
            pack=PackedSequence(
                data=scores,
                batch_sizes=emissions.batch_sizes,
                sorted_indices=emissions.sorted_indices,
                unsorted_indices=emissions.unsorted_indices,
            ), instr=instr,
        )

        start_scores = log.mul(  # [t, c, 1, n]
            start_transitions[hidx[:, None], cidx[None, :], None, :],
            emissions.data[hidx, :, None, :],
        )
        end_scores = end_transitions[hidx[:, None], cidx[None, :], :, None]  # [t, c, n, 1]

        return semiring.bmm(
            semiring.bmm(start_scores, scores), end_scores
        )[..., 0, 0]

    return _compute_partitions_fn


compute_log_partitions = compute_partitions(log)
compute_max_partitions = compute_partitions(max)


class CrfDistribution(distributions.Distribution):
    def __init__(self, emissions: PackedSequence, instr: BatchedInstr,
                 transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.instr = instr

        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions

    def log_prob(self, tags: PackedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: PackedSequence) -> Tensor:
        return compute_log_scores(
            emissions=self.emissions, tags=tags,
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_log_partitions(
            emissions=self.emissions, instr=self.instr,
            unit=log.fill_unit(self.transitions),
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @lazy_property
    def marginals(self) -> Tensor:
        partitions = self.log_partitions
        grad, = autograd.grad(
            partitions, self.emissions.data, torch.ones_like(partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        return grad

    @lazy_property
    def argmax(self) -> PackedSequence:
        partitions = compute_max_partitions(
            emissions=self.emissions, instr=self.instr,
            unit=max.fill_unit(self.transitions),
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

        grad, = torch.autograd.grad(
            partitions, self.emissions.data, torch.ones_like(partitions),
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

    def reset_parameters(self, bound: float = 0.01) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])

    @staticmethod
    def _validate(emissions: PackedSequence, tags: Optional[PackedSequence], instr: Optional[BatchedInstr]):
        if instr is None:
            lengths = packed_sequence_to_lengths(pack=emissions, unsort=True)
            instr = build_crf_batched_instr(
                lengths=lengths, device=emissions.data.device,
                sorted_indices=emissions.sorted_indices,
            )

        return emissions, tags, instr

    def obtain_parameters(self, *args, **kwargs):
        return self.transitions, self.start_transitions, self.end_transitions

    def forward(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                instr: Optional[BatchedInstr] = None):
        emissions, tags, instr = self._validate(emissions=emissions, tags=tags, instr=instr)
        transitions, start_transitions, end_transitions = self.obtain_parameters(
            emissions=emissions, tags=tags, instr=instr)

        dist = CrfDistribution(
            emissions=emissions, instr=instr,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

        return dist, tags

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            instr: Optional[BatchedInstr] = None, reduction: str = 'none') -> Tensor:
        dist, tags = self(emissions=emissions, tags=tags, instr=instr)

        log_prob = dist.log_prob(tags)

        if reduction == 'none':
            return log_prob
        if reduction == 'sum':
            return log_prob.sum()
        if reduction == 'mean':
            return log_prob.mean()
        raise NotImplementedError(f'{reduction} is not supported')

    def decode(self, emissions: PackedSequence, instr: Optional[BatchedInstr] = None) -> PackedSequence:
        dist, _ = self(emissions=emissions, tags=None, instr=instr)

        return dist.argmax

    def marginals(self, emissions: PackedSequence, instr: Optional[BatchedInstr] = None) -> Tensor:
        dist, _ = self(emissions=emissions, tags=None, instr=instr)

        return dist.marginals


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoder, self).__init__(num_tags=num_tags, num_conjugates=num_conjugates)

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


class ConjugatedCrfDecoder(CrfDecoderABC):
    def __init__(self, *crf_decoders: CrfDecoderABC) -> None:
        super(ConjugatedCrfDecoder, self).__init__(num_tags=crf_decoders[0].num_tags, num_conjugates=0)

        self.crf_decoders = nn.ModuleList(crf_decoders)
        for crf_decoder in self.crf_decoders:
            assert self.num_tags == crf_decoder.num_tags
            self.num_conjugates += crf_decoder.num_conjugates

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        for crf_decoder in self.crf_decoders:
            crf_decoder.reset_parameters(bound=bound)

    def obtain_parameters(self, *args, **kwargs):
        transitions, start_transitions, end_transitions = zip(*[
            crf_decoder.obtain_parameters(*args, **kwargs)
            for crf_decoder in self.crf_decoders
        ])
        transitions = torch.cat(transitions, dim=1)
        start_transitions = torch.cat(start_transitions, dim=1)
        end_transitions = torch.cat(end_transitions, dim=1)
        return transitions, start_transitions, end_transitions
