from abc import ABCMeta
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn, autograd, distributions
from torch.distributions.utils import lazy_property
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torchrua import batch_indices, pack_to_lengths
from torchrua import roll_packed_sequence
from torchrua.indexing import select_head, select_last

from torchlatent.instr import BatchedInstr, build_crf_batched_instr
from torchlatent.semiring import log, max


def compute_log_scores(
        emissions: PackedSequence, tags: PackedSequence, batch_ptr: PackedSequence,
        transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    shifted_tags = roll_packed_sequence(tags, offset=1)

    emissions = emissions.data.gather(dim=-1, index=tags.data[:, None])[:, 0]
    transitions = transitions[shifted_tags.data, tags.data]
    transitions[:tags.batch_sizes[0].item()] = log.one

    src = select_head(tags, unsort=True)
    dst = select_last(tags, unsort=True)

    scores = log.mul(start_transitions[src], end_transitions[dst])
    return scores.scatter_add(dim=0, index=batch_ptr.data, src=log.mul(emissions, transitions))


def compute_partitions(semiring):
    def _compute_partitions_fn(
            emissions: PackedSequence, instr: BatchedInstr,
            transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> Tensor:
        start = semiring.mul(start_transitions[None, :], emissions.data[emissions.unsorted_indices, :])  # [bsz, tag]
        end = end_transitions  # [tag]

        transitions = semiring.mul(transitions[None, :, :], emissions.data[:, None, :])  # [pln,  tag, tag]
        transitions[:emissions.batch_sizes[0]] = unit[None, :, :]

        transitions = semiring.reduce(
            pack=PackedSequence(
                data=transitions,
                batch_sizes=emissions.batch_sizes,
                sorted_indices=emissions.sorted_indices,
                unsorted_indices=emissions.unsorted_indices,
            ), instr=instr,
        )
        return semiring.mv(semiring.vm(start, transitions), end)

    return _compute_partitions_fn


compute_log_partitions = compute_partitions(log)
compute_max_partitions = compute_partitions(max)


class CrfDistribution(distributions.Distribution):
    def __init__(self, emissions: PackedSequence, batch_ptr: PackedSequence, instr: BatchedInstr,
                 transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.batch_ptr = batch_ptr
        self.instr = instr

        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions
        self.unit = unit

    def log_prob(self, tags: PackedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: PackedSequence) -> Tensor:
        return compute_log_scores(
            emissions=self.emissions, tags=tags, batch_ptr=self.batch_ptr,
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_log_partitions(
            emissions=self.emissions, instr=self.instr, unit=self.unit,
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
            emissions=self.emissions, instr=self.instr, unit=self.unit,
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
    def reset_parameters(self, bound: float = 0.01) -> None:
        raise NotImplementedError

    @staticmethod
    def _validate(emissions: PackedSequence,
                  tags: Optional[PackedSequence], lengths: Optional[Tensor],
                  batch_ptr: Optional[PackedSequence], instr: Optional[BatchedInstr]):
        if batch_ptr is None:
            if lengths is None:
                lengths = pack_to_lengths(pack=emissions)

            batch_ptr = batch_indices(pack=emissions)

        if instr is None:
            if lengths is None:
                lengths = pack_to_lengths(pack=emissions)
            instr = build_crf_batched_instr(lengths=lengths)

        return emissions, tags, batch_ptr, instr

    def _obtain_parameters(self, *args, **kwargs):
        return self.transitions, self.start_transitions, self.end_transitions, self.unit

    def forward(self, emissions: PackedSequence,
                tags: Optional[PackedSequence] = None, lengths: Optional[Tensor] = None,
                batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None) \
            -> Tuple[CrfDistribution, Optional[PackedSequence]]:
        emissions, tags, batch_ptr, instr = self._validate(
            emissions=emissions, tags=tags, lengths=lengths,
            batch_ptr=batch_ptr, instr=instr,
        )
        transitions, start_transitions, end_transitions, unit = self._obtain_parameters(
            emissions=emissions, tags=tags,
            batch_ptr=batch_ptr, instr=instr,
        )

        dist = CrfDistribution(
            emissions=emissions, batch_ptr=batch_ptr, instr=instr, unit=unit,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

        return dist, tags

    def fit(self, emissions: PackedSequence, tags: PackedSequence, lengths: Optional[Tensor] = None,
            batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None) -> Tensor:
        dist, tags = self(
            emissions=emissions, tags=tags, lengths=lengths,
            batch_ptr=batch_ptr, instr=instr,
        )

        return dist.log_prob(tags)

    def decode(self, emissions: PackedSequence, lengths: Optional[Tensor] = None,
               batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None) -> PackedSequence:
        dist, _ = self.forward(
            emissions=emissions, tags=None, lengths=lengths,
            batch_ptr=batch_ptr, instr=instr,
        )

        return dist.argmax

    def marginals(self, emissions: PackedSequence, lengths: Optional[Tensor] = None,
                  batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None) -> Tensor:
        dist, _ = self.forward(
            emissions=emissions, tags=None, lengths=lengths,
            batch_ptr=batch_ptr, instr=instr,
        )

        return dist.marginals


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int) -> None:
        super(CrfDecoder, self).__init__()

        self.num_tags = num_tags

        self.transitions = nn.Parameter(torch.empty((self.num_tags, self.num_tags)), requires_grad=True)
        self.start_transitions = nn.Parameter(torch.empty((self.num_tags,)), requires_grad=True)
        self.end_transitions = nn.Parameter(torch.empty((self.num_tags,)), requires_grad=True)

        self.register_buffer('unit', log.build_unit(self.transitions))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.start_transitions, -bound, +bound)
        init.uniform_(self.end_transitions, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_ner={self.num_tags}',
        ])
