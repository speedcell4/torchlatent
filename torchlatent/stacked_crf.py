from abc import ABCMeta
from typing import Optional

import torch
from torch import Tensor
from torch import nn, autograd, distributions
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence
from torchrua import batch_indices, packed_sequence_to_lengths
from torchrua import roll_packed_sequence
from torchrua.indexing import select_head, select_last

from torchlatent import CrfDecoder
from torchlatent.instr import BatchedInstr, build_crf_batched_instr
from torchlatent.semiring import log, max


def compute_stacked_log_scores(
        emissions: PackedSequence, tags: PackedSequence, batch_ptr: PackedSequence, pack_ptr: Optional[Tensor],
        transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    num_heads = emissions.batch_sizes[0].item()

    shifted_tags = roll_packed_sequence(tags, offset=1)
    emissions = emissions.data.gather(dim=-1, index=tags.data[:, None])[:, 0]  # [p]

    transitions = transitions[pack_ptr, shifted_tags.data, tags.data]  # [p]
    transitions[:num_heads] = start_transitions[pack_ptr[:num_heads], select_head(tags, unsort=False)]

    scores = end_transitions[pack_ptr[:num_heads], select_last(tags, unsort=True)]  # [b]
    return scores.scatter_add(dim=0, index=batch_ptr.data, src=log.mul(emissions, transitions))


def compute_stacked_partitions(semiring):
    def _compute_stacked_partitions_fn(
            emissions: PackedSequence, instr: BatchedInstr, pack_ptr: Tensor,
            transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> Tensor:
        num_heads = emissions.batch_sizes[0].item()

        start = semiring.mul(start_transitions[pack_ptr[:num_heads]],
                             emissions.data[emissions.unsorted_indices, :])  # [p, t]
        end = end_transitions[pack_ptr[:num_heads]]  # [p, t]

        transitions = semiring.mul(transitions[pack_ptr], emissions.data[:, None, :])  # [p, t, t]
        transitions[:emissions.batch_sizes[0]] = unit[None, :, :]

        transitions = semiring.reduce(
            pack=PackedSequence(
                data=transitions,
                batch_sizes=emissions.batch_sizes,
                sorted_indices=emissions.sorted_indices,
                unsorted_indices=emissions.unsorted_indices,
            ), instr=instr,
        )
        return semiring.bmm(semiring.bmm(start[:, None, :], transitions), end[:, :, None])[:, 0, 0]

    return _compute_stacked_partitions_fn


compute_stacked_log_partitions = compute_stacked_partitions(log)
compute_stacked_max_partitions = compute_stacked_partitions(max)


class CrfDistribution(distributions.Distribution):
    def __init__(self, emissions: PackedSequence, batch_ptr: PackedSequence, instr: BatchedInstr, pack_ptr: Tensor,
                 transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.batch_ptr = batch_ptr
        self.instr = instr

        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions
        self.pack_ptr = pack_ptr
        self.unit = unit

    def log_prob(self, tags: PackedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: PackedSequence) -> Tensor:
        return compute_stacked_log_scores(
            emissions=self.emissions, tags=tags, batch_ptr=self.batch_ptr,
            pack_ptr=self.pack_ptr,
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_stacked_log_partitions(
            emissions=self.emissions, instr=self.instr, unit=self.unit,
            pack_ptr=self.pack_ptr,
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
        partitions = compute_stacked_max_partitions(
            emissions=self.emissions, instr=self.instr, unit=self.unit,
            pack_ptr=self.pack_ptr,
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


class StackedCrfDecoderABC(nn.Module, metaclass=ABCMeta):
    num_packs: int

    def reset_parameters(self, bound: float = 0.01) -> None:
        raise NotImplementedError

    @staticmethod
    def _validate(emissions: PackedSequence,
                  tags: Optional[PackedSequence], lengths: Optional[Tensor],
                  pack_ptr: Tensor, batch_ptr: Optional[PackedSequence], instr: Optional[BatchedInstr]):
        if batch_ptr is None:
            batch_ptr = PackedSequence(
                data=batch_indices(pack=emissions),
                batch_sizes=emissions.batch_sizes,
                sorted_indices=emissions.sorted_indices,
                unsorted_indices=emissions.unsorted_indices,
            )

        if instr is None:
            if lengths is None:
                lengths = packed_sequence_to_lengths(pack=emissions, unsort=True)
            instr = build_crf_batched_instr(
                lengths=lengths, device=emissions.data.device,
                sorted_indices=emissions.sorted_indices,
            )

        pack_ptr = pack_ptr.repeat((emissions.data.size(0) // pack_ptr.size(0),))
        return emissions, tags, pack_ptr, batch_ptr, instr

    def _obtain_parameters(self, *args, **kwargs):
        return self.transitions, self.start_transitions, self.end_transitions, self.unit

    def forward(self, emissions: PackedSequence,
                tags: Optional[PackedSequence] = None, lengths: Optional[Tensor] = None,
                batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None):
        emissions, tags, pack_ptr, batch_ptr, instr = self._validate(
            emissions=emissions, tags=tags, lengths=lengths,
            pack_ptr=torch.arange(self.num_packs, device=emissions.data.device, dtype=torch.long),
            batch_ptr=batch_ptr, instr=instr,
        )
        transitions, start_transitions, end_transitions, unit = self._obtain_parameters(
            emissions=emissions, tags=tags,
            batch_ptr=batch_ptr, instr=instr,
        )

        dist = CrfDistribution(
            emissions=emissions, batch_ptr=batch_ptr, instr=instr, unit=unit,
            pack_ptr=pack_ptr,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

        return dist, tags

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            reduction: str = 'none', lengths: Optional[Tensor] = None,
            batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None) -> Tensor:
        dist, tags = self(
            emissions=emissions, tags=tags, lengths=lengths,
            batch_ptr=batch_ptr, instr=instr,
        )

        loss = dist.log_prob(tags)

        if reduction == 'none':
            return loss
        if reduction == 'sum':
            return loss.sum()
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'batch_mean':
            if lengths is None:
                lengths = packed_sequence_to_lengths(pack=emissions, unsort=True, dtype=torch.float32)
            return (loss / lengths.float()).mean()
        if reduction == 'token_mean':
            return loss.sum() / emissions.data.size(0)
        raise NotImplementedError(f'{reduction} is not supported')

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


class StackedCrfDecoder(StackedCrfDecoderABC):
    def __init__(self, *decoders: CrfDecoder) -> None:
        super(StackedCrfDecoder, self).__init__()

        self.decoders = nn.ModuleList(decoders)

        self.num_tags = decoders[0].num_tags
        self.num_packs = len(decoders)

        self.register_buffer('unit', decoders[0].unit)

        self.reset_parameters()

    def reset_parameters(self, bound: float = 0.01) -> None:
        pass

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_packs={self.num_packs}',
        ])

    def _obtain_parameters(self, *args, **kwargs):
        transitions, start_transitions, end_transitions, unit = zip(*[
            decoder._obtain_parameters(*args, **kwargs) for decoder in self.decoders
        ])
        transitions = torch.stack(transitions, dim=0)
        start_transitions = torch.stack(start_transitions, dim=0)
        end_transitions = torch.stack(end_transitions, dim=0)
        return transitions, start_transitions, end_transitions, unit[0]
