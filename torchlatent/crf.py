from abc import ABCMeta
from typing import Optional

import torch
from torch import Tensor
from torch import nn, autograd, distributions
from torch.distributions.utils import lazy_property
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torchrua import batch_indices, packed_sequence_to_lengths
from torchrua import roll_packed_sequence
from torchrua.indexing import select_head, select_last

from torchlatent.instr import BatchedInstr, build_crf_batched_instr
from torchlatent.semiring import log, max, ent


def compute_log_scores(
        emissions: PackedSequence, tags: PackedSequence, batch_ptr: PackedSequence, pack_ptr: Optional[Tensor],
        transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    num_heads = emissions.batch_sizes[0].item()
    shifted_tags = roll_packed_sequence(tags, offset=1)

    if pack_ptr is None:
        transitions_indices = shifted_tags.data, tags.data,
        start_indices = select_head(tags, unsort=False),
        end_indices = select_last(tags, unsort=True),
    else:
        transitions_indices = pack_ptr, shifted_tags.data, tags.data,
        start_indices = pack_ptr[:num_heads], select_head(tags, unsort=False),
        end_indices = pack_ptr[:num_heads], select_last(tags, unsort=True),

    emissions = emissions.data.gather(dim=-1, index=tags.data[:, None])[:, 0]  # [p]

    transitions = transitions[transitions_indices]  # [p]
    transitions[:num_heads] = start_transitions[start_indices]

    scores = end_transitions[end_indices]  # [p]
    return scores.scatter_add(dim=0, index=batch_ptr.data, src=log.mul(emissions, transitions))


def compute_partitions(semiring):
    def _compute_partitions_fn(
            emissions: PackedSequence, instr: BatchedInstr, pack_ptr: Optional[Tensor],
            transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> Tensor:
        num_heads = emissions.batch_sizes[0].item()

        if pack_ptr is None:
            transitions_indices = ...,
            start_indices = None,
            end_indices = None,
        else:
            transitions_indices = pack_ptr,
            start_indices = pack_ptr[:num_heads],
            end_indices = pack_ptr[:num_heads],

        start = semiring.convert(semiring.mul(
            start_transitions[start_indices],
            emissions.data[emissions.unsorted_indices],
        ))  # [p, c, t]
        end = semiring.convert(end_transitions[end_indices])  # [p, c, t]

        transitions = semiring.convert(semiring.mul(
            transitions[transitions_indices],
            emissions.data[:, None, :],
        ))  # [p, c, t, t]
        transitions[:num_heads] = unit[None, ..., :, :]

        transitions = semiring.tree_reduce(
            pack=PackedSequence(
                data=transitions,
                batch_sizes=emissions.batch_sizes,
                sorted_indices=emissions.sorted_indices,
                unsorted_indices=emissions.unsorted_indices,
            ), instr=instr,
        )

        return semiring.bmm(
            semiring.bmm(start[..., None, :], transitions),
            end[..., :, None],
        )[..., 0, 0]

    return _compute_partitions_fn


compute_log_partitions = compute_partitions(log)
compute_max_partitions = compute_partitions(max)
compute_ent_partitions = compute_partitions(ent)


class CrfDistribution(distributions.Distribution):
    def __init__(self, emissions: PackedSequence, batch_ptr: PackedSequence, instr: BatchedInstr,
                 pack_ptr: Optional[Tensor],
                 transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.batch_ptr = batch_ptr
        self.instr = instr

        self.pack_ptr = pack_ptr
        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions

    def log_prob(self, tags: PackedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: PackedSequence) -> Tensor:
        return compute_log_scores(
            emissions=self.emissions, tags=tags, batch_ptr=self.batch_ptr,
            pack_ptr=self.pack_ptr,
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_log_partitions(
            emissions=self.emissions, instr=self.instr,
            unit=log.fill_unit(self.transitions),
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
    def _entropy(self) -> Tensor:
        return compute_ent_partitions(
            emissions=self.emissions, instr=self.instr,
            unit=ent.fill_unit(self.transitions),
            pack_ptr=self.pack_ptr,
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @lazy_property
    def entropy(self) -> Tensor:
        return ent.unconvert(self._entropy)

    @lazy_property
    def argmax(self) -> PackedSequence:
        partitions = compute_max_partitions(
            emissions=self.emissions, instr=self.instr,
            unit=max.fill_unit(self.transitions),
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


class CrfDecoderABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_packs: int = None):
        super(CrfDecoderABC, self).__init__()

        self.num_packs = num_packs
        if num_packs is None:
            self.pack_ptr = None
        else:
            self.register_buffer('pack_ptr', torch.arange(num_packs, dtype=torch.long))

    def reset_parameters(self, bound: float = 0.01) -> None:
        raise NotImplementedError

    def _validate(self, emissions: PackedSequence,
                  tags: Optional[PackedSequence], lengths: Optional[Tensor],
                  batch_ptr: Optional[PackedSequence], instr: Optional[BatchedInstr]):
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

        if self.pack_ptr is None:
            pack_ptr = None
        else:
            pack_ptr = self.pack_ptr.repeat((emissions.data.size(0) // self.pack_ptr.size(0),))
        return emissions, tags, pack_ptr, batch_ptr, instr

    def _obtain_parameters(self, *args, **kwargs):
        return self.transitions, self.start_transitions, self.end_transitions

    def forward(self, emissions: PackedSequence,
                tags: Optional[PackedSequence] = None, lengths: Optional[Tensor] = None,
                batch_ptr: Optional[PackedSequence] = None, instr: Optional[BatchedInstr] = None):
        emissions, tags, pack_ptr, batch_ptr, instr = self._validate(
            emissions=emissions, tags=tags, lengths=lengths,
            batch_ptr=batch_ptr, instr=instr,
        )
        transitions, start_transitions, end_transitions = self._obtain_parameters(
            emissions=emissions, tags=tags,
            batch_ptr=batch_ptr, instr=instr,
        )

        dist = CrfDistribution(
            emissions=emissions, batch_ptr=batch_ptr, instr=instr,
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


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int) -> None:
        super(CrfDecoder, self).__init__(num_packs=None)

        self.num_tags = num_tags

        self.transitions = nn.Parameter(torch.empty((self.num_tags, self.num_tags)), requires_grad=True)
        self.start_transitions = nn.Parameter(torch.empty((self.num_tags,)), requires_grad=True)
        self.end_transitions = nn.Parameter(torch.empty((self.num_tags,)), requires_grad=True)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.start_transitions, -bound, +bound)
        init.uniform_(self.end_transitions, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
        ])


class StackedCrfDecoder(CrfDecoderABC):
    def __init__(self, *decoders: CrfDecoderABC) -> None:
        super(StackedCrfDecoder, self).__init__(num_packs=len(decoders))

        self.num_tags = decoders[0].num_tags
        self.decoders = nn.ModuleList(decoders)

        for decoder in decoders:
            assert self.num_tags == decoder.num_tags

    def reset_parameters(self, bound: float = 0.01) -> None:
        for decoder in self.decoders:
            decoder.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_packs={self.num_packs}',
        ])

    def _obtain_parameters(self, *args, **kwargs):
        transitions, start_transitions, end_transitions = zip(*[
            decoder._obtain_parameters(*args, **kwargs) for decoder in self.decoders
        ])
        transitions = torch.stack(transitions, dim=0)
        start_transitions = torch.stack(start_transitions, dim=0)
        end_transitions = torch.stack(end_transitions, dim=0)
        return transitions, start_transitions, end_transitions


if __name__ == '__main__':
    pack = pack_sequence([
        torch.tensor([
            [1, 3],
            [2, 4.]
        ], requires_grad=True).log(),
    ], enforce_sorted=False)

    decoder = CrfDecoder(num_tags=2)
    init.zeros_(decoder.transitions)
    init.zeros_(decoder.start_transitions)
    init.zeros_(decoder.end_transitions)

    dist, _ = decoder.forward(emissions=pack)
    print(dist.entropy)
    print(dist.log_partitions)

    import torch

    a = torch.tensor([2, 4, 6, 12.])
    prob = a / a.sum()

    print((prob * prob.log()).sum().neg())
    print(a.sum().log())
