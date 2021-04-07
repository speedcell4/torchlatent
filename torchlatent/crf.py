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
from torchrua.indexing import select_head, select_last, batch_indices

from torchlatent.instr import BatchedInstr, build_crf_batched_instr
from torchlatent.semiring import log, max


def compute_log_scores(
        emissions: PackedSequence, tags: PackedSequence,
        transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    """

    Args:
        emissions: [t1, c1, n]
        tags: [t1, c1]
        transitions: [t2, c2, n, n]
        start_transitions: [t2, c2, n]
        end_transitions: [t2, c2, n]

    Returns:
        [b, c]
    """

    batch_ptr = batch_indices(emissions)  # [t1]

    batch_size = emissions.batch_sizes[0].item()
    time = torch.arange(transitions.size(0), device=transitions.device)  # [t2]
    conj = torch.arange(transitions.size(1), device=transitions.device)  # [c2]

    src = roll_packed_sequence(tags, offset=1).data  # [t1, c1]
    dst = tags.data  # [t1, c1]

    emissions = emissions.data.gather(dim=-1, index=tags.data[..., None])[..., 0]  # [t1, c1]

    transitions = transitions[time[:, None], conj[None, :], src, dst]  # [t, c]

    transitions[:batch_size] = start_transitions[
        time[:batch_size, None], conj[None, :], select_head(tags, unsort=False)]  # [b, c]
    end_transitions = end_transitions[time[:batch_size, None], conj[None, :], select_last(tags, unsort=True)]  # [b, c]

    scores = log.mul(emissions, transitions)
    return end_transitions.scatter_add(dim=0, index=batch_ptr[:, None].expand_as(scores), src=scores)


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

        start = semiring.mul(
            start_transitions[start_indices],
            emissions.data[emissions.unsorted_indices],
        )  # [p, c, t]
        end = end_transitions[end_indices]  # [p, c, t]

        transitions = semiring.mul(
            transitions[transitions_indices],
            emissions.data[:, None, :],
        )  # [p, c, t, t]
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


class CrfDistribution(distributions.Distribution):
    def __init__(self, emissions: PackedSequence, instr: BatchedInstr,
                 pack_ptr: Optional[Tensor],
                 transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.emissions = emissions
        self.instr = instr

        self.pack_ptr = pack_ptr
        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions

    def log_prob(self, tags: PackedSequence) -> Tensor:
        return self.log_scores(tags=tags) - self.log_partitions

    def log_scores(self, tags: PackedSequence) -> Tensor:
        return compute_log_scores(
            emissions=self.emissions, tags=tags,
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
    def entropy(self) -> Tensor:
        marginals = torch.masked_fill(self.marginals, self.marginals == 0, 1.)
        src = (marginals * marginals.log()).sum(dim=-1).neg()
        index = batch_indices(pack=self.emissions)
        zeros = torch.zeros(
            (self.emissions.batch_sizes[0],),
            dtype=torch.float32, device=self.emissions.data.device,
        )
        return torch.scatter_add(zeros, src=src, index=index, dim=0)

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
                  tags: Optional[PackedSequence], lengths: Optional[Tensor], instr: Optional[BatchedInstr]):

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
        return emissions, tags, pack_ptr, instr

    def _obtain_parameters(self, *args, **kwargs):
        return self.transitions, self.start_transitions, self.end_transitions

    def forward(self, emissions: PackedSequence,
                tags: Optional[PackedSequence] = None, lengths: Optional[Tensor] = None,
                instr: Optional[BatchedInstr] = None):
        emissions, tags, pack_ptr, instr = self._validate(
            emissions=emissions, tags=tags, lengths=lengths,
            instr=instr,
        )
        transitions, start_transitions, end_transitions = self._obtain_parameters(
            emissions=emissions, tags=tags,
            instr=instr,
        )

        dist = CrfDistribution(
            emissions=emissions, instr=instr,
            pack_ptr=pack_ptr,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

        return dist, tags

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            reduction: str = 'none', lengths: Optional[Tensor] = None,
            instr: Optional[BatchedInstr] = None) -> Tensor:
        dist, tags = self(
            emissions=emissions, tags=tags, lengths=lengths,
            instr=instr,
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
               instr: Optional[BatchedInstr] = None) -> PackedSequence:
        dist, _ = self.forward(
            emissions=emissions, tags=None, lengths=lengths,
            instr=instr,
        )

        return dist.argmax

    def marginals(self, emissions: PackedSequence, lengths: Optional[Tensor] = None,
                  instr: Optional[BatchedInstr] = None) -> Tensor:
        dist, _ = self.forward(
            emissions=emissions, tags=None, lengths=lengths,
            instr=instr,
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
