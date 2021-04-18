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

    assert emissions.data.dim() == 3, f'{emissions.data.size()}'
    assert tags.data.dim() == 2, f'{tags.data.size()}'
    assert transitions.dim() == 4, f'{transitions.size()}'
    assert start_transitions.dim() == 3, f'{start_transitions.size()}'
    assert end_transitions.dim() == 3, f'{end_transitions.size()}'

    device = transitions.device
    batch_ptr, _, _ = batch_sizes_to_ptr(
        batch_sizes=emissions.batch_sizes.to(device=device),
        sorted_indices=None, unsorted_indices=None,
        total_length=None, device=device,
    )  # [t1]

    batch_size = emissions.batch_sizes[0].item()
    t2 = torch.arange(transitions.size(0), device=device)  # [t2]
    c2 = torch.arange(transitions.size(1), device=device)  # [c2]

    src = roll_packed_sequence(tags, offset=1).data  # [t1, c1]
    dst = tags.data  # [t1, c1]

    sorted_emissions = emissions.data.gather(dim=-1, index=tags.data[..., None])[..., 0]  # [t1, c1]

    sorted_transitions = transitions[
        t2[:, None], c2[None, :], src, dst]  # [t, c]
    sorted_transitions[:batch_size] = start_transitions[
        t2[:batch_size, None], c2[None, :], select_head(tags, unsort=False)]  # [b, c]

    scores = log.mul(sorted_emissions, sorted_transitions)

    end_transitions = end_transitions[
        t2[:batch_size, None], c2[None, :], select_last(tags, unsort=False)]  # [b, c]
    scores = end_transitions.scatter_add(dim=0, index=batch_ptr[:, None].expand_as(scores), src=scores)

    if emissions.unsorted_indices is not None:
        scores = scores[emissions.unsorted_indices]
    return scores


def compute_partitions(semiring):
    def _compute_partitions_fn(
            emissions: PackedSequence, instr: BatchedInstr,
            transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor, unit: Tensor) -> Tensor:
        """

        Args:
            emissions: [t1, c1, n]
            instr:
            transitions: [t2, c2, n, n]
            start_transitions: [t2, c2, n]
            end_transitions: [t2, c2, n]
            unit: [n, n]

        Returns:
            [b, c]
        """

        assert emissions.data.dim() == 3, f'{emissions.data.size()}'
        assert transitions.dim() == 4, f'{transitions.size()}'
        assert start_transitions.dim() == 3, f'{start_transitions.size()}'
        assert end_transitions.dim() == 3, f'{end_transitions.size()}'

        batch_size = emissions.batch_sizes[0].item()
        t2 = torch.arange(transitions.size(0), device=transitions.device)  # [t2]
        c2 = torch.arange(transitions.size(1), device=transitions.device)  # [c2]

        scores = log.mul(transitions, emissions.data[..., None, :])  # [t, c, n, n]
        scores[:batch_size] = unit[None, None, :, :]

        start_scores = log.mul(  # [t, c, 1, n]
            start_transitions[t2[:batch_size, None], c2[None, :], None, :],
            emissions.data[:batch_size, :, None, :],
        )
        end_scores = end_transitions[t2[:batch_size, None], c2[None, :], :, None]  # [t, c, n, 1]

        if emissions.unsorted_indices is not None:
            start_scores = start_scores[emissions.unsorted_indices]
            if end_scores.size(0) > 1:
                end_scores = end_scores[emissions.unsorted_indices]

        scores = semiring.tree_reduce(
            pack=PackedSequence(
                data=scores,
                batch_sizes=emissions.batch_sizes,
                sorted_indices=emissions.sorted_indices,
                unsorted_indices=emissions.unsorted_indices,
            ), instr=instr,
        )

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
    def entropy(self) -> Tensor:
        marginals = torch.masked_fill(self.marginals, self.marginals == 0, 1.)
        src = (marginals * marginals.log()).sum(dim=-1).neg()
        batch_ptr, _, _ = batch_sizes_to_ptr(
            batch_sizes=self.emissions.batch_sizes,
            sorted_indices=self.emissions.sorted_indices,
            unsorted_indices=self.emissions.unsorted_indices,
            total_length=None, device=self.emissions.data.device,
        )  # [t1]
        zeros = torch.zeros(
            (self.emissions.batch_sizes[0],),
            dtype=torch.float32, device=self.emissions.data.device,
        )
        return torch.scatter_add(zeros, src=src, index=batch_ptr, dim=0)

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
