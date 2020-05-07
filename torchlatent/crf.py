from typing import Optional, List, Tuple, Union

import torch
from torch import Tensor
from torch import nn, autograd, distributions
from torch.distributions.utils import lazy_property
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from torchlatent.functional import build_mask, build_seq_ptr
from torchlatent.instr import BatchInstr, build_crf_batch_instr
from torchlatent.semiring import log


def obtain_indices(pack: PackedSequence) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    data, lengths = pad_packed_sequence(
        pack, batch_first=True, padding_value=-1,
    )
    index = torch.arange(0, lengths.size(0), dtype=torch.long, device=data.device)

    src: Tensor = data[:, 0]
    dst: Tensor = data[index, lengths - 1]

    lhs: Tensor = pack_padded_sequence(
        F.pad(data, [1, 0, 0, 0], value=-1),
        lengths=lengths, batch_first=True, enforce_sorted=False,
    ).data
    rhs: Tensor = pack.data

    return src, dst, lhs.clamp_min(0), rhs, lhs == -1


def compute_log_scores(
        log_potentials: PackedSequence, target: PackedSequence, seq_ptr: PackedSequence,
        transition: Tensor, start_transition: Tensor, end_transition: Tensor) -> Tensor:
    src, dst, lhs, rhs, padding_mask = obtain_indices(target)

    e = log_potentials.data.gather(dim=-1, index=target.data[:, None])[:, 0]
    t = transition[lhs, rhs].masked_fill(padding_mask, log.one)

    return (start_transition[src] + end_transition[dst]).scatter_add(dim=0, index=seq_ptr.data, src=log.mul(e, t))


def compute_log_partitions(
        log_potentials: PackedSequence, instr: BatchInstr,
        transition: Tensor, start_transition: Tensor, end_transition: Tensor, unit: Tensor) -> Tensor:
    src, dst, lhs, rhs, padding_mask = obtain_indices(log_potentials._replace(
        data=torch.arange(log_potentials.data.size(0), dtype=torch.long, device=log_potentials.data.device),
    ))

    start = log.mul(start_transition[None, :], log_potentials.data[src, :])  # [bsz, tag]
    end = end_transition  # [tag]

    log_partitions = log.mul(transition[None, :, :], log_potentials.data[:, None, :])  # [pln,  tag, tag]
    log_partitions = torch.where(padding_mask[:, None, None], unit[None, :, :], log_partitions)

    log_partitions = log.reduce(
        pack=log_potentials._replace(data=log_partitions), instr=instr,
    )
    return log.mv(log.vm(start, log_partitions), end)


def viterbi(log_potentials: Tensor, padding_mask: Tensor, length: Tensor,
            transition: Tensor, start_transition: Tensor, end_transition: Tensor) -> List[List[int]]:
    transition = log.mul(transition[None, None, :, :], log_potentials[:, :, None, :])
    scores = log.mul(start_transition[None, :], log_potentials[0])

    back_pointers = []
    for i in range(1, length.max().item()):
        new_scores, back_pointer = torch.max(log.mul(scores[..., None], transition[i]), dim=-2)
        scores = torch.where(padding_mask[i, ..., None], scores, new_scores)
        back_pointers.append(back_pointer)

    ret = [torch.argmax(log.mul(scores, end_transition[None, :]), dim=-1, keepdim=True)]
    for i, back_pointer in reversed(list(enumerate(back_pointers, start=1))):
        new_cur = back_pointer.gather(dim=-1, index=ret[-1])
        ret.append(torch.where(padding_mask[i, ..., None], ret[-1], new_cur))

    ret = torch.cat(ret[::-1], dim=1).detach().cpu().tolist()
    return [ret[i][:l] for i, l in enumerate(length.tolist())]


class CrfDistribution(distributions.Distribution):
    def __init__(self, log_potentials: PackedSequence, seq_ptr: PackedSequence, instr: BatchInstr,
                 transition: Tensor, start_transition: Tensor, end_transition: Tensor, unit: Tensor) -> None:
        super(CrfDistribution, self).__init__()
        self.log_potentials = log_potentials
        self.seq_ptr = seq_ptr
        self.instr = instr

        self.transition = transition
        self.start_transition = start_transition
        self.end_transition = end_transition
        self.unit = unit

    def log_prob(self, target: PackedSequence) -> Tensor:
        return self.log_scores(target) - self.log_partitions

    def log_scores(self, target: PackedSequence) -> Tensor:
        return compute_log_scores(
            log_potentials=self.log_potentials, target=target, seq_ptr=self.seq_ptr,
            transition=self.transition, start_transition=self.start_transition,
            end_transition=self.end_transition,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_log_partitions(
            log_potentials=self.log_potentials, instr=self.instr,
            transition=self.transition, start_transition=self.start_transition,
            end_transition=self.end_transition, unit=self.unit,
        )

    @lazy_property
    def marginals(self) -> Tensor:
        ret, = autograd.grad(
            self.log_partitions, self.log_potentials.data, torch.ones_like(self.log_partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        return ret

    @lazy_property
    def argmax(self) -> List[List[int]]:
        log_potentials, length = pad_packed_sequence(self.log_potentials, batch_first=False)
        padding_mask = build_mask(
            length=length, padding_mask=True, batch_first=False,
            device=log_potentials.device,
        )

        return viterbi(
            log_potentials=log_potentials, padding_mask=padding_mask, length=length,
            transition=self.transition, start_transition=self.start_transition, end_transition=self.end_transition,
        )


class CrfDecoderABC(nn.Module):
    def __init__(self, batch_first: bool) -> None:
        super(CrfDecoderABC, self).__init__()
        self.batch_first = batch_first

    def reset_parameters(self, bound: float = 0.01) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'batch_first={self.batch_first}'

    def _validate(self, log_potentials: Union[PackedSequence, Tensor], target: Optional[Union[PackedSequence, Tensor]],
                  seq_ptr: Optional[PackedSequence], lengths: Optional[Tensor], instr: Optional[BatchInstr]):

        if torch.is_tensor(log_potentials):
            assert lengths is not None

            log_potentials = pack_padded_sequence(
                log_potentials, lengths=lengths,
                batch_first=self.batch_first, enforce_sorted=False,
            )

        if seq_ptr is None:
            assert lengths is not None

            seq_ptr = build_seq_ptr(lengths=lengths, device=log_potentials.data.device)

        if instr is None:
            assert lengths is not None

            instr = build_crf_batch_instr(lengths=lengths, device=log_potentials.data.device)

        if torch.is_tensor(target):
            assert lengths is not None

            target = pack_padded_sequence(
                target, lengths=lengths,
                batch_first=self.batch_first, enforce_sorted=False,
            )

        return log_potentials, seq_ptr, instr, target

    def _obtain_parameters(self, *args, **kwargs) -> Tuple[nn.Parameter, ...]:
        return self.transition, self.start_transition, self.end_transition, self.unit

    def forward(self, log_potentials: Union[PackedSequence, Tensor], target: Union[PackedSequence, Tensor] = None,
                seq_ptr: PackedSequence = None, lengths: Tensor = None, instr: BatchInstr = None):
        log_potentials, seq_ptr, instr, target = self._validate(
            log_potentials=log_potentials, target=target,
            seq_ptr=seq_ptr, lengths=lengths, instr=instr,
        )
        transition, start_transition, end_transition, unit = self._obtain_parameters(
            log_potentials=log_potentials, target=target,
            seq_ptr=seq_ptr, instr=instr,
        )

        dist = CrfDistribution(
            log_potentials=log_potentials, seq_ptr=seq_ptr, instr=instr,
            transition=transition, start_transition=start_transition,
            end_transition=end_transition, unit=unit,
        )

        return dist, target

    def fit(self, log_potentials: Union[PackedSequence, Tensor], target: Union[PackedSequence, Tensor],
            seq_ptr: PackedSequence = None, lengths: Tensor = None, instr: BatchInstr = None) -> List[List[int]]:
        dist, target = self.forward(
            log_potentials=log_potentials, target=target,
            seq_ptr=seq_ptr, lengths=lengths, instr=instr,
        )

        return dist.log_prob(target)

    def decode(self, log_potentials: Union[PackedSequence, Tensor],
               seq_ptr: PackedSequence = None, lengths: Tensor = None, instr: BatchInstr = None) -> List[List[int]]:
        dist, _ = self.forward(
            log_potentials=log_potentials, target=None,
            seq_ptr=seq_ptr, lengths=lengths, instr=instr,
        )

        return dist.argmax

    def marginals(self, log_potentials: Union[PackedSequence, Tensor],
                  seq_ptr: PackedSequence = None, lengths: Tensor = None, instr: BatchInstr = None) -> Tensor:
        dist, _ = self.forward(
            log_potentials=log_potentials, target=None,
            seq_ptr=seq_ptr, lengths=lengths, instr=instr,
        )

        return dist.marginals


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int, batch_first: bool) -> None:
        super(CrfDecoder, self).__init__(batch_first=batch_first)

        self.num_tags = num_tags

        self.transition = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        self.start_transition = nn.Parameter(torch.Tensor(self.num_tags))
        self.end_transition = nn.Parameter(torch.Tensor(self.num_tags))
        self.unit = nn.Parameter(log.build_unit(self.transition), requires_grad=False)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transition, -bound, +bound)
        init.uniform_(self.start_transition, -bound, +bound)
        init.uniform_(self.end_transition, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_ner={self.num_tags}',
            f'batch_first={self.batch_first}',
        ])
