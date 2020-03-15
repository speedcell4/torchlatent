from typing import Optional, List

import torch
from torch import Tensor
from torch import nn, autograd
from torch.distributions.utils import lazy_property
from torch.nn import init

from torchlatent.abc import LatentDistribution
from torchlatent.semiring import Log


def compute_score(
        emission: Tensor, length: Tensor, padding_mask: Tensor, target: Tensor,
        transition: Tensor, start_transition: Tensor, end_transition: Tensor):
    """
    :param emission: [sln, bsz, tag]
    :param length: [bsz]
    :param padding_mask: [sln, bsz]
    :param target: [sln, bsz]

    :param transition: [tag, tag]
    :param start_transition: [tag]
    :param end_transition: [tag]
    """

    semiring = Log

    s = start_transition[target[0]]

    p = emission.masked_fill(padding_mask[:, :, None], semiring.unit)
    p = p.gather(dim=-1, index=target[:, :, None])[:, :, 0]
    p = semiring.prod(p, dim=0)

    t = transition[target[:-1, :], target[+1:, :]]
    t = t.masked_fill(padding_mask[1:, :], semiring.unit)
    t = semiring.prod(t, dim=0)

    e = end_transition[target.gather(dim=0, index=length[None, :] - 1)[0]]

    return semiring.mul(semiring.mul(s, p), semiring.mul(t, e))


def compute_partition(
        emission: Tensor, length: Tensor, padding_mask: Tensor,
        target_filling_mask: Optional[Tensor], unit: Tensor,
        transition: Tensor, start_transition: Tensor, end_transition: Tensor):
    """
    :param emission: [sln, bsz, tag]
    :param length: [bsz]
    :param padding_mask: [sln, bsz]

    :param target_filling_mask: [sln-1, bsz, tag, tag]
    :param unit: [tag, tag]

    :param transition: [tag, tag]
    :param start_transition: [tag]
    :param end_transition: [tag]
    """

    semiring = Log

    s = semiring.mul(start_transition[None, :], emission[0])  # [bsz, tag]
    p = semiring.mul(transition[None, None], emission[1:, :, None, :])  # [sln - 1, bsz, tag, tag]
    e = end_transition  # [tag]

    if target_filling_mask is not None:
        p = torch.masked_fill(p, ~target_filling_mask, semiring.zero)
    p = torch.where(padding_mask[1:, :, None, None], unit[None, None, :, :], p)

    p = semiring.single_reduce(p, length.clamp_min(2) - 1)
    return semiring.mv(semiring.vm(s, p), e)


def viterbi(emission: Tensor, padding_mask: Tensor, length: Tensor,
            start_transition: Tensor, transition: Tensor, end_transition: Tensor):
    semiring = Log

    padding_mask = padding_mask[..., None]
    transition = semiring.mul(transition[None, None, :, :], emission[:, :, None, :])
    scores = semiring.mul(start_transition[None, :], emission[0])

    back_pointers = []
    for i in range(1, length.max().item()):
        new_scores, back_pointer = torch.max(semiring.mul(scores[..., None], transition[i]), dim=-2)
        scores = torch.where(padding_mask[i], scores, new_scores)
        back_pointers.append(back_pointer)

    ans = [torch.argmax(semiring.mul(scores, end_transition[None, :]), dim=-1, keepdim=True)]
    for i, back_pointer in reversed(list(enumerate(back_pointers, start=1))):
        new_cur = back_pointer.gather(dim=-1, index=ans[-1])
        ans.append(torch.where(padding_mask[i], ans[-1], new_cur))

    ans = torch.cat(ans[::-1], dim=1).detach().cpu().tolist()
    return [ans[i][:l] for i, l in enumerate(length.tolist())]


class ConditionalRandomField(LatentDistribution):
    def __init__(self, emission: Tensor, length: Tensor, padding_mask: Tensor,
                 unit: Tensor, transition: Tensor,
                 start_transition: Tensor, end_transition: Tensor) -> None:
        super(ConditionalRandomField, self).__init__()

        self.emission = emission
        self.length = length
        self.padding_mask = padding_mask

        self.unit = unit

        self.transition = transition
        self.start_transition = start_transition
        self.end_transition = end_transition

    def log_scores(self, target: Tensor) -> Tensor:
        if target.dtype == torch.bool:
            return compute_partition(
                emission=self.emission,
                length=self.length,
                padding_mask=self.padding_mask,
                target_filling_mask=target,
                unit=self.unit,
                transition=self.transition,
                start_transition=self.start_transition,
                end_transition=self.end_transition,
            )
        else:
            return compute_score(
                emission=self.emission,
                padding_mask=self.padding_mask,
                length=self.length,
                target=target,
                transition=self.transition,
                start_transition=self.start_transition,
                end_transition=self.end_transition,
            )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_partition(
            emission=self.emission,
            length=self.length,
            padding_mask=self.padding_mask,
            target_filling_mask=None,
            unit=self.unit,
            transition=self.transition,
            start_transition=self.start_transition,
            end_transition=self.end_transition,
        )

    @lazy_property
    def marginals(self) -> Tensor:
        mar, = autograd.grad(
            self.log_partitions, self.emission, torch.ones_like(self.log_partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        if self.batch_shape:
            return mar.transpose(0, 1)
        return mar

    @lazy_property
    def argmax(self) -> List[List[int]]:
        return viterbi(
            emission=self.emission, padding_mask=self.padding_mask,
            length=self.length, transition=self.transition,
            start_transition=self.start_transition,
            end_transition=self.end_transition,
        )


class CRFDecoder(nn.Module):
    def __init__(self, num_tags: int,
                 batch_first: bool, reduction: str = 'sum',
                 equiv: Optional[Tensor] = None) -> None:
        super(CRFDecoder, self).__init__()
        assert reduction in ('sum', 'mean', 'none')

        self.num_tags = num_tags
        self.batch_first = batch_first
        self.reduction = reduction

        self.new_parameters(equiv)
        self.reset_parameters()

    def new_parameters(self, equiv: Optional[Tensor]) -> None:
        self.start_transition = nn.Parameter(torch.Tensor(self.num_tags))
        self.transition = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        self.end_transition = nn.Parameter(torch.Tensor(self.num_tags))

        self.unit = nn.Parameter(Log.matmul_unit(self.transition), requires_grad=False)

        if equiv is not None:
            self.equiv = nn.Parameter(equiv, requires_grad=False)
        else:
            self.register_parameter('equiv', None)

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.start_transition, -bound, +bound)
        init.uniform_(self.transition, -bound, +bound)
        init.uniform_(self.end_transition, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'batch_first={self.batch_first}',
            f'reduction={self.reduction}',
        ])

    def forward(self, emission: Tensor, length: Tensor, padding_mask: Tensor, target: Tensor = None):
        if self.batch_first:
            emission = emission.transpose(0, 1).contiguous()
            if target is not None:
                target = target.transpose(0, 1).contiguous()
            padding_mask = padding_mask.transpose(0, 1).contiguous()

        dist = ConditionalRandomField(
            emission=emission, length=length, padding_mask=padding_mask,
            unit=self.unit, transition=self.transition,
            start_transition=self.start_transition,
            end_transition=self.end_transition,
        )

        if target is not None:
            if self.equiv is not None:
                mask = self.equiv[target]
                target = mask[:-1, :, :, None] & mask[+1:, :, None, :]
        return dist, target

    def fit(self, emission: Tensor, target: Tensor, length: Tensor, padding_mask: Tensor) -> Tensor:
        dist, target = self(emission=emission, length=length, padding_mask=padding_mask, target=target)
        log_prob = -dist.log_prob(target)

        if self.reduction == 'none':
            return log_prob
        elif self.reduction == 'sum':
            return log_prob.sum()
        elif self.reduction == 'mean':
            return log_prob.mean()
        else:
            raise ValueError(f'unsupported reduction method {self.reduction}')

    def decode(self, emission: Tensor, length: Tensor, padding_mask: Tensor) -> List[List[int]]:
        dist, _ = self(emission=emission, length=length, padding_mask=padding_mask, target=None)
        return dist.argmax
