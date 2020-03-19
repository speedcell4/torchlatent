from abc import ABCMeta, abstractmethod
from typing import Optional, List

import torch
from torch import Tensor
from torch import nn, autograd
from torch.distributions.utils import lazy_property
from torch.nn import init

from torchlatent.abc import LatentDistribution
from torchlatent.semiring import Log
from torchlatent.utilities import reduce_tensor


class ConditionalRandomField(LatentDistribution):
    def __init__(self, emission: Tensor, length: Tensor, padding_mask: Tensor,
                 transition: Tensor, start_transition: Tensor, end_transition: Tensor, unit: Tensor) -> None:
        super(ConditionalRandomField, self).__init__(log_potentials=emission)

        self.emission = emission
        self.length = length
        self.padding_mask = padding_mask

        self.transition = transition
        self.start_transition = start_transition
        self.end_transition = end_transition
        self.unit = unit

    def log_scores(self, target: Tensor) -> Tensor:
        if target.dtype == torch.bool:
            return compute_log_partitions(
                emission=self.emission, length=self.length,
                padding_mask=self.padding_mask, target_filling_mask=target,
                transition=self.transition, start_transition=self.start_transition,
                end_transition=self.end_transition, unit=self.unit,
            )
        else:
            return compute_log_scores(
                emission=self.emission, length=self.length,
                padding_mask=self.padding_mask, target=target,
                transition=self.transition, start_transition=self.start_transition,
                end_transition=self.end_transition,
            )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return compute_log_partitions(
            emission=self.emission, length=self.length,
            padding_mask=self.padding_mask, target_filling_mask=None,
            transition=self.transition, start_transition=self.start_transition,
            end_transition=self.end_transition, unit=self.unit,
        )

    @lazy_property
    def argmax(self) -> List[List[int]]:
        return viterbi_decode(
            emission=self.emission, length=self.length,
            padding_mask=self.padding_mask,
            transition=self.transition, start_transition=self.start_transition,
            end_transition=self.end_transition,
        )


class CRFDecoderABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_tags: int, batch_first: bool, reduction: str = 'sum',
                 equiv: Optional[Tensor] = None) -> None:
        super(CRFDecoderABC, self).__init__()
        assert reduction in ('sum', 'mean', 'none')

        self.num_tags = num_tags
        self.batch_first = batch_first
        self.reduction = reduction

        if equiv is not None:
            self.equiv = nn.Parameter(equiv, requires_grad=False)
        else:
            self.register_parameter('equiv', None)

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'batch_first={self.batch_first}',
            f'reduction={self.reduction}',
        ])

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raise NotImplementedError

    def _verify(self, emission: Tensor, length: Tensor, target: Optional[Tensor] = None):
        if self.batch_first:
            emission = emission.transpose(0, 1)
            if target is not None:
                target = target.transpose(0, 1)

                if self.equiv is not None:
                    target_equiv = self.equiv[target]
                    target = target_equiv[:-1, :, :, None] & target_equiv[+1:, :, None, :]

        padding_mask = build_mask(
            length=length, padding_mask=True,
            batch_first=False, device=emission.device,
        )
        return emission, length, target, padding_mask

    @abstractmethod
    def _obtain_parameters(self, emission, length, padding_mask, target):
        raise NotImplementedError

    def forward(self, emission: Tensor, length: Tensor, target: Optional[Tensor] = None):
        emission, length, target, padding_mask = self._verify(
            emission=emission, length=length, target=target,
        )
        transition, start_transition, end_transition, unit = self._obtain_parameters(
            emission=emission, length=length, target=target, padding_mask=padding_mask,
        )

        dist = ConditionalRandomField(
            emission=emission, length=length, padding_mask=padding_mask,
            transition=transition, start_transition=start_transition,
            end_transition=end_transition, unit=unit,
        )

        return dist, target

    def fit(self, emission: Tensor, target: Tensor, length: Tensor) -> Tensor:
        dist, target = self(emission=emission, length=length, target=target)
        return reduce_tensor(-dist.log_prob(target), self.reduction)

    def decode(self, emission: Tensor, length: Tensor) -> List[List[int]]:
        dist, _ = self(emission=emission, length=length, target=None)
        return dist.argmax

    def marginals(self, emission: Tensor, length: Tensor) -> Tensor:
        dist, _ = self(emission=emission, length=length, target=None)
        marginals = dist.marginals
        if not self.batch_first:
            marginals = marginals.transpose(0, 1)
        return marginals


class CRFDecoder(CRFDecoderABC):
    def __init__(self, num_tags: int, batch_first: bool, reduction: str = 'sum',
                 equiv: Optional[Tensor] = None) -> None:
        super(CRFDecoder, self).__init__(
            num_tags=num_tags, batch_first=batch_first,
            reduction=reduction, equiv=equiv,
        )

        self.transition = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transition = nn.Parameter(torch.Tensor(num_tags))
        self.end_transition = nn.Parameter(torch.Tensor(num_tags))
        self.unit = nn.Parameter(Log.matmul_unit(self.transition), requires_grad=False)

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transition, -bound, +bound)
        init.uniform_(self.start_transition, -bound, +bound)
        init.uniform_(self.end_transition, -bound, +bound)

    def _obtain_parameters(self, emission, length, padding_mask, target):
        return self.transition, self.start_transition, self.end_transition, self.unit


def compute_log_scores(
        emission: Tensor, length: Tensor,
        padding_mask: Tensor, target: Tensor,
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


def compute_log_partitions(
        emission: Tensor, length: Tensor,
        padding_mask: Tensor, target_filling_mask: Optional[Tensor],
        transition: Tensor, start_transition: Tensor, end_transition: Tensor, unit: Tensor):
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


def viterbi_decode(
        emission: Tensor, length: Tensor, padding_mask: Tensor,
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


@torch.no_grad()
def build_mask(length: Tensor, padding_mask: bool = True, batch_first: bool = True,
               max_length: int = None, dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if max_length is None:
        max_length = length.max().item()
    indices = torch.arange(0, max_length, device=length.device)

    ls = indices[None, :] < length[:, None]
    if padding_mask:
        ls = ~ls
    if not batch_first:
        ls = ls.transpose(0, 1)
    return ls.type(dtype).to(device or length.device).contiguous()
