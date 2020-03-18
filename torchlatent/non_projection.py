from typing import Any, Tuple

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property

from torchlatent.abc import LatentDistribution
from torchlatent.semiring import Log, Std


class NonProjectionDistribution(LatentDistribution):
    def __init__(self, log_potential: Tensor, length: Tensor) -> None:
        super(NonProjectionDistribution, self).__init__()
        assert log_potential.dim() == 4

        self.log_potential = log_potential
        self.unlabeled = Log.sum(self.log_potential, dim=1)

        self.length = length
        self.padding_edge, self.padding_diag = build_mask(
            length=length, device=log_potential.device,
        )
        self.laplacian = build_laplacian(
            potential=self.unlabeled.exp(),
            padding_mask=self.padding_edge[..., 1:, 1:],
            padding_diag=self.padding_diag[..., 1:, 1:],
        )

    def log_score(self, target: Tuple[Tensor, Tensor]) -> Tensor:
        head, drel = target
        unlabeled = self.log_potential.gather(
            dim=1, index=drel[:, None, :, None].expand(
                (-1, -1, -1, self.log_potential.size(-1))
            )
        )
        unlabeled = unlabeled[:, 0, :, :]

        unlabeled = unlabeled.masked_fill(self.padding_edge, Std.zero)
        unlabeled[:, 0, :] = Std.zero

        scores = unlabeled.gather(dim=-1, index=head[:, :, None])
        return Std.sum(Std.sum(scores, dim=-1), dim=-1)

    @lazy_property
    def log_partitions(self) -> Tensor:
        _, ret = self.laplacian.slogdet()
        return ret

    @lazy_property
    def marginals(self) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def argmax(self) -> Any:
        raise NotImplementedError


@torch.no_grad()
def build_mask(length: Tensor, device, dim1: int = -2, dim2: int = -1) -> Tuple[Tensor, Tensor]:
    max_length = length.max().item()

    index = torch.arange(max_length, dtype=torch.long, device=length.device)
    ls = index[None, :] < length[:, None]  # [bsz, sln]
    filling_diag = ls.diag_embed(dim1=dim1, dim2=dim2)  # [bsz, sln, sln]
    filling_edge = ls[..., :, None] & ls[..., None, :]  # [bsz, sln, sln]
    padding_edge = ~filling_edge | filling_diag
    padding_diag = (~ls).diag_embed(dim1=dim1, dim2=dim2)
    return padding_edge.to(device), padding_diag.to(device)


def build_laplacian(potential: Tensor, padding_mask: Tensor, padding_diag: Tensor,
                    dim1: int = -2, dim2: int = -1) -> Tensor:
    """
    :param potential: [bsz, sln, sln]
    :param padding_mask: [bsz]
    :param padding_diag:
    :param dim1:
    :param dim2:
    """
    root = potential[:, 1:, 0]
    edge = potential[:, 1:, 1:]
    edge = edge.masked_fill(padding_mask, value=0)

    lap = edge.sum(dim=dim2).diag_embed(dim1=dim1, dim2=dim2) - edge
    lap[:, :, 0] = root
    return lap.masked_fill(padding_diag, 1)
