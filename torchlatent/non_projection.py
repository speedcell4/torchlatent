from typing import Any
from typing import Tuple

import torch
from torch import Tensor
from torch import distributions
from torch.distributions.utils import lazy_property

from torchlatent.semiring import log, std


class NonProjectionDistribution(distributions.Distribution):
    def __init__(self, energy: Tensor, length: Tensor) -> None:
        super(NonProjectionDistribution, self).__init__()
        assert energy.dim() == 4
        assert energy.size(-2) == energy.size(-1)

        self.energy = energy
        self.log_potentials = energy
        self.unlabeled = log.sum(energy, dim=1)

        self.length = length
        self.padding_edge, self.padding_diag = build_mask(
            length=length, device=energy.device,
        )
        self.laplacian = build_laplacian(
            potential=self.unlabeled.exp(),
            padding_mask=self.padding_edge[..., 1:, 1:],
            padding_diag=self.padding_diag[..., 1:, 1:],
        )

    def log_score(self, target: Tuple[Tensor, Tensor]) -> Tensor:
        head, drel = target
        unlabeled = self.energy.gather(
            dim=1, index=drel[:, None, :, None].expand(
                (-1, -1, -1, self.energy.size(-1))
            )
        )
        unlabeled = unlabeled[:, 0, :, :]

        unlabeled = unlabeled.masked_fill(self.padding_edge, std.zero)
        unlabeled[:, 0, :] = std.zero

        scores = unlabeled.gather(dim=-1, index=head[:, :, None])
        return std.sum(std.sum(scores, dim=-1), dim=-1)

    @lazy_property
    def log_partitions(self) -> Tensor:
        _, ret = self.laplacian.slogdet()
        return ret

    @lazy_property
    def argmax(self) -> Any:
        raise NotImplementedError


@torch.no_grad()
def build_mask(length: Tensor, device: torch.device, dim1: int = -2, dim2: int = -1) -> Tuple[Tensor, Tensor]:
    max_length = length.max().item()

    index = torch.arange(max_length, dtype=torch.long, device=length.device)
    ls = index[None, :] < length[:, None]  # [bsz, sln]
    filling_edge = ls[..., None, :] & ls[..., :, None]  # [bsz, sln, sln]
    filling_diag = ls.diag_embed(dim1=dim1, dim2=dim2)  # [bsz, sln, sln]
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
