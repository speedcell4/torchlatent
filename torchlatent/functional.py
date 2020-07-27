import torch
from torch import Tensor
from torch import jit
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


@torch.no_grad()
def build_mask(length: Tensor, padding_mask: bool = True, batch_first: bool = True,
               max_length: int = None, dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if max_length is None:
        max_length = length.max().item()
    indices = torch.arange(0, max_length, device=length.device)

    mask = indices[None, :] < length[:, None]
    if padding_mask:
        mask = ~mask
    if not batch_first:
        mask = mask.transpose(0, 1)
    return mask.type(dtype).to(device or length.device).contiguous()


@torch.no_grad()
def build_seq_ptr(lengths: Tensor, device: torch.device) -> PackedSequence:
    seq_ptr = torch.arange(lengths.size(0), device=device)[:, None].expand((-1, lengths.max().item()))
    return pack_padded_sequence(seq_ptr, lengths=lengths, batch_first=True, enforce_sorted=False)
