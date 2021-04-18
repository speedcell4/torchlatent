from itertools import zip_longest
from typing import Tuple, List, Optional, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence

Instr = Tuple[Tensor, int, List[Tensor]]
BatchedInstr = Tuple[Tensor, Optional[Tensor], Tensor, List[int], int]


def build_crf_instr(length: int) -> Instr:
    dst = length
    indices = list(range(dst))

    instr = []
    while len(indices) > 1:
        ins = []
        out = []
        for lhs, rhs in zip(indices[0::2], indices[1::2]):
            ins.append((lhs, rhs, dst))
            out.append(dst)
            dst += 1
        if len(indices) % 2 == 1:
            out.append(indices[-1])
        indices = out
        instr.append(torch.tensor(ins, dtype=torch.long))
    src = torch.arange(length, dtype=torch.long)
    return src, dst, instr[::-1]


def collate_crf_instr(
        collected_instr: List[Instr],
        sorted_indices: Tensor = None,
        device: torch.device = torch.device('cpu')) -> BatchedInstr:
    cnt, batch_src, batch_instr, batch_dst = 0, [], [], []
    for src, dst, instr in collected_instr:
        batch_src.append(src + cnt)
        batch_instr.append([ins + cnt for ins in instr])
        cnt += dst
        batch_dst.append(cnt - 1)

    if sorted_indices is not None:
        sorted_indices = sorted_indices.detach().cpu().tolist()
        batch_src = [batch_src[index] for index in sorted_indices]
        src = pack_sequence(batch_src, enforce_sorted=True)
    else:
        src = pack_sequence(batch_src, enforce_sorted=False)
    instr = [
        torch.cat(instr, dim=0)
        for instr in reversed(list(zip_longest(
            *batch_instr, fillvalue=torch.tensor([], dtype=torch.long)))
        )
    ]
    batch_sizes: List[int] = [i.size(0) for i in instr]
    if len(instr) == 0:
        instr = None
    else:
        instr = torch.cat(instr, dim=0).to(device=device)
    batch_dst = torch.tensor(batch_dst, dtype=torch.long, device=device)
    return src.data.to(device=device), instr, batch_dst, batch_sizes, cnt


def build_crf_batched_instr(lengths: Union[List[int], Tensor],
                            sorted_indices: Tensor = None,
                            device: torch.device = torch.device('cpu')) -> BatchedInstr:
    if torch.is_tensor(lengths):
        lengths = lengths.detach().cpu().tolist()

    collected_instr = [build_crf_instr(length=length) for length in lengths]
    return collate_crf_instr(
        collected_instr=collected_instr,
        sorted_indices=sorted_indices,
        device=device,
    )
