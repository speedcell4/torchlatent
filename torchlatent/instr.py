from itertools import zip_longest
from typing import Tuple, List, Optional, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence

Instr = Tuple[Tensor, int, List[Tensor]]
BatchedInstr = Tuple[Tensor, Optional[Tensor], Tensor, List[int], int]


def build_crf_instr(length: int) -> Instr:
    tgt = length
    src = inp = list(range(tgt))

    instr = []
    while len(inp) > 1:
        ins = []
        out = []
        for lhs, rhs in zip(inp[0::2], inp[1::2]):
            ins.append((lhs, rhs, tgt))
            out.append(tgt)
            tgt += 1
        if len(inp) % 2 == 1:
            out.append(inp[-1])
        inp = out
        instr.append(torch.tensor(ins, dtype=torch.long))
    src = torch.tensor(src, dtype=torch.long)
    dst = tgt
    return src, dst, instr[::-1]


def collate_crf_instr(collected_instr: List[Instr], device: torch.device = torch.device('cpu')) -> BatchedInstr:
    cnt, batch_src, batch_instr, batch_dst = 0, [], [], []
    for src, dst, instr in collected_instr:
        batch_src.append(src + cnt)
        batch_instr.append([ins + cnt for ins in instr])
        cnt += dst
        batch_dst.append(cnt - 1)

    src = pack_sequence(batch_src, enforce_sorted=False)
    instr = [
        torch.cat(instr, dim=0).to(device=device)
        for instr in reversed(list(zip_longest(
            *batch_instr, fillvalue=torch.tensor([], dtype=torch.long)))
        )
    ]
    batch_sizes: List[int] = [i.size(0) for i in instr]
    if len(instr) == 0:
        instr = None
    else:
        instr = torch.cat(instr, dim=0)
    batch_dst = torch.tensor(batch_dst, dtype=torch.long, device=device)
    return src.data.to(device=device), instr, batch_dst, batch_sizes, cnt


def build_crf_batched_instr(lengths: Union[List[int], Tensor],
                            device: torch.device = torch.device('cpu')) -> BatchedInstr:
    if torch.is_tensor(lengths):
        lengths = lengths.detach().cpu().tolist()

    return collate_crf_instr(
        collected_instr=[build_crf_instr(length=length) for length in lengths], device=device,
    )
