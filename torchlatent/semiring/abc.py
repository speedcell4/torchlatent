from typing import Optional, List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def compile_fill_unit(zero: float, one: float, dtype: torch.dtype = torch.float32):
    def build_unit(x: Tensor) -> Tensor:
        mask = torch.eye(x.size(-1), device=x.device, dtype=torch.bool)
        ones = torch.full(mask.size(), fill_value=one, device=x.device, dtype=dtype)
        zeros = torch.full(mask.size(), fill_value=zero, device=x.device, dtype=dtype)
        return torch.where(mask, ones, zeros)

    return build_unit


def compile_bmm(mul, sum):
    def bmm_fn(x: Tensor, y: Tensor):
        return sum(mul(x.unsqueeze(-1), y.unsqueeze(-3)), -2)

    return bmm_fn


def compile_tree_reduction(bmm):
    def reduce_fn(pack: PackedSequence, instr: Tuple[Tensor, Optional[Tensor], Tensor, List[int], int]) -> Tensor:
        src, instr, dst, batch_sizes, num_steps = instr

        data: Tensor = torch.zeros(
            (num_steps,) + pack.data.size()[1:], requires_grad=False,
            dtype=pack.data.dtype, device=pack.data.device,
        )
        data[src] = pack.data

        if instr is not None:
            start, end = 0, 0
            for batch_size in batch_sizes:
                start, end = end, end + batch_size
                lhs = instr[start:end, 0]
                rhs = instr[start:end, 1]
                tgt = instr[start:end, 2]
                data[tgt] = bmm(data[lhs], data[rhs])

        return data[dst]

    return reduce_fn
