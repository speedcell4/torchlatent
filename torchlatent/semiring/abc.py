from typing import Optional, List, Tuple

import torch
from torch import Tensor
from torch import jit
from torch.nn.utils.rnn import PackedSequence


def build_build_unit(zero: float, one: float):
    def build_unit(x: Tensor) -> Tensor:
        c = torch.eye(x.size(-1), device=x.device, dtype=torch.bool)
        o = torch.full(c.size(), fill_value=one, device=x.device, dtype=torch.float32)
        z = torch.full(c.size(), fill_value=zero, device=x.device, dtype=torch.float32)
        return torch.where(c, o, z)

    return build_unit


def build_vm_fn(mul_fn, sum_fn):
    @jit.script
    def vm_fn(x: Tensor, y: Tensor) -> Tensor:
        return sum_fn(mul_fn(x.unsqueeze(-1), y), -2)

    return vm_fn


def build_mv_fn(mul_fn, sum_fn):
    @jit.script
    def mv_fn(x: Tensor, y: Tensor) -> Tensor:
        return sum_fn(mul_fn(x, y.unsqueeze(-2)), -1)

    return mv_fn


def build_mm_fn(mul_fn, sum_fn):
    @jit.script
    def mm_fn(x: Tensor, y: Tensor) -> Tensor:
        return sum_fn(mul_fn(x.unsqueeze(-1), y.unsqueeze(-3)), -2)

    return mm_fn


def build_reduce_fn(mm_fn):
    @jit.script
    def reduce_fn(pack: PackedSequence, instr: Tuple[Tensor, Optional[Tensor], Tensor, List[int], int]) -> Tensor:
        src, instr, dst, batch_sizes, num_steps = instr

        data = torch.empty(
            (num_steps,) + pack.data.size()[1:],
            dtype=pack.data.dtype, device=pack.data.device,
        ).requires_grad_(False)
        data[src] = pack.data

        if instr is not None:
            start, end = 0, 0
            for batch_size in batch_sizes:
                start, end = end, end + batch_size
                lhs = instr[start:end, 0]
                rhs = instr[start:end, 1]
                tgt = instr[start:end, 2]
                data[tgt] = mm_fn(data[lhs], data[rhs])

        return data[dst]

    return reduce_fn
