from typing import Optional, List, Tuple

import torch
from torch import Tensor
from torch import jit
from torch.nn.utils.rnn import PackedSequence


def build_unit_fn(zero: float, one: float):
    def build_unit(x: Tensor) -> Tensor:
        mask = torch.eye(x.size(-1), device=x.device, dtype=torch.bool)
        ones = torch.full(mask.size(), fill_value=one, device=x.device, dtype=torch.float32)
        zeros = torch.full(mask.size(), fill_value=zero, device=x.device, dtype=torch.float32)
        return torch.where(mask, ones, zeros)

    return build_unit


def build_bvm_fn(mul_fn, sum_fn):
    @jit.script
    def bvm_fn(x: Tensor, y: Tensor):
        return sum_fn(mul_fn(x.unsqueeze(-1), y), -2)

    return bvm_fn


def build_bmv_fn(mul_fn, sum_fn):
    @jit.script
    def bmv_fn(x: Tensor, y: Tensor):
        return sum_fn(mul_fn(x, y.unsqueeze(-2)), -1)

    return bmv_fn


def build_bmm_fn(mul_fn, sum_fn):
    @jit.script
    def bmm_fn(x: Tensor, y: Tensor):
        return sum_fn(mul_fn(x.unsqueeze(-1), y.unsqueeze(-3)), -2)

    return bmm_fn


def build_reduce_fn(mm_fn):
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
