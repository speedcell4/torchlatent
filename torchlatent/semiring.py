from typing import Callable, List

import torch
from torch import Tensor
from torch import jit

# Semiring
from torch.nn.utils.rnn import PackedSequence

BIN_FN = Callable[[Tensor, Tensor], Tensor]
SUM_FN = Callable[[Tensor, int], Tensor]
REDUCE_FN = Callable[[Tensor], Tensor]


def build_semiring_matmul_unit(zero: float, unit: float):
    def semiring_matmul_unit(x: Tensor) -> Tensor:
        c = torch.eye(x.size(-1), device=x.device, dtype=torch.bool)
        u = torch.full(c.size(), fill_value=unit, device=c.device, dtype=torch.float32)
        z = torch.full(c.size(), fill_value=zero, device=c.device, dtype=torch.float32)
        return torch.where(c, u, z)

    return semiring_matmul_unit


def build_semiring_vm(semiring_mul: BIN_FN, semiring_sum: SUM_FN):
    @jit.script
    def semiring_vm(lhs: Tensor, rhs: Tensor) -> Tensor:
        return semiring_sum(semiring_mul(lhs.unsqueeze(-1), rhs), -2)

    return semiring_vm


def build_semiring_mv(semiring_mul: BIN_FN, semiring_sum: SUM_FN):
    @jit.script
    def semiring_mv(lhs: Tensor, rhs: Tensor) -> Tensor:
        return semiring_sum(semiring_mul(lhs, rhs.unsqueeze(-2)), -1)

    return semiring_mv


def build_semiring_mm(semiring_mul: BIN_FN, semiring_sum: SUM_FN):
    @jit.script
    def semiring_mm(lhs: Tensor, rhs: Tensor) -> Tensor:
        return semiring_sum(semiring_mul(lhs.unsqueeze(-1), rhs.unsqueeze(-3)), -2)

    return semiring_mm


def build_semiring_batch_reduce(reduce_fn: BIN_FN):
    @jit.script
    def semiring_batch_reduce(x: Tensor) -> Tensor:
        while x.size(0) > 1:
            lhs = x[0:x.size(0) // 2 * 2:2]
            rhs = x[1:x.size(0) // 2 * 2:2]
            z = reduce_fn(lhs, rhs)
            if x.size(0) % 2 == 1:
                z = torch.cat([z, x[x.size(0) - 1:]], dim=0)
            x = z
        return x[0]

    return semiring_batch_reduce


def build_semiring_single_reduce(reduce_fn: REDUCE_FN):
    @jit.script
    def semiring_single_reduce(a: Tensor, length: Tensor) -> Tensor:
        return torch.stack([
            reduce_fn(a[:l, i])
            for i, l in enumerate(length.clamp_min(1).detach().cpu())
        ], dim=0)

    return semiring_single_reduce


def build_semiring_fold(bin_fn: BIN_FN):
    @jit.script
    def semiring_fold(a: Tensor) -> Tensor:
        ret = a[0]
        for i in range(1, a.size(0)):
            ret = bin_fn(ret, a[i])
        return ret

    return semiring_fold


def build_semiring_scan(bin_fn: BIN_FN):
    @jit.script
    def semiring_scan(a: Tensor) -> Tensor:
        ret = [a[0]]
        for i in range(1, a.size(0)):
            ret.append(bin_fn(ret[-1], a[i]))
        return torch.stack(ret, dim=0)

    return semiring_scan


def build_semiring_pack_reduce(bin_fn: BIN_FN):
    @jit.script
    def semiring_pack_reduce(pack: PackedSequence, ins: Tensor, res: Tensor, batch_sizes: List[int], cnt: int):
        zero = torch.zeros(
            (cnt, pack.data.size(-2), pack.data.size(-1)),
            dtype=pack.data.dtype, device=pack.data.device,
        ).requires_grad_(False)
        data = torch.cat([pack.data, zero], dim=0)

        start, end = 0, 0
        for batch_size in batch_sizes:
            start, end = end, end + batch_size
            lhs = ins[start:end, 0]
            rhs = ins[start:end, 1]
            tgt = ins[start:end, 2]
            data[tgt] = bin_fn(data[lhs], data[rhs])

        return data[res]

    return semiring_pack_reduce


class Semiring(object):
    def __init__(self, zero: float, unit: float,
                 add_fn: BIN_FN, mul_fn: BIN_FN,
                 sum_fn: SUM_FN, prod_fn: SUM_FN) -> None:
        super(Semiring, self).__init__()

        self.zero = zero
        self.unit = unit
        self.matmul_unit = build_semiring_matmul_unit(zero, unit)

        self.add = add_fn
        self.mul = mul_fn

        self.sum = sum_fn
        self.prod = prod_fn

        self.vm = build_semiring_vm(mul_fn, sum_fn)
        self.mv = build_semiring_mv(mul_fn, sum_fn)
        self.mm = build_semiring_mm(mul_fn, sum_fn)

        self.batch_reduce = build_semiring_batch_reduce(self.mm)
        self.single_reduce = build_semiring_single_reduce(self.batch_reduce)
        self.fold = build_semiring_fold(self.mm)
        self.scan = build_semiring_scan(self.mm)
        self.pack_reduce = build_semiring_pack_reduce(self.mm)


# Standard Semiring


@jit.script
def std_add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


@jit.script
def std_mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs * rhs


@jit.script
def std_sum(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


@jit.script
def std_prod(x: Tensor, dim: int) -> Tensor:
    return x.prod(dim=dim)


Std = Semiring(
    zero=0., unit=1.,
    add_fn=std_add, mul_fn=std_mul,
    sum_fn=std_sum, prod_fn=std_prod,
)


# Log Semiring


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill(mask, 1).log() + m.masked_fill(mask, -float('inf'))


@jit.script
def log_softmax(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    z = s.masked_fill(mask, 1).log() + m.masked_fill(mask, 0)
    return x.masked_fill(mask.unsqueeze(dim=dim), -float('inf')) - z.unsqueeze(dim=dim)


@jit.script
def softmax(x: Tensor, dim: int) -> Tensor:
    return log_softmax(x=x, dim=dim).exp()


@jit.script
def log_add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return logsumexp(torch.stack([lhs, rhs], dim=-1), dim=-1)


@jit.script
def log_mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


@jit.script
def log_sum(x: Tensor, dim: int) -> Tensor:
    return logsumexp(x, dim=dim)


@jit.script
def log_prod(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


Log = Semiring(
    zero=-float('inf'), unit=0.,
    add_fn=log_add, mul_fn=log_mul,
    sum_fn=log_sum, prod_fn=log_prod,
)
