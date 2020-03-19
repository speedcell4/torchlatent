import torch
from hypothesis import given, strategies as st

from torchlatent.semiring import Std, Log

RTOL = ATOL = 1e-5
BATCH_SIZES = st.lists(st.integers(1, 12), min_size=1, max_size=5)


@given(batch_sizes=BATCH_SIZES)
def test_add(
        batch_sizes
):
    lhs = torch.randn(batch_sizes)
    rhs = torch.randn(batch_sizes)

    std = Std.add(lhs.exp(), rhs.exp())
    log = Log.add(lhs, rhs).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_mul(
        batch_sizes
):
    lhs = torch.randn(batch_sizes)
    rhs = torch.randn(batch_sizes)

    std = Std.mul(lhs.exp(), rhs.exp())
    log = Log.mul(lhs, rhs).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_sum(
        batch_sizes
):
    x = torch.randn(batch_sizes)
    dim = torch.randint(0, x.dim(), ()).item()

    std = Std.sum(x.exp(), dim=dim)
    log = Log.sum(x, dim=dim).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_prod(
        batch_sizes
):
    x = torch.randn(batch_sizes)
    dim = torch.randint(0, x.dim(), ()).item()

    std = Std.prod(x.exp(), dim=dim)
    log = Log.prod(x, dim=dim).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
