import torch
from hypothesis import given, strategies as st

from torchlatent.semiring import Std, Log

RTOL = ATOL = 1e-5

BATCH_SIZE = st.integers(1, 12)
BATCH_SIZES = st.lists(BATCH_SIZE, min_size=1, max_size=4)
SENTENCE_LENGTH = st.integers(1, 24)
DIM = st.integers(1, 12)


@given(batch_sizes=BATCH_SIZES)
def test_add(
        batch_sizes
):
    lhs = torch.randn(batch_sizes)
    rhs = torch.randn(batch_sizes)

    lhs1 = lhs.clone().requires_grad_(True)
    lhs2 = lhs.clone().requires_grad_(True)
    rhs1 = rhs.clone().requires_grad_(True)
    rhs2 = rhs.clone().requires_grad_(True)

    std = Std.add(lhs1.exp(), rhs1.exp())
    log = Log.add(lhs2, rhs2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(lhs1.grad, lhs2.grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(rhs1.grad, rhs2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_mul(
        batch_sizes
):
    lhs = torch.randn(batch_sizes)
    rhs = torch.randn(batch_sizes)

    lhs1 = lhs.clone().requires_grad_(True)
    lhs2 = lhs.clone().requires_grad_(True)
    rhs1 = rhs.clone().requires_grad_(True)
    rhs2 = rhs.clone().requires_grad_(True)

    std = Std.mul(lhs1.exp(), rhs1.exp())
    log = Log.mul(lhs2, rhs2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(lhs1.grad, lhs2.grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(rhs1.grad, rhs2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_sum(
        batch_sizes
):
    x = torch.randn(batch_sizes)
    dim = torch.randint(0, x.dim(), ()).item()
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    std = Std.sum(x1.exp(), dim=dim)
    log = Log.sum(x2, dim=dim).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_prod(
        batch_sizes
):
    x = torch.randn(batch_sizes)
    dim = torch.randint(0, x.dim(), ()).item()
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    std = Std.prod(x1.exp(), dim=dim)
    log = Log.prod(x2, dim=dim).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, input_dim=DIM)
def test_vm(
        batch_sizes, input_dim
):
    lhs = torch.randn((*batch_sizes, input_dim))
    rhs = torch.randn((*batch_sizes, input_dim, input_dim))

    std = Std.vm(lhs.exp(), rhs.exp())
    log = Log.vm(lhs, rhs).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, input_dim=DIM)
def test_mv(
        batch_sizes, input_dim
):
    lhs = torch.randn((*batch_sizes, input_dim, input_dim))
    rhs = torch.randn((*batch_sizes, input_dim))

    std = Std.mv(lhs.exp(), rhs.exp())
    log = Log.mv(lhs, rhs).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, input_dim=DIM)
def test_mm(
        batch_sizes, input_dim
):
    lhs = torch.randn((*batch_sizes, input_dim, input_dim))
    rhs = torch.randn((*batch_sizes, input_dim, input_dim))

    std = Std.mm(lhs.exp(), rhs.exp())
    log = Log.mm(lhs, rhs).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_batch_reduce(
        batch_sizes, sentence_length, input_dim
):
    x = torch.randn((sentence_length, *batch_sizes, input_dim, input_dim))

    std = Std.batch_reduce(x.exp())
    log = Log.batch_reduce(x).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_size=BATCH_SIZE, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_single_reduce(
        batch_size, sentence_length, input_dim
):
    x = torch.randn((sentence_length, batch_size, input_dim, input_dim))
    l = torch.randint(1, 1 + sentence_length, (batch_size,))

    std = Std.single_reduce(x.exp(), l)
    log = Log.single_reduce(x, l).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_fold(
        batch_sizes, sentence_length, input_dim
):
    x = torch.randn((sentence_length, *batch_sizes, input_dim, input_dim))

    std = Std.fold(x.exp())
    log = Log.fold(x).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_scan(
        batch_sizes, sentence_length, input_dim
):
    x = torch.randn((sentence_length, *batch_sizes, input_dim, input_dim))

    std = Std.scan(x.exp())
    log = Log.scan(x).exp()

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)


@given(batch_size=BATCH_SIZE, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_single_reduce_scan_equivalence(
        batch_size, sentence_length, input_dim
):
    x = torch.randn((sentence_length, batch_size, input_dim, input_dim))
    l = torch.randint(1, 1 + sentence_length, (batch_size,))
    b = torch.arange(batch_size)

    std_reduce = Std.single_reduce(x.exp(), l)
    log_reduce = Log.single_reduce(x, l).exp()
    std_scan = Std.scan(x.exp())[l - 1, b]
    log_scan = Log.scan(x).exp()[l - 1, b]

    assert torch.allclose(std_reduce, log_reduce, rtol=RTOL, atol=ATOL)
    assert torch.allclose(std_scan, log_scan, rtol=RTOL, atol=ATOL)
    assert torch.allclose(std_reduce, std_scan, rtol=RTOL, atol=ATOL)
