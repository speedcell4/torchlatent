import torch
from hypothesis import given, strategies as st

from torchlatent.semiring import Std, Log, logsumexp, log_softmax, softmax

RTOL = ATOL = 1e-5

BATCH_SIZE = st.integers(1, 8)
BATCH_SIZES = st.lists(BATCH_SIZE, min_size=1, max_size=3)
SENTENCE_LENGTH = st.integers(1, 8)
DIM = st.integers(1, 7)


@given(batch_sizes=BATCH_SIZES)
def test_logsumexp(batch_sizes):
    dim = torch.randint(0, len(batch_sizes), (), dtype=torch.long).item()

    x = torch.rand(batch_sizes)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    y1 = logsumexp(x1, dim=dim)
    y2 = torch.logsumexp(x2, dim=dim)
    y1.backward(torch.ones_like(y1))
    y2.backward(torch.ones_like(y2))

    assert torch.allclose(y1, y2, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)

    x = torch.full(batch_sizes, fill_value=-float('inf'), requires_grad=True)
    y = logsumexp(x, dim=dim)
    y.backward(torch.ones_like(y))

    assert torch.allclose(y, torch.full_like(y, -float('inf')), rtol=RTOL, atol=ATOL)
    assert torch.allclose(x.grad, torch.full_like(x.grad, 0), rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_log_softmax(batch_sizes):
    dim = torch.randint(0, len(batch_sizes), (), dtype=torch.long).item()

    x = torch.rand(batch_sizes)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    y1 = log_softmax(x1, dim=dim)
    y2 = torch.log_softmax(x2, dim=dim)
    y1.backward(torch.ones_like(y1))
    y2.backward(torch.ones_like(y2))

    assert torch.allclose(y1, y2, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)

    x = torch.full(batch_sizes, fill_value=-float('inf'), requires_grad=True)
    y = log_softmax(x, dim=dim)
    y.backward(torch.ones_like(y))

    assert torch.allclose(y, torch.full_like(y, -float('inf')), rtol=RTOL, atol=ATOL)
    assert torch.allclose(x.grad, torch.full_like(x.grad, 0), rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES)
def test_softmax(batch_sizes):
    dim = torch.randint(0, len(batch_sizes), (), dtype=torch.long).item()

    x = torch.rand(batch_sizes)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    y1 = softmax(x1, dim=dim)
    y2 = torch.softmax(x2, dim=dim)
    y1.backward(torch.ones_like(y1))
    y2.backward(torch.ones_like(y2))

    assert torch.allclose(y1, y2, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)

    x = torch.full(batch_sizes, fill_value=-float('inf'), requires_grad=True)
    y = softmax(x, dim=dim)
    y.backward(torch.ones_like(y))

    assert torch.allclose(y, torch.full_like(y, 0), rtol=RTOL, atol=ATOL)
    assert torch.allclose(x.grad, torch.full_like(x.grad, 0), rtol=RTOL, atol=ATOL)


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

    lhs1 = lhs.clone().requires_grad_(True)
    lhs2 = lhs.clone().requires_grad_(True)
    rhs1 = rhs.clone().requires_grad_(True)
    rhs2 = rhs.clone().requires_grad_(True)

    std = Std.vm(lhs1.exp(), rhs1.exp())
    log = Log.vm(lhs2, rhs2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(lhs1.grad, lhs2.grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(rhs1.grad, rhs2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, input_dim=DIM)
def test_mv(
        batch_sizes, input_dim
):
    lhs = torch.randn((*batch_sizes, input_dim, input_dim))
    rhs = torch.randn((*batch_sizes, input_dim))

    lhs1 = lhs.clone().requires_grad_(True)
    lhs2 = lhs.clone().requires_grad_(True)
    rhs1 = rhs.clone().requires_grad_(True)
    rhs2 = rhs.clone().requires_grad_(True)

    std = Std.mv(lhs1.exp(), rhs1.exp())
    log = Log.mv(lhs2, rhs2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(lhs1.grad, lhs2.grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(rhs1.grad, rhs2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, input_dim=DIM)
def test_mm(
        batch_sizes, input_dim
):
    lhs = torch.randn((*batch_sizes, input_dim, input_dim))
    rhs = torch.randn((*batch_sizes, input_dim, input_dim))

    lhs1 = lhs.clone().requires_grad_(True)
    lhs2 = lhs.clone().requires_grad_(True)
    rhs1 = rhs.clone().requires_grad_(True)
    rhs2 = rhs.clone().requires_grad_(True)

    std = Std.mm(lhs1.exp(), rhs1.exp())
    log = Log.mm(lhs2, rhs2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(lhs1.grad, lhs2.grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(rhs1.grad, rhs2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_batch_reduce(
        batch_sizes, sentence_length, input_dim
):
    x = torch.randn((sentence_length, *batch_sizes, input_dim, input_dim))
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    std = Std.batch_reduce(x1.exp())
    log = Log.batch_reduce(x2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)


@given(batch_size=BATCH_SIZE, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_single_reduce(
        batch_size, sentence_length, input_dim
):
    x = torch.randn((sentence_length, batch_size, input_dim, input_dim))
    l = torch.randint(1, 1 + sentence_length, (batch_size,))
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    std = Std.single_reduce(x1.exp(), l)
    log = Log.single_reduce(x2, l).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_fold(
        batch_sizes, sentence_length, input_dim
):
    x = torch.randn((sentence_length, *batch_sizes, input_dim, input_dim))
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    std = Std.fold(x1.exp())
    log = Log.fold(x2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)


@given(batch_sizes=BATCH_SIZES, sentence_length=SENTENCE_LENGTH, input_dim=DIM)
def test_scan(
        batch_sizes, sentence_length, input_dim
):
    x = torch.randn((sentence_length, *batch_sizes, input_dim, input_dim))
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    std = Std.scan(x1.exp())
    log = Log.scan(x2).exp()
    std.backward(torch.ones_like(std))
    log.backward(torch.ones_like(log))

    assert torch.allclose(std, log, rtol=RTOL, atol=ATOL)
    assert torch.allclose(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)


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
