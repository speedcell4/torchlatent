import torch
from hypothesis import given, strategies as st

from torchlatent.semiring import Std, Log


@given(
    batch_sizes=st.lists(st.integers(1, 10), min_size=1, max_size=4),
)
def test_add(
        batch_sizes
):
    lhs = torch.randn(batch_sizes)
    rhs = torch.randn(batch_sizes)

    std = Std.add(lhs.exp(), rhs.exp())
    log = Log.add(lhs, rhs).exp()

    assert torch.allclose(std, log, rtol=1e-5, atol=1e-5)
