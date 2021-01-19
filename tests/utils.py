import torch
from torch import Tensor


def assert_equal(x: Tensor, y: Tensor) -> None:
    assert x.size() == y.size(), f'{x.size()} != {y.size()}'
    assert torch.allclose(x, y, rtol=1e-5, atol=1e-5), f'{x.view(-1)} != {y.view(-1)}'