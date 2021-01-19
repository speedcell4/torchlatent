import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.utils import assert_equal, device
from torchlatent import CrfDecoder


@given(
    batch_size=st.integers(1, 4),
    total_length=st.integers(1, 12),
    num_tags=st.integers(1, 10),
)
def test_entropy(batch_size, total_length, num_tags):
    decoder = CrfDecoder(num_tags=num_tags).to(device=device)

    lengths = torch.randint(0, total_length, (batch_size,), device=device) + 1
    emissions = pack_sequence([
        torch.randn((length, num_tags), device=device, dtype=torch.float32, requires_grad=True)
        for length in lengths.detach().cpu().tolist()
    ], enforce_sorted=False)

    dist, _ = decoder.forward(emissions=emissions)
    assert_equal(dist.log_partitions, dist._entropy[:, 0])

    grad1, = torch.autograd.grad(
        dist.log_partitions,
        emissions.data,
        torch.ones_like(dist.log_partitions),
        create_graph=True, allow_unused=False,
    )
    grad2, = torch.autograd.grad(
        dist._entropy[:, 0],
        emissions.data,
        torch.ones_like(dist._entropy[:, 0]),
        create_graph=True, allow_unused=False,
    )
    assert_equal(grad1, grad2)

    assert not torch.isnan(dist.entropy).any().detach().cpu().item()
