import torch
from hypothesis import given
from hypothesis import strategies as st
from torch_struct import TreeCRF

from torchlatent.cky import CkyDecoder
from torchlatent.cky import cky_partitions
from torchlatent.cky import cky_scores
from torchlatent.semiring import Log
from torchnyan import BATCH_SIZE
from torchnyan import TINY_TOKEN_SIZE
from torchnyan import TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import device
from torchnyan import sizes
from torchrua import C
from torchrua import CattedSequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_targets=st.sampled_from([C.cat, C.pad, C.pack]),
)
def test_cky_scores(token_sizes, num_targets, rua_targets):
    emissions = torch.randn((len(token_sizes), max(token_sizes), max(token_sizes), num_targets), requires_grad=True)
    token_sizes = torch.tensor(token_sizes, device=device)

    expected_cky = TreeCRF(emissions, lengths=token_sizes)

    mask = expected_cky.argmax > 0
    _, t, _, n = mask.size()

    index = torch.arange(t, device=mask.device)
    x = torch.masked_select(index[None, :, None, None], mask=mask)
    y = torch.masked_select(index[None, None, :, None], mask=mask)

    index = torch.arange(n, device=mask.device)
    z = torch.masked_select(index[None, None, None, :], mask=mask)

    expected = expected_cky.max

    targets = CattedSequence(data=torch.stack([x, y, z], dim=-1), token_sizes=token_sizes * 2 - 1)
    actual = cky_scores(
        emissions=CattedSequence(emissions, token_sizes),
        targets=rua_targets(targets),
        semiring=Log,
    )

    assert_close(actual=actual, expected=expected)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
)
def test_cky_partitions(token_sizes, num_targets):
    emissions = torch.randn((len(token_sizes), max(token_sizes), max(token_sizes), num_targets), requires_grad=True)
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(emissions, lengths=token_sizes).partition

    actual_emissions = CattedSequence(
        data=emissions.logsumexp(dim=-1),
        token_sizes=token_sizes,
    )
    actual = cky_partitions(actual_emissions, Log)

    assert_close(actual=actual, expected=expected)


@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_argmax(token_sizes, num_targets):
    emissions = torch.randn((len(token_sizes), max(token_sizes), max(token_sizes), num_targets), requires_grad=True)
    token_sizes = torch.tensor(token_sizes, device=device)

    expected_cky = TreeCRF(emissions, lengths=token_sizes)

    mask = expected_cky.argmax > 0
    _, t, _, n = mask.size()

    index = torch.arange(t, device=mask.device)
    x = torch.masked_select(index[None, :, None, None], mask=mask)
    y = torch.masked_select(index[None, None, :, None], mask=mask)

    index = torch.arange(n, device=mask.device)
    z = torch.masked_select(index[None, None, None, :], mask=mask)

    expected = CattedSequence(data=torch.stack([x, y, z], dim=-1), token_sizes=token_sizes * 2 - 1)

    actual_cky = CkyDecoder(num_targets=num_targets)
    actual = actual_cky(emissions=CattedSequence(emissions, token_sizes)).argmax

    for actual, expected in zip(actual.tolist(), expected.tolist()):
        assert set(map(tuple, actual)) == set(map(tuple, expected))
