import torch
from hypothesis import given, strategies as st
from torch_struct import TreeCRF
from torchnyan import BATCH_SIZE, TINY_TOKEN_SIZE, assert_close, assert_grad_close, device, sizes
from torchrua import C

from torchlatent.cky import CkyDecoder, cky_partitions, cky_scores
from torchlatent.semiring import Log


def get_argmax(cky):
    argmax = cky.argmax
    mask = argmax > 0

    _, t, _, n = mask.size()
    index = torch.arange(t, device=mask.device)
    x = torch.masked_select(index[None, :, None, None], mask=mask)
    y = torch.masked_select(index[None, None, :, None], mask=mask)

    index = torch.arange(n, device=mask.device)
    z = torch.masked_select(index[None, None, None, :], mask=mask)

    return argmax, x, y, z


@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
    rua_targets=st.sampled_from([C.cat, C.pad, C.pack]),
)
def test_cky_scores(token_sizes, num_targets, rua_targets):
    emissions = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)
    expected_cky = TreeCRF(emissions, lengths=token_sizes)

    argmax, x, y, z = get_argmax(expected_cky)

    emissions = torch.randn_like(emissions, requires_grad=True)

    expected_cky = TreeCRF(emissions, lengths=token_sizes)
    expected = expected_cky.log_prob(argmax) + expected_cky.partition

    targets = C(data=torch.stack([x, y, z], dim=-1), token_sizes=token_sizes * 2 - 1)
    actual = cky_scores(
        emissions=C(emissions, token_sizes),
        targets=rua_targets(targets),
        semiring=Log,
    )

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(emissions,))


@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_partitions(token_sizes, num_targets):
    emissions = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(emissions, lengths=token_sizes).partition

    actual_emissions = C(
        data=emissions.logsumexp(dim=-1),
        token_sizes=token_sizes,
    )
    actual = cky_partitions(actual_emissions, Log)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(emissions,))


@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_argmax(token_sizes, num_targets):
    emissions = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected_cky = TreeCRF(emissions, lengths=token_sizes)

    _, x, y, z = get_argmax(expected_cky)

    expected = C(data=torch.stack([x, y, z], dim=-1), token_sizes=token_sizes * 2 - 1)

    actual_cky = CkyDecoder(num_targets=num_targets)
    actual = actual_cky(emissions=C(emissions, token_sizes)).argmax

    for actual, expected in zip(actual.tolist(), expected.tolist()):
        assert set(map(tuple, actual)) == set(map(tuple, expected))
