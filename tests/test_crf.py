import torch
from hypothesis import given, settings, strategies as st
from torchcrf import CRF
from torchnyan import BATCH_SIZE, TOKEN_SIZE, assert_close, assert_grad_close, assert_sequence_close, device, sizes
from torchrua import C, D, P

from torchlatent.crf import CrfDecoder, crf_partitions, crf_scores
from torchlatent.semiring import Log


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_logits=st.sampled_from([C.new, D.new, P.new]),
    rua_targets=st.sampled_from([C.new, D.new, P.new]),
)
def test_crf_scores(token_sizes, num_targets, rua_logits, rua_targets):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    targets = [
        torch.randint(0, num_targets, (token_size,), device=device)
        for token_size in token_sizes
    ]

    expected_crf = CRF(num_tags=num_targets, batch_first=False).to(device=device)

    expected_logits = D.new(inputs)
    expected_tags = D.new(targets)

    expected = expected_crf._compute_score(
        expected_logits.data.transpose(0, 1),
        expected_tags.data.transpose(0, 1),
        expected_logits.mask().transpose(0, 1),
    )

    actual = crf_scores(
        logits=rua_logits(inputs),
        targets=rua_targets(targets),
        bias=(expected_crf.transitions, expected_crf.start_transitions, expected_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_logits=st.sampled_from([C.new, D.new, P.new]),
)
def test_crf_partitions(token_sizes, num_targets, rua_logits):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected_crf = CRF(num_tags=num_targets, batch_first=False).to(device=device)

    expected_logits = D.new(inputs)

    expected = expected_crf._compute_normalizer(
        expected_logits.data.transpose(0, 1),
        expected_logits.mask().t(),
    )

    actual = crf_partitions(
        logits=rua_logits(inputs),
        bias=(expected_crf.transitions, expected_crf.start_transitions, expected_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=expected, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs, rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_logits=st.sampled_from([C.new, D.new, P.new]),
)
def test_crf_argmax(token_sizes, num_targets, rua_logits):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected_crf = CRF(num_tags=num_targets, batch_first=False).to(device=device)

    expected_logits = D.new(inputs)

    expected = expected_crf.decode(
        expected_logits.data.transpose(0, 1),
        expected_logits.mask().t(),
    )
    expected = C.new([torch.tensor(tensor, device=device) for tensor in expected])

    actual_crf = CrfDecoder(num_targets=num_targets)
    actual_crf.bias = expected_crf.transitions
    actual_crf.head_bias = expected_crf.start_transitions
    actual_crf.last_bias = expected_crf.end_transitions

    actual = actual_crf(rua_logits(inputs)).argmax.cat()

    assert_sequence_close(actual=actual, expected=expected)
