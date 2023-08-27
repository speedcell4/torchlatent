import torch
from hypothesis import given
from hypothesis import strategies as st
from torchcrf import CRF
from torchnyan import BATCH_SIZE
from torchnyan import TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import assert_grad_close
from torchnyan import assert_sequence_close
from torchnyan import device
from torchnyan import sizes
from torchrua import C
from torchrua import D
from torchrua import P

from torchlatent.crf import CrfDecoder
from torchlatent.crf import crf_partitions
from torchlatent.crf import crf_scores
from torchlatent.semiring import Log


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_emissions=st.sampled_from([C.new, D.new, P.new]),
    rua_targets=st.sampled_from([C.new, D.new, P.new]),
)
def test_crf_scores(token_sizes, num_targets, rua_emissions, rua_targets):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    targets = [
        torch.randint(0, num_targets, (token_size,), device=device)
        for token_size in token_sizes
    ]

    expected_crf = CRF(num_tags=num_targets, batch_first=False).to(device=device)

    expected_emissions = D.new(inputs)
    expected_tags = D.new(targets)

    expected = expected_crf._compute_score(
        expected_emissions.data.transpose(0, 1),
        expected_tags.data.transpose(0, 1),
        expected_emissions.mask().transpose(0, 1),
    )

    actual = crf_scores(
        emissions=rua_emissions(inputs),
        targets=rua_targets(targets),
        transitions=(expected_crf.transitions, expected_crf.start_transitions, expected_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_emissions=st.sampled_from([C.new, D.new, P.new]),
)
def test_crf_partitions(token_sizes, num_targets, rua_emissions):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected_crf = CRF(num_tags=num_targets, batch_first=False).to(device=device)

    expected_emissions = D.new(inputs)

    expected = expected_crf._compute_normalizer(
        expected_emissions.data.transpose(0, 1),
        expected_emissions.mask().t(),
    )

    actual = crf_partitions(
        emissions=rua_emissions(inputs),
        transitions=(expected_crf.transitions, expected_crf.start_transitions, expected_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=expected, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_emissions=st.sampled_from([C.new, D.new, P.new]),
)
def test_crf_argmax(token_sizes, num_targets, rua_emissions):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected_crf = CRF(num_tags=num_targets, batch_first=False).to(device=device)

    expected_emissions = D.new(inputs)

    expected = expected_crf.decode(
        expected_emissions.data.transpose(0, 1),
        expected_emissions.mask().t(),
    )
    expected = C.new([torch.tensor(tensor, device=device) for tensor in expected])

    actual_crf = CrfDecoder(num_targets=num_targets)
    actual_crf.transitions = expected_crf.transitions
    actual_crf.head_transitions = expected_crf.start_transitions
    actual_crf.last_transitions = expected_crf.end_transitions

    actual = actual_crf(rua_emissions(inputs)).argmax.cat()

    assert_sequence_close(actual=actual, expected=expected)
