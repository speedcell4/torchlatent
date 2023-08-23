import torch
from hypothesis import given
from hypothesis import strategies as st
from torchcrf import CRF

from torchlatent.linear_crf import CrfDecoder
from torchlatent.linear_crf import crf_partitions
from torchlatent.linear_crf import crf_scores
from torchlatent.semiring import Log
from torchnyan import BATCH_SIZE
from torchnyan import TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import assert_grad_close
from torchnyan import assert_sequence_close
from torchnyan import device
from torchnyan import sizes
from torchrua import cat_sequence
from torchrua import pack_sequence
from torchrua import pad_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_emissions=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
    rua_targets=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
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

    excepted_crf = CRF(num_tags=num_targets, batch_first=False)

    excepted_emissions = pad_sequence(inputs)
    excepted_tags = pad_sequence(targets)

    excepted = excepted_crf._compute_score(
        excepted_emissions.data.transpose(0, 1),
        excepted_tags.data.transpose(0, 1),
        excepted_emissions.mask().transpose(0, 1),
    )

    actual = crf_scores(
        emissions=rua_emissions(inputs),
        targets=rua_targets(targets),
        transitions=(excepted_crf.transitions, excepted_crf.start_transitions, excepted_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_emissions=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
)
def test_crf_partitions(token_sizes, num_targets, rua_emissions):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted_crf = CRF(num_tags=num_targets, batch_first=False)

    excepted_emissions = pad_sequence(inputs)

    excepted = excepted_crf._compute_normalizer(
        excepted_emissions.data.transpose(0, 1),
        excepted_emissions.mask().t(),
    )

    actual = crf_partitions(
        emissions=rua_emissions(inputs),
        transitions=(excepted_crf.transitions, excepted_crf.start_transitions, excepted_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
    rua_emissions=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
)
def test_crf_argmax(token_sizes, num_targets, rua_emissions):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted_crf = CRF(num_tags=num_targets, batch_first=False)

    excepted_emissions = pad_sequence(inputs)

    excepted = excepted_crf.decode(
        excepted_emissions.data.transpose(0, 1),
        excepted_emissions.mask().t(),
    )
    excepted = cat_sequence([torch.tensor(tensor, device=device) for tensor in excepted])

    actual_crf = CrfDecoder(num_targets=num_targets)
    actual_crf.transitions = excepted_crf.transitions
    actual_crf.head_transitions = excepted_crf.start_transitions
    actual_crf.last_transitions = excepted_crf.end_transitions

    actual = actual_crf(rua_emissions(inputs)).argmax.cat()

    assert_sequence_close(actual=actual, expected=excepted)
