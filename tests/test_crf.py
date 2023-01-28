import torch
import torchcrf
from hypothesis import given
from torchrua import cat_sequence, pad_catted_indices, pack_catted_indices
from torchrua import pad_sequence, pad_packed_indices, pack_sequence

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import device, sizes, TOKEN_SIZE, TINY_BATCH_SIZE, NUM_CONJUGATES, TINY_TOKEN_SIZE
from torchlatent.crf import CrfLayer


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_catted_scores(token_sizes, num_tags):
    actual_decoder = CrfLayer(num_tags).to(device=device)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False).to(device=device)

    excepted_decoder.transitions.data = torch.randn_like(excepted_decoder.transitions)
    excepted_decoder.start_transitions.data = torch.randn_like(excepted_decoder.start_transitions)
    excepted_decoder.end_transitions.data = torch.randn_like(excepted_decoder.end_transitions)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence([x[:, None] for x in emissions])
    catted_targets = cat_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr = pad_catted_indices(token_sizes=catted_emissions.token_sizes, batch_first=False)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.forward(emissions=catted_emissions).log_scores(targets=catted_targets)[:, 0]
    excepted = excepted_decoder._compute_score(
        emissions=padded_emissions, tags=padded_targets.long(),
        mask=mask.byte(),
    )

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_catted_fit(token_sizes, num_tags):
    actual_decoder = CrfLayer(num_tags).to(device=device)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False).to(device=device)

    excepted_decoder.transitions.data = torch.randn_like(excepted_decoder.transitions)
    excepted_decoder.start_transitions.data = torch.randn_like(excepted_decoder.start_transitions)
    excepted_decoder.end_transitions.data = torch.randn_like(excepted_decoder.end_transitions)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence([x[:, None] for x in emissions])
    catted_targets = cat_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr = pad_catted_indices(token_sizes=catted_emissions.token_sizes, batch_first=False)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.fit(emissions=catted_emissions, targets=catted_targets)[:, 0]
    excepted = excepted_decoder.forward(
        emissions=padded_emissions, tags=padded_targets.long(),
        mask=mask.byte(), reduction='none',
    ).neg()

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_catted_decode(token_sizes, num_tags):
    actual_decoder = CrfLayer(num_tags).to(device=device)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False).to(device=device)

    excepted_decoder.transitions.data = torch.randn_like(excepted_decoder.transitions)
    excepted_decoder.start_transitions.data = torch.randn_like(excepted_decoder.start_transitions)
    excepted_decoder.end_transitions.data = torch.randn_like(excepted_decoder.end_transitions)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence([x[:, None] for x in emissions])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)

    size, ptr = pad_catted_indices(token_sizes=catted_emissions.token_sizes, batch_first=False)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual, actual_token_sizes = actual_decoder.decode(emissions=catted_emissions)

    excepted = excepted_decoder.decode(emissions=padded_emissions, mask=mask.byte())
    excepted, excepted_token_sizes = cat_sequence([torch.tensor(x, device=device) for x in excepted])

    assert_close(actual=actual[:, 0], expected=excepted)
    assert_close(actual=actual_token_sizes, expected=excepted_token_sizes)


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_packed_scores(token_sizes, num_tags):
    actual_decoder = CrfLayer(num_tags).to(device=device)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False).to(device=device)

    excepted_decoder.transitions.data = torch.randn_like(excepted_decoder.transitions)
    excepted_decoder.start_transitions.data = torch.randn_like(excepted_decoder.start_transitions)
    excepted_decoder.end_transitions.data = torch.randn_like(excepted_decoder.end_transitions)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    packed_emissions = pack_sequence([x[:, None] for x in emissions])
    packed_targets = pack_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr, _ = pad_packed_indices(
        batch_sizes=packed_emissions.batch_sizes,
        sorted_indices=packed_emissions.sorted_indices,
        unsorted_indices=packed_emissions.unsorted_indices,
        batch_first=False,
    )
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.forward(emissions=packed_emissions).log_scores(targets=packed_targets)[:, 0]
    excepted = excepted_decoder._compute_score(
        emissions=padded_emissions, tags=padded_targets.long(),
        mask=mask.byte(),
    )

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_packed_fit(token_sizes, num_tags):
    actual_decoder = CrfLayer(num_tags).to(device=device)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False).to(device=device)

    excepted_decoder.transitions.data = torch.randn_like(excepted_decoder.transitions)
    excepted_decoder.start_transitions.data = torch.randn_like(excepted_decoder.start_transitions)
    excepted_decoder.end_transitions.data = torch.randn_like(excepted_decoder.end_transitions)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    packed_emissions = pack_sequence([x[:, None] for x in emissions])
    packed_targets = pack_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr, _ = pad_packed_indices(
        batch_sizes=packed_emissions.batch_sizes,
        sorted_indices=packed_emissions.sorted_indices,
        unsorted_indices=packed_emissions.unsorted_indices,
        batch_first=False,
    )
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.fit(emissions=packed_emissions, targets=packed_targets)[:, 0]
    excepted = excepted_decoder.forward(
        emissions=padded_emissions, tags=padded_targets.long(),
        mask=mask.byte(), reduction='none',
    ).neg()

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_packed_decode(token_sizes, num_tags):
    actual_decoder = CrfLayer(num_tags).to(device=device)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False).to(device=device)

    excepted_decoder.transitions.data = torch.randn_like(excepted_decoder.transitions)
    excepted_decoder.start_transitions.data = torch.randn_like(excepted_decoder.start_transitions)
    excepted_decoder.end_transitions.data = torch.randn_like(excepted_decoder.end_transitions)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    packed_emissions = pack_sequence([x[:, None] for x in emissions])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)

    size, ptr, _ = pad_packed_indices(
        batch_sizes=packed_emissions.batch_sizes,
        sorted_indices=packed_emissions.sorted_indices,
        unsorted_indices=packed_emissions.unsorted_indices,
        batch_first=False,
    )
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.decode(emissions=packed_emissions)

    excepted = excepted_decoder.decode(emissions=padded_emissions, mask=mask.byte())
    excepted = pack_sequence([torch.tensor(x, device=device) for x in excepted])

    assert_close(actual=actual.data[:, 0], expected=excepted.data)
    assert_close(actual=actual.batch_sizes, expected=excepted.batch_sizes)
    assert_close(actual=actual.sorted_indices, expected=excepted.sorted_indices)
    assert_close(actual=actual.unsorted_indices, expected=excepted.unsorted_indices)


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
    num_conjugates=sizes(NUM_CONJUGATES),
    num_tags=sizes(TINY_TOKEN_SIZE),
)
def test_conjugated_catted_fit(token_sizes, num_conjugates, num_tags):
    decoder = CrfLayer(num_targets=num_tags, num_conjugates=num_conjugates).to(device=device)
    decoders = [CrfLayer(num_targets=num_tags, num_conjugates=1).to(device=device) for _ in range(num_conjugates)]

    for index in range(num_conjugates):
        decoders[index].transitions.data = torch.randn_like(decoders[index].transitions)
        decoders[index].head_transitions.data = torch.randn_like(decoders[index].head_transitions)
        decoders[index].last_transitions.data = torch.randn_like(decoders[index].last_transitions)

        decoder.transitions.data[:, index] = decoders[index].transitions
        decoder.head_transitions.data[:, index] = decoders[index].head_transitions
        decoder.last_transitions.data[:, index] = decoders[index].last_transitions

    emissions = [[
        torch.randn((token_size, 1, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ] for _ in range(num_conjugates)]

    targets = [[
        torch.randint(0, num_tags, (token_size, 1), device=device)
        for token_size in token_sizes
    ] for _ in range(num_conjugates)]

    actual = decoder.fit(
        emissions=cat_sequence([torch.cat(sequences, dim=1) for sequences in zip(*emissions)], device=device),
        targets=cat_sequence([torch.cat(sequences, dim=1) for sequences in zip(*targets)], device=device),
    )

    expected = torch.cat([
        decoders[index].fit(
            emissions=cat_sequence(emissions[index], device=device),
            targets=cat_sequence(targets[index], device=device),
        )
        for index in range(num_conjugates)
    ], dim=1)

    assert_close(actual=actual, expected=expected, rtol=1e-4, atol=1e-4)
    assert_grad_close(
        actual=actual, expected=expected,
        inputs=[x for xs in emissions for x in xs],
        rtol=1e-4, atol=1e-4, check_stride=False,
    )


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
    num_conjugates=sizes(NUM_CONJUGATES),
    num_tags=sizes(TINY_TOKEN_SIZE),
)
def test_conjugated_packed_fit(token_sizes, num_conjugates, num_tags):
    decoder = CrfLayer(num_targets=num_tags, num_conjugates=num_conjugates).to(device=device)
    decoders = [CrfLayer(num_targets=num_tags, num_conjugates=1).to(device=device) for _ in range(num_conjugates)]

    for index in range(num_conjugates):
        decoders[index].transitions.data = torch.randn_like(decoders[index].transitions)
        decoders[index].head_transitions.data = torch.randn_like(decoders[index].head_transitions)
        decoders[index].last_transitions.data = torch.randn_like(decoders[index].last_transitions)

        decoder.transitions.data[:, index] = decoders[index].transitions
        decoder.head_transitions.data[:, index] = decoders[index].head_transitions
        decoder.last_transitions.data[:, index] = decoders[index].last_transitions

    emissions = [[
        torch.randn((token_size, 1, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ] for _ in range(num_conjugates)]

    targets = [[
        torch.randint(0, num_tags, (token_size, 1), device=device)
        for token_size in token_sizes
    ] for _ in range(num_conjugates)]

    actual = decoder.fit(
        emissions=pack_sequence([torch.cat(sequences, dim=1) for sequences in zip(*emissions)], device=device),
        targets=pack_sequence([torch.cat(sequences, dim=1) for sequences in zip(*targets)], device=device),
    )

    expected = torch.cat([
        decoders[index].fit(
            emissions=pack_sequence(emissions[index], device=device),
            targets=pack_sequence(targets[index], device=device),
        )
        for index in range(num_conjugates)
    ], dim=1)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(
        actual=actual, expected=expected,
        inputs=[x for xs in emissions for x in xs],
        rtol=1e-4, atol=1e-4, check_stride=False,
    )


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
    num_conjugates=sizes(NUM_CONJUGATES),
    num_tags=sizes(TINY_TOKEN_SIZE),
)
def test_dynamic_fit(token_sizes, num_conjugates, num_tags):
    packed_decoder = CrfLayer(num_targets=num_tags, num_conjugates=num_conjugates).to(device=device)
    catted_decoder = CrfLayer(num_targets=num_tags, num_conjugates=num_conjugates).to(device=device)

    emissions = [
        torch.randn((token_size, 1, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    targets = [
        torch.randint(0, num_tags, (token_size, 1), device=device)
        for token_size in token_sizes
    ]

    catted_decoder.transitions.data = torch.randn((sum(token_sizes), num_conjugates, num_tags, num_tags), device=device)
    catted_decoder.head_transitions.data = torch.randn((len(token_sizes), num_conjugates, num_tags), device=device)
    catted_decoder.last_transitions.data = torch.randn((len(token_sizes), num_conjugates, num_tags), device=device)

    token_sizes = torch.tensor(token_sizes, device=device)
    indices, _, sorted_indices, _ = pack_catted_indices(token_sizes=token_sizes, device=device)

    packed_decoder.transitions.data = catted_decoder.transitions[indices]
    packed_decoder.head_transitions.data = catted_decoder.head_transitions[sorted_indices]
    packed_decoder.last_transitions.data = catted_decoder.last_transitions[sorted_indices]

    packed_fit = packed_decoder.fit(
        emissions=pack_sequence(emissions, device=device),
        targets=pack_sequence(targets, device=device),
    )

    catted_fit = catted_decoder.fit(
        emissions=cat_sequence(emissions, device=device),
        targets=cat_sequence(targets, device=device),
    )

    assert_close(actual=packed_fit, expected=catted_fit, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=catted_fit, expected=catted_fit, inputs=emissions, rtol=1e-4, atol=1e-4)
