import torch
import torchcrf
from hypothesis import given
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchrua import pad_packed_sequence, token_sizes_to_mask, pack_sequence

from tests.strategies import devices, token_size_lists, conjugate_sizes, tag_sizes
from tests.utils import assert_close, assert_grad_close, assert_packed_equal
from torchlatent.crf import CrfDecoder


class ThirdPartyCrfDecoder(nn.Module):
    def __init__(self, num_tags: int, num_conjugates: int) -> None:
        super(ThirdPartyCrfDecoder, self).__init__()
        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

        self.decoders = nn.ModuleList([
            torchcrf.CRF(num_tags=num_tags, batch_first=False)
            for _ in range(num_conjugates)
        ])

    @torch.no_grad()
    def reset_parameters_with_(self, decoder: CrfDecoder) -> None:
        assert self.num_tags == decoder.num_tags
        assert self.num_conjugates == decoder.num_conjugates

        for index in range(self.num_conjugates):
            self.decoders[index].transitions.data[::] = decoder.transitions[:, index, :, :]
            self.decoders[index].start_transitions.data[::] = decoder.head_transitions[:, index, :]
            self.decoders[index].end_transitions.data[::] = decoder.tail_transitions[:, index, :]

    def fit(self, emissions: PackedSequence, tags: PackedSequence, **kwargs) -> Tensor:
        num_emissions_conjugates = emissions.data.size()[1]
        num_decoders_conjugates = self.num_conjugates
        num_conjugates = max(num_emissions_conjugates, num_decoders_conjugates)

        emissions, token_sizes = pad_packed_sequence(emissions, batch_first=False)
        tags, _ = pad_packed_sequence(tags, batch_first=False)
        mask = token_sizes_to_mask(token_sizes=token_sizes, batch_first=False)

        log_probs = []
        for index in range(num_conjugates):
            decoder = self.decoders[index % num_decoders_conjugates]
            emission = emissions[:, :, index % num_emissions_conjugates]
            tag = tags[:, :, index % num_emissions_conjugates]

            log_probs.append(decoder(emissions=emission, tags=tag, mask=mask, reduction='none'))

        return torch.stack(log_probs, dim=-1)

    def decode(self, emissions: PackedSequence, **kwargs) -> PackedSequence:
        num_emissions_conjugates = emissions.data.size()[1]
        num_decoders_conjugates = self.num_conjugates
        num_conjugates = max(num_emissions_conjugates, num_decoders_conjugates)

        emissions, token_sizes = pad_packed_sequence(emissions, batch_first=False)
        mask = token_sizes_to_mask(token_sizes=token_sizes, batch_first=False)

        predictions = []
        for index in range(num_conjugates):
            decoder = self.decoders[index % num_decoders_conjugates]
            emission = emissions[:, :, index % num_emissions_conjugates]

            prediction = decoder.decode(emissions=emission, mask=mask)
            predictions.append(pack_sequence([torch.tensor(p) for p in prediction], device=emissions.device))

        return PackedSequence(
            torch.stack([prediction.data for prediction in predictions], dim=1),
            batch_sizes=predictions[0].batch_sizes,
            sorted_indices=predictions[0].sorted_indices,
            unsorted_indices=predictions[0].unsorted_indices,
        )


@given(
    device=devices(),
    token_sizes=token_size_lists(),
    num_conjugate=conjugate_sizes(),
    num_tags=tag_sizes(),
)
def test_crf_decoder_fit(device, token_sizes, num_conjugate, num_tags):
    emissions = pack_sequence([
        torch.randn((token_size, num_conjugate, num_tags), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    tags = pack_sequence([
        torch.randint(0, num_tags, (token_size, num_conjugate), device=device)
        for token_size in token_sizes
    ], device=device)

    third_party_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate)
    rua_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate)
    third_party_decoder.reset_parameters_with_(decoder=rua_decoder)

    target = third_party_decoder.fit(emissions=emissions, tags=tags)
    prediction = rua_decoder.fit(emissions=emissions, tags=tags)

    assert_close(prediction, target)
    assert_grad_close(prediction, target, (emissions.data,))


@given(
    device=devices(),
    token_sizes=token_size_lists(),
    num_conjugate=conjugate_sizes(),
    num_tags=tag_sizes(),
)
def test_crf_decoder_decode(device, token_sizes, num_conjugate, num_tags):
    emissions = pack_sequence([
        torch.randn((token_size, num_conjugate, num_tags), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    third_party_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate)
    rua_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate)
    third_party_decoder.reset_parameters_with_(decoder=rua_decoder)

    target = third_party_decoder.decode(emissions=emissions)
    prediction = rua_decoder.decode(emissions=emissions)

    assert_packed_equal(prediction, target)
