import torch
import torchcrf
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import pad_catted_indices, pad_packed_sequence, pack_sequence


@torch.no_grad()
def token_sizes_to_mask(sizes: Tensor, batch_first: bool, device: Device = None) -> Tensor:
    if device is None:
        device = sizes.device

    size, ptr = pad_catted_indices(sizes, batch_first=batch_first, device=device)
    mask = torch.zeros(size, device=device, dtype=torch.bool)
    mask[ptr] = True
    return mask


class CrfDecoder(nn.Module):
    def __init__(self, num_tags: int, num_conjugates: int) -> None:
        super(CrfDecoder, self).__init__()
        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

        self.decoders = nn.ModuleList([
            torchcrf.CRF(num_tags=num_tags, batch_first=False)
            for _ in range(num_conjugates)
        ])

    @torch.no_grad()
    def reset_parameters_with_(self, decoder) -> None:
        assert self.num_tags == decoder.num_tags
        assert self.num_conjugates == decoder.num_conjugates

        for index in range(self.num_conjugates):
            self.decoders[index].transitions.data[::] = decoder.transitions[:, index, :, :]
            self.decoders[index].start_transitions.data[::] = decoder.head_transitions[:, index, :]
            self.decoders[index].end_transitions.data[::] = decoder.last_transitions[:, index, :]

    def fit(self, emissions: PackedSequence, tags: PackedSequence, **kwargs) -> Tensor:
        num_emissions_conjugates = emissions.data.size()[1]
        num_decoders_conjugates = self.num_conjugates
        num_conjugates = max(num_emissions_conjugates, num_decoders_conjugates)

        emissions, token_sizes = pad_packed_sequence(emissions, batch_first=False)
        tags, _ = pad_packed_sequence(tags, batch_first=False)
        mask = token_sizes_to_mask(sizes=token_sizes, batch_first=False)

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
        mask = token_sizes_to_mask(sizes=token_sizes, batch_first=False)

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
