from datetime import datetime

import torch
from aku import App
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchcrf import CRF
from torchrua import lengths_to_mask
from tqdm import tqdm

from torchlatent import CrfDecoder


class Timer(object):
    def __init__(self):
        self.seconds = 0

    def __enter__(self):
        self.start_tm = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds += (datetime.now() - self.start_tm).total_seconds()


def gen_pad(lengths: Tensor, num_tag: int, device: torch.device):
    emissions = torch.randn(
        (lengths.size(0), lengths.max().item(), num_tag),
        dtype=torch.float32, device=device, requires_grad=True,
    )
    tags = torch.randint(0, num_tag, (lengths.size(0), lengths.max().item()), dtype=torch.long, device=device)
    mask = lengths_to_mask(lengths=lengths, filling_mask=True, batch_first=True, device=device)
    return emissions, tags, mask


def gen_pack(lengths: Tensor, num_tag: int, device: torch.device):
    emissions = torch.randn(
        (lengths.size(0), lengths.max().item(), num_tag),
        dtype=torch.float32, device=device, requires_grad=True,
    )
    emissions = pack_padded_sequence(
        emissions, lengths=lengths, batch_first=True, enforce_sorted=False,
    )
    emissions.data.requires_grad_(True)

    tags = torch.randint(0, num_tag, (lengths.size(0), lengths.max().item()), dtype=torch.long, device=device)
    tags = pack_padded_sequence(
        tags, lengths=lengths, batch_first=True, enforce_sorted=False,
    )

    return emissions, tags


def check_pad(decoder: CRF, batched_lengths, num_tags, device):
    data_timer, forward_timer, backward_timer, decode_timer = Timer(), Timer(), Timer(), Timer()
    for lengths in tqdm(batched_lengths):
        with data_timer:
            emissions, tags, mask = gen_pad(lengths=lengths, num_tag=num_tags, device=device)
        with forward_timer:
            loss = decoder(emissions=emissions, tags=tags, mask=mask, reduction='sum')
        with backward_timer:
            decoder.zero_grad()
            loss.backward()
        with decode_timer:
            decoder.decode(emissions=emissions, mask=mask)

    print(f'torchcrf.forward => {forward_timer.seconds:.4f}')
    print(f'torchcrf.backward => {backward_timer.seconds:.4f}')
    print(f'torchcrf.decode => {decode_timer.seconds:.4f}')


def check_pack(decoder: CrfDecoder, batched_lengths, num_tags, device):
    data_timer, compile_timer, forward_timer, backward_timer, decode_timer = Timer(), Timer(), Timer(), Timer(), Timer()
    for lengths in tqdm(batched_lengths):
        try:
            with data_timer:
                emissions, tags = gen_pack(lengths=lengths, num_tag=num_tags, device=device)
            with compile_timer:
                emissions, tags, batch_ptr, instr = decoder._validate(
                    emissions, tags, lengths=None, batch_ptr=None, instr=None)
            with forward_timer:
                loss = decoder.fit(emissions, tags, batch_ptr=batch_ptr, instr=instr).sum()
            with backward_timer:
                decoder.zero_grad()
                loss.backward()
            with decode_timer:
                predictions = decoder.decode(emissions, batch_ptr=batch_ptr, instr=instr)
                predictions, lengths = pad_packed_sequence(
                    predictions, batch_first=True,
                )
                predictions = predictions.detach().cpu()
                _ = [
                    predictions[i][:length].tolist()
                    for i, length in enumerate(lengths.detach().cpu().tolist())
                ]
        except RuntimeError as error:
            print(lengths)
            raise error

    print(f'torchlatent.compile => {compile_timer.seconds:.4f}')
    print(f'torchlatent.forward => {forward_timer.seconds:.4f}')
    print(f'torchlatent.backward => {backward_timer.seconds:.4f}')
    print(f'torchlatent.decode => {decode_timer.seconds:.4f}')


app = App()


@app.register
def main(num_examples: int = 100, batch_size: int = 10, total_length: int = 120, num_tags: int = 10, device: int = -1):
    if device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device}')

    batched_lengths = [
        torch.randint(0, total_length, (batch_size,), device=device) + 1
        for _ in range(num_examples)
    ]
    our_decoder = CrfDecoder(num_tags=num_tags).to(device=device)
    their_decoder = CRF(num_tags=num_tags, batch_first=True).to(device=device)

    check_pack(our_decoder, batched_lengths=batched_lengths, num_tags=num_tags, device=device)
    check_pad(their_decoder, batched_lengths=batched_lengths, num_tags=num_tags, device=device)


if __name__ == '__main__':
    app.run()
