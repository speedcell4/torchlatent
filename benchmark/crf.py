import torch
from torchrua import pack_sequence
from tqdm import tqdm

from benchmark.meter import TimeMeter
from tests.third_party import ThirdPartyCrfDecoder
from torchlatent.crf import CrfDecoder


def benchmark_crf(num_tags: int = 50, num_conjugates: int = 1, num_runs: int = 100,
                  batch_size: int = 32, max_token_size: int = 512):
    j1, f1, b1, d1, = TimeMeter(), TimeMeter(), TimeMeter(), TimeMeter()
    j2, f2, b2, d2, = TimeMeter(), TimeMeter(), TimeMeter(), TimeMeter()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'device => {device}')

    decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugates).to(device=device)
    print(f'decoder => {decoder}')

    third_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugates).to(device=device)
    print(f'third_decoder => {third_decoder}')

    for _ in tqdm(range(num_runs)):
        token_sizes = torch.randint(1, max_token_size + 1, (batch_size,), device=device).detach().cpu().tolist()

        emissions = pack_sequence([
            torch.randn((token_size, num_conjugates, num_tags), device=device, requires_grad=True)
            for token_size in token_sizes
        ])

        tags = pack_sequence([
            torch.randint(0, num_tags, (token_size, num_conjugates), device=device)
            for token_size in token_sizes
        ])

        with j1:
            indices = decoder.compile_indices(emissions=emissions, tags=tags)

        with f1:
            loss = decoder.fit(emissions=emissions, tags=tags, indices=indices).neg().mean()

        with b1:
            _, torch.autograd.grad(loss, emissions.data, torch.ones_like(loss))

        with d1:
            _ = decoder.decode(emissions=emissions, indices=indices)

        with f2:
            loss = third_decoder.fit(emissions=emissions, tags=tags).neg().mean()

        with b2:
            _, torch.autograd.grad(loss, emissions.data, torch.ones_like(loss))

        with d2:
            _ = third_decoder.decode(emissions=emissions)

    print(f'TorchLatent ({j1.merit + f1.merit + b1.merit:.6f}) => {j1} {f1} {b1} {d1}')
    print(f'Third       ({j2.merit + f2.merit + b2.merit:.6f}) => {j2} {f2} {b2} {d2}')
