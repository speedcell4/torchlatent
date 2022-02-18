import torch
from torchrua import cat_sequence
from tqdm import tqdm

from benchmark.meter import TimeMeter
from torchlatent.crf import CrfDecoder


def benchmark_crf(num_tags: int = 32, num_conjugates: int = 4, num_runs: int = 100,
                  batch_size: int = 120, max_token_size: int = 512):
    jit_timer, fwd_timer, bwd_timer, dec_timer, = TimeMeter(), TimeMeter(), TimeMeter(), TimeMeter()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'device => {device}')

    decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugates).to(device=device)
    print(f'decoder => {decoder}')

    for _ in tqdm(range(num_runs)):
        token_sizes = torch.randint(1, max_token_size + 1, (batch_size,), device=device).detach().cpu().tolist()

        emissions = cat_sequence([
            torch.randn((token_size, num_conjugates, num_tags), device=device, requires_grad=True)
            for token_size in token_sizes
        ])

        tags = cat_sequence([
            torch.randint(0, num_tags, (token_size, num_conjugates), device=device)
            for token_size in token_sizes
        ])

        with jit_timer:
            indices = decoder.compile_indices(emissions=emissions, tags=tags)

        with fwd_timer:
            loss = decoder.fit(emissions=emissions, tags=tags, indices=indices).neg().mean()

        with bwd_timer:
            _, torch.autograd.grad(loss, emissions.data, torch.ones_like(loss))

        with dec_timer:
            _ = decoder.decode(emissions=emissions, indices=indices)

    print(f'compile => {jit_timer}')
    print(f'forward => {fwd_timer}')
    print(f'backward => {bwd_timer}')
    print(f'decode => {dec_timer}')
