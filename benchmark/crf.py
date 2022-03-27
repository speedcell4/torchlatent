import torch
from torchrua import pack_sequence, cat_sequence
from tqdm import tqdm

from benchmark.meter import TimeMeter
from third.crf import CrfDecoder as ThirdPartyCrfDecoder
from torchlatent.crf import CrfDecoder


def benchmark_crf(num_tags: int = 50, num_conjugates: int = 1, num_runs: int = 100,
                  batch_size: int = 32, max_token_size: int = 512):
    jit1, fwd1, bwd1, dec1, = TimeMeter(), TimeMeter(), TimeMeter(), TimeMeter()
    jit2, fwd2, bwd2, dec2, = TimeMeter(), TimeMeter(), TimeMeter(), TimeMeter()
    jit3, fwd3, bwd3, dec3, = TimeMeter(), TimeMeter(), TimeMeter(), TimeMeter()

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

        catted_emissions = cat_sequence([
            torch.randn((token_size, num_conjugates, num_tags), device=device, requires_grad=True)
            for token_size in token_sizes
        ])
        catted_tags = cat_sequence([
            torch.randint(0, num_tags, (token_size, num_conjugates), device=device)
            for token_size in token_sizes
        ])

        packed_emissions = pack_sequence([
            torch.randn((token_size, num_conjugates, num_tags), device=device, requires_grad=True)
            for token_size in token_sizes
        ])
        packed_tags = pack_sequence([
            torch.randint(0, num_tags, (token_size, num_conjugates), device=device)
            for token_size in token_sizes
        ])

        with jit1:
            indices = decoder.compile_indices(emissions=packed_emissions, tags=packed_tags)

        with fwd1:
            loss = decoder.fit(emissions=packed_emissions, tags=packed_tags, indices=indices).neg().mean()

        with bwd1:
            _, torch.autograd.grad(loss, packed_emissions.data, torch.randn_like(loss))

        with dec1:
            _ = decoder.decode(emissions=packed_emissions, indices=indices)

        with jit2:
            indices = decoder.compile_indices(emissions=catted_emissions, tags=catted_tags)

        with fwd2:
            loss = decoder.fit(emissions=catted_emissions, tags=catted_tags, indices=indices).neg().mean()

        with bwd2:
            _, torch.autograd.grad(loss, catted_emissions.data, torch.randn_like(loss))

        with dec2:
            _ = decoder.decode(emissions=catted_emissions, indices=indices)

        with fwd3:
            loss = third_decoder.fit(emissions=packed_emissions, tags=packed_tags).neg().mean()

        with bwd3:
            _, torch.autograd.grad(loss, packed_emissions.data, torch.randn_like(loss))

        with dec3:
            _ = third_decoder.decode(emissions=packed_emissions)

    print(f'PackedLatent ({jit1.merit + fwd1.merit + bwd1.merit:.6f}) => {jit1} {fwd1} {bwd1} {dec1}')
    print(f'CattedLatent ({jit2.merit + fwd2.merit + bwd2.merit:.6f}) => {jit2} {fwd2} {bwd2} {dec2}')
    print(f'Third        ({jit3.merit + fwd3.merit + bwd3.merit:.6f}) => {jit3} {fwd3} {bwd3} {dec3}')
