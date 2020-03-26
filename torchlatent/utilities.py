from typing import Optional, List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def reduce_tensor(x: Tensor, reduction: Optional[str]) -> Tensor:
    if reduction is None or reduction == 'none':
        return x
    if reduction == 'sum':
        return x.sum()
    if reduction == 'mean':
        return x.mean()
    raise NotImplementedError(f'reduction {reduction} is not supported')


def build_ins(pack: PackedSequence) -> Tuple[Tensor, Tensor, List[int], int]:
    indices, lengths = pad_packed_sequence(pack, batch_first=True)
    indices = [
        index[:length] for index, length in zip(
            indices.detach().cpu().tolist(),
            lengths.detach().cpu().tolist(),
        )
    ]

    ins, batch_sizes = [], []
    target = lengths.sum().detach().cpu().item()

    while True:
        batch_size = 0
        for i, index in enumerate(indices):
            new_datum = []
            for lhs, rhs in zip(index[0::2], index[1::2]):
                ins.append((lhs, rhs, target))
                new_datum.append(target)
                target += 1
                batch_size += 1
            if len(index) % 2 == 1:
                new_datum.append(index[-1])
            indices[i] = new_datum
        if batch_size == 0:
            break
        batch_sizes.append(batch_size)

    ins = torch.tensor(ins, dtype=torch.long, device=pack.data.device)
    res = torch.tensor(indices, dtype=torch.long, device=pack.data.device)
    return ins, res[..., 0], batch_sizes, target - pack.data.size(0)
