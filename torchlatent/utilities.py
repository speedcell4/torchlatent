from typing import Optional

from torch import Tensor


def reduce_tensor(x: Tensor, reduction: Optional[str]) -> Tensor:
    if reduction is None or reduction == 'none':
        return x
    if reduction == 'sum':
        return x.sum()
    if reduction == 'mean':
        return x.mean()
    raise NotImplementedError(f'reduction {reduction} is not supported')
