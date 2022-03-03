import torch

from hypothesis import strategies as st

TINY_BATCH_SIZE = 6
TINY_TOKEN_SIZE = 12

BATCH_SIZE = 24
TOKEN_SIZE = 50
NUM_TAGS = 8
NUM_CONJUGATES = 5


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    _ = torch.empty((1,), device=device)
    return device


@st.composite
def sizes(draw, *size: int, min_size: int = 1):
    max_size, *size = size

    if len(size) == 0:
        return draw(st.integers(min_value=min_size, max_value=max_size))
    else:
        return [
            draw(sizes(*size, min_size=min_size))
            for _ in range(draw(st.integers(min_value=min_size, max_value=max_size)))
        ]
