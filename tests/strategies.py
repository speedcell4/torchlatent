import torch

from hypothesis import strategies as st

BATCH_SIZE = 25
TOKEN_SIZE = 50
NUM_CONJUGATES = 5
NUM_TAGS = 15
EMBEDDING_DIM = 16

TINY_BATCH_SIZE = 5
TINY_TOKEN_SIZE = 10
TINY_NUM_CONJUGATES = 3
TINY_NUM_TAGS = 3
TINY_EMBEDDING_DIM = 4

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
_ = torch.empty((1,), device=device)


@st.composite
def sizes(draw, *max_sizes: int, min_size: int = 1):
    max_size, *max_sizes = max_sizes
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    if len(max_sizes) == 0:
        return n
    return [draw(sizes(*max_sizes, min_size=min_size)) for _ in range(n)]
