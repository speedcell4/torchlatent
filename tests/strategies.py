from typing import List

import torch
from hypothesis import strategies as st
from torch.nn.utils.rnn import pack_sequence

MAX_BATCH_SIZE = 7
TOTAL_LENGTH = 11
MAX_NUM_TAGS = 13


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device('cuda')


@st.composite
def batch_size_integers(draw, max_batch_size: int = MAX_BATCH_SIZE):
    return draw(st.integers(min_value=1, max_value=max_batch_size))


@st.composite
def length_integers(draw, total_length: int = TOTAL_LENGTH):
    return draw(st.integers(min_value=1, max_value=total_length))


@st.composite
def length_lists(draw, total_length: int = TOTAL_LENGTH, batch_sizes: int = MAX_BATCH_SIZE):
    return draw(st.lists(length_integers(total_length=total_length), min_size=1, max_size=batch_sizes))


@st.composite
def num_tags_integers(draw, max_num_tags: int = MAX_NUM_TAGS):
    return draw(st.integers(min_value=1, max_value=max_num_tags))





@st.composite
def tags_packs(draw, lengths: List[int], num_tags: int):
    return pack_sequence([
        torch.randint(0, num_tags, (length,), device=draw(devices()))
        for length in lengths
    ], enforce_sorted=False)


@st.composite
def conjugated_tags_packs(draw, lengths: List[int], num_tags: int, num_conjugates: int):
    return pack_sequence([
        torch.randint(0, num_tags, (length, num_conjugates), device=draw(devices()))
        for length in lengths
    ], enforce_sorted=False)
