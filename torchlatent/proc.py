import warnings
from typing import List

import torch

from torchlatent.instr import build_crf_instr, collate_crf_instr, BatchedInstr, Instr

try:
    from torchglyph.proc import Proc


    class BuildCrfInstr(Proc):
        def __call__(self, length: int, *args, **kwargs) -> Instr:
            return build_crf_instr(length=length)


    class CollateCrfInstr(Proc):
        def __init__(self, device: torch.device) -> None:
            super(CollateCrfInstr, self).__init__()
            self.device = device

        def extra_repr(self) -> str:
            return f'{self.device}'

        def __call__(self, collected_instr: List[Instr], *args, **kwargs) -> BatchedInstr:
            return collate_crf_instr(collected_instr=collected_instr, device=self.device)

except ImportError:
    warnings.warn(f'torchglyph is required')
