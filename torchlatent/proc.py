import warnings
from typing import Any
from typing import List

from torchlatent.instr import build_crf_instr, collate_crf_instr, BatchInstr, Instr

try:
    from torchglyph.proc import Proc


    class BuildCrfInstr(Proc):
        def __call__(self, data: List[Any], *args, **kwargs) -> Instr:
            return build_crf_instr(length=len(data))


    class CollateCrfInstr(Proc):
        def __call__(self, collected_instr: List[Instr], *args, **kwargs) -> BatchInstr:
            return collate_crf_instr(collected_instr=collected_instr)

except ImportError:
    warnings.warn(f'torchglyph is required')
