from aku import Aku

from benchmark.crf import benchmark_crf

aku = Aku()

aku.option(benchmark_crf)

aku.run()
