import numpy as np
from pytest import mark

from libcaf import histogram, histogram_parallel, histogram_parallel_64bit, histogram_fast, huffman_tree

SIZES = [
    2 ** 4,  # 16 B
    2 ** 8,  # 256 B
    2 ** 12,  # 4 KiB
    2 ** 16,  # 64 KiB
    2 ** 18,  # 256 KiB
    2 ** 19,  # 512 KiB
    2 ** 20,  # 1 MiB
    2 ** 21,  # 2 MiB
    2 ** 22,  # 4 MiB
    2 ** 23,  # 8 MiB
    2 ** 24,  # 16 MiB
    2 ** 25,  # 32 MiB
    2 ** 26,  # 64 MiB
    2 ** 27,  # 128 MiB
    2 ** 28,  # 256 MiB
    2 ** 29,  # 512 MiB
    # 2 ** 30,  # 1 GiB
    # 2 ** 31,  # 2 GiB
    # 2 ** 32,  # 4 GiB
]


# @mark.parametrize('histogram_func', [histogram, histogram_parallel, histogram_parallel_64bit, histogram_fast], ids=['histogram', 'histogram_parallel', 'histogram_parallel_64bit', 'histogram_fast'])
@mark.parametrize('histogram_func', [histogram, histogram_parallel], ids=['histogram', 'histogram_parallel'])
@mark.parametrize('payload_size', SIZES)
def test_benchmark_huffman_tree(random_payload: np.ndarray, benchmark, histogram_func) -> None:  # type: ignore[no-untyped-def]
    def create_tree():
        hist = histogram_func(random_payload)
        huffman_tree(hist)

    benchmark.pedantic(create_tree, warmup_rounds=3, rounds=10, iterations=1)
