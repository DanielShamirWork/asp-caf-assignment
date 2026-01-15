import numpy as np
from pytest import mark

from libcaf import (
    calculate_compressed_size_in_bits,
    histogram,
    huffman_tree,
    huffman_dict,
    huffman_encode_span,
    huffman_encode_span_parallel,
    huffman_encode_span_parallel_twopass,
)

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
    2 ** 30,  # 1 GiB
]


@mark.parametrize('encode_func', [
    huffman_encode_span,
    huffman_encode_span_parallel,
    huffman_encode_span_parallel_twopass,
], ids=['sequential', 'parallel', 'parallel_twopass'])
@mark.parametrize('payload_size', SIZES)
def test_benchmark_huffman_encode_span(random_payload: np.ndarray, benchmark, encode_func) -> None:  # type: ignore[no-untyped-def]
    """Benchmark different huffman_encode_span implementations."""
    # Pre-compute histogram, tree, and dictionary (not part of benchmark)
    hist = histogram(random_payload)
    tree = huffman_tree(hist)
    dict_codes = huffman_dict(tree)

    # Calculate output buffer size
    compressed_bits = calculate_compressed_size_in_bits(hist, dict_codes)
    compressed_bytes = (compressed_bits + 7) // 8

    def encode():
        # Create fresh output buffer for each iteration
        output = np.zeros(compressed_bytes, dtype=np.uint8)
        encode_func(random_payload, output, dict_codes)
        return output

    benchmark.pedantic(encode, warmup_rounds=3, rounds=10, iterations=1)
