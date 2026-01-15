import numpy as np
from pathlib import Path
from pytest import mark

from libcaf import huffman_encode_file

SIZES = [
    # files under 4 KiB are have >1.0 compression ratio due to overhead from saving the histogram (2kb)
    # so we ignore them to not skew the results
    # 2 ** 4,   # 16 B
    # 2 ** 8,   # 256 B
    # 2 ** 12,  # 4 KiB
    2 ** 16,  # 64 KiB
    2 ** 18,  # 256 KiB
    2 ** 20,  # 1 MiB
    2 ** 22,  # 4 MiB
    2 ** 24,  # 16 MiB
    2 ** 26,  # 64 MiB
    2 ** 28,  # 256 MiB
    2 ** 30,  # 1 GiB
    # 2 ** 32,  # 4 GiB
]


@mark.parametrize('payload_type', ['random', 'repetitive', 'uniform'], ids=['random', 'repetitive', 'uniform'])
@mark.parametrize('payload_size', SIZES)
def test_compression_ratio(payload: np.ndarray, benchmark, tmp_path: Path) -> None:
    """Test and measure compression ratios for different data types and sizes."""
    input_file = tmp_path / "input.bin"
    output_file = tmp_path / "output.huff"

    payload.tofile(input_file)
    original_size = len(payload)

    def compress_file():
        compressed_size = huffman_encode_file(str(input_file), str(output_file))
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': compressed_size / original_size
        }

    result = benchmark.pedantic(compress_file, warmup_rounds=1, rounds=5, iterations=1)

    benchmark.extra_info.update(result)
