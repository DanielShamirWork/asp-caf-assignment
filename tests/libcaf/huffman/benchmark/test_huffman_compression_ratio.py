import numpy as np
import tempfile
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
    # 2 ** 26,  # 64 MiB
    # 2 ** 28,  # 256 MiB
    # 2 ** 30,  # 1 GiB
    # 2 ** 32,  # 4 GiB
]


@mark.parametrize('data_type', ['random', 'repetitive', 'uniform'], ids=['random', 'repetitive', 'uniform'])
@mark.parametrize('payload_size', SIZES)
def test_compression_ratio(random_payload: np.ndarray, benchmark, data_type: str) -> None:  # type: ignore[no-untyped-def]
    """Test and measure compression ratios for different data types and sizes."""

    # Generate different types of data based on the parameter
    if data_type == 'random':
        # Use the random payload from fixture
        data = random_payload
    elif data_type == 'repetitive':
        # Create highly repetitive data (should compress well)
        # 90% of one value, 10% random
        rng = np.random.default_rng(0xBEEF + len(random_payload))
        data = np.full(len(random_payload), ord('A'), dtype=np.uint8)
        num_variations = max(1, len(random_payload) // 10)
        if len(random_payload) > 0:
            variation_indices = rng.choice(len(random_payload), min(num_variations, len(random_payload)), replace=False)
            data[variation_indices] = rng.integers(0, 256, len(variation_indices), dtype=np.uint8)
    elif data_type == 'uniform':
        # All same byte (best compression)
        data = np.full(len(random_payload), 42, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    def compress_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.bin"
            output_file = Path(tmpdir) / "output.huff"

            # Write data to input file
            data.tofile(input_file)

            # Compress
            compressed_size = huffman_encode_file(str(input_file), str(output_file))

            # Calculate ratio
            original_size = len(data)
            if original_size == 0:
                ratio = 0.0
            else:
                ratio = compressed_size / original_size

            # Store in benchmark extra info for later plotting
            return {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'ratio': ratio
            }

    result = benchmark.pedantic(compress_file, warmup_rounds=1, rounds=5, iterations=1)

    # Store the compression stats in benchmark extra_info
    # This will be accessible in the JSON output
    benchmark.extra_info.update(result)
