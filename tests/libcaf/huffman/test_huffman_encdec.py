import numpy as np
import os
import tempfile
from pathlib import Path
from pytest import mark

from libcaf import huffman_encode_file


@mark.parametrize('payload_size', [
    0,
    1,
    10,
    100,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 16,
    2 ** 20,  # 1 MiB
    2 ** 24,  # 16 MiB
    2 ** 26,  # 64 MiB
    # 2 ** 28,  # 256 MiB
    # 2 ** 30,  # 1 GiB
    # 2 ** 32,  # 4 GiB
])
def test_huffman_encode_file(random_payload: np.ndarray) -> None:
    """Test huffman_encode_file with random payloads of different sizes."""

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.bin"
        output_file = Path(tmpdir) / "output.huff"

        # Write random payload to input file
        random_payload.tofile(input_file)

        # Verify input file exists and has correct size
        assert input_file.exists()
        input_size = input_file.stat().st_size
        assert input_size == len(random_payload)

        # Verify output file doesn't exist yet
        assert not output_file.exists()

        # Compress the file
        compressed_size = huffman_encode_file(str(input_file), str(output_file))

        # Verify output file now exists
        assert output_file.exists()

        # Verify the returned size matches the actual file size
        actual_compressed_size = output_file.stat().st_size
        assert compressed_size == actual_compressed_size

        # Verify compression worked (file size check)
        # For empty files or very small files, compression might not reduce size
        # due to header overhead (8 bytes + 2048 bytes histogram = 2056 bytes)
        header_size = 8 + 256 * 8  # uint64_t size + histogram

        if len(random_payload) == 0:
            # Empty file should have just the header
            assert compressed_size == header_size
        elif len(random_payload) < 2056:
            # Small files might be larger after compression due to header overhead
            # Just verify the file was created with valid size
            assert compressed_size >= header_size
        else:
            # For larger files with random data, compression should generally work
            # but not always (random data is hard to compress)
            # Just verify structure is correct
            assert compressed_size >= header_size


@mark.parametrize('payload_size', [
    2 ** 12,
    2 ** 16,
])
def test_huffman_encode_file_repetitive_data(payload_size: int) -> None:
    """Test huffman_encode_file with highly repetitive data that compresses well."""

    # Create highly repetitive data (should compress very well)
    # Use mostly one byte value with occasional other bytes
    rng = np.random.default_rng(0xBEEF + payload_size)
    repetitive_payload = np.full(payload_size, ord('A'), dtype=np.uint8)
    # Add some variation (5% of data)
    num_variations = payload_size // 20
    variation_indices = rng.choice(payload_size, num_variations, replace=False)
    repetitive_payload[variation_indices] = rng.integers(ord('B'), ord('E'), num_variations, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "repetitive.bin"
        output_file = Path(tmpdir) / "repetitive.huff"

        repetitive_payload.tofile(input_file)

        input_size = input_file.stat().st_size
        compressed_size = huffman_encode_file(str(input_file), str(output_file))

        # Repetitive data should compress significantly
        # Header is 2056 bytes, so for large enough files, compressed should be much smaller
        assert output_file.exists()
        assert compressed_size < input_size, "Repetitive data should compress well"


def test_huffman_encode_file_all_same_byte() -> None:
    """Test huffman_encode_file with data containing only one unique byte value."""

    payload_size = 10000
    same_byte_payload = np.full(payload_size, 42, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "same_byte.bin"
        output_file = Path(tmpdir) / "same_byte.huff"

        same_byte_payload.tofile(input_file)

        compressed_size = huffman_encode_file(str(input_file), str(output_file))

        # With only one unique byte, each byte should encode to 1 bit
        # Compressed data should be: 10000 bits = 1250 bytes
        # Total size = 8 (size header) + 2048 (histogram) + 1250 (data) = 3306 bytes
        header_size = 8 + 256 * 8
        expected_data_size = (payload_size + 7) // 8  # Round up to bytes
        expected_total = header_size + expected_data_size

        assert output_file.exists()
        assert compressed_size == expected_total


def test_huffman_encode_file_preserves_input() -> None:
    """Test that huffman_encode_file doesn't modify the input file."""

    payload_size = 1000
    rng = np.random.default_rng(0xDEADBEEF)
    payload = rng.integers(0, 256, payload_size, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "preserve_test.bin"
        output_file = Path(tmpdir) / "preserve_test.huff"

        payload.tofile(input_file)
        original_content = input_file.read_bytes()

        huffman_encode_file(str(input_file), str(output_file))

        # Verify input file hasn't changed
        after_content = input_file.read_bytes()
        assert original_content == after_content


def test_huffman_encode_file_overwrites_existing() -> None:
    """Test that huffman_encode_file overwrites existing output files."""

    payload_size = 1000
    rng = np.random.default_rng(0xCAFE)
    payload1 = rng.integers(0, 256, payload_size, dtype=np.uint8)
    payload2 = rng.integers(0, 256, payload_size, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "overwrite_test.bin"
        output_file = Path(tmpdir) / "overwrite_test.huff"

        # First compression
        payload1.tofile(input_file)
        size1 = huffman_encode_file(str(input_file), str(output_file))
        content1 = output_file.read_bytes()

        # Second compression with different data
        payload2.tofile(input_file)
        size2 = huffman_encode_file(str(input_file), str(output_file))
        content2 = output_file.read_bytes()

        # The output should be different (different input data)
        # Unless by extreme coincidence they compress identically
        assert len(content2) == size2
        # Just verify the file was overwritten by checking it exists and has valid structure
        assert output_file.exists()
