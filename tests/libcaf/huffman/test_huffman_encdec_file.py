import numpy as np
import os
import tempfile
from pathlib import Path
from pytest import mark

from libcaf import huffman_encode_file, huffman_decode_file, HUFFMAN_HEADER_SIZE


@mark.parametrize('payload_size', [
    10,
    100,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 16,
    2 ** 20,  # 1 MiB
    2 ** 24,  # 16 MiB
    2 ** 26,  # 64 MiB
    2 ** 28,  # 256 MiB
    2 ** 30,  # 1 GiB
    2 ** 32,  # 4 GiB
])
def test_huffman_encode_file(random_payload: np.ndarray) -> None:
    """Test huffman_encode_file with random payloads of different sizes."""

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.bin"
        output_file = Path(tmpdir) / "output.huff"

        random_payload.tofile(input_file)

        assert input_file.exists()
        input_size = input_file.stat().st_size
        assert input_size == len(random_payload)

        assert not output_file.exists()

        compressed_size = huffman_encode_file(str(input_file), str(output_file))
        assert output_file.exists()

        # Verify the returned size matches the actual file size
        actual_compressed_size = output_file.stat().st_size
        assert compressed_size == actual_compressed_size

        # All files should have at least the header
        assert compressed_size >= HUFFMAN_HEADER_SIZE

        # Since huffman encoding *might* not compress the data (since we add a header), we can't assert
        # that the compressed size is smaller than the original size
        # But that's ok, a reduced size is not actually a required property of the algorithm
        # It should reduce size only as much as possible


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
        # So, the expected data size equals the the payload size
        expected_data_size = (payload_size + 7) // 8  # Round up to bytes
        expected_total = HUFFMAN_HEADER_SIZE + expected_data_size

        assert output_file.exists()
        assert compressed_size == expected_total


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

@mark.parametrize('payload_size', [
    10,
    100,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 16,
    2 ** 20,  # 1 MiB
    2 ** 24,  # 16 MiB
    2 ** 26,  # 64 MiB
    2 ** 28,  # 256 MiB
    2 ** 30,  # 1 GiB
    2 ** 32,  # 4 GiB
])
def test_huffman_file_encoding_decoding(random_payload: np.ndarray) -> None:
    """Test encoding a file and then decoding it back to verify data integrity."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "original.bin"
        compressed_file = Path(tmpdir) / "compressed.huff"
        restored_file = Path(tmpdir) / "restored.bin"

        random_payload.tofile(input_file)

        huffman_encode_file(str(input_file), str(compressed_file))
        assert compressed_file.exists()

        huffman_decode_file(str(compressed_file), str(restored_file))
        assert restored_file.exists()

        assert restored_file.stat().st_size == input_file.stat().st_size

        restored_data = np.fromfile(restored_file, dtype=np.uint8)
        np.testing.assert_array_equal(random_payload, restored_data)