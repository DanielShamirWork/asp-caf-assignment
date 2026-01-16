import numpy as np
from pytest import mark

from libcaf import (
    huffman_encode_span,
    huffman_decode_span,
    histogram_parallel,
    huffman_tree,
    huffman_dict,
    huffman_build_reverse_dict,
    canonicalize_huffman_dict,
    calculate_compressed_size_in_bits,
    MAX_CODE_LEN,
)

empty_dict = [[] for _ in range(256)]

dict_1 = empty_dict.copy()
dict_1[61] = [False]
dict_1[62] = [True, False]
dict_1[63] = [True, True]

dict_2 = empty_dict.copy()
dict_2[61] = [False, False]
dict_2[62] = [False, True]
dict_2[63] = [True, False]
dict_2[64] = [True, True, False]
dict_2[65] = [True, True, True]

@mark.parametrize("dict", [
    empty_dict,
    dict_1,
    dict_2,
])
def test_huffman_build_reverse_dict(dict: list[list[bool]]) -> None:
    """Test huffman_encode_file with random payloads of different sizes."""
    reverse_dict = huffman_build_reverse_dict(dict, MAX_CODE_LEN)
    # print(reverse_dict)
    assert len(reverse_dict) == 2 ** MAX_CODE_LEN

    # All values in the reverse dictionary should be bytes
    for i in range(2 ** MAX_CODE_LEN):
        assert reverse_dict[i] in range(256)

    for symbol in range(256):
        code = dict[symbol]
        if not code:
            continue

        code_val = 0
        for bit in code:
            code_val = (code_val << 1) | int(bit)
        
        # All codes should appear 2^(code length) times
        shift = MAX_CODE_LEN - len(code)
        start_idx = code_val << shift
        num_entries = 1 << shift

        for i in range(start_idx, start_idx + num_entries):
            assert reverse_dict[i] == symbol, f"Index {i} should map to symbol {symbol} (Code {code}), but got {reverse_dict[i]}"

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
def test_huffman_encdec(random_payload: np.ndarray) -> None:
    """Test huffman_encode_file with random payloads of different sizes."""

    hist = histogram_parallel(random_payload)
    tree = huffman_tree(hist)
    dictionary = huffman_dict(tree)
    canonicalize_huffman_dict(dictionary)
    
    total_bits = calculate_compressed_size_in_bits(hist, dictionary)
    total_bytes = (total_bits + 7) // 8
    
    encoded_data = np.zeros(total_bytes, dtype=np.uint8)
    huffman_encode_span(random_payload, encoded_data, dictionary)
    
    decoded_data = np.zeros(len(random_payload), dtype=np.uint8)
    huffman_decode_span(encoded_data, total_bits, decoded_data, dictionary)
    
    np.testing.assert_array_equal(random_payload, decoded_data)