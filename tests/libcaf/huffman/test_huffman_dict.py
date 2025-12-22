import numpy as np
from pytest import mark

from libcaf import histogram, huffman_tree, huffman_dict


@mark.parametrize('payload_size', [
    0,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 20,  # 1 MiB
    2 ** 30,  # 1 GiB
    # 2 ** 32,  # 4 GiB # My computer ran out of memory on this test
])
def test_huffman_dict_invariants(random_payload: np.ndarray) -> None:
    huff_hist = histogram(random_payload)
    huff_nodes = huffman_tree(huff_hist)
    huff_dict = huffman_dict(huff_nodes)

    if len(random_payload) == 0:
        assert huff_dict == {}
        return
    
    # Ensure that all symbols with non-zero frequency are present in the dictionary
    for byte_val in range(256):
        if huff_hist[byte_val] > 0:
            assert byte_val in huff_dict
        else:
            assert byte_val not in huff_dict
        
    # Ensure that all codes are non-empty and unique
    codes = set()
    for code in huff_dict.values():
        assert len(code) > 0
        code_tuple = tuple(code)
        assert code_tuple not in codes
        codes.add(code_tuple)

    # Verify that the codes satisfy the prefix property
    code_list = list(huff_dict.values())
    for i in range(len(code_list)):
        for j in range(len(code_list)):
            if i != j:
                code_i = code_list[i]
                code_j = code_list[j]
                min_len = min(len(code_i), len(code_j))
                assert code_i[:min_len] != code_j[:min_len]