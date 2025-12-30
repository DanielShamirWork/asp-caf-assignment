import numpy as np
from pytest import mark

from libcaf import histogram, huffman_tree, huffman_dict


@mark.parametrize('payload_size', [
    0,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 20,  # 1 MiB
    # 2 ** 30,  # 1 GiB # takes too long
    # 2 ** 32,  # 4 GiB # My computer ran out of memory on this test
])
def test_huffman_dict_invariants(random_payload: np.ndarray) -> None:
    huff_hist = histogram(random_payload)
    huff_nodes = huffman_tree(huff_hist)
    huff_dict = huffman_dict(huff_nodes)

    if len(random_payload) == 0:
        assert huff_dict == [[] for _ in range(256)]
        return
    
    # Ensure that all symbols with non-zero frequency are present in the dictionary
    for byte_val in range(256):
        if huff_hist[byte_val] > 0:
            assert huff_dict[byte_val] != []
        else:
            assert huff_dict[byte_val] == []
        
    # Ensure that all codes that are non-empty are unique
    codes = set()
    for i in range(256):
        code = huff_dict[i]
        if len(code) > 0:
            code_tuple = tuple(code)
            assert code_tuple not in codes
            codes.add(code_tuple)

    # Verify that the codes satisfy the prefix property
    # i.e. for all pairs of codes a and b, a is not a prefix of b
    # this ensures that the codes can be uniquely decoded for decompression
    for i in range(len(huff_dict)):
        for j in range(len(huff_dict)):
            if i != j:
                code_i = huff_dict[i]
                code_j = huff_dict[j]
                if len(code_i) == 0 or len(code_j) == 0:
                    continue

                min_len = min(len(code_i), len(code_j))
                assert code_i[:min_len] != code_j[:min_len]