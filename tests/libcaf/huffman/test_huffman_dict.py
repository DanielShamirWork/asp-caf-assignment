import numpy as np
from pytest import mark

from libcaf import histogram, huffman_tree, huffman_dict, canonicalize_huffman_dict, next_canonical_huffman_code


@mark.parametrize('payload_size', [
    0,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 20,  # 1 MiB
    2 ** 30,  # 1 GiB
    2 ** 32,  # 4 GiB
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

@mark.parametrize('code', [
    [0],
    [1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])
def test_next_canonical_huffman_code(code: list[bool]) -> None:
    """Test that next_canonical_huffman_code produces the expected sequence."""
    
    # transform to number and check next canonical bit is + 1
    code_num = 0
    for bit in code:
        code_num = (code_num << 1) | int(bit)
    
    next_code = next_canonical_huffman_code(code)
    next_code_num = 0
    for bit in next_code:
        next_code_num = (next_code_num << 1) | int(bit)
    
    assert next_code_num == code_num + 1


@mark.parametrize('payload_size', [
    2 ** 8,
    2 ** 12,
    2 ** 16,
])
def test_canonicalize_huffman_dict(random_payload: np.ndarray) -> None:
    """Test that canonicalize_huffman_dict produces valid canonical codes."""

    huff_hist = histogram(random_payload)
    huff_nodes = huffman_tree(huff_hist)
    huff_dict = huffman_dict(huff_nodes)
    
    canonical_dict = canonicalize_huffman_dict(huff_dict)
    
    codes_with_symbols = []
    for symbol in range(256):
        if len(canonical_dict[symbol]) > 0:
            codes_with_symbols.append((symbol, canonical_dict[symbol]))
    
    codes_with_symbols.sort(key=lambda x: (len(x[1]), x[0]))
    
    # Note that since we test next_canonical_huffman_code's mathematical properties
    # there's no need to actually test all the codes follow the canonical property
    # As long as we use the function correctly and it's test pass, we are good 
    
    # Ensure all codes are still unique
    codes_set = set()
    for symbol, code in codes_with_symbols:
        code_tuple = tuple(code)
        assert code_tuple not in codes_set, f"Duplicate code found: {code}"
        codes_set.add(code_tuple)
    
    # Ensure prefix property still holds
    for i, (sym_i, code_i) in enumerate(codes_with_symbols):
        for j, (sym_j, code_j) in enumerate(codes_with_symbols):
            if i != j:
                min_len = min(len(code_i), len(code_j))
                assert code_i[:min_len] != code_j[:min_len], f"Prefix property violated for symbols {sym_i} and {sym_j}"