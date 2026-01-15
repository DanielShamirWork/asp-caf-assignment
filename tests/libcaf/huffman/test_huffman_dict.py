import numpy as np
from pytest import mark

from libcaf import histogram, huffman_tree, huffman_dict, canonicalize_huffman_dict, next_canonical_huffman_code


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


def test_next_canonical_huffman_code() -> None:
    """Test that next_canonical_huffman_code produces the expected sequence."""

    # Test incrementing codes
    assert next_canonical_huffman_code([False]) == [True]
    assert next_canonical_huffman_code([True]) == [True, False]  # overflow adds a bit
    
    assert next_canonical_huffman_code([False, False]) == [False, True]
    assert next_canonical_huffman_code([False, True]) == [True, False]
    assert next_canonical_huffman_code([True, False]) == [True, True]
    assert next_canonical_huffman_code([True, True]) == [True, False, False]
    
    # Test longer codes
    assert next_canonical_huffman_code([True, False, True]) == [True, True, False]
    assert next_canonical_huffman_code([True, True, True]) == [True, False, False, False]


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
    
    if len(codes_with_symbols) <= 1:
        return
    
    codes_with_symbols.sort(key=lambda x: (len(x[1]), x[0]))
    
    # Verify canonical properties:
    # 1. Codes of the same length are sequential when interpreted as binary numbers
    # 2. When length increases, the code is incremented and shifted left
    prev_code = None
    prev_len = 0
    for symbol, code in codes_with_symbols:
        if prev_code is None:
            # First code should be all zeros
            assert all(bit == False for bit in code), f"First canonical code should be all 0s, got {code}"
        else:
            if len(code) == prev_len:
                # Same length: should be prev_code + 1
                expected = next_canonical_huffman_code(prev_code)
                if len(expected) > prev_len:
                    expected = expected[1:]
                assert code == expected, f"Code for symbol {symbol} should be {expected}, got {code}"
            else:
                # Length increased: should be prev_code + 1, then padded with zeros
                incremented = next_canonical_huffman_code(prev_code)
                assert len(code) > len(prev_code), "Code length should increase"
        
        prev_code = code
        prev_len = len(code)
    
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