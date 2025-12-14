import pytest
from libcaf import Huffman

@pytest.mark.parametrize("input_str, expected_freqs", [
    ("aaa", {'a': 3}),
    ("aab", {'a': 2, 'b': 1}),
    ("", {}),
    ("mississippi", {'m': 1, 'i': 4, 's': 4, 'p': 2}),
])
def test_huffman_frequencies(input_str, expected_freqs):
    h = Huffman()
    freqs = h.frequencies(input_str)

    # Check that we get a 256-element array
    assert len(freqs) == 256

    # Check specific character frequencies
    for char, count in expected_freqs.items():
        assert freqs[ord(char)] == count

    # Check that all other characters have 0 frequency
    expected_chars = set(expected_freqs.keys())
    for i in range(256):
        if chr(i) not in expected_chars:
            assert freqs[i] == 0
