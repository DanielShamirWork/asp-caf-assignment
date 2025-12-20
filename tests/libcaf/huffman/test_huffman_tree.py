import numpy as np
from pytest import mark

from libcaf import histogram, histogram_fast, histogram_parallel, histogram_parallel_64bit, huffman_tree


@mark.parametrize('payload_size', [
    0,
    2 ** 4,
    2 ** 8,
    2 ** 12,
    2 ** 20,  # 1 MiB
    2 ** 30,  # 1 GiB
    2 ** 32,  # 4 GiB
])
@mark.parametrize('histogram_func', [
    histogram,
    histogram_parallel,
    histogram_parallel_64bit,
    histogram_fast
])
def test_huffman_invariants(random_payload: np.ndarray, histogram_func) -> None:
    hist = histogram_func(random_payload)

    assert len(hist) == 256
    assert sum(hist) == len(random_payload)

    nodes = huffman_tree(hist)

    if len(random_payload) == 0:
        assert nodes == []
        return

    assert len(nodes) >= 1

    # The root node is always the last node created
    total_weight = nodes[-1].frequency
    assert total_weight == len(random_payload)

    seen_nodes: set[int] = set()
    leaf_freqs = {i: 0 for i in range(256)}
    stack = [len(nodes) - 1]

    while stack:
        idx = stack.pop()
        assert 0 <= idx < len(nodes)
        assert idx not in seen_nodes
        seen_nodes.add(idx)

        node = nodes[idx]
        if node.is_leaf:
            symbol = node.symbol
            assert symbol is not None
            leaf_freqs[symbol] += node.frequency
        else:
            left = node.left_index
            right = node.right_index
            assert left is not None and right is not None
            stack.extend([left, right])
            assert node.frequency == nodes[left].frequency + nodes[right].frequency

    assert seen_nodes == set(range(len(nodes)))

    for byte_val, expected in enumerate(hist):
        assert leaf_freqs[byte_val] == expected
