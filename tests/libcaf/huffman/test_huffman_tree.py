import numpy as np
from pytest import mark

from libcaf import histogram, huffman_tree


def test_huffman_tree_empty_histogram_returns_no_root() -> None:
    nodes, root_index = huffman_tree(histogram(np.array([], dtype=np.uint8)))

    assert nodes == []
    assert root_index is None


@mark.parametrize('payload_size', [
    0, 1,
    2 ** 4, 2 ** 4 + 1,
    2 ** 8,
    2 ** 12,
    2 ** 20,  # 1 MiB
    2 ** 30,  # 1 GiB
    2 ** 32,  # 4 GiB
])
def test_huffman_invariants(random_payload: np.ndarray) -> None:
    hist = histogram(random_payload)

    assert len(hist) == 256
    assert sum(hist) == len(random_payload)

    nodes, root_index = huffman_tree(hist)

    if len(random_payload) == 0:
        assert nodes == []
        assert root_index is None
        return

    assert root_index is not None
    total_weight = nodes[root_index].frequency
    assert total_weight == len(random_payload)

    seen_nodes: set[int] = set()
    leaf_freqs = {i: 0 for i in range(256)}
    stack = [root_index]

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
