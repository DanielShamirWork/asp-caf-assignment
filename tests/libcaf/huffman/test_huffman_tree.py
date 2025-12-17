import pytest
from libcaf import HuffmanTree, HUFFMAN_NODE_NULL_INDEX

def assert_is_index_valid(index: int, huff_tree: HuffmanTree) -> bool:
    assert index is not None
    assert isinstance(index, int)
    assert index >= 0
    assert index < huff_tree.get_num_nodes()

def test_huffman_tree() -> None:
    input_bytes = b"aaaabbbccd"

    huff_tree = HuffmanTree()
    huff_tree.build_tree_and_get_root_index(input_bytes)

    for index in range(huff_tree.get_num_nodes()):
        assert_is_index_valid(index, huff_tree)

        symbol = huff_tree.get_symbol(index)
        frequency = huff_tree.get_frequency(index)
        left_index = huff_tree.get_left_index(index)
        right_index = huff_tree.get_right_index(index)
        is_leaf = huff_tree.is_leaf_node(index)

        assert frequency > 0

        if is_leaf: # leaf node
            assert left_index == HUFFMAN_NODE_NULL_INDEX
            assert right_index == HUFFMAN_NODE_NULL_INDEX
            assert symbol in input_bytes
            assert frequency == input_bytes.count(bytes([symbol]))
        else: # internal node
            # NOTE(daniel): i believe huffman trees are always complete, but i don't want to assume
            # NOTE(daniel): an internal node's symbol is unused, no need to check it
            assert left_index != HUFFMAN_NODE_NULL_INDEX or right_index != HUFFMAN_NODE_NULL_INDEX
            assert frequency == huff_tree.get_frequency(left_index) + huff_tree.get_frequency(right_index)

if __name__ == "__main__":
    test_huffman_tree()
    print("test_huffman_tree passed!")
