import pytest
from libcaf import histogram, huffman_tree, HuffmanNode

def assert_is_node_valid(node: HuffmanNode) -> bool:
    assert node is not None
    assert isinstance(node.frequency, int)
    assert node.frequency >= 0

def test_huffman_tree() -> None:
    input_bytes = b"aaaabbbccd"

    hist = histogram(input_bytes)
    nodes, root_index = huffman_tree(hist)

    for index, node in enumerate(nodes):
        assert_is_node_valid(node)

        # symbol = node.symbol
        # frequency = node.frequency
        # left_index = node.left_child_index
        # right_index = node.right_child_index
        # is_leaf = (left_index == HUFFMAN_NODE_NULL_INDEX) and (right_index == HUFFMAN_NODE_NULL_INDEX)

        # assert frequency > 0

        # if is_leaf: # leaf node
        #     assert left_index == HUFFMAN_NODE_NULL_INDEX
        #     assert right_index == HUFFMAN_NODE_NULL_INDEX
        #     assert symbol in input_bytes
        #     assert frequency == input_bytes.count(bytes([symbol]))
        # else: # internal node
        #     # NOTE(daniel): i believe huffman trees are always complete, but i don't want to assume
        #     # NOTE(daniel): an internal node's symbol is unused, no need to check it
        #     assert left_index != HUFFMAN_NODE_NULL_INDEX or right_index != HUFFMAN_NODE_NULL_INDEX
        #     assert frequency == nodes[left_index].frequency + nodes[right_index].frequency

if __name__ == "__main__":
    test_huffman_tree()
    print("test_huffman_tree passed!")
