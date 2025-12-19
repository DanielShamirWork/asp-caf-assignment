import pytest
from libcaf import histogram, huffman_tree, HuffmanNode

def assert_node_is_valid(node: HuffmanNode) -> bool:
    assert node is not None
    assert isinstance(node.frequency, int)
    assert node.frequency >= 0

def test_huffman_tree() -> None:
    input_bytes = b"aaaabbbccd"

    hist = histogram(input_bytes)
    nodes, root_index = huffman_tree(hist)

    root_is_not_child = True

    for node in nodes:
        assert_node_is_valid(node)

        if node.is_leaf: # leaf node
            assert node.symbol in input_bytes
            assert node.frequency == input_bytes.count(bytes([node.symbol]))
        else: # internal node
            assert node.frequency == nodes[node.left_index].frequency + nodes[node.right_index].frequency
            if node.left_index == root_index or node.right_index == root_index:
                root_is_not_child = False

    assert root_is_not_child

if __name__ == "__main__":
    test_huffman_tree()
    print("test_huffman_tree passed!")
