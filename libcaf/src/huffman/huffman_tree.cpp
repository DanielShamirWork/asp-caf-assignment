#include "huffman_tree.h"

#include "huffman.h"

#include <assert.h>
#include <queue>

// const unsigned char get_symbol() const;
const unsigned char HuffmanNode::get_symbol() const {
    return symbol;
}

const uint64_t HuffmanNode::get_frequency() const {
    return frequency;
}
const bool HuffmanNode::is_leaf() const {
    return left_index == HUFFMAN_NODE_NULL_INDEX && right_index == HUFFMAN_NODE_NULL_INDEX;
}

const uint64_t HuffmanNode::get_left_index() const {
    return left_index;
}

const uint64_t HuffmanNode::get_right_index() const {
    return right_index;
}

const HuffmanNode &HuffmanTree::build_tree(const unsigned char * const str, size_t length) {
    assert(str != nullptr);
    assert(length > 0);

    std::array<uint64_t, 256> hist = histogram(str, length);
    std::priority_queue<const HuffmanNode &, std::vector<const HuffmanNode &>, CompareHuffmanNode> min_heap;

    // if there are no symbols or only one symbol, no huffman compression is possible
    // can revert to RLE or other compression in this case
    if (hist.size() == 0 || hist.size() == 1) {
        return;
    }

    for (unsigned char i = 0; i < hist.size(); i++) {
        if (hist[i] > 0) {
            const HuffmanNode &node = *(new HuffmanNode(i, hist[i]));
            min_heap.push(node);
        }
    }

    while (min_heap.size() > 1) {
        const HuffmanNode &left = min_heap.top();
        min_heap.pop();
        const HuffmanNode &right = min_heap.top();
        min_heap.pop();

        const uint64_t left_index = nodes_list.size();
        nodes_list.push_back(left);

        const uint64_t right_index = nodes_list.size();
        nodes_list.push_back(right);

        HuffmanNode parent('\0', left.get_frequency() + right.get_frequency(), left_index, right_index);
        min_heap.push(parent);
    }

    const HuffmanNode &root = min_heap.top();
    const uint64_t root_index = nodes_list.size();
    nodes_list.push_back(root);

    return root;
}