#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <variant>

using TreeIndex = size_t;

struct LeafNodeData {
    std::byte symbol;
};

struct InternalNodeData {
    TreeIndex left_index;
    TreeIndex right_index;
};

struct HuffmanNode {
    uint64_t frequency;
    std::variant<LeafNodeData, InternalNodeData> data;

    HuffmanNode(uint64_t freq, std::byte symbol)
        : frequency(freq), data(LeafNodeData{symbol}) {}

    HuffmanNode(uint64_t freq, TreeIndex left, TreeIndex right)
        : frequency(freq), data(InternalNodeData{left, right}) {}
};

#endif // HUFFMAN_TREE_H