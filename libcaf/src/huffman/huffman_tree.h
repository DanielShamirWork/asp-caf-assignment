#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <cstdint>
#include <vector>

#define HUFFMAN_NODE_NULL_INDEX -1

class HuffmanNode {
private:
    const unsigned char symbol;
    const uint64_t frequency;
    const uint64_t left_index;
    const uint64_t right_index;

public:
    HuffmanNode(const unsigned char symbol, uint64_t frequency, const uint64_t left_index, const uint64_t right_index)
        : symbol(symbol),
          frequency(frequency),
          left_index(left_index),
          right_index(right_index) {}

    const unsigned char get_symbol() const;
    const uint64_t get_frequency() const;
    const bool is_leaf() const;
    const uint64_t get_left_index() const;
    const uint64_t get_right_index() const;
};

struct CompareHuffmanNode {
    bool operator()(const HuffmanNode &l, const HuffmanNode &r) {
        return l.get_frequency() > r.get_frequency();
    }
};

class HuffmanTree {
private:
    std::vector<const HuffmanNode &> nodes_list;

public:
    const HuffmanNode &build_tree(const unsigned char * const str, size_t length);
    
    // TODO(daniel): add a function that builds a huffman dictionary from this tree
};

#endif // HUFFMAN_TREE_H