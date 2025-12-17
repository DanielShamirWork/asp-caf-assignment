#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <cstdint>
#include <vector>

// define the null index to be the maximum value of uint64_t
#define HUFFMAN_NODE_NULL_INDEX UINT64_MAX

class HuffmanTree {
private:
    std::vector<unsigned char> symbols_list;
    std::vector<uint64_t> frequencies_list;
    std::vector<uint64_t> left_indices_list;
    std::vector<uint64_t> right_indices_list;
    uint64_t num_nodes = 0;

public:
    uint64_t build_tree_and_get_root_index(const unsigned char * const str, size_t length);

    unsigned char get_symbol(uint64_t index) const;
    uint64_t get_frequency(uint64_t index) const;
    uint64_t get_left_index(uint64_t index) const; // returns the left child of the node at index
    uint64_t get_right_index(uint64_t index) const; // returns the right child of the node at index
    bool is_leaf_node(uint64_t index) const;
    uint64_t get_num_nodes() const;

private:
    void add_node(const unsigned char sym, const uint64_t frequency, const uint64_t left_index, const uint64_t right_index);
};

#endif // HUFFMAN_TREE_H