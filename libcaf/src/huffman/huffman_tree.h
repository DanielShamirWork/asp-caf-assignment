#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <cstdint>
#include <vector>

#define HUFFMAN_NODE_NULL_INDEX -1

class HuffmanTree {
private:
    std::vector<const unsigned char> symbols_list;
    std::vector<const uint64_t> frequencies_list;
    std::vector<const uint64_t> left_indices_list;
    std::vector<const uint64_t> right_indices_list;

public:
    const uint64_t build_tree_and_get_root_index(const unsigned char * const str, size_t length);

    const unsigned char get_symbol(uint64_t index) const;
    const uint64_t get_frequency(uint64_t index) const;
    const uint64_t get_left_index(uint64_t index) const; // returns the left child of the node at index
    const uint64_t get_right_index(uint64_t index) const; // returns the right child of the node at index
    
    // TODO(daniel): add a function that builds a huffman dictionary from this tree

private:
    void add_node(const unsigned char sym, const uint64_t frequency, const uint64_t left_index, const uint64_t right_index);
};

#endif // HUFFMAN_TREE_H