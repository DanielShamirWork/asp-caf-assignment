#include "huffman_tree.h"

#include "huffman.h"

#include <assert.h>
#include <queue>

uint64_t HuffmanTree::build_tree_and_get_root_index(const unsigned char * const str, size_t length) {
    assert(str != nullptr);
    assert(length > 0);

    std::array<uint64_t, 256> hist = histogram(str, length);
    std::priority_queue<uint64_t, std::vector<uint64_t>> min_heap;

    // create all leaf nodes and insert them into the minimum heap and the nodes data arrays
    this->num_nodes = 0;
    for (size_t i = 0; i < hist.size(); i++) {
        if (hist[i] > 0) {
            min_heap.push(this->num_nodes);
            this->add_node(i, hist[i], HUFFMAN_NODE_NULL_INDEX, HUFFMAN_NODE_NULL_INDEX);
            this->num_nodes++;
        }
    }

    // build all the internal nodes, until there is only one node left in the heap
    while (min_heap.size() > 1) {
        uint64_t left_index = min_heap.top();
        min_heap.pop();
        uint64_t right_index = min_heap.top();
        min_heap.pop();

        uint64_t parent_frequency = this->get_frequency(left_index) + this->get_frequency(right_index);
        
        // create a new internal node
        // NOTE(daniel): it doesn't actually matter what symbol we assign to internal nodes, since they are defined by having children
        this->add_node('\0', parent_frequency, left_index, right_index);
        uint64_t parent_index = this->num_nodes;
        this->num_nodes++;
        min_heap.push(parent_index);
    }

    // the remaining node is the root
    uint64_t root_index = min_heap.top();
    return root_index;
}

uint64_t HuffmanTree::get_num_nodes() const {
    return this->num_nodes;
}

uint64_t HuffmanTree::get_frequency(uint64_t index) const {
    return frequencies_list[index];
}

uint64_t HuffmanTree::get_left_index(uint64_t index) const {
    return left_indices_list[index];
}

uint64_t HuffmanTree::get_right_index(uint64_t index) const {
    return right_indices_list[index];
}

bool HuffmanTree::is_leaf_node(uint64_t index) const {
    return left_indices_list[index] == HUFFMAN_NODE_NULL_INDEX && right_indices_list[index] == HUFFMAN_NODE_NULL_INDEX;
}

unsigned char HuffmanTree::get_symbol(uint64_t index) const {
    return symbols_list[index];
}

void HuffmanTree::add_node(const unsigned char sym, const uint64_t frequency, const uint64_t left_index, const uint64_t right_index) {
    symbols_list.push_back(sym);
    frequencies_list.push_back(frequency);
    left_indices_list.push_back(left_index);
    right_indices_list.push_back(right_index);
}