#include "huffman.h"

#include <queue>

// populates freqs by the frequency of each char in syms, based on str
// OPTIMIZATION: go over all chars in parallel/SIMD, also try loop unrolling
std::array<uint64_t, 256> histogram(const unsigned char* data, size_t length) {
    std::array<uint64_t, 256> freqs = {0};
    
    // count frequencies, unsigned char to avoid negative indices
    for (size_t i = 0; i < length; ++i) {
        freqs[data[i]]++;
    }
    
    return freqs;
}

std::pair<std::vector<HuffmanNode *>, size_t> huffman_tree(const std::array<uint64_t, 256>& hist) {
    std::priority_queue<uint64_t, std::vector<uint64_t>> min_heap;
    std::vector<HuffmanNode *> nodes;

    // create all leaf nodes and insert them into the minimum heap and the nodes data arrays
    size_t num_nodes = 0;
    for (size_t i = 0; i < hist.size(); i++) {
        if (hist[i] > 0) {
            min_heap.push(num_nodes);
            HuffmanNode *node = new HuffmanNode(hist[i], static_cast<unsigned char>(i));
            nodes.push_back(node);
            num_nodes++;
        }
    }

    // build all the internal nodes, until there is only one node left in the heap
    while (min_heap.size() > 1) {
        uint64_t left_index = min_heap.top();
        min_heap.pop();
        uint64_t right_index = min_heap.top();
        min_heap.pop();

        HuffmanNode *left_node = nodes[left_index];
        HuffmanNode *right_node = nodes[right_index];

        uint64_t parent_frequency = left_node->frequency + right_node->frequency;
        
        // create a new internal node
        HuffmanNode *parent_node = new HuffmanNode(parent_frequency, left_index, right_index);
        nodes.push_back(parent_node);
        uint64_t parent_index = nodes.size() - 1;
        min_heap.push(parent_index);
    }

    // the remaining node is the root
    size_t root_index = min_heap.top();
    return {nodes, root_index};
}