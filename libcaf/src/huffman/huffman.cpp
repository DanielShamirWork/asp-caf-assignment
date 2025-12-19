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

std::pair<std::vector<HuffmanNode>, size_t> huffman_tree(const std::array<uint64_t, 256>& hist) {
    std::priority_queue<size_t, std::vector<size_t>> min_heap;
    std::vector<HuffmanNode> nodes;
    nodes.reserve(2 * 256 - 1); // Max number of nodes in a full binary tree with 256 leaves

    // Create all leaf nodes
    size_t next_node_idx = 0;
    for (size_t i = 0; i < hist.size(); i++) {
        if (hist[i] == 0)
            continue;

        min_heap.push(next_node_idx);
        nodes.emplace_back(hist[i], i);
        next_node_idx++;
    }

    // build all the internal nodes, until there is only one node left in the heap
    while (min_heap.size() > 1) {
        const size_t left_index = min_heap.top();
        min_heap.pop();
        const size_t right_index = min_heap.top();
        min_heap.pop();

        const uint64_t parent_frequency = nodes[left_index].frequency + nodes[right_index].frequency;
        nodes.emplace_back(parent_frequency, left_index, right_index);

        const size_t parent_index = nodes.size() - 1;
        min_heap.push(parent_index);
    }

    // the remaining node is the root
    const size_t root_index = min_heap.top();
    return {nodes, root_index};
}