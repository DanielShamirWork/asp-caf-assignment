#include "huffman.h"

#include <queue>
#include <omp.h>

struct NodeComparator {
    const std::vector<HuffmanNode>& nodes;

    NodeComparator(const std::vector<HuffmanNode>& nodes) : nodes(nodes) {}

    bool operator()(size_t lhs, size_t rhs) const {
        return nodes[lhs].frequency > nodes[rhs].frequency;
    }
};

std::vector<HuffmanNode> huffman_tree(const std::array<uint64_t, 256>& hist) {
    std::vector<HuffmanNode> nodes;
    nodes.reserve(2 * 256 - 1); // Max number of nodes in a full binary tree with 256 leaves

    std::priority_queue<size_t, std::vector<size_t>, NodeComparator> min_heap{NodeComparator(nodes)};

    // Create all leaf nodes
    for (size_t i = 0; i < hist.size(); i++) {
        if (hist[i] == 0)
            continue;

        nodes.emplace_back(hist[i], static_cast<std::byte>(i));
        min_heap.push(nodes.size() - 1);
    }

    if (min_heap.empty()) {
        return nodes;
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

    // the remaining node is the root, always at nodes.size() - 1
    return nodes;
}