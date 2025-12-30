#include "huffman.h"
#include <stack>

#include <iostream>

// builds a dictionary based on the huffman tree nodes
std::array<std::vector<bool>, 256> huffman_dict(const std::vector<HuffmanNode>& nodes) {
    std::array<std::vector<bool>, 256> dict{};

    if (nodes.empty()) {
        return dict;
    }

    // iterative DFS algorithm to build the dictionary of huffman codes

    // the stack will hold for each node its index and the path taken to reach it
    struct StackItem {
        size_t index;
        std::vector<bool> path;
    };

    std::stack<StackItem> stack;
    stack.push({nodes.size() - 1, {}}); // start from the root node, at index size - 1

    while (!stack.empty()) {
        auto [index, path] = std::move(stack.top());
        stack.pop();

        const auto& node = nodes[index];

        if (std::holds_alternative<LeafNodeData>(node.data)) {
            const auto& leaf = std::get<LeafNodeData>(node.data);
            dict[std::to_integer<size_t>(leaf.symbol)] = path;
        } else {
            const auto& internal = std::get<InternalNodeData>(node.data);

            // Push right child
            auto right_path = path; // copy current path
            right_path.push_back(true);
            stack.push({internal.right_index, right_path});

            // Push left child
            auto left_path = path; // copy current path
            left_path.push_back(false);
            stack.push({internal.left_index, left_path});
        }
    }

    return dict;
}