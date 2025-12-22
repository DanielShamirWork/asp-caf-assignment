#include "huffman.h"
#include <stack>

// builds a dictionary based on the huffman tree nodes
std::unordered_map<std::byte, std::vector<bool>> huffman_dict(const std::vector<HuffmanNode>& nodes) {
    std::unordered_map<std::byte, std::vector<Bit>> dict;
    dict.reserve(256); // reserve space for all possible byte values

    // iterative DFS to build the dictionary

    // the stack will hold for each node its index and the path taken to reach it
    struct StackItem {
        size_t index;
        std::vector<Bit> path;
    };

    std::stack<StackItem> stack;
    stack.push({nodes.size() - 1, {}}); // start from the root node, at index size - 1

    while (!stack.empty()) {
        auto [index, path] = std::move(stack.top());
        stack.pop();

        const auto& node = nodes[index];

        if (std::holds_alternative<LeafNodeData>(node.data)) {
            const auto& leaf = std::get<LeafNodeData>(node.data);
            dict[leaf.symbol] = path;
        } else {
            const auto& internal = std::get<InternalNodeData>(node.data);

            // Push right child
            std::vector<Bit> right_path = path;
            right_path.push_back(true);
            stack.push({internal.right_index, std::move(right_path)});

            // Push left child
            path.push_back(false);
            stack.push({internal.left_index, std::move(path)});
        }
    }

    return dict;
}