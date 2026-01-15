#include "huffman.h"
#include <stack>

#include <iostream>
#include <algorithm>

// builds a dictionary based on the huffman tree nodes
std::array<std::vector<bool>, 256> huffman_dict(const std::vector<HuffmanNode>& nodes) {
    std::array<std::vector<bool>, 256> dict{};

    if (nodes.empty()) {
        return dict;
    }

    if (nodes.size() == 1) {
        // special case: only one symbol in the tree
        const auto& leaf = std::get<LeafNodeData>(nodes[0].data);
        dict[std::to_integer<size_t>(leaf.symbol)] = {false}; // assign code '0'
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

void canonicalize_huffman_dict(std::array<std::vector<bool>, 256> &dict) {
    // Create list of (symbol, length) for non-empty codes
    std::vector<std::pair<uint8_t, size_t>> symbols_by_length;
    for (size_t i = 0; i < 256; i++) {
        if (!dict[i].empty()) {
            symbols_by_length.emplace_back(static_cast<uint8_t>(i), dict[i].size());
        }
    }
    
    if (symbols_by_length.empty()) return;
    
    // Sort by length, then by symbol value
    std::sort(
        symbols_by_length.begin(),
        symbols_by_length.end(),
        [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second < b.second;
        }
    );
    
    // Assign canonical codes - first code is all zeros
    std::vector<bool> code(symbols_by_length[0].second, false);
    dict[symbols_by_length[0].first] = code;
    
    // Generate subsequent canonical codes
    for (size_t i = 1; i < symbols_by_length.size(); i++) {
        code = next_canonical_huffman_code(code);
        size_t target_len = symbols_by_length[i].second;
        
        // Pad with zeros if length increases
        while (code.size() < target_len) {
            code.push_back(false);
        }
        
        dict[symbols_by_length[i].first] = code;
    }
}

std::vector<bool> next_canonical_huffman_code(const std::vector<bool>& code) {
    std::vector<bool> new_code = code;
    for (size_t i = code.size(); i > 0; i--) {
        if (code[i - 1] == false) {
            new_code[i - 1] = true;
            return new_code;
        }
        new_code[i - 1] = false;
    }

    // Deal with overflow
    new_code.insert(new_code.begin(), true);
    return new_code;
}