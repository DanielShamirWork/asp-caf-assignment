#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "huffman_node.h"

#include <cstdint>          // for uint64_t
#include <array>            // for std::array
#include <cstddef>          // for size_t
#include <vector>           // for std::vector
#include <span>             // for std::span
#include <unordered_map>    // for std::unordered_map

// huffman_histogram.cpp
std::array<uint64_t, 256> histogram(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel_64bit(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_fast(std::span<const std::byte> data);

// huffman_tree.cpp
std::vector<HuffmanNode> huffman_tree(const std::array<uint64_t, 256>& hist);

// huffman_dict.cpp
std::array<std::vector<bool>, 256> huffman_dict(const std::vector<HuffmanNode>& nodes);

#endif // HUFFMAN_H