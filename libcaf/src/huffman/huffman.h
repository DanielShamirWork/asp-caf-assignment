#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "huffman_node.h"

#include <cstdint>  // for uint64_t
#include <array>    // for std::array
#include <cstddef>  // for size_t
#include <vector>   // for std::vector
#include <utility>  // for std::pair
#include <span>     // for std::span
#include <optional> // for std::optional

std::array<uint64_t, 256> histogram(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_fast(std::span<const std::byte> data);

std::pair<std::vector<HuffmanNode>, std::optional<size_t>> huffman_tree(const std::array<uint64_t, 256>& hist);

#endif // HUFFMAN_H