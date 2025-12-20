#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "huffman_node.h"

#include <cstdint>  // for uint64_t
#include <array>    // for std::array
#include <cstddef>  // for size_t
#include <vector>   // for std::vector
#include <span>     // for std::span

std::array<uint64_t, 256> histogram(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel_64bit(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_fast(std::span<const std::byte> data);

std::vector<HuffmanNode> huffman_tree(const std::array<uint64_t, 256>& hist);

#endif // HUFFMAN_H