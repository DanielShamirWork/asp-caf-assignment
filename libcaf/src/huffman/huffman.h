#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "huffman_node.h"

#include <cstdint>          // for uint64_t
#include <array>            // for std::array
#include <cstddef>          // for size_t
#include <vector>           // for std::vector
#include <span>             // for std::span
#include <unordered_map>    // for std::unordered_map
#include <string>           // for std::string

// huffman_histogram.cpp
std::array<uint64_t, 256> histogram(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel_64bit(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_fast(std::span<const std::byte> data);

// huffman_tree.cpp
std::vector<HuffmanNode> huffman_tree(const std::array<uint64_t, 256>& hist);

// huffman_dict.cpp
std::array<std::vector<bool>, 256> huffman_dict(const std::vector<HuffmanNode>& nodes);

// huffman_encdec.cpp
uint64_t calculate_compressed_size_in_bits(const std::array<uint64_t, 256>& hist, const std::array<std::vector<bool>, 256>& dict);
void huffman_encode_span(const std::span<const std::byte> from, const std::span<std::byte> to, const std::array<std::vector<bool>, 256>& dict);
void huffman_encode_span_parallel(const std::span<const std::byte> from, const std::span<std::byte> to, const std::array<std::vector<bool>, 256>& dict);
void huffman_encode_span_parallel_twopass(const std::span<const std::byte> from, const std::span<std::byte> to, const std::array<std::vector<bool>, 256>& dict);
uint64_t huffman_encode_file(const std::string& input_file, const std::string& output_file);

#endif // HUFFMAN_H