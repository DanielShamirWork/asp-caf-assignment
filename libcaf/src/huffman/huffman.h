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

#define MAX_CODE_LEN 9 // there can be 511 possible codes so 9 bits are needed to represent them

// huffman_histogram.cpp
std::array<uint64_t, 256> histogram(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_parallel_64bit(std::span<const std::byte> data);
std::array<uint64_t, 256> histogram_fast(std::span<const std::byte> data);

// huffman_tree.cpp
std::vector<HuffmanNode> huffman_tree(const std::array<uint64_t, 256>& hist);

// huffman_dict.cpp
std::array<std::vector<bool>, 256> huffman_dict(const std::vector<HuffmanNode>& nodes);
void canonicalize_huffman_dict(std::array<std::vector<bool>, 256>& dict);
std::vector<bool> next_canonical_huffman_code(const std::vector<bool>& code);

// huffman_encdec.cpp

/*
    huffman compressed file layout:
    
    [8 bytes]   : uint64_t original file size
    [8 bytes]   : uint64_t compressed data size (in bits)
    [512 bytes] : code lengths (256 * sizeof(uint16_t))
    [n bytes]   : compressed data
*/
struct HuffmanHeader {
    uint64_t original_file_size;
    uint64_t compressed_data_size;
    std::array<uint16_t, 256> code_lengths;
};
constexpr size_t HUFFMAN_HEADER_SIZE = sizeof(HuffmanHeader);

uint64_t calculate_compressed_size_in_bits(const std::array<uint64_t, 256>& hist, const std::array<std::vector<bool>, 256>& dict);
void huffman_encode_span(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict);
void huffman_encode_span_parallel(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict);
void huffman_encode_span_parallel_twopass(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict);

std::array<uint16_t, 512> huffman_build_reverse_dict(const std::array<std::vector<bool>, 256>& dict, const size_t max_code_len);
void huffman_decode_span(const std::span<const std::byte> source, const size_t source_size_in_bits, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict);

uint64_t huffman_encode_file(const std::string& input_file, const std::string& output_file);
uint64_t huffman_decode_file(const std::string& input_file, const std::string& output_file);
#endif // HUFFMAN_H