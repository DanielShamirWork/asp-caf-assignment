#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "huffman_node.h"

#include <cstdint>  // for uint64_t
#include <array>    // for std::array
#include <cstddef>  // for size_t
#include <vector>   // for std::vector
#include <utility>  // for std::pair

std::array<uint64_t, 256> histogram(const unsigned char* data, size_t length);
std::pair<std::vector<HuffmanNode *>, size_t> huffman_tree(const std::array<uint64_t, 256>& hist);

#endif // HUFFMAN_H