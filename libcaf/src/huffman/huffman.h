#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <cstdint>  // for uint64_t
#include <array>    // for std::array
#include <cstddef>  // for size_t

std::array<uint64_t, 256> histogram(const unsigned char* data, size_t length);

#endif // HUFFMAN_H