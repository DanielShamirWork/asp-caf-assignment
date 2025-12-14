#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <cstdint>
#include <array>
#include <string>

std::array<uint64_t, 256> histogram(const std::string &str);

#endif // HUFFMAN_H