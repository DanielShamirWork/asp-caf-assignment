#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <cstdint>

class Huffman {
public:
    Huffman();
    ~Huffman();

    const uint64_t *frequencies(const char* const str);

private:
    uint64_t freqs[256];
};

#endif // HUFFMAN_H