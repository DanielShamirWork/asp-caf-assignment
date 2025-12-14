#include "huffman.h"
#include <cstring>

Huffman::Huffman() {
    memset(freqs, 0, sizeof(freqs));
}

Huffman::~Huffman() {
}

// popoulates freqs by the frequency of each char in syms, based on str
// OPTIMIZATION: go over all chars in parallel/SIMD, also try loop unrolling
const uint64_t *Huffman::frequencies(const char* const str) {
    memset(freqs, 0, sizeof(freqs));
    
    if (str == nullptr) {
        return freqs;
    }

    // Count character frequencies, casting to unsigned char to avoid negative indices
    for (const char * s = str; *s != '\0'; s++) {
        freqs[static_cast<unsigned char>(*s)]++;
    }

    return freqs;
}