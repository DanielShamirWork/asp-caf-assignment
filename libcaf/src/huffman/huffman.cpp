#include "huffman.h"

// popoulates freqs by the frequency of each char in syms, based on str
// OPTIMIZATION: go over all chars in parallel/SIMD, also try loop unrolling
const uint64_t *Huffman::frequencies(const char* const str) {
    if (str == nullptr) {
        return {};
    }

    uint64_t freqs[256] = {0};
    for (const char * s = str; *s != '\0'; s++) {
        freqs[*s]++;
    }

    return freqs;
}