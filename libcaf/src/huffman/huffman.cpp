#include "huffman.h"

// populates freqs by the frequency of each char in syms, based on str
// OPTIMIZATION: go over all chars in parallel/SIMD, also try loop unrolling
std::array<uint64_t, 256> histogram(const unsigned char* data, size_t length) {
    std::array<uint64_t, 256> freqs = {0};
    
    // count frequencies, unsigned char to avoid negative indices
    for (size_t i = 0; i < length; ++i) {
        freqs[data[i]]++;
    }
    
    return freqs;
}