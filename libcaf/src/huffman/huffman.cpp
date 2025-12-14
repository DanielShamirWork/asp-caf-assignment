#include "huffman.h"

// popoulates freqs by the frequency of each char in syms, based on str
// OPTIMIZATION: go over all chars in parallel/SIMD, also try loop unrolling
std::array<uint64_t, 256> histogram(const std::string &str) {
    std::array<uint64_t, 256> freqs = {0};
    
    if (str.empty()) {
        return freqs;
    }
    
    // count frequencies, unsigned char to avoid negative indices
    for (unsigned char c : str) {
        freqs[c]++;
    }
    
    return freqs;
}