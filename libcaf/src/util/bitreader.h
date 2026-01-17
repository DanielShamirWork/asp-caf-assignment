#ifndef BITREADER_H
#define BITREADER_H

#include <cstddef>
#include <cstdint>
#include <span>

class BitReader {
public:
    BitReader(const std::span<const std::byte> data, const size_t data_size_in_bits);
    uint64_t read(const size_t n_bits) const;
    void advance(const size_t n_bits);
    bool done() const;

private:
    std::span<const std::byte> data;
    size_t bit_pos;
    size_t data_size_in_bits;
};

#endif // BITREADER_H
