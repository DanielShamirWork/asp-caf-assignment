#include "bitreader.h"

#include <stdexcept>
#include <algorithm>

BitReader::BitReader(std::span<const std::byte> data, const size_t data_size_in_bits)
    : data(data), bit_pos(0), data_size_in_bits(data_size_in_bits) {}

uint64_t BitReader::read(const size_t n_bits) const {
    if (n_bits == 0) {
        return 0;
    }
    if (n_bits > 64) {
        throw std::invalid_argument("Cannot read more than 64 bits at once");
    }
    if (bit_pos + n_bits > data_size_in_bits * 8) {
        throw std::out_of_range("Not enough bits remaining to read");
    }

    uint64_t result = 0;
    size_t cur_bit = bit_pos;
    size_t bits_remaining = n_bits;

    while (bits_remaining > 0) {
        size_t byte_index = cur_bit / 8;
        size_t bit_offset = cur_bit % 8;
        
        size_t bits_in_this_byte = std::min(8 - bit_offset, bits_remaining);
        
        uint8_t byte_val = static_cast<uint8_t>(data[byte_index]);
        
        // Shift to align desired bits to LSB, then mask
        uint8_t shift = 8 - bit_offset - bits_in_this_byte;
        uint8_t mask = (1 << bits_in_this_byte) - 1;
        uint8_t extracted = (byte_val >> shift) & mask;
        
        // Append to result
        result = (result << bits_in_this_byte) | extracted;
        
        cur_bit += bits_in_this_byte;
        bits_remaining -= bits_in_this_byte;
    }

    return result;
}

void BitReader::advance(const size_t n_bits) {
    if (bit_pos + n_bits > data_size_in_bits * 8) {
        throw std::out_of_range("Cannot advance past end of data");
    }
    bit_pos += n_bits;
}

bool BitReader::done() const {
    return bit_pos >= data_size_in_bits;
}