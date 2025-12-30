#include "huffman.h"

#include <stdexcept>
#include <iostream>
#include <fstream>

/*
    huffman compressed file layout:
    
    [8 bytes]   : uint64_t compressed data size (excluding this header and histogram)
    [2048 bytes]: histogram (256 * sizeof(uint64_t))
    [n bytes]   : compressed data
*/

uint64_t calculate_compressed_size_in_bits(const std::array<uint64_t, 256>& hist, const std::array<std::vector<bool>, 256>& dict) {
    uint64_t total_bits = 0;

    // add bits required for the compressed data
    for (size_t i = 0; i < hist.size(); ++i) {
        total_bits += hist[i] * dict[i].size();
    }

    return total_bits;
}

void huffman_encode_span(const std::span<const std::byte> from, const std::span<std::byte> to, const std::array<std::vector<bool>, 256>& dict) {
    uint64_t bit_position = 0;
    
    for (size_t i = 0; i < from.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(from[i]);
        const std::vector<bool>& code = dict[byte];

        for (size_t j = 0; j < code.size(); ++j) {
            size_t bit_index = bit_position;
            size_t byte_index = bit_index / 8;
            size_t bit_offset = 7 - (bit_index % 8); // Store bits from MSB to LSB

            // assume span is zeroed, so only set bits when code[j] is true
            std::byte code_bit = code[j] ? std::byte{1} : std::byte{0};
            to[byte_index] |= static_cast<std::byte>(code_bit << bit_offset);

            bit_position++;
        }
    }
}

uint64_t huffman_encode_file(const std::string& input_file, const std::string& output_file) {
    // read input file
    std::ifstream in(input_file, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open input file");
    }

    const uint64_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<std::byte> input_data;
    input_data.resize(file_size);

    // read entire file, we need to do that to build the histogram
    if (!in.read(reinterpret_cast<char*>(input_data.data()), file_size)) {
        throw std::runtime_error("Failed to read input file");
    } 
    in.close();

    const std::array<uint64_t, 256> hist = histogram_parallel(input_data);
    const std::vector<HuffmanNode> tree = huffman_tree(hist);
    const std::array<std::vector<bool>, 256> dict = huffman_dict(tree);

    const uint64_t compressed_size_in_bits = calculate_compressed_size_in_bits(hist, dict);
    const uint64_t compressed_size_in_bytes = (compressed_size_in_bits + 7) / 8; // round up to full bytes

    std::vector<std::byte> compressed_data(compressed_size_in_bytes, std::byte{0});
    huffman_encode_span(input_data, compressed_data, dict);

    // create or truncate output file
    std::ofstream out(output_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open output file");
    }

    // write compressed data to output file
    out.write(reinterpret_cast<const char*>(&compressed_size_in_bits), sizeof(compressed_size_in_bits)); // write compressed data size
    out.write(reinterpret_cast<const char*>(hist.data()), hist.size() * sizeof(uint64_t)); // write histogram
    out.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_size_in_bytes); // write compressed data
    out.close();

    // return total size of the compressed file, including header and histogram
    return sizeof(uint64_t) + hist.size() * sizeof(uint64_t) + compressed_size_in_bytes;
}