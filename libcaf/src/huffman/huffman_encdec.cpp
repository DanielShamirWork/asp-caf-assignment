#include "huffman.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <cstring>

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

void huffman_encode_span(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict) {
    uint64_t bitstream_position = 0;

    for (size_t i = 0; i < source.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(source[i]);
        const std::vector<bool>& code = dict[byte];

        for (size_t j = 0; j < code.size(); ++j) {
            size_t bit_idx_in_current_byte = bitstream_position;
            size_t byte_idx = bit_idx_in_current_byte / 8;
            size_t bit_offset = 7 - (bit_idx_in_current_byte % 8); // Store bits from MSB to LSB

            // assume span is zeroed, so only set bits when code[j] is true
            std::byte code_bit = code[j] ? std::byte{1} : std::byte{0};
            destination[byte_idx] |= static_cast<std::byte>(code_bit << bit_offset);

            bitstream_position++;
        }
    }
}

void huffman_encode_span_parallel(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict) {
    const int num_threads = omp_get_max_threads();
    const size_t chunk_size = (source.size() + num_threads - 1) / num_threads;

    // Per-thread buffers to hold encoded chunks
    std::vector<std::vector<std::byte>> thread_buffers(num_threads);
    std::vector<uint64_t> thread_bit_sizes(num_threads, 0);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        const size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, source.size());

        if (start < end) {
            // Calculate size needed for this chunk
            uint64_t chunk_bits = 0;
            for (size_t i = start; i < end; ++i) {
                uint8_t byte = static_cast<uint8_t>(source[i]);
                chunk_bits += dict[byte].size();
            }
            thread_bit_sizes[thread_id] = chunk_bits;

            // Allocate buffer for this chunk
            const size_t chunk_bytes = (chunk_bits + 7) / 8;
            thread_buffers[thread_id].resize(chunk_bytes, std::byte{0});

            // Encode this chunk
            uint64_t bitstream_position = 0;
            for (size_t i = start; i < end; ++i) {
                uint8_t byte = static_cast<uint8_t>(source[i]);
                const std::vector<bool>& code = dict[byte];

                for (size_t j = 0; j < code.size(); ++j) {
                    size_t bit_idx_in_current_byte = bitstream_position;
                    size_t byte_idx = bit_idx_in_current_byte / 8;
                    size_t bit_offset = 7 - (bit_idx_in_current_byte % 8);

                    std::byte code_bit = code[j] ? std::byte{1} : std::byte{0};
                    thread_buffers[thread_id][byte_idx] |= static_cast<std::byte>(code_bit << bit_offset);

                    bitstream_position++;
                }
            }
        }
    }

    // Combine all thread buffers into the output buffer
    uint64_t current_bit_offset = 0;
    for (int t = 0; t < num_threads; ++t) {
        const uint64_t chunk_bits = thread_bit_sizes[t];
        if (chunk_bits == 0) continue;

        const auto& thread_buffer = thread_buffers[t];

        // Copy bits from thread buffer to output buffer
        for (uint64_t bit_idx = 0; bit_idx < chunk_bits; ++bit_idx) {
            size_t src_byte_idx = bit_idx / 8;
            size_t src_bit_offset = 7 - (bit_idx % 8);

            size_t dst_bit_pos = current_bit_offset + bit_idx;
            size_t dst_byte_idx = dst_bit_pos / 8;
            size_t dst_bit_offset = 7 - (dst_bit_pos % 8);

            // Read bit from source
            std::byte bit = (thread_buffer[src_byte_idx] >> src_bit_offset) & std::byte{1};

            // Write bit to destination
            if (bit != std::byte{0}) {
                destination[dst_byte_idx] |= static_cast<std::byte>(std::byte{1} << dst_bit_offset);
            }
        }

        current_bit_offset += chunk_bits;
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
    huffman_encode_span_parallel(input_data, compressed_data, dict);

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