#include "huffman.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

#include "../util/bitreader.h"

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

std::array<uint16_t, 512> huffman_build_reverse_dict(const std::array<std::vector<bool>, 256>& dict, const size_t max_code_len) {
    std::array<uint16_t, 512> reverse_dict = {};

    for (size_t symbol = 0; symbol < 256; symbol++) {
        const std::vector<bool>& code = dict[symbol];
        size_t len = code.size();
        if(len == 0)
            continue;

        uint16_t symbol_code = 0;
        for (size_t i = 0; i < len; ++i) {
            symbol_code = (symbol_code << 1) | code[i];
        }
        symbol_code <<= max_code_len - len;

        // For a bit string of length len, the prefix uniquely defines the symbol
        // so we can have multiple entries that all point to the same symbol
        // without conflicts
        // e.g. if 'A' has huffman code 00 then
        // [0000] => 'A', [0001] => 'A', [0010] => 'A', [0011] => 'A'
        uint64_t num_entries = (1 << (max_code_len - len));

        for (size_t i = 0; i < num_entries; ++i) {
            reverse_dict[symbol_code + i] = symbol;
        }
    }

    return reverse_dict;
}

void huffman_decode_span(const std::span<const std::byte> source, const size_t source_size_in_bits, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict) {
    std::array<uint16_t, 512> reverse_dict = huffman_build_reverse_dict(dict, MAX_CODE_LEN);

    BitReader reader(source, source_size_in_bits);
    
    size_t dst_byte_idx = 0;
    while (!reader.done()) {
        uint64_t code = reader.read(MAX_CODE_LEN);
        uint8_t symbol = reverse_dict[code];
        size_t symbol_len = dict[symbol].size();
        
        destination[dst_byte_idx++] = static_cast<std::byte>(symbol);
        reader.advance(symbol_len);
    }
}


void huffman_encode_span_parallel(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict) {
    const int num_threads = omp_get_max_threads();
    const size_t chunk_size = (source.size() + num_threads - 1) / num_threads;

    std::vector<std::vector<std::byte>> thread_buffers(num_threads);
    std::vector<uint64_t> thread_code_lengths(num_threads, 0);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        const size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, source.size());

        if (start < end) {
            uint64_t chunk_bits = 0;
            for (size_t i = start; i < end; ++i) {
                uint8_t byte = static_cast<uint8_t>(source[i]);
                chunk_bits += dict[byte].size();
            }
            thread_code_lengths[thread_id] = chunk_bits;

            const size_t chunk_bytes = (chunk_bits + 7) / 8;
            thread_buffers[thread_id].resize(chunk_bytes, std::byte{0});

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
    uint64_t cur_bit_offset = 0;
    for (int t = 0; t < num_threads; ++t) {
        const uint64_t chunk_bits = thread_code_lengths[t];
        if (chunk_bits == 0) continue;

        const auto& thread_buffer = thread_buffers[t];
        const size_t dst_start_bit = cur_bit_offset;
        const size_t dst_bit_in_byte = dst_start_bit % 8;

        if (dst_bit_in_byte == 0) { // Destination is byte-aligned - we can use memcpy for full bytes
            const size_t full_bytes = chunk_bits / 8;
            const size_t remaining_bits = chunk_bits % 8;
            const size_t dst_byte_idx = dst_start_bit / 8;

            if (full_bytes > 0) {
                std::memcpy(&destination[dst_byte_idx], thread_buffer.data(), full_bytes);
            }

            // Handle remaining bits (last partial byte)
            if (remaining_bits > 0) {
                std::byte src_byte = thread_buffer[full_bytes];
                // Copy top 'remaining_bits' bits from source to destination
                uint8_t mask = static_cast<uint8_t>(0xFF << (8 - remaining_bits));
                destination[dst_byte_idx + full_bytes] |= src_byte & static_cast<std::byte>(mask);
            }
        } else { // Destination is not byte-aligned - use 64-bit word operations for speed
            const size_t dst_byte_idx = dst_start_bit / 8;
            const size_t shift_right = dst_bit_in_byte;  // bits to shift right
            const size_t shift_left = 8 - shift_right;   // bits to shift left for carry
            const size_t src_bytes = (chunk_bits + 7) / 8;

            // Process 8 bytes at a time using 64-bit operations
            size_t i = 0;
            const size_t aligned_end = (src_bytes >= 8) ? (src_bytes - 7) : 0;
            
            for (; i < aligned_end; i += 8) {
                uint64_t src_val;
                std::memcpy(&src_val, &thread_buffer[i], 8);
                
                uint64_t dst_val;
                std::memcpy(&dst_val, &destination[dst_byte_idx + i], 8);
                dst_val |= (src_val >> shift_right);
                std::memcpy(&destination[dst_byte_idx + i], &dst_val, 8);
                
                destination[dst_byte_idx + i + 8] |= static_cast<std::byte>(
                    static_cast<uint8_t>(thread_buffer[i + 7]) << shift_left);
            }

            // Handle remaining bytes
            for (; i < src_bytes; ++i) {
                uint8_t src_val = static_cast<uint8_t>(thread_buffer[i]);
                destination[dst_byte_idx + i] |= static_cast<std::byte>(src_val >> shift_right);
                if ((dst_byte_idx + i + 1) < destination.size()) {
                    destination[dst_byte_idx + i + 1] |= static_cast<std::byte>(src_val << shift_left);
                }
            }

            // Mask off any extra bits written past chunk_bits
            const size_t total_bits_written = dst_start_bit + chunk_bits;
            const size_t last_byte_idx = (total_bits_written - 1) / 8;
            const size_t valid_bits_in_last_byte = ((total_bits_written - 1) % 8) + 1;
            if (valid_bits_in_last_byte < 8) {
                uint8_t mask = static_cast<uint8_t>(0xFF << (8 - valid_bits_in_last_byte));
                destination[last_byte_idx] &= static_cast<std::byte>(mask);
            }
        }

        cur_bit_offset += chunk_bits;
    }
}

// In order to avoid joining/copying at the end, we calculate the code lengths one pass and then write directly to the destination a second pass
void huffman_encode_span_parallel_twopass(const std::span<const std::byte> source, const std::span<std::byte> destination, const std::array<std::vector<bool>, 256>& dict) {
    const int num_threads = omp_get_max_threads();
    const size_t chunk_size = (source.size() + num_threads - 1) / num_threads;

    std::vector<uint64_t> thread_code_lengths(num_threads, 0);
    std::vector<uint64_t> thread_bit_offsets(num_threads, 0);

    // Pass 1: calculate the code lengths for each thread
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        const size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, source.size());

        uint64_t chunk_bits = 0;
        for (size_t i = start; i < end; ++i) {
            uint8_t byte = static_cast<uint8_t>(source[i]);
            chunk_bits += dict[byte].size();
        }
        thread_code_lengths[thread_id] = chunk_bits;
    }

    // calculate the bit offsets to the start of each write thread's chunk
    for (int t = 1; t < num_threads; ++t) {
        thread_bit_offsets[t] = thread_bit_offsets[t - 1] + thread_code_lengths[t - 1];
    }

    // Pass 2: write the compressed data using buffered writes
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        const size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, source.size());

        if (start < end) {
            const uint64_t bit_start = thread_bit_offsets[thread_id];
            const uint64_t bit_end = bit_start + thread_code_lengths[thread_id];
            
            // Calculate byte boundaries
            const size_t first_byte = bit_start / 8;
            const size_t last_byte = (bit_end > 0) ? ((bit_end - 1) / 8) : first_byte;
            
            // Determine if first/last bytes are shared with other threads
            const bool first_byte_shared = (bit_start % 8) != 0;
            const bool last_byte_shared = (bit_end % 8) != 0;
            
            uint64_t bitstream_position = bit_start;
            uint8_t current_byte = 0;
            
            for (size_t i = start; i < end; ++i) {
                uint8_t byte = static_cast<uint8_t>(source[i]);
                const std::vector<bool>& code = dict[byte];

                for (size_t j = 0; j < code.size(); ++j) {
                    const size_t byte_idx = bitstream_position / 8;
                    const size_t bit_offset = 7 - (bitstream_position % 8);
                    
                    // Accumulate bit into current_byte
                    current_byte |= static_cast<uint8_t>(code[j]) << bit_offset;
                    
                    // Check if we've completed a byte (bit_offset == 0 means we just wrote the LSB)
                    if (bit_offset == 0) { // 
                        if ((byte_idx == first_byte && first_byte_shared) ||
                            (byte_idx == last_byte && last_byte_shared)) {
                            // Use atomic for shared boundary bytes
                            #pragma omp atomic
                            reinterpret_cast<uint8_t&>(destination[byte_idx]) |= current_byte;
                        } else {
                            // Direct write for non-shared bytes
                            destination[byte_idx] = static_cast<std::byte>(current_byte);
                        }
                        current_byte = 0;
                    }
                    
                    bitstream_position++;
                }
            }
            
            // Write any remaining partial byte
            if ((bitstream_position % 8) != 0) {
                const size_t byte_idx = (bitstream_position - 1) / 8;
                if (byte_idx == first_byte && first_byte_shared) {
                    #pragma omp atomic
                    reinterpret_cast<uint8_t&>(destination[byte_idx]) |= current_byte;
                } else {
                    destination[byte_idx] |= static_cast<std::byte>(current_byte);
                }
            }
        }
    }
}

uint64_t huffman_encode_file(const std::string& input_file, const std::string& output_file) {
    // read input file
    int in_fd = open(input_file.c_str(), O_RDONLY);
    if (in_fd < 0)
        throw std::runtime_error("Failed to open input file");

    struct stat file_stat;
    if (fstat(in_fd, &file_stat) < 0) {
        close(in_fd);
        throw std::runtime_error("Failed to get file size");
    }

    size_t file_size = file_stat.st_size;
    void *in_ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, in_fd, 0);
    close(in_fd);

    if (in_ptr == MAP_FAILED)
        throw std::runtime_error("Failed to map file");

    std::span<const std::byte> input_data(static_cast<const std::byte*>(in_ptr), file_size);
    const std::array<uint64_t, 256> hist = histogram_parallel(input_data);
    const std::vector<HuffmanNode> tree = huffman_tree(hist);
    std::array<std::vector<bool>, 256> dict = huffman_dict(tree);
    canonicalize_huffman_dict(dict);

    const uint64_t compressed_size_in_bits = calculate_compressed_size_in_bits(hist, dict);
    const uint64_t compressed_size_in_bytes = (compressed_size_in_bits + 7) / 8; // round up to full bytes

    HuffmanHeader header;
    header.original_file_size = file_size;
    header.compressed_data_size = compressed_size_in_bits;
    for (size_t i = 0; i < 256; i++) {
        header.code_lengths[i] = dict[i].size();
    }

    size_t total_output_size = sizeof(HuffmanHeader) + compressed_size_in_bytes;
    
    int out_fd = open(output_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        munmap(in_ptr, file_size);
        throw std::runtime_error("Failed to open output file");
    }

    if (ftruncate(out_fd, total_output_size) < 0) {
        close(out_fd);
        munmap(in_ptr, file_size);
        throw std::runtime_error("Failed to truncate output file");
    }

    void *out_ptr = mmap(nullptr, total_output_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    close(out_fd);

    if (out_ptr == MAP_FAILED) {
        munmap(in_ptr, file_size);
        throw std::runtime_error("Failed to map file");
    }

    std::memcpy(out_ptr, &header, sizeof(HuffmanHeader));

    std::byte *data_start = static_cast<std::byte*>(out_ptr) + sizeof(HuffmanHeader);
    std::span<std::byte> output_data(data_start, compressed_size_in_bytes);
    
    // The actual encoding is done here
    huffman_encode_span(input_data, output_data, dict);

    // Cleanup
    munmap(in_ptr, file_size);
    munmap(out_ptr, total_output_size);

    return total_output_size;
}

std::array<std::vector<bool>, 256> reconstruct_canonical_dict(const std::array<uint16_t, 256>& code_lengths) {
    std::array<std::vector<bool>, 256> dict;
    
    struct Entry { uint16_t len; uint16_t sym; };
    std::vector<Entry> entries;
    entries.reserve(256);
    
    for (size_t sym = 0; sym < 256; ++sym) {
        if (code_lengths[sym] > 0) {
            entries.emplace_back(code_lengths[sym], static_cast<uint16_t>(sym));
        }
    }
    
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.len != b.len) return a.len < b.len;
        return a.sym < b.sym;
    });
    
    uint64_t current_code = 0;
    uint16_t current_len = 0;
    
    for (const auto& entry : entries) {
        while (current_len < entry.len) {
            current_code <<= 1;
            current_len++;
        }
        
        std::vector<bool> code_vec;
        code_vec.reserve(current_len);
        for (int i = current_len - 1; i >= 0; --i) {
            code_vec.emplace_back((current_code >> i) & 1);
        }
        
        dict[entry.sym] = code_vec;
        
        current_code++;
    }
    
    return dict;
}

uint64_t huffman_decode_file(const std::string& input_file, const std::string& output_file) {
    int in_fd = open(input_file.c_str(), O_RDONLY);
    if (in_fd < 0) 
        throw std::runtime_error("Failed to open input file");

    struct stat in_stat;
    if (fstat(in_fd, &in_stat) < 0) {
        close(in_fd);
        throw std::runtime_error("Failed to get input file size");
    }
    size_t in_size = in_stat.st_size;

    if (in_size < sizeof(HuffmanHeader)) {
        close(in_fd);
        throw std::runtime_error("Input file too small to contain header");
    }

    void* in_ptr = mmap(nullptr, in_size, PROT_READ, MAP_PRIVATE, in_fd, 0);
    close(in_fd);

    if (in_ptr == MAP_FAILED)
        throw std::runtime_error("Failed to map input file");

    const HuffmanHeader* header = static_cast<const HuffmanHeader*>(in_ptr);
    size_t original_file_size = header->original_file_size;
    size_t compressed_data_bits = header->compressed_data_size;

    std::array<std::vector<bool>, 256> dict = reconstruct_canonical_dict(header->code_lengths);

    int out_fd = open(output_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        munmap(in_ptr, in_size);
        throw std::runtime_error("Failed to open output file");
    }

    if (ftruncate(out_fd, original_file_size) < 0) {
        close(out_fd);
        munmap(in_ptr, in_size);
        throw std::runtime_error("Failed to resize output file");
    }

    void* out_ptr = mmap(nullptr, original_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    close(out_fd);

    if (out_ptr == MAP_FAILED) {
        munmap(in_ptr, in_size);
        throw std::runtime_error("Failed to map output file");
    }

    const std::byte* compressed_start = static_cast<const std::byte*>(in_ptr) + sizeof(HuffmanHeader);
    size_t compressed_bytes = (compressed_data_bits + 7) / 8;

    std::span<const std::byte> source_span(compressed_start, compressed_bytes);
    std::span<std::byte> dest_span(static_cast<std::byte*>(out_ptr), original_file_size);

    // The actual decoding is done here
    huffman_decode_span(source_span, compressed_data_bits, dest_span, dict);

    // Cleanup
    munmap(out_ptr, original_file_size);
    munmap(in_ptr, in_size);

    return original_file_size;
}