#include "huffman.h"
#include "huffman_node.h"

#include <omp.h>

/*
 * Histogram function variants with different optimization levels:
 *
 * | Function                | Parallelism | 64-bit Loading | SIMD Merge |
 * |-------------------------|-------------|----------------|------------|
 * | histogram               | ❌          | ❌             | ❌         |
 * | histogram_parallel      | ✅          | ❌             | ❌         |
 * | histogram_parallel_64bit| ✅          | ✅             | ❌         |
 * | histogram_fast          | ✅          | ✅             | ✅         |
 *
 * SIMD Merge: Uses #pragma omp simd to vectorize the histogram merge loop
 */

std::array<uint64_t, 256> histogram(std::span<const std::byte> data) {
    std::array<uint64_t, 256> freqs = {0};
    
    for (std::byte b : data) {
        freqs[static_cast<unsigned char>(b)]++;
    }
    
    return freqs;
}

std::array<uint64_t, 256> histogram_parallel(std::span<const std::byte> data) {
    const int num_threads = omp_get_max_threads();
    std::vector<std::array<uint64_t, 256>> partial_freqs(num_threads);

    const size_t chunk_size = (data.size() + num_threads - 1) / num_threads;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_freqs = partial_freqs[thread_id];
        local_freqs.fill(0);

        const size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, data.size());

        // Simple byte-by-byte processing, no 64-bit loading
        for (size_t i = start; i < end; ++i) {
            local_freqs[static_cast<unsigned char>(data[i])]++;
        }
    }

    // Merge partial histograms - no SIMD
    std::array<uint64_t, 256> freqs = partial_freqs[0];
    for (int t = 1; t < num_threads; ++t) {
        for (size_t bin = 0; bin < 256; ++bin) {
            freqs[bin] += partial_freqs[t][bin];
        }
    }

    return freqs;
}

std::array<uint64_t, 256> histogram_parallel_64bit(std::span<const std::byte> data) {
    const int num_threads = omp_get_max_threads();
    std::vector<std::array<uint64_t, 256>> partial_freqs(num_threads);

    const size_t chunk_size = (data.size() + num_threads - 1) / num_threads;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_freqs = partial_freqs[thread_id];
        local_freqs.fill(0);

        const size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, data.size());

        const auto* ptr = reinterpret_cast<const uint64_t*>(data.data() + start);
        size_t remaining = end - start;

        // Process 8 bytes at a time
        while (remaining >= 8) {
            uint64_t word = *ptr;

            local_freqs[(word >>  0) & 0xFF]++;
            local_freqs[(word >>  8) & 0xFF]++;
            local_freqs[(word >> 16) & 0xFF]++;
            local_freqs[(word >> 24) & 0xFF]++;
            local_freqs[(word >> 32) & 0xFF]++;
            local_freqs[(word >> 40) & 0xFF]++;
            local_freqs[(word >> 48) & 0xFF]++;
            local_freqs[(word >> 56) & 0xFF]++;

            ptr++;
            remaining -= 8;
        }

        // Handle remaining bytes
        const auto* byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
        while (remaining > 0) {
            local_freqs[*byte_ptr]++;
            byte_ptr++;
            remaining--;
        }
    }

    // Merge partial histograms - no SIMD
    std::array<uint64_t, 256> freqs = partial_freqs[0];
    for (int t = 1; t < num_threads; ++t) {
        for (size_t bin = 0; bin < 256; ++bin) {
            freqs[bin] += partial_freqs[t][bin];
        }
    }

    return freqs;
}

std::array<uint64_t, 256> histogram_fast(std::span<const std::byte> data) {
    const int num_threads = omp_get_max_threads();
    std::vector<std::array<uint64_t, 256>> partial_freqs(num_threads);

    const size_t chunk_size = (data.size() + num_threads - 1) / num_threads;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_freqs = partial_freqs[thread_id];
        local_freqs.fill(0);

        size_t start = thread_id * chunk_size;
        const size_t end = std::min(start + chunk_size, data.size());

        const auto* byte_ptr = reinterpret_cast<const uint8_t*>(data.data() + start);
        size_t remaining = end - start;

        // Align to 8-byte boundary
        while (remaining > 0 && (reinterpret_cast<uintptr_t>(byte_ptr) & 7) != 0) {
            local_freqs[*byte_ptr]++;
            byte_ptr++;
            remaining--;
        }

        // Process 8 bytes at a time (now aligned)
        const auto* ptr = reinterpret_cast<const uint64_t*>(byte_ptr);
        while (remaining >= 8) {
            uint64_t word = *ptr;

            local_freqs[(word >>  0) & 0xFF]++;
            local_freqs[(word >>  8) & 0xFF]++;
            local_freqs[(word >> 16) & 0xFF]++;
            local_freqs[(word >> 24) & 0xFF]++;
            local_freqs[(word >> 32) & 0xFF]++;
            local_freqs[(word >> 40) & 0xFF]++;
            local_freqs[(word >> 48) & 0xFF]++;
            local_freqs[(word >> 56) & 0xFF]++;

            ptr++;
            remaining -= 8;
        }

        // Handle remaining bytes
        byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
        while (remaining > 0) {
            local_freqs[*byte_ptr]++;
            byte_ptr++;
            remaining--;
        }
    }

    std::array<uint64_t, 256> freqs = partial_freqs[0];
    for (int t = 1; t < num_threads; ++t) {
        #pragma omp simd
        for (size_t bin = 0; bin < 256; ++bin) {
            freqs[bin] += partial_freqs[t][bin];
        }
    }

    return freqs;
}