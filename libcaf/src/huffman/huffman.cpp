#include "huffman.h"

#include <queue>
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

    std::array<uint64_t, 256> freqs = partial_freqs[0];
    for (int t = 1; t < num_threads; ++t) {
        #pragma omp simd
        for (size_t bin = 0; bin < 256; ++bin) {
            freqs[bin] += partial_freqs[t][bin];
        }
    }

    return freqs;
}

struct NodeComparator {
    const std::vector<HuffmanNode>& nodes;

    NodeComparator(const std::vector<HuffmanNode>& nodes) : nodes(nodes) {}

    bool operator()(size_t lhs, size_t rhs) const {
        return nodes[lhs].frequency > nodes[rhs].frequency;
    }
};

std::vector<HuffmanNode> huffman_tree(const std::array<uint64_t, 256>& hist) {
    std::vector<HuffmanNode> nodes;
    nodes.reserve(2 * 256 - 1); // Max number of nodes in a full binary tree with 256 leaves

    std::priority_queue<size_t, std::vector<size_t>, NodeComparator> min_heap{NodeComparator(nodes)};

    // Create all leaf nodes
    for (size_t i = 0; i < hist.size(); i++) {
        if (hist[i] == 0)
            continue;

        nodes.emplace_back(hist[i], static_cast<std::byte>(i));
        min_heap.push(nodes.size() - 1);
    }

    if (min_heap.empty()) {
        return nodes;
    }

    // build all the internal nodes, until there is only one node left in the heap
    while (min_heap.size() > 1) {
        const size_t left_index = min_heap.top();
        min_heap.pop();
        const size_t right_index = min_heap.top();
        min_heap.pop();

        const uint64_t parent_frequency = nodes[left_index].frequency + nodes[right_index].frequency;
        nodes.emplace_back(parent_frequency, left_index, right_index);

        const size_t parent_index = nodes.size() - 1;
        min_heap.push(parent_index);
    }

    // the remaining node is the root, always at nodes.size() - 1
    return nodes;
}