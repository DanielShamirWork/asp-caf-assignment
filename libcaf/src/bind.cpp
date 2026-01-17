#include "caf.h"
#include "hash_types.h"
#include "object_io.h"
#include "huffman/huffman.h"
#include "util/bitreader.h"

#include <span>
#include <stdexcept>
#include <vector>
#include <cstring>

#include <pybind11/pybind11.h>

// Custom type caster for std::byte
// must be defined before pybind11/stl.h!
namespace pybind11 { namespace detail {
    template <> struct type_caster<std::byte> {
    public:
        PYBIND11_TYPE_CASTER(std::byte, const_name("int"));

        bool load(handle src, bool) {
            PyObject *source = src.ptr();
            if (!PyLong_Check(source)) return false;
            unsigned long val = PyLong_AsUnsignedLong(source);
            if (PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            if (val > 255) return false;
            value = static_cast<std::byte>(val);
            return true;
        }

        static handle cast(std::byte src, return_value_policy /* policy */, handle /* parent */) {
            return PyLong_FromUnsignedLong(static_cast<unsigned char>(src));
        }
    };
}}

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

// Helper to convert py::buffer to std::span (requires C++20)
BitReader create_reader(py::buffer b, size_t data_size_in_bits) {
    py::buffer_info info = b.request();
    if (info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::runtime_error("Incompatible buffer format!");
    }
    // Create span from the buffer's raw pointer
    auto span = std::span<const std::byte>(
        static_cast<const std::byte*>(info.ptr), 
        static_cast<size_t>(info.size)
    );
    return BitReader(span, data_size_in_bits);
}

PYBIND11_MODULE(_libcaf, m) {
    // caf
    m.def("hash_file", hash_file);
    m.def("hash_string", hash_string);
    m.def("hash_length", hash_length);
    m.def("save_file_content", save_file_content);
    m.def("open_content_for_writing", open_content_for_writing);
    m.def("delete_content", delete_content);
    m.def("open_content_for_reading", open_content_for_reading);

    // huffman constants
    m.attr("HUFFMAN_HEADER_SIZE") = HUFFMAN_HEADER_SIZE;

    // hash_types
    m.def("hash_object", py::overload_cast<const Blob&>(&hash_object), py::arg("blob"));
    m.def("hash_object", py::overload_cast<const Tree&>(&hash_object), py::arg("tree"));
    m.def("hash_object", py::overload_cast<const Commit&>(&hash_object), py::arg("commit"));

    // object_io
    m.def("save_commit", &save_commit);
    m.def("load_commit", &load_commit);
    m.def("save_tree", &save_tree);
    m.def("load_tree", &load_tree);

    py::class_<Blob>(m, "Blob")
    .def(py::init<std::string>())
    .def_readonly("hash", &Blob::hash);

    py::enum_<TreeRecord::Type>(m, "TreeRecordType")
    .value("TREE", TreeRecord::Type::TREE)
    .value("BLOB", TreeRecord::Type::BLOB)
    .value("COMMIT", TreeRecord::Type::COMMIT)
    .export_values();

    py::class_<TreeRecord>(m, "TreeRecord")
    .def(py::init<TreeRecord::Type, std::string, std::string>())
    .def_readonly("type", &TreeRecord::type)
    .def_readonly("hash", &TreeRecord::hash)
    .def_readonly("name", &TreeRecord::name)
    .def("__eq__", [](const TreeRecord &self, const TreeRecord &other) {
        return self.type == other.type && self.hash == other.hash && self.name == other.name;
    });

    py::class_<Tree>(m, "Tree")
    .def(py::init<const std::map<std::string, TreeRecord>&>())
    .def_readonly("records", &Tree::records);

    py::class_<Commit>(m, "Commit")
        .def(py::init<const string &, const string&, const string&, time_t, const std::optional<std::string>&>())
        .def_readonly("tree_hash", &Commit::tree_hash)
        .def_readonly("author", &Commit::author)
        .def_readonly("message", &Commit::message)
        .def_readonly("timestamp", &Commit::timestamp)
        .def_readonly("parent", &Commit::parent);

    // histogram for huffman compression
    m.def("histogram", [](py::array_t<uint8_t, py::array::c_style> array) {
        auto info = array.request();
        if (info.ndim != 1) {
            throw std::runtime_error("histogram expects a 1-D numpy array");
        }
        auto* ptr = static_cast<const std::byte*>(info.ptr);
        return histogram(std::span<const std::byte>(ptr, static_cast<size_t>(info.shape[0])));
    }, py::arg("data"));

    m.def("histogram_parallel", [](py::array_t<uint8_t, py::array::c_style> array) {
        auto info = array.request();
        if (info.ndim != 1) {
            throw std::runtime_error("histogram_parallel expects a 1-D numpy array");
        }
        auto* ptr = static_cast<const std::byte*>(info.ptr);
        return histogram_parallel(std::span<const std::byte>(ptr, static_cast<size_t>(info.shape[0])));
    }, py::arg("data"));

    m.def("histogram_parallel_64bit", [](py::array_t<uint8_t, py::array::c_style> array) {
        auto info = array.request();
        if (info.ndim != 1) {
            throw std::runtime_error("histogram_parallel_64bit expects a 1-D numpy array");
        }
        auto* ptr = static_cast<const std::byte*>(info.ptr);
        return histogram_parallel_64bit(std::span<const std::byte>(ptr, static_cast<size_t>(info.shape[0])));
    }, py::arg("data"));

    m.def("histogram_fast", [](py::array_t<uint8_t, py::array::c_style> array) {
        auto info = array.request();
        if (info.ndim != 1) {
            throw std::runtime_error("histogram_fast expects a 1-D numpy array");
        }
        auto* ptr = static_cast<const std::byte*>(info.ptr);
        return histogram_fast(std::span<const std::byte>(ptr, static_cast<size_t>(info.shape[0])));
    }, py::arg("data"));

    // huffman_tree bindings
    py::class_<LeafNodeData>(m, "LeafNodeData")
        .def_readonly("symbol", &LeafNodeData::symbol);

    py::class_<InternalNodeData>(m, "InternalNodeData")
        .def_readonly("left_index", &InternalNodeData::left_index)
        .def_readonly("right_index", &InternalNodeData::right_index);

    py::class_<HuffmanNode>(m, "HuffmanNode")
        .def_readonly("frequency", &HuffmanNode::frequency)
        .def_property_readonly("is_leaf", [](const HuffmanNode& n) {
            return std::holds_alternative<LeafNodeData>(n.data);
        })
        .def_property_readonly("symbol", [](const HuffmanNode& n) -> std::optional<uint8_t> {
            if (auto* val = std::get_if<LeafNodeData>(&n.data)) {
                return static_cast<uint8_t>(val->symbol);
            }
            return std::nullopt;
        })
        .def_property_readonly("left_index", [](const HuffmanNode& n) -> std::optional<TreeIndex> {
            if (auto* val = std::get_if<InternalNodeData>(&n.data)) {
                return val->left_index;
            }
            return std::nullopt;
        })
        .def_property_readonly("right_index", [](const HuffmanNode& n) -> std::optional<TreeIndex> {
            if (auto* val = std::get_if<InternalNodeData>(&n.data)) {
                return val->right_index;
            }
            return std::nullopt;
        });

    m.def("huffman_tree", &huffman_tree);

    // huffman_dict bindings
    m.def("huffman_dict", &huffman_dict);

    m.def("canonicalize_huffman_dict", [](std::array<std::vector<bool>, 256> dict) {
        canonicalize_huffman_dict(dict);
        return dict;
    }, py::arg("dict"));

    m.def("next_canonical_huffman_code", &next_canonical_huffman_code, py::arg("code"));

    m.def("calculate_compressed_size_in_bits", [](py::array_t<uint64_t, py::array::c_style> hist,
                                                   const std::array<std::vector<bool>, 256>& dict) {
        auto info = hist.request();
        if (info.ndim != 1 || info.shape[0] != 256) {
            throw std::runtime_error("calculate_compressed_size_in_bits expects a 1-D numpy array of size 256");
        }
        std::array<uint64_t, 256> hist_arr;
        std::memcpy(hist_arr.data(), info.ptr, 256 * sizeof(uint64_t));
        return calculate_compressed_size_in_bits(hist_arr, dict);
    }, py::arg("hist"), py::arg("dict"));

    // huffman_encode_span bindings (for benchmarking different implementations)
    m.def("huffman_encode_span", [](py::array_t<uint8_t, py::array::c_style> source,
                                    py::array_t<uint8_t, py::array::c_style> destination,
                                    const std::array<std::vector<bool>, 256>& dict) {
        auto src_info = source.request();
        auto dst_info = destination.request(true);  // writable
        if (src_info.ndim != 1 || dst_info.ndim != 1) {
            throw std::runtime_error("huffman_encode_span expects 1-D numpy arrays");
        }
        auto* src_ptr = static_cast<const std::byte*>(src_info.ptr);
        auto* dst_ptr = static_cast<std::byte*>(dst_info.ptr);
        huffman_encode_span(
            std::span<const std::byte>(src_ptr, static_cast<size_t>(src_info.shape[0])),
            std::span<std::byte>(dst_ptr, static_cast<size_t>(dst_info.shape[0])),
            dict);
    }, py::arg("source"), py::arg("destination"), py::arg("dict"));

    m.def("huffman_build_reverse_dict", 
        [](const std::array<std::vector<bool>, 256>& dict, size_t max_code_len) {
            auto result_array = huffman_build_reverse_dict(dict, max_code_len);
            return std::vector<uint16_t>(result_array.begin(), result_array.end());
        },
        py::arg("dict"), py::arg("max_code_len")
    );

    m.def("huffman_decode_span", [](py::array_t<uint8_t, py::array::c_style> source,
                                    const size_t source_size_in_bits,
                                    py::array_t<uint8_t, py::array::c_style> destination,
                                    const std::array<std::vector<bool>, 256>& dict) {
        auto src_info = source.request();
        auto dst_info = destination.request(true);
        if (src_info.ndim != 1 || dst_info.ndim != 1) {
            throw std::runtime_error("huffman_decode_span expects 1-D numpy arrays");
        }
        auto* src_ptr = static_cast<const std::byte*>(src_info.ptr);
        auto* dst_ptr = static_cast<std::byte*>(dst_info.ptr);
        huffman_decode_span(
            std::span<const std::byte>(src_ptr, static_cast<size_t>(src_info.shape[0])),
            source_size_in_bits,
            std::span<std::byte>(dst_ptr, static_cast<size_t>(dst_info.shape[0])),
            dict);
    }, py::arg("source"), py::arg("source_size_in_bits"), py::arg("destination"), py::arg("dict"));

    m.def("huffman_encode_span_parallel", [](py::array_t<uint8_t, py::array::c_style> source,
                                              py::array_t<uint8_t, py::array::c_style> destination,
                                              const std::array<std::vector<bool>, 256>& dict) {
        auto src_info = source.request();
        auto dst_info = destination.request(true);
        if (src_info.ndim != 1 || dst_info.ndim != 1) {
            throw std::runtime_error("huffman_encode_span_parallel expects 1-D numpy arrays");
        }
        auto* src_ptr = static_cast<const std::byte*>(src_info.ptr);
        auto* dst_ptr = static_cast<std::byte*>(dst_info.ptr);
        huffman_encode_span_parallel(
            std::span<const std::byte>(src_ptr, static_cast<size_t>(src_info.shape[0])),
            std::span<std::byte>(dst_ptr, static_cast<size_t>(dst_info.shape[0])),
            dict);
    }, py::arg("source"), py::arg("destination"), py::arg("dict"));

    m.def("huffman_encode_span_parallel_twopass", [](py::array_t<uint8_t, py::array::c_style> source,
                                                      py::array_t<uint8_t, py::array::c_style> destination,
                                                      const std::array<std::vector<bool>, 256>& dict) {
        auto src_info = source.request();
        auto dst_info = destination.request(true);
        if (src_info.ndim != 1 || dst_info.ndim != 1) {
            throw std::runtime_error("huffman_encode_span_parallel_twopass expects 1-D numpy arrays");
        }
        auto* src_ptr = static_cast<const std::byte*>(src_info.ptr);
        auto* dst_ptr = static_cast<std::byte*>(dst_info.ptr);
        huffman_encode_span_parallel_twopass(
            std::span<const std::byte>(src_ptr, static_cast<size_t>(src_info.shape[0])),
            std::span<std::byte>(dst_ptr, static_cast<size_t>(dst_info.shape[0])),
            dict);
    }, py::arg("source"), py::arg("destination"), py::arg("dict"));

    // huffman_encdec bindings
    m.def("huffman_encode_file", &huffman_encode_file);
    m.def("huffman_decode_file", &huffman_decode_file);

    m.attr("MAX_CODE_LEN") = MAX_CODE_LEN;

    // Utils bindings
    py::class_<BitReader>(m, "BitReader")
        .def(py::init(&create_reader), 
             py::arg("data"), 
             py::arg("data_size_in_bits"),
             py::keep_alive<1, 2>())
        .def("read", &BitReader::read, py::arg("n_bits"))
        .def("advance", &BitReader::advance, py::arg("n_bits"))
        .def("done", &BitReader::done);
}