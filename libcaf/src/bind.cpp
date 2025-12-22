#include "caf.h"
#include "hash_types.h"
#include "object_io.h"
#include "huffman/huffman.h"

#include <span>
#include <stdexcept>
#include <vector>
#include <cstring>

#include <pybind11/pybind11.h>

// Custom type caster for std::byte
// must be defined pybind11/stl.h!
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

PYBIND11_MODULE(_libcaf, m) {
    // caf
    m.def("hash_file", hash_file);
    m.def("hash_string", hash_string);
    m.def("hash_length", hash_length);
    m.def("save_file_content", save_file_content);
    m.def("open_content_for_writing", open_content_for_writing);
    m.def("delete_content", delete_content);
    m.def("open_content_for_reading", open_content_for_reading);

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
    .def(py::init<const std::unordered_map<std::string, TreeRecord>&>())
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
}