#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "caf.h"
#include "hash_types.h"
#include "object_io.h"
#include "huffman/huffman.h"
#include "huffman/huffman_tree.h"

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
    m.def("histogram", [](std::optional<py::bytes> str_obj) {
        if (!str_obj) {
            return histogram(nullptr, 0);
        }

        char* buffer;
        Py_ssize_t length;
        
        if (PyBytes_AsStringAndSize(str_obj->ptr(), &buffer, &length) != 0) {
            throw py::error_already_set();
        }
        
        return histogram(reinterpret_cast<const unsigned char*>(buffer), static_cast<size_t>(length));
    }, py::arg("str"));

    // HuffmanTree bindings
    m.attr("HUFFMAN_NODE_NULL_INDEX") = static_cast<uint64_t>(HUFFMAN_NODE_NULL_INDEX);
    py::class_<HuffmanTree>(m, "HuffmanTree")
        .def(py::init<>())
        .def("build_tree_and_get_root_index", 
            [](HuffmanTree* self, std::optional<py::bytes> str_obj) {
                if (!str_obj) {
                    return self->build_tree_and_get_root_index(nullptr, 0);
                }

                char* buffer;
                Py_ssize_t length;
                if (PyBytes_AsStringAndSize(str_obj->ptr(), &buffer, &length) != 0) {
                    throw py::error_already_set();
                }

                return self->build_tree_and_get_root_index(reinterpret_cast<const unsigned char*>(buffer), static_cast<size_t>(length));
            }, py::arg("str"))
        .def("get_num_nodes", &HuffmanTree::get_num_nodes)
        .def("get_symbol", &HuffmanTree::get_symbol, py::arg("index"))
        .def("get_frequency", &HuffmanTree::get_frequency, py::arg("index"))
        .def("get_left_index", &HuffmanTree::get_left_index, py::arg("index"))
        .def("get_right_index", &HuffmanTree::get_right_index, py::arg("index"))
        .def("is_leaf_node", &HuffmanTree::is_leaf_node, py::arg("index"));

    
}