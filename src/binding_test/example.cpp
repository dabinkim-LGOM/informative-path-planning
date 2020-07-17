#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
// #include <pybind11/eigen.h>

namespace py = pybind11;

int add(int i, int j)
{
    return i+j;
}

PYBIND11_MODULE(ex_py2, m)
{
    m.doc()= "pybind11 example";
    m.def("add", &add, "A function which adds two numbers");
}