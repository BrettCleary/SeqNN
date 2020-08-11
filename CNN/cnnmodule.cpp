//#include <Python.h>
#include <Windows.h>
#include <cmath>

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}
/*
PyObject* tanh_impl(PyObject*, PyObject* o) {
    double x = PyFloat_AsDouble(o);
    double tanh_x = sinh_impl(x) / cosh_impl(x);
    return PyFloat_FromDouble(tanh_x);
}*/
/*
static PyMethodDef CNN_methods[] = {
    // The first property is the name exposed to Python, fast_tanh, the second is the C++
    // function name that contains the implementation.
    { "fast_tanh", (PyCFunction)tanh_impl, METH_O, nullptr },

    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef CNN_module = {
    PyModuleDef_HEAD_INIT,
    "CNN",                        // Module name to use with Python import statements
    "Provides some functions, but faster",  // Module description
    0,
    CNN_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_CNN() {
    return PyModule_Create(&CNN_module);
}*/

/*namespace py = pybind11;

PYBIND11_MODULE(CNN, m) {
    m.def("fast_tanh2", &tanh_impl, R"pbdoc(
        Compute a hyperbolic tangent of a single argument expressed in radians.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}*/