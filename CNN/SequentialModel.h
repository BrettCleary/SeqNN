#ifndef CNN_SEQMODEL_H
#define CNN_SEQMODEL_H
#include "Layer.h"
#include <vector>
//#include <numpy/ndarraytypes.h>
//#include <boost/python/numpy.hpp>

//namespace py = boost::python;
//namespace np = boost::python::numpy;

class SequentialModel
{
	//PyObject_HEAD
	std::vector<Layer> allLayers;
	double weightStepSize = 0.025;

	void FwdProp(const std::vector<std::vector<double>>* input) {
		for (Layer& layer_i : allLayers) {
			input = layer_i.FwdProp(*input);
		}
	}

	void BackProp(const std::vector<std::vector<double>>& error) {
		const std::vector<std::vector<double>>* errorPtr = &error;
		for (int i = allLayers.size() - 1; i >= 0; --i) {
			errorPtr = allLayers[i].BackProp(*errorPtr);
		}
	}

	std::vector<std::vector<double>> CalcError(const std::vector<std::vector<double>>& target) {
		auto dError_dAct = target;
		auto& lastLayer = allLayers[allLayers.size() - 1];
		const std::vector<std::vector<double>>& output = *(lastLayer.GetOutput());

		for (int i = 0; i < target.size(); ++i) {
			for (int j = 0; j < target[0].size(); ++j) {
				dError_dAct[i][j] = (output[i][j] - target[i][j]) / (output[i][j] * (1 - output[i][j]));
			}
		}
		return move(dError_dAct);
	}
	
	void UpdateWeights() {
		for (Layer& layer_i : allLayers) {
			layer_i.UpdateWeights(weightStepSize);
		}
	}

	const std::vector<int> Predict(const std::vector<std::vector<std::vector<double>>>& inputData) {
		for (auto& input_i : inputData) {
			FwdProp(&input_i);
			auto& lastLayer = allLayers[allLayers.size() - 1];
			const std::vector<std::vector<double>>& output = *(lastLayer.GetOutput());

			double maxProb = -1;
			int maxI = -1;
			int maxJ = -1;
			for (int i = 0; i < output.size(); ++i) {
				for (int j = 0; j < output[0].size(); ++j) {
					if (output[i][j] > maxProb) {
						maxProb = output[i][j];
						maxI = i;
						maxJ = j;
					}
				}
			}
			return { maxI, maxJ };
		}
	}

public:
	SequentialModel(std::vector<Layer*> layerList) {
		//convert from pyobject* to list
		//allLayers = layerListConverted
	}

	void Add(Layer& layer) {
		
		//allLayers.push_back(layerConverted);
	}

	void Train(const std::vector<std::vector<std::vector<double>>>& inputData, const std::vector<std::vector<std::vector<double>>>& targets, int batchSize, int numEpochs) {

		int numBatches = numEpochs * std::min(inputData.size(), targets.size()) / batchSize;

		for (int batch_k = 0; batch_k < numBatches; ++batch_k) {
			for (int i = batch_k * batchSize; i < (batch_k - 1) * batchSize; ++i) {
				FwdProp(&inputData[i]);
				auto error = CalcError(targets[i]);
				BackProp(error);
			}
			UpdateWeights();
		}
	}


	/*std::vector<std::vector<char>> ConvertNumpyToVector2DByte(np::ndarray const & ndArry) {
		int numRows = ndArry.shape(0);
		int numCols = ndArry.shape(1);
		std::vector<std::vector<char>> ret;

	}*/

};
//
//typedef struct {
//	PyObject_HEAD
//} MyObject;
//
//static PyTypeObject MyObject_Type;
//
////static PyTypeObject MyObject_Type {
////	PyVarObject_HEAD_INIT(NULL, 0)
////	"CNN.MyObject",               /* tp_name */
////	sizeof(MyObject),         /* tp_basicsize */
////	0,                              /* tp_itemsize */
////	0,      /* tp_dealloc */
////	0,                              /* tp_vectorcall_offset */
////	0,                              /* tp_getattr */
////	0,                              /* tp_setattr */
////	0,                              /* tp_as_async */
////	0,           /* tp_repr */
////	0,                              /* tp_as_number */
////	0,                              /* tp_as_sequence */
////	0,                              /* tp_as_mapping */
////	0,                              /* tp_hash */
////	0,                              /* tp_call */
////	0,                              /* tp_str */
////	0,                              /* tp_getattro */
////	0,                              /* tp_setattro */
////	0,                              /* tp_as_buffer */
////	Py_TPFLAGS_DEFAULT,                              /* tp_flags */
////	"Sequental CNN Model",                   /* tp_doc */
////	0,                              /* tp_traverse */
////	0,                              /* tp_clear */
////	0,                              /* tp_richcompare */
////	0,                              /* tp_weaklistoffset */
////	0,                              /* tp_iter */
////	0,                              /* tp_iternext */
////	0,                              /* tp_methods */
////	0,                              /* tp_members */
////	0,                              /* tp_getset */
////	0,                              /* tp_base */
////	0,                              /* tp_dict */
////	0,                              /* tp_descr_get */
////	0,                              /* tp_descr_set */
////	0,                              /* tp_dictoffset */
////	0,                              /* tp_init */
////	0,                              /* tp_alloc */
////	PyType_GenericNew,                      /* tp_new */
////};
//
//static PyMethodDef CNN_methods[] = {
//	// The first property is the name exposed to Python, fast_tanh, the second is the C++
//	// function name that contains the implementation.
//	//{ "fast_tanh", (PyCFunction)Fit, METH_O, nullptr },
//
//	// Terminate the array with an object containing nulls.
//	{ nullptr, nullptr, 0, nullptr }
//};
//
//static PyModuleDef CNN_module = {
//	PyModuleDef_HEAD_INIT,
//	"CNN",                        // Module name to use with Python import statements
//	"Sequential NN Library",  // Module description
//	0,
//	CNN_methods                   // Structure that defines the methods of the module
//};
//
//PyMODINIT_FUNC PyInit_CNN() {
//	MyObject_Type.tp_name = "";
//
//		.tp_name = "custom.Custom",
//		.tp_doc = "Custom objects",
//		.tp_basicsize = sizeof(CustomObject),
//		.tp_itemsize = 0,
//		.tp_flags = Py_TPFLAGS_DEFAULT,
//		.tp_new = PyType_GenericNew,
//
//
//
//	PyObject* m;
//	if (PyType_Ready(&SeqModelObject_Type) < 0)
//		return NULL;
//
//	m = PyModule_Create(&CNN_module);
//	if (m == NULL)
//		return NULL;
//
//	Py_INCREF(&SeqModelObject_Type);
//	if (PyModule_AddObject(m, "SequentialModel", (PyObject*)&SeqModelObject_Type) < 0) {
//		Py_DECREF(&SeqModelObject_Type);
//		Py_DECREF(m);
//		return NULL;
//	}
//
//	return m;
//}
#endif // CNN_SEQMODEL_H