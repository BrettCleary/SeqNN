#ifndef CNN_SEQMODEL_H
#define CNN_SEQMODEL_H
#include "Layer.h"
#include "Conv2DLayer.h"
#include "DenseLayer.h"
#include "Pool2DLayer.h"
#include <vector>
#include <iostream>
//#include <numpy/ndarraytypes.h>
//#include <boost/python/numpy.hpp>

//namespace py = boost::python;
//namespace np = boost::python::numpy;

class SequentialModel
{
	//PyObject_HEAD
	//std::vector<Layer*> allLayers;
	Layer* allLayers[3];
	double weightStepSize = .025;

	std::vector<std::vector<std::vector<double>>> inputData;
	//[dataPoint][row][col]
	std::vector<std::vector<std::vector<double>>> targets;
	int batchSize = 1;
	int numEpochs = 1;




	void FwdProp(const std::vector<std::vector<double>>* input) {
		/*for (Layer* layer_i : allLayers) {
			input = layer_i->FwdProp(*input);
		}*/
		//std::cout << std::endl;
		for (int i = 0; i < 3; ++i) {
			input = allLayers[i]->FwdProp(*input);
			//std::cout << "fwdprop layer i: " << i << " completed." << std::endl;
		}
		//std::cout << std::endl;
	}

	void BackProp(const std::vector<std::vector<double>>& error) {
		const std::vector<std::vector<double>>* errorPtr = &error;
		/*for (int i = allLayers.size() - 1; i >= 0; --i) {
			errorPtr = allLayers[i]->BackProp(*errorPtr);
		}*/
		//std::cout << std::endl;
		for (int i = 2; i >= 0; --i) {
			errorPtr = allLayers[i]->BackProp(*errorPtr);
			//std::cout << " backprop layer i: " << i << " completed." << std::endl;
		}
		//std::cout << std::endl;
	}

	std::vector<std::vector<double>> CalcError(const std::vector<std::vector<double>>& target) {
		auto dError_dAct = target;
		//auto& lastLayer = allLayers[allLayers.size() - 1];
		auto& lastLayer = allLayers[2];
		const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());

		for (int i = 0; i < target.size(); ++i) {
			for (int j = 0; j < target[0].size(); ++j) {
				dError_dAct[i][j] = (output[i][j] - target[i][j]) / (output[i][j] * (1 - output[i][j]));
				//std::cout << dError_dAct[i][j] << std::endl;
			}
		}
		return move(dError_dAct);
	}
	
	void UpdateWeights() {
		/*for (Layer* layer_i : allLayers) {
			layer_i->UpdateWeights(weightStepSize);
		}*/

		for (int i = 0; i < 3; ++i) {
			allLayers[i]->UpdateWeights(weightStepSize);
			//std::cout << "Updated Weights for i = " << i << std::endl;
		}
	}

	void _AddInputDataPoint(std::vector<std::vector<double>>& dataPoint) {
		inputData.push_back(dataPoint);
	}

	const std::vector<int> _Predict(const std::vector<std::vector<std::vector<double>>>& inputData) {
		std::vector<int> maxOutputs;
		for (auto& input_i : inputData) {
			FwdProp(&input_i);
			//std::cout << "fwdprop completed" << std::endl;
			//auto& lastLayer = allLayers[allLayers.size() - 1];
			auto& lastLayer = allLayers[2];
			const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());

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
			maxOutputs.push_back(maxJ);
		}
		return maxOutputs;
	}

	std::vector<std::vector<std::vector<double>>> ConvertNpToVector(int len1_, int len2_, int len3_, double* vec_) {
		std::vector<std::vector<std::vector<double>>> v(len3_);
		bool display = false;
		for (int k = 0; k < len3_; ++k) {
			if (k == 0)
				display = false;
			else
				display = false;
			std::vector<std::vector<double>> arry(len1_);
			v[k] = std::move(arry);
			for (int i = 0; i < len1_; ++i) {
				//std::cout << "len1_: " << len1_ << " len2_: " << len2_ << " len3_: " << len3_ << std::endl;
				if (display) {
					//std::cout << "i: " << i << std::endl;
					for (int j = 0; j < len2_; ++j) {
						//std::cout << " j: " << j << " val: " << *(vec_ + k * len1_ * len2_ + i * len2_ + j) << " ";
					}
				}

				v[k][i].insert(v[k][i].end(), vec_ + k * len1_ * len2_ + i * len2_, vec_ + k * len1_ * len2_ + (i + 1) * len2_);
			}
		}
		return std::move(v);
	}

public:
	//SequentialModel(std::vector<Layer*> layerList) {
	SequentialModel(){
		//convert from pyobject* to list
		//allLayers = layerListConverted

		//testClass b;
		//Conv2DLayer a;
		//testing
		//allLayers.push_back(new Conv2DLayer());
		//allLayers.push_back(new Pool2DLayer());
		//allLayers.push_back(new DenseLayer());

		allLayers[0] = new Conv2DLayer();
		allLayers[1] = new Pool2DLayer();
		allLayers[2] = new DenseLayer();
	}

	void Add(Layer& layer) {
		
		//allLayers.push_back(layerConverted);
	}

	

	int AddInputDataPoint(int len1_, int len2_, double* vec_) {
		std::vector< std::vector<double> > v(len1_);
		for (int i = 0; i < len1_; ++i) {
			v[i].insert(v[i].end(), vec_ + i * len2_, vec_ + (i + 1) * len2_);
		}
		_AddInputDataPoint(v);

		return 1;
	}

	int AddInputDataPoints(int len1_, int len2_, int len3_, double* vec_) {
		//return 3;
		inputData = ConvertNpToVector(len1_, len2_, len3_, vec_);
		return 1;
	}

	int AddTargetVector(int len1_, double* vec_) {
		std::vector<double> v(len1_);
		for (int i = 0; i < len1_; ++i) {
			v.insert(v.end(), vec_, vec_ + len1_);
		}
		std::vector<std::vector<double>> v2D{ v };
		targets.push_back(v2D);

		return 2;
	}

	int AddTargetVectors(int len1_, int len2_, int len3_, double* vec_) {
		//return 4;
		targets = ConvertNpToVector(len1_, len2_, len3_, vec_);
		return 1;
	}

	void SetBatchSize(int sz) {
		batchSize = sz;
	}

	void SetNumEpochs(int num) {
		numEpochs = num;
	}

	void Train() {
		//std::cout << "\n entering train" << std::endl;
		//std::cout << "batches: " << batchSize << std::endl;
		int numBatches = numEpochs * std::min(inputData.size(), targets.size()) / batchSize;
		int n = 0;
		for (int batch_k = 0; batch_k < numBatches; ++batch_k) {
			for (int i = 0; i < batchSize; ++i) {
				FwdProp(&inputData[n]);
				//std::cout << "finished fwdprop for batch: " << batch_k << " and data element i: " << i << std::endl;
				auto error = CalcError(targets[n]);
				//std::cout << "finished error for batch: " << batch_k << " and data element i: " << i << std::endl;
				BackProp(error);
				//std::cout << "finished backprop for batch: " << batch_k << " and data element i: " << i << std::endl;
				++n;
				n = (n + 1) % inputData.size();
			}
			//std::cout << "updating weights: " << std::endl;
			UpdateWeights();
			//std::cout << "done updating weights: " << std::endl;
		}
		//std::cout << "leaving train" << std::endl;
	}

	std::vector<int> Predict(int len1_, int len2_, int len3_, double* vec_) {
		auto v = ConvertNpToVector(len1_, len2_, len3_, vec_);
		//std::cout << "converted" << std::endl;
		return _Predict(v);
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