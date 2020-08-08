#include <Python.h>
#include "Layer.h"
#include <vector>
#include <numpy/ndarraytypes.h>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

class SequentialModel
{

	std::vector<Layer> allLayers;


	void FwdProp() {

	}

	void BackProp() {

	}




public:
	SequentialModel(PyObject* layerList) {
		//convert from pyobject* to list
		//allLayers = layerListConverted
	}

	void Add(PyObject* layer) {
		//convert layer
		allLayers.push_back(layerConverted);
	}

	void Fit(PyObject* trainData, PyObject* trainLabels, PyObject* batchSize, PyObject* numEpochs) {

	}

	std::vector<std::vector<char>> ConvertNumpyToVector2DByte(np::ndarray const & ndArry) {
		int numRows = ndArry.shape(0);
		int numCols = ndArry.shape(1);
		std::vector<std::vector<char>> ret

	}


};

