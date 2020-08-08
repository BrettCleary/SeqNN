#include <Python.h>
#include <string>
#include <vector>

class Layer
{
protected:
	std::string name;

	//row indexes are previous layer vertice indexes and column indexes are current layer vertice index
	//improves speed of dot product with next layer error vector during backprop
	//std::vector<std::vector<double>> weights;

	std::vector<double> output;

	virtual std::vector<std::vector<double>>* fwdProp(const std::vector<std::vector<char>>& input) = 0;

	virtual std::vector<double> backProp(std::vector<double> errors) = 0;


public:
	PyObject* GetOutput() {
		return output;
	}

	PyObject* GetName() {
		return name;//
	}

	Layer(PyObject* outputSize, PyObject* activationFxn, PyObject* nameIn, PyObject* regMethod) {

	}
};

