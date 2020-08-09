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

	int numInputRows = -1;
	int numInputCols = -1;
	int numOutputRows = -1;
	int numOutputCols = -1;
	bool initialized = false;

	std::vector<double> output;

	

	double logSig(double a) {
		return 1 / (1 + exp(a));
	}

public:

	virtual std::vector<std::vector<double>>* fwdProp(const std::vector<std::vector<double>>& input) = 0;

	virtual std::vector<std::vector<double>> backProp(std::vector<std::vector<double>> errors) = 0;

	std::string GetName() {
		return name;
	}

	Layer(PyObject* outputSize, PyObject* activationFxn, PyObject* nameIn, PyObject* regMethod) {

	}
};

