#ifndef CNN_LAYER_H
#define CNN_LAYER_H

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
	int numPropsSinceLastUpdate = 0;

	//first [row][col] is for current layer indexes. second [row][col] is for input
	std::vector<std::vector<std::vector<std::vector<double>>>> weights;
	std::vector<std::vector<std::vector<std::vector<double>>>> weightDer;
	//first [row][col] is for current layer indices to access bias
	std::vector<std::vector<double>> bias;
	std::vector<std::vector<double>> biasDer;


	std::vector<std::vector<double>> output;

	double LogSig(double a) {
		return 1 / (1 + exp(a));
	}

public:

	void UpdateWeights(double step) {
		if (weights.empty() || weightDer.empty() || bias.empty() || biasDer.empty()) {
			return;
		}

		for (int i = 0; i < numOutputRows; ++i) {
			for (int j = 0; j < numOutputCols; ++j) {
				for (int m = 0; m < numInputRows; ++m) {
					for (int n = 0; n < numInputCols; ++n) {
						weights[i][j][m][n] += - step * weightDer[i][j][m][n];
					}
				}
				bias[i][j] += - step * biasDer[i][j];
			}
		}
	}

	const std::vector<std::vector<double>>* GetOutput() {
		return &output;
	}

	virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) = 0;

	virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) = 0;

	std::string GetName() {
		return name;
	}
	/*
	Layer(PyObject* outputSize, PyObject* activationFxn, PyObject* nameIn, PyObject* regMethod) {

	}*/
};

#endif // CNN_LAYER_H