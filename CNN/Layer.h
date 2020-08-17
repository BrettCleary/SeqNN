#ifndef CNN_LAYER_H
#define CNN_LAYER_H

#include <string>
#include <vector>
#include <iostream>
//#include "SequentialModel.h"

class SequentialModel;

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
	//current layer [row][col]
	std::vector<std::vector<double>> error;

	//input layer [row][col]
	const std::vector<std::vector<double>>* inputValues;


	//input layer[row][col]
	std::vector<std::vector<double>> backPropError;


	//for use in calculating numerical derivatives in O(W^2) time to verify backprop procedure which runs in O(W) time
	std::vector<std::vector<std::vector<std::vector<double>>>> weightDerNumerical;
	std::vector<std::vector<double>> biasDerNumerical;
	double epsilon = 0.00001;
	double errorLimitWeightDer = 0.01;

	bool usingWeights = true;

	std::vector<std::vector<double>> output;

	double LogSig(double a) {
		return 1 / (1 + exp(-a));
	}

public:

	void ResetWeights() {
		if (!usingWeights)
			return;

		for (int i = 0; i < weights.size(); ++i) {
			for (int j = 0; j < weights[0].size(); ++j) {
				for (int m = 0; m < weights[0][0].size(); ++m) {
					for (int n = 0; n < weights[0][0][0].size(); ++n) {
						weightDer[i][j][m][n] = 0;
						weightDerNumerical[i][j][m][n] = 0;
					}
				}
				biasDer[i][j] = 0;
				biasDerNumerical[i][j] = 0;
			}
		}
	}

	/*void InitializeWeightArrays() {
		if (!usingWeights)
			return;
	}*/

	bool GradientCorrect(SequentialModel* model, int startIndex, int endIndex);

	void UpdateWeights(double step);

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