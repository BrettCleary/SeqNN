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

	double weightStepSize;

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
	bool displayWeights = false;

	Layer(double step) : weightStepSize(step) {

	}

	void ResetWeights();

	bool GradientCorrect(SequentialModel* model, int startIndex, int endIndex);

	void UpdateWeights();

	const std::vector<std::vector<double>>* GetOutput() {
		return &output;
	}

	virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) = 0;

	virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) = 0;

	std::string GetName() {
		return name;
	}
};

#endif // CNN_LAYER_H