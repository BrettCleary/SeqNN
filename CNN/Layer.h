#ifndef CNN_LAYER_H
#define CNN_LAYER_H

#include <string>
#include <vector>
#include <iostream>
//#include "SequentialModel.h"

class SequentialModel;

enum class ActFxn {
	logSig,
	softmax
};

enum class Regularizer {
	none,
	weightDecay,
	softWeightSharing
};

class Layer
{
private:
	void CalculateNumericalDerivatives(SequentialModel*, int, int);
	bool CheckNumDerEqualsBackProp();
	void UpdateMainWeights();
	void UpdateSoftWeightSharingParams();

protected:
	std::string name;
	int numInputRows = -1;
	int numInputCols = -1;
	int numOutputRows = -1;
	int numOutputCols = -1;
	bool initialized = false;
	double momentum = 0.9;
	double regCoef = 0;
	ActFxn actFxn = ActFxn::logSig;
	Regularizer regType = Regularizer::none;

	//soft weight sharing
	int numGaussians = 0;
	std::vector<double> gaussianMeans;
	std::vector<double> gaussianStdDevs;
	std::vector<double> gaussianStdDevAuxVars;
	std::vector<double> gaussianMixingCoefs;
	std::vector<double> gaussianMixingCoefsAuxiliaryVars;
	std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> gaussianPosteriors;
	std::vector<double> gaussianMeansDer;
	std::vector<double> gaussianStdDevAuxVarsDer;
	std::vector<double> gaussianMixingCoefsAuxiliaryVarsDer;
	double gausMeanStepSize = 0.1, gausStdDevStepSize = 0.1, gausMixingCoefStepSize = 0.1;

	//first [row][col] is for current layer indexes. second [row][col] is for input
	std::vector<std::vector<std::vector<std::vector<double>>>> weights;
	std::vector<std::vector<std::vector<std::vector<double>>>> weightDer;
	std::vector<std::vector<std::vector<std::vector<double>>>> weightDerMomentum;

	//first [row][col] is for current layer indices to access bias
	std::vector<std::vector<double>> bias;
	std::vector<std::vector<double>> biasDer;
	std::vector<std::vector<double>> biasDerMomentum;

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

	double Gaussian(double w, double u, double o) {
		//missing 1 / (2*pi)^0.5 normalization term since it factors out in soft weight sharing
		return exp(-1 / (2 * pow(o, 2)) * pow((w - u), 2)) / o;
	}

	void InitalizeWeights(int outRows, int outCols, int inRows, int inCols);
	void InitializeErrorAndOutput();

public:
	Layer(double step, double momentumParam) : weightStepSize(step), momentum(momentumParam) {

	}

	Layer(double step, double momentumParam, double gausMeanStepSizeParam, double gausStdDevStepSizeParam, double gausMixingCoefStepSizeParam) :
		weightStepSize(step), momentum(momentumParam), gausMeanStepSize(gausMeanStepSizeParam), gausStdDevStepSize(gausStdDevStepSizeParam), gausMixingCoefStepSize(gausMixingCoefStepSizeParam) {

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