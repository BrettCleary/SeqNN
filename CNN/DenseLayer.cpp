#include "DenseLayer.h"

void DenseLayer::Initialize(const std::vector<std::vector<double>>& input) {
    numInputRows = input.size();
    numInputCols = input[0].size();

    //init weights to 0
    InitalizeWeights(numOutputRows, numOutputCols, numInputRows, numInputCols);

    //init backPropError, error, and output to 0
    InitializeErrorAndOutput();

    initialized = true;
}

std::vector<std::vector<double>>* DenseLayer::FwdProp(const std::vector<std::vector<double>>& input) {
    if (!initialized) {
        Initialize(input);
    }

    inputValues = &input;

    double expSum = 0;

    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            double activation = 0;
            for (int m = 0; m < numInputRows; ++m) {
                for (int n = 0; n < numInputCols; ++n) {
                    activation += input[m][n] * weights[i][j][m][n];
                }
            }
            activation += bias[i][j];
            if (actFxn == ActFxn::logSig)
                output[i][j] = LogSig(activation);
            else if (actFxn == ActFxn::softmax) {
                output[i][j] = exp(activation);
                expSum += output[i][j];
            }
        }
    }
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            output[i][j] /= expSum;
        }
    }

    return &output;
}

const std::vector<std::vector<double>>* DenseLayer::BackProp(const std::vector<std::vector<double>>& backPropErrorSum) {
    //calculate errors
    CalcErrors(backPropErrorSum);

    //calculate weight derivatives
    CalcWeightDers();

    //calculate backPropErrorSum for the input layer
    CalcBackPropErrorSum();
    return &backPropError;
}

void DenseLayer::CalcErrors(const std::vector<std::vector<double>>& backPropErrorSum) {
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            //same for logSig squared loss and crossEntropy loss for softmax
            error[i][j] = output[i][j] * (1 - output[i][j]) * backPropErrorSum[i][j];
        }
    }
}

void DenseLayer::CalcBackPropErrorSum() {
    for (int m = 0; m < numInputRows; ++m) {
        for (int n = 0; n < numInputCols; ++n) {
            double errorSum = 0;
            for (int i = 0; i < numOutputRows; ++i) {
                for (int j = 0; j < numOutputCols; ++j) {
                    errorSum += error[i][j] * weights[i][j][m][n];
                }
            }
            backPropError[m][n] = errorSum;
        }
    }
}

void DenseLayer::CalcWeightDers() {
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            for (int m = 0; m < numInputRows; ++m) {
                for (int n = 0; n < numInputCols; ++n) {
                    //add weight derivatives until next weight update at end of batch
                    double regTerm = 0;
                    if (regType == Regularizer::weightDecay)
                        regTerm = regCoef * weights[i][j][m][n];
                    weightDer[i][j][m][n] += error[i][j] * (*inputValues)[m][n] + regTerm;
                }
            }
            biasDer[i][j] += error[i][j];
        }
    }
}