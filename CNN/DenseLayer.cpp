#include "DenseLayer.h"

void DenseLayer::Initialize(const std::vector<std::vector<double>>& input) {
    numInputRows = input.size();
    numInputCols = input[0].size();

    //init weights to 0
    for (int i = 0; i < numOutputRows; ++i) {
        std::vector<std::vector<std::vector<double>>> row_pool;
        for (int j = 0; j < numOutputCols; ++j) {
            std::vector<std::vector<double>> col_pool;
            for (int k = 0; k < numInputRows; ++k) {
                std::vector<double> row_i(numInputCols, 0);
                col_pool.push_back(row_i);
            }
            row_pool.push_back(col_pool);
        }
        weights.push_back(row_pool);
    }

    //init weight derivatives to 0
    for (int i = 0; i < numOutputRows; ++i) {
        std::vector<std::vector<std::vector<double>>> row_pool;
        std::vector<std::vector<std::vector<double>>> row_poolMom;
        std::vector<std::vector<std::vector<double>>> row_poolNum;
        for (int j = 0; j < numOutputCols; ++j) {
            std::vector<std::vector<double>> col_pool;
            std::vector<std::vector<double>> col_poolMom;
            std::vector<std::vector<double>> col_poolNum;
            for (int k = 0; k < numInputRows; ++k) {
                std::vector<double> row_i(numInputCols, 0);
                col_pool.push_back(row_i);
                std::vector<double> rowMom_i(numInputCols, 0);
                col_poolMom.push_back(rowMom_i);
                std::vector<double> row_iNum(numInputCols, 0);
                col_poolNum.push_back(row_iNum);
            }
            row_pool.push_back(col_pool);
            row_poolMom.push_back(col_poolMom);
            row_poolNum.push_back(col_poolNum);
        }
        weightDer.push_back(row_pool);
        weightDerNumerical.push_back(row_poolNum);
        weightDerMomentum.push_back(row_poolMom);
    }

    //init backPropError to 0
    for (int k = 0; k < numInputRows; ++k) {
        std::vector<double> row_i(numInputCols, 0);
        backPropError.push_back(row_i);
    }

    //init output, bias, and bias der to 0
    for (int i = 0; i < numOutputRows; ++i) {
        std::vector<double> row_i(numOutputCols, 0);
        output.push_back(row_i);
        std::vector<double> bias_i(numOutputCols, 0);
        bias.push_back(bias_i);
        std::vector<double> biasDer_i(numOutputCols, 0);
        biasDer.push_back(biasDer_i);
        std::vector<double> biasDerMom_i(numOutputCols, 0);
        biasDerMomentum.push_back(biasDerMom_i);
        std::vector<double> biasDerNum_i(numOutputCols, 0);
        biasDerNumerical.push_back(biasDerNum_i);
        std::vector<double> error_i(numOutputCols, 0);
        error.push_back(biasDer_i);
    }

    initialized = true;
}

std::vector<std::vector<double>>* DenseLayer::FwdProp(const std::vector<std::vector<double>>& input) {
    if (!initialized) {
        Initialize(input);
    }

    inputValues = &input;

    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            double activation = 0;
            for (int m = 0; m < numInputRows; ++m) {
                for (int n = 0; n < numInputCols; ++n) {
                    activation += input[m][n] * weights[i][j][m][n];
                }
            }
            activation += bias[i][j];
            output[i][j] = LogSig(activation);
        }
    }
    return &output;
}

const std::vector<std::vector<double>>* DenseLayer::BackProp(const std::vector<std::vector<double>>& backPropErrorSum) {
    //calculate errors
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            error[i][j] = output[i][j] * (1 - output[i][j]) * backPropErrorSum[i][j];
        }
    }

    //calculate weight derivatives
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            for (int m = 0; m < numInputRows; ++m) {
                for (int n = 0; n < numInputCols; ++n) {
                    //add weight derivatives until next weight update at end of batch
                    weightDer[i][j][m][n] += error[i][j] * (*inputValues)[m][n];
                }
            }
            biasDer[i][j] += error[i][j];
        }
    }

    //calculate backPropErrorSum for the input layer
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

    ++numPropsSinceLastUpdate;

    return &backPropError;
}