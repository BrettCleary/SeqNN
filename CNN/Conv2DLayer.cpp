#include "Conv2DLayer.h"


std::vector<std::vector<double>>* Conv2DLayer::FwdProp(const std::vector<std::vector<double>>& input) {
    if (!initialized) {
        numInputRows = windowRows;
        numInputCols = windowCols;

        numOutputCols = (input[0].size() - windowCols) / strideCol + 1;
        numOutputRows = (input.size() - windowRows) / strideRow + 1;

        //init weights and weight derivatives to 0
        for (int i = 0; i < numOutputRows; ++i) {
            std::vector<std::vector<std::vector<double>>> row_pool;
            std::vector<std::vector<std::vector<double>>> row_poolDer;
            std::vector<std::vector<std::vector<double>>> row_poolDerNum;
            for (int j = 0; j < numOutputCols; ++j) {
                std::vector<std::vector<double>> col_pool;
                std::vector<std::vector<double>> col_poolDer;
                std::vector<std::vector<double>> col_poolDerNum;
                for (int k = 0; k < windowRows; ++k) {
                    std::vector<double> row_i(windowCols, 0);
                    col_pool.push_back(row_i);
                    std::vector<double> row_iDer(windowCols, 0);
                    col_poolDer.push_back(row_iDer);
                    std::vector<double> row_iDerNum(windowCols, 0);
                    col_poolDerNum.push_back(row_iDerNum);
                }
                row_pool.push_back(col_pool);
                row_poolDer.push_back(col_poolDer);
                row_poolDerNum.push_back(col_poolDerNum);
            }
            weights.push_back(row_pool);
            weightDer.push_back(row_poolDer);
            weightDerNumerical.push_back(row_poolDerNum);
        }

        //init output and bias to 0
        for (int i = 0; i < numOutputRows; ++i) {
            std::vector<double> row_i(numOutputCols, 0);
            output.push_back(row_i);
            std::vector<double> bias_i(numOutputCols, 0);
            bias.push_back(bias_i);
            std::vector<double> biasDer_i(numOutputCols, 0);
            biasDer.push_back(biasDer_i);
            std::vector<double> biasDerNum_i(numOutputCols, 0);
            biasDerNumerical.push_back(biasDerNum_i);
            std::vector<double> error_i(numOutputCols, 0);
            error.push_back(error_i);
        }

        //init backPropError to 0
        for (int k = 0; k < numInputRows; ++k) {
            std::vector<double> row_i(numInputCols, 0);
            backPropError.push_back(row_i);
        }

        initialized = true;
    }
    
    inputValues = &input;

    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            double activation = 0;
            for (int m = 0; m < windowRows; ++m) {
                for (int n = 0; n < windowCols; ++n) {
                    activation += input[i * strideRow + m][j * strideCol + n] * weights[i][j][m][n];
                }
            }
            activation += bias[i][j];
            output[i][j] = LogSig(activation);
        }
    }
    return &output;
}

const std::vector<std::vector<double>>* Conv2DLayer::BackProp(const std::vector<std::vector<double>>& backPropErrorSum) {
    if (backPropErrorSum.size() != numOutputRows) {
        std::cout << "ERROR: Back propagation error rows = " << backPropErrorSum.size() << " does not equal numOutputRows = " << numOutputRows << std::endl;
    }
    if (!backPropErrorSum.empty() && backPropErrorSum[0].size() != numOutputCols) {
        std::cout << "ERROR: Back propagation error rows = " << backPropErrorSum[0].size() << " does not equal numOutputCols = " << numOutputCols << std::endl;
    }

    //calculate errors
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            error[i][j] = output[i][j] * (1 - output[i][j]) * backPropErrorSum[i][j];}
    }

    //calculate weight derivatives
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            for (int m = 0; m < windowRows; ++m) {
                for (int n = 0; n < windowCols; ++n) {
                    weightDer[i][j][m][n] += error[i][j] * (*inputValues)[i * strideRow + m][j * strideCol + n];
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