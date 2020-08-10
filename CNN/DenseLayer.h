#pragma once
#include "Layer.h"
class DenseLayer :
    public Layer
{
    //first 2 dim are current layer output [row][col], next 2 are input [row][col] 
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    std::vector<std::vector<std::vector<std::vector<double>>>> weightDer;
    //first [row][col] is for current layer indices to access bias
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<double>> biasDer;
    //current layer [row][col]
    std::vector<std::vector<double>> error;

    //input layer[row][col]
    std::vector<std::vector<double>> backPropError;

public:

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override {
        if (!initialized) {
            numOutputRows = 1;
            numOutputCols = 10;

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
                for (int j = 0; j < numOutputCols; ++j) {
                    std::vector<std::vector<double>> col_pool;
                    for (int k = 0; k < numInputRows; ++k) {
                        std::vector<double> row_i(numInputCols, 0);
                        col_pool.push_back(row_i);
                    }
                    row_pool.push_back(col_pool);
                }
                weightDer.push_back(row_pool);
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
            }

            initialized = true;
        }

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

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override {

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
                        weightDer[i][j][m][n] += error[i][j] * weights[i][j][m][n];
                    }
                }
                biasDer[i][j] = error[i][j];
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
};

