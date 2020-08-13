#ifndef CNN_DENSELAYER_H
#define CNN_DENSELAYER_H


#pragma once
#include "Layer.h"
class DenseLayer :
    public Layer
{
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

            //init weight derivatives to 1. if 0, then weights will be stuck at 0
            for (int i = 0; i < numOutputRows; ++i) {
                std::vector<std::vector<std::vector<double>>> row_pool;
                for (int j = 0; j < numOutputCols; ++j) {
                    std::vector<std::vector<double>> col_pool;
                    for (int k = 0; k < numInputRows; ++k) {
                        std::vector<double> row_i(numInputCols, 1);
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
                std::vector<double> error_i(numOutputCols, 0);
                error.push_back(biasDer_i);
            }

            initialized = true;
        }

        //std::cout << "finished initializing denselayer" << std::endl;

        //std::cout << " numInputRows: " << numInputRows << " numInputCols: " << numInputCols << " numOutputRows: " << numOutputRows << " numOutputCols: " << numOutputCols << std::endl;

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
        /*std::cout << "backprop dense layer" << std::endl;
        for (auto vec : backPropErrorSum) {
            for (auto element : vec) {
                std::cout << element << std::endl;
            }
        }*/
        //calculate errors
        for (int i = 0; i < numOutputRows; ++i) {
            for (int j = 0; j < numOutputCols; ++j) {
                error[i][j] = output[i][j] * (1 - output[i][j]) * backPropErrorSum[i][j];
                //std::cout << "error for i: " << i << " j: " << j << " is " << error[i][j] << " backproperrorsum: " << backPropErrorSum[i][j] << " output: " << output[i][j] << std::endl;
            }
        }
        //std::cout << "calculated errors for backprop" << std::endl;

        //calculate weight derivatives
        for (int i = 0; i < numOutputRows; ++i) {
            for (int j = 0; j < numOutputCols; ++j) {
                for (int m = 0; m < numInputRows; ++m) {
                    for (int n = 0; n < numInputCols; ++n) {
                        weightDer[i][j][m][n] += error[i][j] * weights[i][j][m][n];
                        //std::cout << "error for i: " << i << " j: " << j << " is " << error[i][j] << " weights: " << weights[i][j][m][n] << " weightDer: " << weightDer[i][j][m][n] << std::endl;
                    }
                }
                biasDer[i][j] = error[i][j];
            }
        }
        //std::cout << "calculated weight derivatives for backprop" << std::endl;

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
       // std::cout << "calculated backproperrorsum for backprop" << std::endl;

        ++numPropsSinceLastUpdate;

        return &backPropError;
    }
};


#endif // CNN_DENSELAYER_H