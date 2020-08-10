#pragma once
#include "Layer.h"
#include <algorithm>

class Pool2DLayer :
    public Layer
{
    int poolCols = 2;
    int poolRows = 2;

    bool isMax = true;

    //first two dim are [row][col] of input 2D matrix, each element is an adjacency list of a pair of indexes to the output matrix {row, col}
    std::vector<std::vector<std::vector<std::vector<char>>>> adjList;

    //input layer[row][col]
    std::vector<std::vector<double>> backPropError;

    double MaxPool(int i, int j, const std::vector<std::vector<double>>& input) {
        double maxAct = DBL_MIN;
        for (int m = 0; m < poolRows; ++m) {
            for (int n = 0; n < poolCols; ++n) {
                maxAct = std::max(maxAct, input[numOutputRows * poolRows + m][numOutputCols * poolCols + n]);
            }
        }
        return maxAct;
    }

    double MinPool(int i, int j, const std::vector<std::vector<double>>& input) {
        double minAct = DBL_MAX;
        for (int m = 0; m < poolRows; ++m) {
            for (int n = 0; n < poolCols; ++n) {
                minAct = std::min(minAct, input[numOutputRows * poolRows + m][numOutputCols * poolCols + n]);
            }
        }
        return minAct;
    }

public:

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override {
        if (!initialized) {
            numInputRows = input.size();
            numInputCols = input[0].size();

            numOutputRows = numInputRows / poolRows;
            numOutputCols = numInputCols / poolCols;

            for (int i = 0; i < numOutputRows; ++i) {
                std::vector<double> row_i(numOutputCols, 0);
                output.push_back(row_i);
            }
            initialized = true;
        }

        for (int i = 0; i < numOutputRows; ++i) {
            for (int j = 0; j < numOutputCols; ++j) {
                if (isMax) {
                    output[i][j] = MaxPool(i, j, input);
                }
                else {
                    output[i][j] = MinPool(i, j, input);
                }
            }
        }
        return &output;
    }

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override {

        //error = backPropErrorSum since da/dz = 1 for pool layer

        //no weights to adjust

        //calculate backPropErrorSum for the input layer
        //this is done by summing the errors over the adjacency list of the input layer since dz/da = 1 for pool layer argmax/argmin and dz/da = 0 if not in the adj list
        for (int m = 0; m < numInputRows; ++m) {
            for (int n = 0; n < numInputCols; ++n) {
                double errorSum = 0;
                for (auto elementVector : adjList[m][n]) {
                    errorSum += backPropErrorSum[elementVector[0]][elementVector[1]];
                }
                backPropError[m][n] = errorSum;
            }
        }
        return &backPropError;
    }
};

