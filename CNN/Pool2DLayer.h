#ifndef CNN_POOL2DLAYER_H
#define CNN_POOL2DLAYER_H

#pragma once
#include "Layer.h"
#include <algorithm>

class Pool2DLayer :
    public Layer
{
    int poolCols = 2;
    int poolRows = 1;

    bool isMax = false;

    //first two dim are [row][col] of input 2D matrix, each element is an adjacency list of a pair of indexes to the output matrix {row, col}
    std::vector<std::vector<std::vector<std::vector<int>>>> adjList;

    double MaxPool(int i, int j, const std::vector<std::vector<double>>& input) {
        double maxAct = DBL_MIN;
        int mMax = -1;
        int nMax = -1;
        for (int m = 0; m < poolRows; ++m) {
            for (int n = 0; n < poolCols; ++n) {
                if (input[i * poolRows + m][j * poolCols + n] > maxAct) {
                    maxAct = input[i * poolRows + m][j * poolCols + n];
                    mMax = m;
                    nMax = n;
                }
            }
        }

        adjList[mMax][nMax].push_back({ i, j });
        return maxAct;
    }

    double MinPool(int i, int j, const std::vector<std::vector<double>>& input) {
        double minAct = DBL_MAX;
        for (int m = 0; m < poolRows; ++m) {
            for (int n = 0; n < poolCols; ++n) {
                minAct = std::min(minAct, input[i * poolRows + m][j * poolCols + n]);
            }
        }
        return minAct;
    }

    void ClearAdjList() {
        for (int i = 0; i < numInputRows; ++i) {
            for (int j = 0; j < numInputCols; ++j) {
                adjList[i][j].clear();
            }
        }
    }

public:

    Pool2DLayer(bool isMaxPool, int poolColsInput, int poolRowsInput) : Layer(0) {
        isMax = isMaxPool;
        poolCols = poolColsInput;
        poolRows = poolRowsInput;
    }

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override {
        if (!initialized) {
            numInputRows = input.size();
            numInputCols = input[0].size();

            numOutputRows = numInputRows / poolRows;
            numOutputCols = numInputCols / poolCols;

            //initializing output
            for (int i = 0; i < numOutputRows; ++i) {
                std::vector<double> row_i(numOutputCols, 0);
                output.push_back(row_i);
            }

            //initializing adjList
            for (int i = 0; i < numInputRows; ++i) {
                std::vector<std::vector<std::vector<int>>> row_pool;
                for (int j = 0; j < numInputCols; ++j) {
                    std::vector<std::vector<int>> col_pool;
                    row_pool.push_back(col_pool);
                }
                adjList.push_back(row_pool);
            }

            //init backPropError to 0
            for (int k = 0; k < numInputRows; ++k) {
                std::vector<double> row_i(numInputCols, 0);
                backPropError.push_back(row_i);
            }

            usingWeights = false;
            initialized = true;
        }

        inputValues = &input;
        //std::cout << "finished initializing pool2dlayer" << std::endl;
        ClearAdjList();
        //std::cout << " numInputRows: " << numInputRows << " numInputCols: " << numInputCols << " numOutputRows: " << numOutputRows << " numOutputCols: " << numOutputCols << std::endl;

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
        //std::cout << "exiting fwd prop pool2dlayer" << std::endl;

        return &output;
    }

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override {

        //error = backPropErrorSum since da/dz = 1 for pool layer

        //no weights to adjust

        //calculate backPropErrorSum for the input layer
        //this is done by summing the errors over the adjacency list of the input layer since dz/da = 1 for pool layer argmax/argmin and dz/da = 0 if not in the adj list
        //std::cout << "backPropErrorSum in pool2dLayer num Rows: " << backPropErrorSum.size() << " and cols: " << backPropErrorSum[0].size() << std::endl;
        for (int m = 0; m < numInputRows; ++m) {
            for (int n = 0; n < numInputCols; ++n) {
                double errorSum = 0;
                for (auto elementVector : adjList[m][n]) {
                    //std::cout << "elementVector pool backprop: " << elementVector[0] << " " << elementVector[1] << std::endl;
                    errorSum += backPropErrorSum[elementVector[0]][elementVector[1]];
                }
                backPropError[m][n] = errorSum;
               // std::cout << "finished m: " << m << " and n: " << n << std::endl;
            }
        }
        //std::cout << "leaving pool2dlayer backprop" << std::endl;
        return &backPropError;
    }
};

#endif // CNN_POOL2DLAYER_H