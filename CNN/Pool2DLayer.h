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

    double MaxPool(int i, int j, const std::vector<std::vector<double>>& input);

    double MinPool(int i, int j, const std::vector<std::vector<double>>& input);

    void ClearAdjList() {
        for (int i = 0; i < numInputRows; ++i) {
            for (int j = 0; j < numInputCols; ++j) {
                adjList[i][j].clear();
            }
        }
    }

    void Initialize(const std::vector<std::vector<double>>& input);

public:

    Pool2DLayer(bool isMaxPool, int poolColsInput, int poolRowsInput) : Layer(0, 0.9) {
        isMax = isMaxPool;
        poolCols = poolColsInput;
        poolRows = poolRowsInput;
    }

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;
};
#endif // CNN_POOL2DLAYER_H