#include "Pool2DLayer.h"

double Pool2DLayer::MaxPool(int i, int j, const std::vector<std::vector<double>>& input) {
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
    adjList[i * poolRows + mMax][j * poolCols + nMax].push_back({ i, j });
    return maxAct;
}

double Pool2DLayer::MinPool(int i, int j, const std::vector<std::vector<double>>& input) {
    double minAct = DBL_MAX;
    int mMin = -1;
    int nMin = -1;
    for (int m = 0; m < poolRows; ++m) {
        for (int n = 0; n < poolCols; ++n) {
            minAct = std::min(minAct, input[i * poolRows + m][j * poolCols + n]);
            mMin = m;
            nMin = n;
        }
    }
    adjList[i * poolRows + mMin][j * poolCols + nMin].push_back({ i, j });
    return minAct;
}

std::vector<std::vector<double>>* Pool2DLayer::FwdProp(const std::vector<std::vector<double>>& input) {
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
    ClearAdjList();

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

const std::vector<std::vector<double>>* Pool2DLayer::BackProp(const std::vector<std::vector<double>>& backPropErrorSum) {

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