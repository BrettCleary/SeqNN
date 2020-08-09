#pragma once
#include "Layer.h"
#include <algorithm>

class Pool2DLayer :
    public Layer
{
    std::vector<std::vector<double>> output;
    int poolCols = 2;
    int poolRows = 2;

    bool isMax = true;

    double maxPool(int i, int j, const std::vector<std::vector<double>>& input) {
        double maxAct = DBL_MIN;
        for (int m = 0; m < poolRows; ++m) {
            for (int n = 0; n < poolCols; ++n) {
                maxAct = std::max(maxAct, input[numOutputRows * poolRows + m][numOutputCols * poolCols + n]);
            }
        }
        return maxAct;
    }

    double minPool(int i, int j, const std::vector<std::vector<double>>& input) {
        double minAct = DBL_MAX;
        for (int m = 0; m < poolRows; ++m) {
            for (int n = 0; n < poolCols; ++n) {
                minAct = std::min(minAct, input[numOutputRows * poolRows + m][numOutputCols * poolCols + n]);
            }
        }
        return minAct;
    }

public:

    virtual std::vector<std::vector<double>>* fwdProp(const std::vector<std::vector<double>>& input) override {
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
                    output[i][j] = maxPool(i, j, input);
                }
                else {
                    output[i][j] = minPool(i, j, input);
                }
            }
        }
        return &output;
    }

    virtual std::vector<double> backProp(std::vector<double> errors) override {

    }
};

