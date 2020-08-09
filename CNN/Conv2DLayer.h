#include "Layer.h"



class Conv2DLayer :
    public Layer
{
    //first [row][col] is for current layer indexes. second [row][col] is for input
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    //first [row][col] is for current layer indices to access bias
    std::vector<std::vector<double>> bias;

    std::vector<std::vector<double>> output;

    int windowRows = 2;
    int windowCols = 2;
    int strideRow = 2;
    int strideCol = 2;
    int padding = 0;
    
public:

    virtual std::vector<std::vector<double>>* fwdProp(const std::vector<std::vector<double>>& input) override {
        if (!initialized) {
            numInputRows = input.size();
            numInputCols = input[0].size();

            numOutputCols = (numInputCols - windowCols) / (strideCol + 1);
            numOutputRows = (numInputRows - windowRows) / (strideRow + 1);

            for (int i = 0; i < numOutputRows; ++i) {
                std::vector<std::vector<std::vector<double>>> row_pool;
                for (int j = 0; j < numOutputCols; ++j) {
                    std::vector<std::vector<double>> col_pool;
                    for (int k = 0; k < windowRows; ++k) {
                        std::vector<double> row_i(windowCols, 0);
                        col_pool.push_back(row_i);
                    }
                    row_pool.push_back(col_pool);
                }
                weights.push_back(row_pool);
            }

            for (int i = 0; i < numOutputRows; ++i) {
                std::vector<double> row_i(numOutputCols, 0);
                output.push_back(row_i);
                std::vector<double> bias_i(numOutputCols, 0);
                bias.push_back(bias_i);
            }

            initialized = true;
        }

        for (int i = 0; i < numOutputRows; ++i) {
            for (int j = 0; j < numOutputCols; ++j) {
                double activation = 0;
                for (int m = 0; m < windowRows; ++m) {
                    for (int n = 0; n < windowCols; ++n) {
                        activation += input[numOutputRows*windowRows + m][numOutputCols*windowCols + n] * weights[i][j][m][n];
                    }
                }
                activation += bias[i][j];
                output[i][j] = logSig(activation);
            }
        }
    
        return &output;
    }

    virtual std::vector<double> backProp(std::vector<double> errors) override {

    }

};

