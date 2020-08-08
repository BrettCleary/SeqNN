#include "Layer.h"



class Conv2DLayer :
    public Layer
{
    //first [row][col] is for pool layer indexes. second [row][col] is for input
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    bool initialized = false;
    int windowRows = 2;
    int windowCols = 2;
    std::vector<std::vector<double>> output;

    int numInputRows = -1;
    int numInputCols = -1;
    int numOutputRows = -1;
    int numOutputCols = -1;


public:


    virtual std::vector<std::vector<double>>* fwdProp(const std::vector<std::vector<char>>& input) override {
        if (!initialized) {
            numInputRows = input.size();
            numInputCols = input[0].size();
            numPoolRows = numInputRows / poolRows;
            numPoolCols = numInputCols / poolCols;
            for (int i = 0; i < numPoolRows; ++i) {
                std::vector<std::vector<std::vector<double>>> row_pool;
                for (int j = 0; j < numPoolCols; ++j) {
                    std::vector<std::vector<double>> col_pool;
                    for (int k = 0; k < numInputRows; ++k) {
                        std::vector<double> row_i(numInputCols, 0);
                        col_pool.push_back(row_i);
                    }
                    row_pool.push_back(col_pool);
                }
                weights.push_back(row_pool);
            }

            for (int i = 0; i < numPoolRows; ++i) {
                std::vector<double> row_i(numPoolCols, 0);
                output.push_back(row_i);
            }


            initialized = true;
        }


        
        for (int i = 0; i < numPoolRows; ++i) {

            for (int j = 0; j < numPoolCols; ++j) {
                double activation = 0;
                for (int m = 0; m < numInputRows; ++m) {

                    for (int n = 0; n < numInputCols; ++n) {
                        activation += input[m][n] * weights[i][j][m][n];


                    }

                }

            }

        }
    



        return &output;
    }


    virtual std::vector<double> backProp(std::vector<double> errors) override {

    }

};

