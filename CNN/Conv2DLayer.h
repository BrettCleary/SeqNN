#ifndef CNN_CONV2DLAYER_H
#define CNN_CONV2DLAYER_HC
#include "Layer.h"

class Conv2DLayer :
    public Layer
{
    int windowRows = 2;
    int windowCols = 2;
    int strideRow = 2;
    int strideCol = 2;
    int padding = 0;

    void Initialize(const int, const int);
    double CalcSoftWeightSharingRegterm(int i, int j, int m, int n);
    void InitializeSoftWeightSharingVectors();
    void CalcMixingCoefs();
    void CalcWeightDers();
    void CalcSoftWeightSharingDers();
    void CalcBackPropErrorSum();
    void CalcErrors(const std::vector<std::vector<double>>& backPropErrorSum);

public:
    Conv2DLayer(int winRows, int winCols, int strideRowInput, int strideColInput, int paddingInput, double step, double momentum,
        int reg, double regCoefInput, int numGaussiansInput, double meanStepSize, double stdDevStepSize, double mixingStepSize) : Layer(step, momentum, meanStepSize, stdDevStepSize, mixingStepSize) {
        windowRows = winRows;
        windowCols = winCols;
        strideRow = strideRowInput;
        strideCol = strideColInput;
        padding = paddingInput;

        regType = (Regularizer)reg;
        regCoef = regCoefInput;
        numGaussians = numGaussiansInput;
    }

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;
};

#endif // CNN_CONV2DLAYER_HC