#ifndef CNN_DENSELAYER_H
#define CNN_DENSELAYER_H
#include "Layer.h"

class DenseLayer :
    public Layer
{
    void Initialize(const std::vector<std::vector<double>>& input);
    void CalcWeightDers();
    void CalcBackPropErrorSum();
    void CalcErrors(const std::vector<std::vector<double>>& backPropErrorSum);

public:
    DenseLayer(double step, int outRows, int outCols, double momentum, int activationFxn, int reg, double regCoefInput) : Layer(step, momentum) {
        numOutputCols = outCols;
        numOutputRows = outRows;

        actFxn = (ActFxn)activationFxn;
        regType = (Regularizer)reg;
        regCoef = regCoefInput;
    }
    
    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;
};
#endif // CNN_DENSELAYER_H