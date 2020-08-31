#ifndef CNN_DENSELAYER_H
#define CNN_DENSELAYER_H


#pragma once
#include "Layer.h"
class DenseLayer :
    public Layer
{

    void Initialize(const std::vector<std::vector<double>>& input);

public:

    DenseLayer(double step, int outRows, int outCols, double momentum) : Layer(step, momentum) {
        numOutputCols = outCols;
        numOutputRows = outRows;
    }

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;
};


#endif // CNN_DENSELAYER_H