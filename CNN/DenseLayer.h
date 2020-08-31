#ifndef CNN_DENSELAYER_H
#define CNN_DENSELAYER_H
#include "Layer.h"

enum ActFxn {
    logSig,
    softmax
};

class DenseLayer :
    public Layer
{
    ActFxn actFxn = ActFxn::logSig;
    void Initialize(const std::vector<std::vector<double>>& input);

public:

    DenseLayer(double step, int outRows, int outCols, double momentum, int activationFxn) : Layer(step, momentum) {
        numOutputCols = outCols;
        numOutputRows = outRows;
        
        if (activationFxn == 0)
            actFxn = ActFxn::logSig;
        else if (activationFxn == 1)
            actFxn = ActFxn::softmax;
    }

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;
};


#endif // CNN_DENSELAYER_H