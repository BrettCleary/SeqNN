#ifndef CNN_CONV2DLAYER_H
#define CNN_CONV2DLAYER_HC
#include "Layer.h"


class testClass {

};


class Conv2DLayer :
    public Layer
{
    int windowRows = 4;
    int windowCols = 4;
    int strideRow = 4;
    int strideCol = 4;
    int padding = 0;
    
public:

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;

};

#endif // CNN_CONV2DLAYER_HC