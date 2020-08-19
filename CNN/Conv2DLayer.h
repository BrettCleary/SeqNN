#ifndef CNN_CONV2DLAYER_H
#define CNN_CONV2DLAYER_HC
#include "Layer.h"


class testClass {

};


class Conv2DLayer :
    public Layer
{
    int windowRows = 2;
    int windowCols = 2;
    int strideRow = 2;
    int strideCol = 2;
    int padding = 0;
    
public:

    Conv2DLayer(int winRows, int winCols, int strideRowInput, int strideColInput, int paddingInput) : Layer() {
        windowRows = winRows;
        windowCols = winCols;
        strideRow = strideRowInput;
        strideCol = strideColInput;
        padding = paddingInput;
        //std::cout << "conv2dLayer initialized" << std::endl;
    }

    virtual std::vector<std::vector<double>>* FwdProp(const std::vector<std::vector<double>>& input) override;

    virtual const std::vector<std::vector<double>>* BackProp(const std::vector<std::vector<double>>& backPropErrorSum) override;

};

#endif // CNN_CONV2DLAYER_HC