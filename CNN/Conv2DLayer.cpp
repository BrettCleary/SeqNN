#include "Conv2DLayer.h"

void Conv2DLayer::Initialize(const int inputRows, const int inputCols) {
    numInputRows = windowRows;
    numInputCols = windowCols;

    numOutputCols = (inputCols - windowCols) / strideCol + 1;
    numOutputRows = (inputRows - windowRows) / strideRow + 1;

    InitalizeWeights(numOutputRows, numOutputCols, windowRows, windowCols);

    InitializeErrorAndOutput();

    //init gaussian mixture vectors for soft weight sharing regularizing
    if (regType == Regularizer::softWeightSharing) {
        InitializeSoftWeightSharingVectors();
    }

    initialized = true;
}

void Conv2DLayer::InitializeSoftWeightSharingVectors() {
    gaussianMeans.insert(gaussianMeans.begin(), numGaussians, 0);
    double avgMixingCoef = 1 / numGaussians;
    gaussianMixingCoefs.insert(gaussianMixingCoefs.begin(), numGaussians, avgMixingCoef);
    gaussianMixingCoefsAuxiliaryVars.insert(gaussianMixingCoefsAuxiliaryVars.begin(), numGaussians, 0);
    gaussianStdDevs.insert(gaussianStdDevs.begin(), numGaussians, 0.1);

    gaussianMeansDer.insert(gaussianMeansDer.begin(), numGaussians, 0);
    gaussianStdDevsDer.insert(gaussianStdDevsDer.begin(), numGaussians, 0);
    gaussianMixingCoefsAuxiliaryVarsDer.insert(gaussianMixingCoefsAuxiliaryVarsDer.begin(), numGaussians, 0);

    for (int gausIndex = 0; gausIndex < numGaussians; ++gausIndex) {
        std::vector<std::vector<std::vector<std::vector<double>>>> gausRow;
        for (int i = 0; i < numOutputRows; ++i) {
            std::vector<std::vector<std::vector<double>>> row_i;
            for (int j = 0; j < numOutputCols; ++j) {
                std::vector<std::vector<double>> col_j;
                for (int k = 0; k < windowRows; ++k) {
                    std::vector<double> row_k(windowCols, 0);
                    col_j.push_back(row_k);
                }
                row_i.push_back(col_j);
            }
            gausRow.push_back(row_i);
        }
        gaussianPosteriors.push_back(gausRow);
    }
}

std::vector<std::vector<double>>* Conv2DLayer::FwdProp(const std::vector<std::vector<double>>& input) {
    if (!initialized) {
        Initialize(input.size(), input[0].size());
    }

    //std::cout << "entering denselayer fwdprop" << std::endl;
    inputValues = &input;
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            double activation = 0;
            for (int m = 0; m < windowRows; ++m) {
                for (int n = 0; n < windowCols; ++n) {
                    activation += input[i * strideRow + m][j * strideCol + n] * weights[i][j][m][n];
                }
            }
            activation += bias[i][j];
            output[i][j] = LogSig(activation);
        }
    }
    //std::cout << "leaving denselayer fwdprop" << std::endl;
    return &output;
}

const std::vector<std::vector<double>>* Conv2DLayer::BackProp(const std::vector<std::vector<double>>& backPropErrorSum) {
    if (backPropErrorSum.size() != numOutputRows) {
        std::cout << "ERROR: Back propagation error rows = " << backPropErrorSum.size() << " does not equal numOutputRows = " << numOutputRows << std::endl;
    }
    if (!backPropErrorSum.empty() && backPropErrorSum[0].size() != numOutputCols) {
        std::cout << "ERROR: Back propagation error rows = " << backPropErrorSum[0].size() << " does not equal numOutputCols = " << numOutputCols << std::endl;
    }

    //calculate errors
    CalcErrors(backPropErrorSum);

    if (regType == Regularizer::softWeightSharing) {
        //calculate mixing coefficients from auxiliary variables
        CalcMixingCoefs();
    }

    //calculate weight derivatives
    CalcWeightDers();

    if (regType == Regularizer::softWeightSharing) {
        //calculate gaussian prior derivatives
        CalcSoftWeightSharingDers();
    }

    //calculate backPropErrorSum for the input layer
    CalcBackPropErrorSum();
    return &backPropError;
}

void Conv2DLayer::CalcErrors(const std::vector<std::vector<double>>& backPropErrorSum) {
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            error[i][j] = output[i][j] * (1 - output[i][j]) * backPropErrorSum[i][j];
        }
    }
}

void Conv2DLayer::CalcBackPropErrorSum() {
    for (int m = 0; m < numInputRows; ++m) {
        for (int n = 0; n < numInputCols; ++n) {
            double errorSum = 0;
            for (int i = 0; i < numOutputRows; ++i) {
                for (int j = 0; j < numOutputCols; ++j) {
                    errorSum += error[i][j] * weights[i][j][m][n];
                }
            }
            backPropError[m][n] = errorSum;
        }
    }
}

void Conv2DLayer::CalcSoftWeightSharingDers() {
    for (int gausIndex = 0; gausIndex < numGaussians; ++gausIndex) {
        double gausMeanDer = 0;
        double gausStdDevDer = 0;
        double gausPriorAuxDer = 0;
        for (int i = 0; i < numOutputRows; ++i) {
            for (int j = 0; j < numOutputCols; ++j) {
                for (int m = 0; m < windowRows; ++m) {
                    for (int n = 0; n < windowCols; ++n) {
                        gausMeanDer += gaussianPosteriors[gausIndex][i][j][m][n] * (gaussianMeans[gausIndex] - weights[i][j][m][n]) / pow(gaussianStdDevs[gausIndex], 2);
                        gausStdDevDer += gaussianPosteriors[gausIndex][i][j][m][n] * (1 / gaussianStdDevs[gausIndex] - pow(weights[i][j][m][n] - gaussianMeans[gausIndex], 2) / pow(gaussianStdDevs[gausIndex], 3));
                        gausPriorAuxDer += gaussianMixingCoefs[gausIndex] - gaussianPosteriors[gausIndex][i][j][m][n];
                    }
                }
            }
        }
        gaussianMeansDer[gausIndex] = regCoef * gausMeanDer;
        gaussianStdDevsDer[gausIndex] = regCoef * gausStdDevDer;
        gaussianMixingCoefsAuxiliaryVarsDer[gausIndex] = regCoef * gausPriorAuxDer;
    }
}

void Conv2DLayer::CalcWeightDers() {
    for (int i = 0; i < numOutputRows; ++i) {
        for (int j = 0; j < numOutputCols; ++j) {
            for (int m = 0; m < windowRows; ++m) {
                for (int n = 0; n < windowCols; ++n) {
                    double regTerm = 0;
                    //calculate regularizer term
                    if (regType == Regularizer::softWeightSharing) {
                        regTerm = CalcSoftWeightSharingRegterm(i, j, m, n);
                    }
                    weightDer[i][j][m][n] += error[i][j] * (*inputValues)[i * strideRow + m][j * strideCol + n] + regTerm;
                }
            }
            biasDer[i][j] += error[i][j];
        }
    }
}

void Conv2DLayer::CalcMixingCoefs() {
    //calculate mixing coefficients from auxiliary variables
    double expSum = 0;
    for (int i = 0; i < numGaussians; ++i) {
        gaussianMixingCoefs[i] = exp(gaussianMixingCoefsAuxiliaryVars[i]);
        expSum += gaussianMixingCoefs[i];
    }
    for (int i = 0; i < numGaussians; ++i) {
        gaussianMixingCoefs[i] /= expSum;
    }
}

double Conv2DLayer::CalcSoftWeightSharingRegterm(int i, int j, int m, int n) {
    double regTerm = 0;
    double posteriorSum = 0;
    for (int gausIndex = 0; gausIndex < numGaussians; ++gausIndex) {
        gaussianPosteriors[gausIndex][i][j][m][n] = gaussianMixingCoefs[gausIndex] * Gaussian(weights[i][j][m][n], gaussianMeans[gausIndex], gaussianStdDevs[gausIndex]);
        posteriorSum += gaussianPosteriors[gausIndex][i][j][m][n];
        regTerm += gaussianPosteriors[gausIndex][i][j][m][n] * (weights[i][j][m][n] - gaussianMeans[gausIndex]) / (pow(gaussianStdDevs[gausIndex], 2));
    }

    for (int gausIndex = 0; gausIndex < numGaussians; ++gausIndex) {
        gaussianPosteriors[gausIndex][i][j][m][n] /= posteriorSum;
    }

    regTerm = regCoef * regTerm / posteriorSum;
    return regTerm;
}