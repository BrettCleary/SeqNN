#include "Layer.h"
#include "SequentialModel.h"

bool Layer::GradientCorrect(SequentialModel* model, int startIndex, int endIndex) {
	if (!usingWeights)
		return true;

	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex) {
		for (int m = 0; m < numOutputRows; ++m) {
			for (int n = 0; n < numOutputCols; ++n) {
				for (int i = 0; i < numInputRows; ++i) {
					for (int j = 0; j < numInputCols; ++j) {
						weights[m][n][i][j] += epsilon;
						double error1 = model->CalcErrorNumerically(dataIndex);
						weights[m][n][i][j] -= 2 * epsilon;
						double error0 = model->CalcErrorNumerically(dataIndex);

						//std::cout << "error1: " << error1 << " error2: " << error0 << std::endl;
						weightDerNumerical[m][n][i][j] += (error1 - error0) / (2 * epsilon);

						weights[m][n][i][j] += epsilon;
					}
				}
				bias[m][n] += epsilon;
				double error1 = model->CalcErrorNumerically(dataIndex);
				bias[m][n] -= 2 * epsilon;
				double error0 = model->CalcErrorNumerically(dataIndex);
				biasDerNumerical[m][n] += (error1 - error0) / (2 * epsilon);
				bias[m][n] += epsilon;
			}
		}
	}
	//std::cout << "updating weights in layer class for startIndex = " << startIndex << " and endIndex = " << endIndex << std::endl;
	for (int m = 0; m < numOutputRows; ++m) {
		for (int n = 0; n < numOutputCols; ++n) {
			for (int i = 0; i < numInputRows; ++i) {
				for (int j = 0; j < numInputCols; ++j) {
					if (abs(weightDerNumerical[m][n][i][j] - weightDer[m][n][i][j]) > errorLimitWeightDer) {
						std::cout << "Numerical weight derivative does not agree with backprop weightDer for mnij: " << m << n << i << j <<
							" Numerical is " << weightDerNumerical[m][n][i][j] << " and backprop der is : " << weightDer[m][n][i][j] << std::endl;
						return false;
					}
				}
			}
			if (abs(biasDerNumerical[m][n] - biasDer[m][n]) > errorLimitWeightDer) {
				std::cout << "Numerical weight derivative for bias does not agree with backprop weightDer for mn: " << m << n <<
					" Numerical is " << biasDerNumerical[m][n] - biasDer[m][n] << " greater than backprop derivative." << std::endl;
				return false;
			}
		}
	}
	return true;
}

void Layer::UpdateWeights(double step) {
	//std::cout << "updating weights in Layer Class" << std::endl;
	if (!usingWeights)
		return;

	if (weights.empty() || weightDer.empty() || bias.empty() || biasDer.empty()) {
		std::cout << "weights, weightDer, bias, or biasDer was empty, so weights were not updated" << std::endl;
		std::cout << "weights.empty() = " << weights.empty() << " weightDer.empty() = " << weightDer.empty() << " bias.empty() = " << bias.empty() << " biasDer.empty() " << biasDer.empty() << std::endl;
		return;
	}

	for (int i = 0; i < numOutputRows; ++i) {
		for (int j = 0; j < numOutputCols; ++j) {
			for (int m = 0; m < numInputRows; ++m) {
				for (int n = 0; n < numInputCols; ++n) {
					weights[i][j][m][n] -= step * weightDer[i][j][m][n];
					/*double rangeLimit = 0.1;
					if (weights[i][j][m][n] < -rangeLimit)
						weights[i][j][m][n] = -rangeLimit;
					else if (weights[i][j][m][n] > rangeLimit)
						weights[i][j][m][n] = rangeLimit;*/
					//std::cout << "i j m n: " << i << j << m << n << " step: " << step << " weightDer[i][j][m][n] = " << weightDer[i][j][m][n] << " weights[i][j][m][n] = " << weights[i][j][m][n] << std::endl;
					weightDer[i][j][m][n] = 0;
				}
			}
			bias[i][j] -= step * biasDer[i][j];
			biasDer[i][j] = 0;
		}
	}
	//std::cout << "finished updating weights in Layer Class" << std::endl;
}