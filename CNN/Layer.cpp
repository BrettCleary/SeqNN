#include "Layer.h"
#include "SequentialModel.h"

bool Layer::GradientCorrect(SequentialModel* model, int startIndex, int endIndex) {
	if (!usingWeights)
		return true;

	CalculateNumericalDerivatives(model, startIndex, endIndex);

	return CheckNumDerEqualsBackProp();
}

bool Layer::CheckNumDerEqualsBackProp() {
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

void Layer::CalculateNumericalDerivatives(SequentialModel* model, int startIndex, int endIndex) {
	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex) {
		for (int m = 0; m < numOutputRows; ++m) {
			for (int n = 0; n < numOutputCols; ++n) {
				for (int i = 0; i < numInputRows; ++i) {
					for (int j = 0; j < numInputCols; ++j) {
						weights[m][n][i][j] += epsilon;
						double error1 = model->CalcErrorNumerically(dataIndex);
						weights[m][n][i][j] -= 2 * epsilon;
						double error0 = model->CalcErrorNumerically(dataIndex);
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
}

void Layer::UpdateWeights() {
	if (!usingWeights)
		return;

	if (weights.empty() || weightDer.empty() || bias.empty() || biasDer.empty()) {
		std::cout << "weights, weightDer, bias, or biasDer was empty, so weights were not updated" << std::endl;
		std::cout << "weights.empty() = " << weights.empty() << " weightDer.empty() = " << weightDer.empty() << " bias.empty() = " << bias.empty() << " biasDer.empty() " << biasDer.empty() << std::endl;
		return;
	}

	//updating weights
	UpdateMainWeights();

	//updating hyperparams for soft weight sharing
	if (regType == Regularizer::softWeightSharing) {
		UpdateSoftWeightSharingParams();
	}
}

void Layer::UpdateSoftWeightSharingParams() {
	for (int i = 0; i < numGaussians; ++i) {
		gaussianMeans[i] -= gausMeanStepSize * gaussianMeansDer[i];
		gaussianStdDevs[i] -= gausStdDevStepSize * gaussianStdDevsDer[i];
		gaussianMixingCoefsAuxiliaryVars[i] -= gausMixingCoefStepSize * gaussianMixingCoefsAuxiliaryVarsDer[i];
	}
}

void Layer::UpdateMainWeights() {
	for (int i = 0; i < numOutputRows; ++i) {
		for (int j = 0; j < numOutputCols; ++j) {
			for (int m = 0; m < numInputRows; ++m) {
				for (int n = 0; n < numInputCols; ++n) {
					weightDerMomentum[i][j][m][n] = weightDerMomentum[i][j][m][n] * momentum + (1 - momentum) * weightDer[i][j][m][n];
					weights[i][j][m][n] -= weightStepSize * weightDerMomentum[i][j][m][n];
					weightDer[i][j][m][n] = 0;
				}
			}
			biasDerMomentum[i][j] = biasDerMomentum[i][j] * momentum + (1 - momentum) * biasDer[i][j];
			bias[i][j] -= weightStepSize * biasDerMomentum[i][j];
			biasDer[i][j] = 0;
		}
	}
}

void Layer::ResetWeights() {
	if (!usingWeights)
		return;

	for (int i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights[0].size(); ++j) {
			for (int m = 0; m < weights[0][0].size(); ++m) {
				for (int n = 0; n < weights[0][0][0].size(); ++n) {
					weightDer[i][j][m][n] = 0;
					weightDerNumerical[i][j][m][n] = 0;
				}
			}
			biasDer[i][j] = 0;
			biasDerNumerical[i][j] = 0;
		}
	}
}

void Layer::InitalizeWeights(int outRows, int outCols, int inRows, int inCols) {
	//init weights and weight derivatives to 0
	for (int i = 0; i < outRows; ++i) {
		std::vector<std::vector<std::vector<double>>> row_pool;
		std::vector<std::vector<std::vector<double>>> row_poolDer;
		std::vector<std::vector<std::vector<double>>> row_poolMomentumDer;
		std::vector<std::vector<std::vector<double>>> row_poolDerNum;
		for (int j = 0; j < outCols; ++j) {
			std::vector<std::vector<double>> col_pool;
			std::vector<std::vector<double>> col_poolDer;
			std::vector<std::vector<double>> col_poolMomentumDer;
			std::vector<std::vector<double>> col_poolDerNum;
			for (int k = 0; k < inRows; ++k) {
				std::vector<double> row_i(inCols, 0);
				col_pool.push_back(row_i);
				std::vector<double> row_iDer(inCols, 0);
				col_poolDer.push_back(row_iDer);
				std::vector<double> row_iMomentumDer(inCols, 0);
				col_poolMomentumDer.push_back(row_iMomentumDer);
				std::vector<double> row_iDerNum(inCols, 0);
				col_poolDerNum.push_back(row_iDerNum);
			}
			row_pool.push_back(col_pool);
			row_poolDer.push_back(col_poolDer);
			row_poolDerNum.push_back(col_poolDerNum);
			row_poolMomentumDer.push_back(col_poolMomentumDer);
		}
		weights.push_back(row_pool);
		weightDer.push_back(row_poolDer);
		weightDerMomentum.push_back(row_poolMomentumDer);
		weightDerNumerical.push_back(row_poolDerNum);
	}

	//init output and bias to 0
	for (int i = 0; i < numOutputRows; ++i) {
		std::vector<double> bias_i(numOutputCols, 0);
		bias.push_back(bias_i);
		std::vector<double> biasDer_i(numOutputCols, 0);
		biasDer.push_back(biasDer_i);
		std::vector<double> biasDerMom_i(numOutputCols, 0);
		biasDerMomentum.push_back(biasDerMom_i);
		std::vector<double> biasDerNum_i(numOutputCols, 0);
		biasDerNumerical.push_back(biasDerNum_i);
	}
}

void Layer::InitializeErrorAndOutput() {
	//init backPropError to 0
	for (int k = 0; k < numInputRows; ++k) {
		std::vector<double> row_i(numInputCols, 0);
		backPropError.push_back(row_i);
	}
	for (int k = 0; k < numOutputRows; ++k) {
		std::vector<double> error_i(numOutputCols, 0);
		error.push_back(error_i);
		std::vector<double> outputRow_i(numOutputCols, 0);
		output.push_back(outputRow_i);
	}
}
