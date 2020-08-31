#include "SequentialModel.h"

const std::vector<int> SequentialModel::_Predict(const std::vector<std::vector<std::vector<double>>>& inputDataPredict) {
	std::vector<int> maxOutputs;
	for (auto& input_i : inputDataPredict) {
		FwdProp(&input_i);
		auto& lastLayer = allLayers[allLayers.size() - 1];
		const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());

		double maxProb = -1;
		int maxI = -1;
		int maxJ = -1;
		for (int i = 0; i < output.size(); ++i) {
			for (int j = 0; j < output[0].size(); ++j) {
				if (output[i][j] > maxProb) {
					maxProb = output[i][j];
					maxI = i;
					maxJ = j;
				}
			}
		}
		maxOutputs.push_back(maxJ);
	}
	return maxOutputs;
}

std::vector<std::vector<std::vector<double>>> SequentialModel::ConvertNpToVector(int len1_, int len2_, int len3_, double* vec_) {
	std::vector<std::vector<std::vector<double>>> v(len3_);
	bool display = false;
	for (int k = 0; k < len3_; ++k) {
		if (k == 0)
			display = false;
		else
			display = false;
		std::vector<std::vector<double>> arry(len1_);
		v[k] = std::move(arry);
		for (int i = 0; i < len1_; ++i) {
			std::vector<double> row_i(len2_);
			v[k][i] = std::move(row_i);
			for (int j = 0; j < len2_; ++j) {
				v[k][i][j] = *(vec_ + k + len3_ * (j + len2_ * i));
			}
		}
	}
	return std::move(v);
}

bool SequentialModel::CheckGradientNumerically() {
	int n = 0;
	while (n < inputData.size() - batchSize) {
		for (int i = 0; i < batchSize; ++i) {
			FwdProp(&inputData[n]);
			auto error = CalcError(targets[n]);
			BackProp(error);
			++n;
		}
		for (int k = 0; k < allLayers.size(); ++k) {
			if (!allLayers[k]->GradientCorrect(this, n - batchSize, n))
				return false;
		}
		for (int p = 0; p < allLayers.size(); ++p) {
			allLayers[p]->ResetWeights();
		}
	}
	return true;
}

void SequentialModel::Train() {
	try {
		int numBatches = numEpochs * std::min(inputData.size(), targets.size()) / batchSize;
		for (int batch_k = 0; batch_k < numBatches; ++batch_k) {
			for (int i = 0; i < batchSize; ++i) {
				FwdProp(&inputData[lastTrainedIndex]);
				auto error = CalcError(targets[lastTrainedIndex]);
				BackProp(error);
				lastTrainedIndex = (lastTrainedIndex + 1) % inputData.size();
			}
			UpdateWeights();
		}
	}
	catch (std::exception& e) {
		std::cout << "exception in Train()" << std::endl;
	}
}

double SequentialModel::CalcErrorNumerically(int dataPointIndex) {
	FwdProp(&SequentialModel::inputData[dataPointIndex]);

	auto& lastLayer = allLayers[allLayers.size() - 1];
	const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());
	double crossEntropyErrorSum = 0;
	for (int i = 0; i < targets[dataPointIndex].size(); ++i) {
		for (int j = 0; j < targets[dataPointIndex][0].size(); ++j) {
			crossEntropyErrorSum -=  targets[dataPointIndex][i][j] * log(output[i][j]) + (1 - targets[dataPointIndex][i][j]) * log(1 - output[i][j]);
		}
	}
	return crossEntropyErrorSum;
}
