#include "SequentialModel.h"

double SequentialModel::CalcErrorNumerically(int dataPointIndex) {
	FwdProp(&SequentialModel::inputData[dataPointIndex]);

	auto& lastLayer = allLayers[numLayers - 1];
	const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());
	double crossEntropyErrorSum = 0;
	for (int i = 0; i < targets[dataPointIndex].size(); ++i) {
		for (int j = 0; j < targets[dataPointIndex][0].size(); ++j) {
			crossEntropyErrorSum -=  targets[dataPointIndex][i][j] * log(output[i][j]) + (1 - targets[dataPointIndex][i][j]) * log(1 - output[i][j]);
		}
	}
	//std::cout << "crossEntropyErrorSum = " << crossEntropyErrorSum << std::endl;
	return crossEntropyErrorSum;
}