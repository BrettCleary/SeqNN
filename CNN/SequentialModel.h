#ifndef CNN_SEQMODEL_H
#define CNN_SEQMODEL_H
#include "Layer.h"
#include "Conv2DLayer.h"
#include "DenseLayer.h"
#include "Pool2DLayer.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <exception>

class SequentialModel
{
	std::vector<Layer*> allLayers;

	std::vector<std::vector<std::vector<double>>> inputData;
	std::vector<std::vector<std::vector<double>>> targets;
	int batchSize = 1;
	int numEpochs = 1;
	int lastTrainedIndex = 0;

	std::vector<std::vector<double>> errorPrev;

	void FwdProp(const std::vector<std::vector<double>>* input) {
		for (int i = 0; i < allLayers.size(); ++i) {
			input = allLayers[i]->FwdProp(*input);
		}
	}

	void BackProp(const std::vector<std::vector<double>>& error) {
		const std::vector<std::vector<double>>* errorPtr = &error;
		for (int i = allLayers.size() - 1; i >= 0; --i) {
			errorPtr = allLayers[i]->BackProp(*errorPtr);
		}
	}

	std::vector<std::vector<double>> CalcError(const std::vector<std::vector<double>>& target) {
		auto dError_dOutput = target;
		auto& lastLayer = allLayers[allLayers.size() - 1];
		const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());

		for (int i = 0; i < target.size(); ++i) {
			for (int j = 0; j < target[0].size(); ++j) {
				dError_dOutput[i][j] = (output[i][j] - target[i][j]) / (output[i][j] * (1 - output[i][j]));}
		}
		return std::move(dError_dOutput);
	}
	
	void UpdateWeights() {
		for (int i = 0; i < allLayers.size(); ++i) {
			allLayers[i]->UpdateWeights();
		}
	}

	void _AddInputDataPoint(std::vector<std::vector<double>>& dataPoint) {
		inputData.push_back(dataPoint);
	}

	const std::vector<int> _Predict(const std::vector<std::vector<std::vector<double>>>& inputDataPredict);

	std::vector<std::vector<std::vector<double>>> ConvertNpToVector(int len1_, int len2_, int len3_, double* vec_);

public:
	SequentialModel(){
	}

	std::vector<std::vector<std::vector<double>>> GetInputDataPointsVector() {
		return inputData;
	}

	std::vector<std::vector<std::vector<double>>> GetTargetVectors() {
		return targets;
	}

	bool CheckGradientNumerically();

	//for use by layers to calculate weight derivatives numerically 
	double CalcErrorNumerically(int dataPointIndex);

	void AddInputDataPoint(int len1_, int len2_, double* vec_) {
		std::vector< std::vector<double> > v(len1_);
		for (int i = 0; i < len1_; ++i) {
			v[i].insert(v[i].end(), vec_ + i * len2_, vec_ + (i + 1) * len2_);
		}
		_AddInputDataPoint(v);
	}

	void AddInputDataPoints(int len1_, int len2_, int len3_, double* vec_) {
		inputData = ConvertNpToVector(len1_, len2_, len3_, vec_);
	}

	void AddTargetVector(int len1_, double* vec_) {
		std::vector<double> v(len1_);
		for (int i = 0; i < len1_; ++i) {
			v.insert(v.end(), vec_, vec_ + len1_);
		}
		std::vector<std::vector<double>> v2D{ v };
		targets.push_back(v2D);
	}

	void AddTargetVectors(int len1_, int len2_, int len3_, double* vec_) {
		targets = ConvertNpToVector(len1_, len2_, len3_, vec_);
	}

	void SetBatchSize(int sz) {
		batchSize = sz;
	}

	void SetNumEpochs(int num) {
		numEpochs = num;
	}

	void AddLayer(Layer* layer) {
		allLayers.push_back(layer);
	}

	void ClearLayers() {
		allLayers.clear();
	}

	void Train();

	std::vector<int> Predict(int len1_, int len2_, int len3_, double* vec_) {
		auto v = ConvertNpToVector(len1_, len2_, len3_, vec_);
		return _Predict(v);
	}
};
#endif // CNN_SEQMODEL_H