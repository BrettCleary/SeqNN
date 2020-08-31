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
//#include <numpy/ndarraytypes.h>
//#include <boost/python/numpy.hpp>

//namespace py = boost::python;
//namespace np = boost::python::numpy;

class SequentialModel
{
	//PyObject_HEAD
	//std::vector<Layer*> allLayers;
	//static const int numLayers = 1;
	//Layer* allLayers[numLayers];
	std::vector<Layer*> allLayers;
	//double weightStepSize = .01;

	std::vector<std::vector<std::vector<double>>> inputData;
	//[dataPoint][row][col]
	std::vector<std::vector<std::vector<double>>> targets;
	int batchSize = 1;
	int numEpochs = 1;
	int lastTrainedIndex = 0;

	//double* crossEntropyErrorSum;

	//bool calcCrossEntropyError = false;

	std::vector<std::vector<double>> errorPrev;
	//int printCounter = 0;


	void PrintErrorChange(const std::vector<std::vector<double>>& error) {
		if (errorPrev.empty())
			return;
		for (int i = 0; i < error.size(); ++i) {
			//std::cout << "dErrror: " << std::endl;
			//std::cout << " i: " << i << std::endl;;
			for (int j = 0; j < error[0].size(); ++j) {
				//std::cout << " j: " << j << " dError = " << error[i][j] - errorPrev[i][j];
			}
		}
		//std::cout << std::endl;
	}


	void FwdProp(const std::vector<std::vector<double>>* input) {
		/*for (Layer* layer_i : allLayers) {
			input = layer_i->FwdProp(*input);
		}*/
		//std::cout << std::endl;
		for (int i = 0; i < allLayers.size(); ++i) {
			input = allLayers[i]->FwdProp(*input);
			//std::cout << "fwdprop layer i: " << i << " completed." << std::endl;
		}
		//std::cout << std::endl;
	}

	void BackProp(const std::vector<std::vector<double>>& error) {
		/*if (printCounter % 10 == 0) {
			//PrintErrorChange(error);
			//errorPrev = error;
			printCounter = 0;
			std::cout << "num of backprops: " << printCounter << std::endl;
		}
		++printCounter;*/
		const std::vector<std::vector<double>>* errorPtr = &error;
		/*for (int i = allLayers.size() - 1; i >= 0; --i) {
			errorPtr = allLayers[i]->BackProp(*errorPtr);
		}*/
		//std::cout << std::endl;
		for (int i = allLayers.size() - 1; i >= 0; --i) {
			errorPtr = allLayers[i]->BackProp(*errorPtr);
			//std::cout << " backprop layer i: " << i << " completed." << std::endl;
		}
		//std::cout << std::endl;
	}

	std::vector<std::vector<double>> CalcError(const std::vector<std::vector<double>>& target) {
		auto dError_dAct = target;
		//auto& lastLayer = allLayers[allLayers.size() - 1];
		auto& lastLayer = allLayers[allLayers.size() - 1];
		const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());

		/*std::cout << "target rows: " << target.size() << " target cols" << target[0].size() << std::endl;

		std::cout << "target vector in calcerror" << std::endl;
		for (int i = 0; i < target.size(); ++i) {
			for (int j = 0; j < target[0].size(); ++j) {
				std::cout << "ij " << i << j << " target = " << target[i][j] << std::endl;
			}
		}*/

		for (int i = 0; i < target.size(); ++i) {
			for (int j = 0; j < target[0].size(); ++j) {
				dError_dAct[i][j] = (output[i][j] - target[i][j]) / (output[i][j] * (1 - output[i][j]));
				//if (i == 0 && j == 0)
				//	std::cout << "for i and j: " << i << j << " dError_dact = " << dError_dAct[i][j] << " output = " << output[i][j] << " target = " << target[i][j] << std::endl;
			}
		}
		return std::move(dError_dAct);
	}
	
	void UpdateWeights() {
		/*for (Layer* layer_i : allLayers) {
			layer_i->UpdateWeights(weightStepSize);
		}*/

		for (int i = 0; i < allLayers.size(); ++i) {
			//std::cout << "Updating Weights for i = " << i << std::endl;
			//if (i == 0)
			//	allLayers[i]->displayWeights = true;
			allLayers[i]->UpdateWeights();
			//if (i == 0)
			//	allLayers[i]->displayWeights = false;
			//std::cout << "Updated Weights for i = " << i << std::endl;
		}
	}

	void _AddInputDataPoint(std::vector<std::vector<double>>& dataPoint) {
		inputData.push_back(dataPoint);
	}

	const std::vector<int> _Predict(const std::vector<std::vector<std::vector<double>>>& inputDataPredict) {
		std::vector<int> maxOutputs;
		for (auto& input_i : inputDataPredict) {
			/*std::cout << "input data array" << std::endl;
			for (int i = 0; i < input_i.size(); ++i) {
				for (int j = 0; j < input_i[0].size(); ++j) {
					std::cout << "ij: " << i << j << " inputData to Predict = " << input_i[i][j] << std::endl;
				}
			}*/

			FwdProp(&input_i);
			//std::cout << "fwdprop completed" << std::endl;
			//auto& lastLayer = allLayers[allLayers.size() - 1];
			auto& lastLayer = allLayers[allLayers.size() - 1];
			const std::vector<std::vector<double>>& output = *(lastLayer->GetOutput());

			double maxProb = -1;
			int maxI = -1;
			int maxJ = -1;
			for (int i = 0; i < output.size(); ++i) {
				for (int j = 0; j < output[0].size(); ++j) {
					//std::cout << "output for predict ij" << i << j << " is " << output[i][j] << std::endl;
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

	std::vector<std::vector<std::vector<double>>> ConvertNpToVector(int len1_, int len2_, int len3_, double* vec_) {
		//std::cout << "len1_ = " << len1_ << " len2_ = " << len2_ << " len3_ = " << len3_ << std::endl;
		/*for (int i = 0; i < len1_ * len2_ * len3_; ++i) {
			std::cout << "element i " << i << " vec value = " << *(vec_ + i) << std::endl;
		}*/
		//std::cout << "converting array to vector" << std::endl;
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
				//std::cout << "len1_: " << len1_ << " len2_: " << len2_ << " len3_: " << len3_ << std::endl;
				std::vector<double> row_i(len2_);
				v[k][i] = std::move(row_i);
				for (int j = 0; j < len2_; ++j) {
					v[k][i][j] = *(vec_ + k + len3_ * (j + len2_ * i));
					//std::cout << "k = " << k << " i = " << i << " j = " << j << " val: " << v[k][i][j] << std::endl;
				}

				//v[k][i].insert(v[k][i].end(), vec_ + k * len1_ * len2_ + i * len2_, vec_ + k * len1_ * len2_ + (i + 1) * len2_);
			}
		}
		//std::cout << "done converting array to vector" << std::endl;
		return std::move(v);
	}

public:
	//SequentialModel(std::vector<Layer*> layerList) {
	SequentialModel(){

		//crossEntropyErrorSum = new double();

		//allLayers[0] = new Conv2DLayer();
		//allLayers[1] = new Pool2DLayer();
		//allLayers[2] = new DenseLayer();
		//allLayers[0] = new Pool2DLayer();
		//allLayers[0] = new DenseLayer();
	}

	//void Add(Layer& layer) {
	/*	
		//allLayers.push_back(layerConverted);
	}*/

	std::vector<std::vector<std::vector<double>>> GetInputDataPointsVector() {
		return inputData;
	}

	std::vector<std::vector<std::vector<double>>> GetTargetVectors() {
		return targets;
	}

	bool CheckGradientNumerically() {
		int n = 0;
		//std::cout << "size" << inputData.size() - batchSize << std::endl;
		while (n < inputData.size() - batchSize) {
			for (int i = 0; i < batchSize; ++i) {
				FwdProp(&inputData[n]);
				auto error = CalcError(targets[n]);
				BackProp(error);
				++n;
			}
			for (int k = 0; k < allLayers.size(); ++k) {
				//std::cout << "layer k = " << k << std::endl;
				if (!allLayers[k]->GradientCorrect(this, n - batchSize, n))
					return false;
			}
			for (int p = 0; p < allLayers.size(); ++p) {
				allLayers[p]->ResetWeights();
			}
		}
		return true;
	}

	//for use by layers to calculate weight derivatives numerically 
	double CalcErrorNumerically(int dataPointIndex);

	int AddInputDataPoint(int len1_, int len2_, double* vec_) {
		std::vector< std::vector<double> > v(len1_);
		for (int i = 0; i < len1_; ++i) {
			v[i].insert(v[i].end(), vec_ + i * len2_, vec_ + (i + 1) * len2_);
		}
		_AddInputDataPoint(v);

		return 1;
	}

	int AddInputDataPoints(int len1_, int len2_, int len3_, double* vec_) {
		//return 3;
		inputData = ConvertNpToVector(len1_, len2_, len3_, vec_);
		return 1;
	}

	int AddTargetVector(int len1_, double* vec_) {
		std::vector<double> v(len1_);
		for (int i = 0; i < len1_; ++i) {
			v.insert(v.end(), vec_, vec_ + len1_);
		}
		std::vector<std::vector<double>> v2D{ v };
		targets.push_back(v2D);

		return 2;
	}

	int AddTargetVectors(int len1_, int len2_, int len3_, double* vec_) {
		//return 4;
		targets = ConvertNpToVector(len1_, len2_, len3_, vec_);

		/*for (int i = 0; i < len1_; ++i) {
			for (int j = 0; j < len2_; ++j) {
				for (int k = 0; k < len3_; ++k) {
					std::cout << "ijk " << i << j << k << " target = " << targets[k][i][j] << std::endl;
				}
			}
		}*/
		return 1;
	}

	void SetBatchSize(int sz) {
		batchSize = sz;
	}

	void SetNumEpochs(int num) {
		numEpochs = num;
	}

	/*void SetStepSize(double step) {
		weightStepSize = step;
	}*/

	void AddLayer(Layer* layer) {
		allLayers.push_back(layer);
	}

	void ClearLayers() {
		allLayers.clear();
	}

	void Train() {
		try {
			/*int z = 0;
			for (auto dataPoint : inputData) {
				std::cout << "image input for index z = " << z << std::endl;
				for (int i = 0; i < dataPoint.size(); ++i) {
					for (int j = 0; j < dataPoint[0].size(); ++j) {
						std::cout << " ij " << i << j << " value = " << dataPoint[i][j] << std::endl;
					}
				}
				++z;
			}*/


			//std::cout << "\n entering train" << std::endl;
			//std::cout << "batches: " << batchSize << std::endl;
			int numBatches = numEpochs * std::min(inputData.size(), targets.size()) / batchSize;
			//int n = 0;
			for (int batch_k = 0; batch_k < numBatches; ++batch_k) {
				for (int i = 0; i < batchSize; ++i) {
					//std::cout << "Entering FwdProp()" << std::endl;
					FwdProp(&inputData[lastTrainedIndex]);
					//std::cout << "finished fwdprop for batch: " << batch_k << " and data element i: " << i << std::endl;
					//std::cout << " n = " << n << std::endl;
					auto error = CalcError(targets[lastTrainedIndex]);
					//std::cout << "finished error for batch: " << batch_k << " and data element i: " << i << std::endl;
					BackProp(error);
					//std::cout << "finished backprop for batch: " << batch_k << " and data element i: " << i << std::endl;
					lastTrainedIndex = (lastTrainedIndex + 1) % inputData.size();
				}
				//std::cout << "updating weights: " << std::endl;
				UpdateWeights();
				//std::cout << "done updating weights: " << std::endl;
			}
			//std::cout << "leaving train" << std::endl;
		}
		catch (std::exception& e) {
			std::cout << "exception in Train()" << std::endl;
		}
	}

	std::vector<int> Predict(int len1_, int len2_, int len3_, double* vec_) {
		auto v = ConvertNpToVector(len1_, len2_, len3_, vec_);
		//std::cout << "converted" << std::endl;
		return _Predict(v);
	}
};
#endif // CNN_SEQMODEL_H