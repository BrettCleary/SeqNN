import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import cProfile
import re
import CNN

class SeqNN(object):
    """description of class"""

    __inputRows = 0
    __inputCols = 0
    __numOutputClasses = 0
    __model = None
    #Numpy arrays
    __trainData = None
    __trainTargets = None
    __validationData = None
    __validationTargets = None
    __testData = None
    __testTargets = None

    __modelTrained = False
    __outputPredicted = None

    def __init__(self, inputRows, inputCols, numOutputClasses):#, strideH, strideV, fieldW, fieldH, poolW, poolH, gradStep):
        self.__inputRows = inputRows
        self.__inputCols = inputCols
        self.__numOutputClasses = numOutputClasses

    def __initNN__(self):
        self.__model = CNN.SequentialModel()
        self.__model.AddInputDataPoints(self.__trainData)
        self.__model.AddTargetVectors(self.__trainTargets)
    
    def __Calc2DErrorRate__(self, target, predicted):
        #print("target: ", type(target), " has shape", target.shape)
        #print("predicted: ", type(predicted), " has len ", len(predicted))
        #print("target: ", target)
        #print("predicted: ", predicted)
        targetClasses = np.zeros(target.shape[2])

        for z in range(target.shape[2]):
            zClass = -1
            for j in range(target.shape[1]):
                if (target[0, j, z] == 1):
                    zClass = j
            targetClasses[z] = zClass


        if len(predicted) != targetClasses.shape[0]:
            print("Target and predicted matrices do not have the same shape. Target shape is ", target.shape, ", and predicted shape is ", targetClasses.shape)


        numWrong = 0
        for z in range(targetClasses.shape[0]):
            #print("Target is ", targetClasses[z], ", and predicted is ", predicted[z])
            if targetClasses[z] != predicted[z]:
                numWrong += 1
                
        return numWrong / len(predicted) * 100

    def setValidationData(self, data):
        self.__validationData = data

    def setValidationTargets(self, targets):
        self.__validationTargets = targets

    def getTrainData(self):
        return self.__trainData

    def getTrainTargets(self):
        return self.__trainTargets

    def __predict(self, data):
        self.__outputPredicted = self.__model.Predict(data)
        return self.__outputPredicted

    def setValidationDataAndTargets(self, validationData, validationTargets):
        self.setValidationData(validationData)
        self.setValidationTargets(validationTargets)
        
    def trainNN(self, batchSize, numEpochs, useEarlyStopping = False,
     stopPercentThreshold = 0.1, numEpochsBetweenChecks = 1, numFailsToStop = 10):
        #if ((not any(self.__trainData)) or (not any(self.__trainTargets))):
        #    print("Training data and targets must be loaded first before training the neural network.")
        #    return
        self.__model.SetBatchSize(batchSize)
        if not useEarlyStopping:
            self.__model.SetNumEpochs(numEpochs)
            print("starting train without early stopping")
            self.__model.Train()
        else:
            self.__model.SetNumEpochs(numEpochsBetweenChecks)
            numEpochsTrained = 0
            numErrorTooLow = 0
            lastErrorPct = 1
            while (numEpochsTrained < numEpochs) and (numErrorTooLow < numFailsToStop):
                self.__model.Train()
                numEpochsTrained += numEpochsBetweenChecks
                predictedOutput = self.__predict(self.__validationData)
                errorPct = self.__Calc2DErrorRate__(self.__validationTargets, predictedOutput)
                print("Error Rate (%) after training ", numEpochsTrained, " number of epochs is ", errorPct)
                if (lastErrorPct - errorPct) < stopPercentThreshold:
                    numErrorTooLow += 1
                else: 
                    numErrorTooLow = 0
                lastErrorPct = errorPct
            
            if numErrorTooLow >= numFailsToStop:
                print("The neural network stopped early during training due to insufficient error reduction.")
        self.__modelTrained = True

    def predict(self, inputData, ignoreFirstColumn):
        if not self.__modelTrained:
            print("Model must be trained first before predicting classes.")
            return
        return self.__predict(inputData)

    def setTrainData(self, trainData, trainTargets):
        self.__trainData = trainData
        self.__trainTargets = trainTargets
        self.__initNN__()

    def calcTestErrorRate(self, testData, testTargets):
        self.__testData = testData
        self.__testTargets = testTargets
        predicted = self.__predict(self.__testData)
        return self.__Calc2DErrorRate__(self.__testTargets, predicted)

    def addLayer(self, layer):
        self.__model.AddLayer(layer)

    def addLayerList(self, layerList):
        for layer in layerList:
            self.__model.AddLayer(layer)

    def clearLayers(self):
        self.__model.ClearLayers()

    def checkGradientNumerically(self):
        return self.__model.CheckGradientNumerically()