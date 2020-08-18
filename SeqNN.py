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
        self.__model.AddInputDataPoints(self.__trainTargets)
        self.__model.AddTargetVectors(self.__trainTargets)
    
    def __createNpImgArray__(self, csvFileName, isTrain):
        dfTrainRaw = pd.read_csv(csvFileName)

        dfTrain = dfTrainRaw
        if isTrain:
            dfTrain = dfTrainRaw.iloc[:, 1:]

        #if len(dfTrain.iterrows()) != self.__inputRows:
        #    print("for file ", csvFileName, " num input rows provided in constructor = \t", len(dfTrain.iterrows()), " does not equal the numRows = \t", len(dfTrain.columns), " and input")

        imagesArray = np.zeros((self.__inputRows, self.__inputCols))
        for rowIndex, row in dfTrain.iterrows():
            numpyArray = np.zeros((self.__inputRows, self.__inputCols))
            for colIndex in range(len(dfTrain.columns)):
                #read pixels into 2d numpy array
                npRow = math.floor(colIndex / self.__inputCols)
                npCol = colIndex % self.__inputCols
                temp = dfTrain.iloc[rowIndex, colIndex]
                numpyArray[npRow, npCol] = temp
            imagesArray = np.dstack((imagesArray, numpyArray))
        returnArray = imagesArray[:,:,1:]
        #print("img array shape:")
        #print (returnArray.shape)
        np.ascontiguousarray(returnArray)
        #print (returnArray.shape)
        return returnArray

    def __createNpImgTargetArray__(self, csvFileName):
        dfTrainRaw = pd.read_csv(csvFileName)

        dfTarget = dfTrainRaw.iloc[:, 0]

        targetArrayNp = np.zeros((1, self.__numOutputClasses))
        for rowIndex, row in dfTarget.iteritems():
            targetArray = np.zeros((1, self.__numOutputClasses))
            targetArray[0, dfTarget.iloc[rowIndex]] = 1
            targetArrayNp = np.dstack((targetArrayNp, targetArray))
        #print ("target array shape: ")
        #print (targetArrayNp.shape)
        returnArray = targetArrayNp[:,:,1:]
        np.ascontiguousarray(returnArray)
        return returnArray

    def getTrainData(self):
        return self.__trainData

    def getTrainTargets(self):
        return self.__trainTargets

    def trainNN(self, batchSize, numEpochs, weightStepSize):
        #if ((not any(self.__trainData)) or (not any(self.__trainTargets))):
        #    print("Training data and targets must be loaded first before training the neural network.")
        #    return
        self.__model.SetBatchSize(batchSize)
        self.__model.SetNumEpochs(numEpochs)
        self.__model.SetStepSize(weightStepSize)
        self.__model.Train()
        self.__modelTrained = True

    def predict(self, csvFileName):
        if not self.__modelTrained:
            print("Model must be trained first before predicting classes.")
            return
        inputData = self.__createNpImgArray__(csvFileName, False)
        self.__outputPredicted = self.__model.Predict(inputData)
        return self.__outputPredicted

    def readData(self, csvFileName):
        self.__trainData = self.__createNpImgArray__(csvFileName, True)
        self.__trainTargets = self.__createNpImgTargetArray__(csvFileName)
        self.__initNN__()

    def addLayer(self, layer):
        self.__model.AddLayer(layer)

    def addLayerList(self, layerList):
        for layer in layerList:
            self.__model.AddLayer(layer)

    def clearLayers(self):
        self.__model.ClearLayers()
