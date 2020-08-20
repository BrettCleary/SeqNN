import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import cProfile
import re
import CNN
import SeqNN


def RunAllTests():
    csvFileNames = ["Data//TestA_fourClassEightBinaryPixels.csv", "Data//TestB_fourClassSixteenBinaryPixels.csv", "Data//TestC_fourClassEightBinarySparsePixels.csv", "Data//TestD_MNIST_tenDigits.csv"]
    numRowsInput = [2, 4, 2, 28]
    numColsInput = [4, 4, 4, 28]
    numOutputClasses = [4, 4, 4, 10]
    batchSizes = [1, 1, 1, 1]
    numEpochs = [100, 100, 100, 1000]
    stepSizes = [0.001, 0.001, 0.001, 0.000001]
    
    for fileIndex in range(len(csvFileNames)):
        DataConvertTest(csvFileNames[fileIndex], True, numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex])

    print("Single Dense Layer Tests: ")
    for fileIndex in range(len(csvFileNames)):
        SingleDenseLayerTest(csvFileNames[fileIndex], numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex], 
        batchSizes[fileIndex], numEpochs[fileIndex], stepSizes[fileIndex])

    numEpochsCnn = [1000, 10000, 1000, 10000]
    denseStepSize = [0.04, 0.04, 0.04, 0.04]
    cnnStepSize = [1, 1, 1, 1]

    print("CNN 2D -> Dense Layer -> Output Tests: ")
    for fileIndex in range(len(csvFileNames)):
        Simple2DCnnTest(csvFileNames[fileIndex], numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex], 
        batchSizes[fileIndex], numEpochsCnn[fileIndex], cnnStepSize[fileIndex], denseStepSize[fileIndex])

    return

def RunOneCnnTest(fileIndex):
    csvFileNames = ["Data//TestA_fourClassEightBinaryPixels.csv", "Data//TestB_fourClassSixteenBinaryPixels.csv", "Data//TestC_fourClassEightBinarySparsePixels.csv", "Data//TestD_MNIST_tenDigits.csv"]
    numRowsInput = [2, 4, 2, 28]
    numColsInput = [4, 4, 4, 28]
    numOutputClasses = [4, 4, 4, 10]
    batchSizes = [1, 1, 1, 1]
    numEpochsCnn = [1000, 10000, 1000, 1000]
    denseStepSize = [0.04, 0.04, 0.04, 0.001]
    cnnStepSize = [1, 1, 1, 0.001]

    print("CNN 2D -> Dense Layer -> Output Tests: ")
    Simple2DCnnTest(csvFileNames[fileIndex], numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex], 
    batchSizes[fileIndex], numEpochsCnn[fileIndex], cnnStepSize[fileIndex], denseStepSize[fileIndex])

    return

def Calc2DErrorRate(target, predicted):

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
    
    return numWrong / (target.shape[0] * target.shape[1])

def DataConvertTest(csvFileName, isTrain, inputRows, inputCols, numOutputClasses):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)

    inputArrays = model.__createNpImgArray__(csvFileName, isTrain)
    targetArrays = model.__createNpImgTargetArray__(csvFileName)

    print("\nImage array convert test for csv file: \t", csvFileName, " is successful: ", imageArrayConvertTest(inputArrays))
    print("Target vector convert test for csv file:", csvFileName, " is successful: ", targetVectorConvertTest(targetArrays))
    print()

def SingleDenseLayerTest(csvFileName, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, stepSize):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.readData(csvFileName)
    denseLayer = CNN.DenseLayer(stepSize)
    model.addLayer(denseLayer)
    #print(model.__model.CheckGradientNumerically())
    model.trainNN(batchSize, numEpochs)

    output = model.predict(csvFileName, True)
    errorRate = Calc2DErrorRate(model.getTrainTargets(), output)
    print("\nThe error rate for ", csvFileName, " is ", errorRate, "\n")

def Simple2DCnnTest(csvFileName, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, cnnStepSize, denseLayerStepSize):

    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.readData(csvFileName)
    #print(model.getTrainData().shape)
    #print(model.getTrainTargets().shape)
    cnnLayer = CNN.Conv2DLayer(2, 2, 2, 2, 0, cnnStepSize)
    #cnnLayer = CNN.Conv2DLayer(1, 1, 1, 1, 0)
    model.addLayer(cnnLayer)
    denseLayer = CNN.DenseLayer(denseLayerStepSize)
    model.addLayer(denseLayer)
    print("Initial numerical gradient agrees with initial backprop gradient :", model.checkGradientNumerically())
    model.trainNN(batchSize, numEpochs)
    print("Final Numerical gradient agrees with final backprop gradient", model.checkGradientNumerically())

    output = model.predict(csvFileName, True)
    errorRate = Calc2DErrorRate(model.getTrainTargets(), output)
    print("\nThe error rate for ", csvFileName, " is ", errorRate, "\n")

def imageArrayConvertTest(inputImageArray):
    modelTest = CNN.SequentialModel()
    modelTest.AddInputDataPoints(inputImageArray)
    checkArray = modelTest.GetInputDataPointsVector()
    
    #print("inputImageArray")
    #print(inputImageArray)
    #print("allimagesarray")
    #print(allImagesArrayTrain)
    #print("checkArray")
    # print(checkArray)

    for z in range(inputImageArray.shape[2]):
        for row_i in range(inputImageArray.shape[0]):
            for col_j in range(inputImageArray.shape[1]):
                #print("z = ", z, "i = ", row_i, "j = ", col_j)
                if checkArray[z][row_i][col_j] != inputImageArray[row_i,col_j,z]:
                    print("ERROR: Input image arrays do not match converted vector arrays in SequentialModel for i = ", row_i, " j = ", col_j, " z = ", z)
                    print("Vector element value = ", checkArray[z][row_i][col_j])
                    print("Input image array value = ", inputImageArray[row_i,col_j,z])
                    return False

    return True

def targetVectorConvertTest(targets):
    modelTest = CNN.SequentialModel()
    modelTest.AddTargetVectors(targets)
    checkArray = modelTest.GetTargetVectors()

    #print(inputImageArray.shape[2])
    #print(len(checkArray))

    for z in range(targets.shape[2]):
        for row_i in range(targets.shape[0]):
            for col_j in range(targets.shape[1]):
                #print("z = ", z, "i = ", row_i, "j = ", col_j)
                if checkArray[z][row_i][col_j] != targets[row_i,col_j,z]:
                    print("ERROR: Input image arrays do not match converted vector arrays in SequentialModel for i = ", row_i, " j = ", col_j, " z = ", z)
                    print("Vector element value = ", checkArray[z][row_i][col_j])
                    print("Input image array value = ", targets[row_i,col_j,z])
                    return False

    return True

