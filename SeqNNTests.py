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
from tempfile import TemporaryFile

def RunAllTests(runOneDenseTest = False, denseIndex = 0, runOneCnnTest = False, cnnIndex = 0):
    csvFileNames = ["Data//TestA_fourClassEightBinaryPixels", "Data//TestB_fourClassSixteenBinaryPixels",
     "Data//TestC_fourClassEightBinarySparsePixels", "Data//TestD_MNIST_tenDigits", "Data//TestF_MNIST_fourDigits_normalized"]
    npArraysData = list()
    npArraysTargets = list()
    numRowsInput = [2, 4, 2, 28, 28]
    numColsInput = [4, 4, 4, 28, 28]
    numOutputClasses = [4, 4, 5, 10, 5]
    batchSizes = [1, 1, 1, 1, 1]
    numEpochs = [100, 100, 100, 100, 100]
    stepSizes = [0.001, 0.001, 0.001, 0.000001, 0.001]
    useEarlyStopping = [False, False, False, True, True]
    stopPercentThreshold = [0.1, 0.1, 0.1, 0.1, 0.1]
    numEpochsBetweenChecks = [1, 1, 1, 1000, 1000]
    numFailsToStop = [1, 1, 1, 10, 10]
    
    for fileIndex in range(len(csvFileNames)):
        npArraysData.append(np.load(csvFileNames[fileIndex] + "_trainData.npy"))
        npArraysTargets.append(np.load(csvFileNames[fileIndex] + "_trainTargets.npy"))
        DataConvertTest(csvFileNames[fileIndex], npArraysData[fileIndex], npArraysTargets[fileIndex])

    print("Single Dense Layer Tests: ")
    if not runOneDenseTest:
        for fileIndex in range(len(csvFileNames)):
            SingleDenseLayerTest(csvFileNames[fileIndex], npArraysData[fileIndex], npArraysTargets[fileIndex], numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex], 
            batchSizes[fileIndex], numEpochs[fileIndex], stepSizes[fileIndex])
    else:
        SingleDenseLayerTest(csvFileNames[denseIndex], numRowsInput[denseIndex], numColsInput[denseIndex], numOutputClasses[denseIndex], 
        batchSizes[denseIndex], numEpochs[denseIndex], stepSizes[denseIndex], useEarlyStopping[denseIndex], stopPercentThreshold[denseIndex],
        numEpochsBetweenChecks[denseIndex], numFailsToStop[denseIndex])

    numEpochsCnn = [1000, 10000, 1000, 10000, 10000]
    denseStepSize = [0.04, 0.04, 0.04, 0.04, 0.04]
    cnnStepSize = [1, 1, 1, 1, 1]
    
    print("CNN 2D -> Dense Layer -> Output Tests: \n")
    if not runOneCnnTest:
        for fileIndex in range(len(csvFileNames)):
            Simple2DCnnTest(csvFileNames[fileIndex], npArraysData[fileIndex], npArraysTargets[fileIndex], numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex], 
            batchSizes[fileIndex], numEpochsCnn[fileIndex], cnnStepSize[fileIndex], denseStepSize[fileIndex])
    else:
            Simple2DCnnTest(csvFileNames[cnnIndex], npArraysData[fileIndex], npArraysTargets[fileIndex], numRowsInput[cnnIndex], numColsInput[cnnIndex], numOutputClasses[cnnIndex], 
            batchSizes[cnnIndex], numEpochsCnn[cnnIndex], cnnStepSize[cnnIndex], denseStepSize[cnnIndex])

    print("CNN 2D -> Pool Layer -> Dense Layer -> Output Tests: \n")
    poolFileIndex = 3
    Simple2DCnnPoolTest(csvFileNames[poolFileIndex], npArraysData[fileIndex], npArraysTargets[fileIndex], numRowsInput[poolFileIndex], numColsInput[poolFileIndex], numOutputClasses[poolFileIndex], 
    batchSizes[poolFileIndex], numEpochsCnn[poolFileIndex], cnnStepSize[poolFileIndex], denseStepSize[poolFileIndex])

def Calc2DErrorRate(target, predicted):
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

def DataConvertTest(csvFileName, inputData, inputTargets):
    print("\nImage array convert test for csv file: \t", csvFileName, " is successful: ", imageArrayConvertTest(inputData))
    print("Target vector convert test for csv file:", csvFileName, " is successful: ", targetVectorConvertTest(inputTargets))
    print()

def SingleDenseLayerTest(csvFileName, trainData, trainTargets, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, stepSize, 
useEarlyStopping = False, stopPercentThreshold = 0.1, numEpochsBetweenChecks = 1, numFailsToStop = 10):
    model = SeqNN.SeqNN([
        CNN.DenseLayer(stepSize, 1, numOutputClasses)
    ])
    model.trainNN(batchSize, numEpochs, trainData, trainTargets, trainData, trainTargets, 
    useEarlyStopping, stopPercentThreshold, numEpochsBetweenChecks, numFailsToStop)

    print("\nThe error rate for ", csvFileName, " is ", model.calcTestErrorRate(trainData, trainTargets), "\n")

def Simple2DCnnTest(csvFileName, trainData, trainTargets, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, cnnStepSize, denseLayerStepSize):
    model = SeqNN.SeqNN([
        CNN.Conv2DLayer(2, 2, 2, 2, 0, cnnStepSize),
        CNN.DenseLayer(denseLayerStepSize, 1, numOutputClasses)
    ])
    #print("Initial numerical gradient agrees with initial backprop gradient :", model.checkGradientNumerically())
    print("starting to train simple2DCNNTest")
    model.trainNN(batchSize, numEpochs, trainData, trainTargets)
    print("Final Numerical gradient agrees with final backprop gradient", model.checkGradientNumerically())
    print("\nThe error rate for ", csvFileName, " is ", model.calcTestErrorRate(trainData, trainTargets), "\n")

def Simple2DCnnPoolTest(csvFileName, trainData, trainTargets, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, cnnStepSize, denseLayerStepSize):
    model = SeqNN.SeqNN([
        CNN.Conv2DLayer(2, 2, 2, 2, 0, cnnStepSize),
        CNN.Pool2DLayer(True, 2, 2),
        CNN.DenseLayer(denseLayerStepSize, 1, numOutputClasses)
    ])
    #print("Initial numerical gradient agrees with initial backprop gradient :", model.checkGradientNumerically())
    model.trainNN(batchSize, numEpochs, trainData, trainTargets)
    print("Final Numerical gradient agrees with final backprop gradient", model.checkGradientNumerically())
    print("\nThe error rate for ", csvFileName, " is ", model.calcTestErrorRate(trainData, trainTargets), "\n")

def imageArrayConvertTest(inputImageArray):
    modelTest = CNN.SequentialModel()
    modelTest.AddInputDataPoints(inputImageArray)
    checkArray = modelTest.GetInputDataPointsVector()

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