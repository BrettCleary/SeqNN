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
    csvFileNames = ["Data//TestA_fourClassEightBinaryPixels.csv", "Data//TestB_fourClassFourBinaryPixels.csv", "Data//TestC_fourClassEightBinarySparsePixels.csv", "Data//TestD_MNIST_tenDigits.csv"]
    numRowsInput = [2, 2, 2, 28]
    numColsInput = [4, 2, 4, 28]
    numOutputClasses = [4, 4, 4, 10]
    
    for fileIndex in range(len(csvFileNames)):
        DataConvertTest(csvFileNames[fileIndex], True, numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex])

    for fileIndex in range(len(csvFileNames)):
        Simple2DCnnTest(csvFileNames[fileIndex], numRowsInput[fileIndex], numColsInput[fileIndex], numOutputClasses[fileIndex])

    return

def Calc2DErrorRate(target, predicted):
    if target.shape != predicted.shape:
        print("Target and predicted matrices do not have the same shape. Target shape is ", target.shape, ", and predicted shape is ", predicted.shape)
    
    numWrong = 0
    for x in range(target.shape[0]):
        for y in range(target.shape[1]):
            if target[x,y] != predicted[x,y]:
                numWrong += 1
    
    return numWrong / (target.shape[0] * target.shape[1])

def DataConvertTest(csvFileName, isTrain, inputRows, inputCols, numOutputClasses):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)

    inputArrays = model.__createNpImgArray__(csvFileName, isTrain)
    targetArrays = model.__createNpImgTargetArray__(csvFileName)

    print("\nImage array convert test for csv file: \t", csvFileName, " is successful: ", imageArrayConvertTest(inputArrays))
    print("Target vector convert test for csv file:", csvFileName, " is successful: ", targetVectorConvertTest(targetArrays))
    print()

def Simple2DCnnTest(csvFileName, inputRows, inputCols, numOutputClasses):

    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.readData(csvFileName)
    cnnLayer = CNN.Conv2DLayer(2, 2, 2, 2, 0)
    model.addLayer(cnnLayer)
    denseLayer = CNN.DenseLayer()
    model.addLayer(denseLayer)

    model.trainNN(1, 2, 0.01)

    output = model.predict(model.getTrainData())
    errorRate = Calc2DErrorRate(model.getTrainTargets(), output)
    print("The error rate for ", csvFileName, " is ", errorRate)

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

