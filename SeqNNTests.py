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


def RunAllTests(csvFileName):
    DataConvertTest(csvFileName)
    return


def DataConvertTest(csvFileName):
    model = SeqNN.SeqNN(2, 4, 4, csvFileName)

    print("image array convert test: ", imageArrayConvertTest(allImagesArrayTrain))
    print("target vector convert test: ", targetVectorConvertTest(targetArrayTrain))

def Simple2DCNN(csvFileName):
    csvFileName = "Data//trainScratchSimple3.csv"

    model = SeqNN.SeqNN(2, 4, 4, csvFileName)
    cnnLayer = CNN.Conv2DLayer(2, 2, 2, 2, 0)
    model.__addLayer__(cnnLayer)
    denseLayer = CNN.DenseLayer()
    model.__addLayer__(denseLayer)

    model.__trainNN__(1,2,0.01)

    output = model.__predict__(model.__getTrainData__())
    print ("output predicted: ", output)


def imageArrayConvertTest(inputImageArray):
    modelTest = CNN.SequentialModel()
    modelTest.AddInputDataPoints(allImagesArrayTrain)
    checkArray = modelTest.GetInputDataPointsVector()
    
    #print("inputImageArray")
    #print(inputImageArray)
    #print("allimagesarray")
    #print(allImagesArrayTrain)
    #print("checkArray")
    # print(checkArray)

    for z in range(inputImageArray.shape[2]):
        for row_i in range(imgHeight):
            for col_j in range(imgWidth):
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
        for row_i in range(1):
            for col_j in range(numOutputClasses):
                #print("z = ", z, "i = ", row_i, "j = ", col_j)
                if checkArray[z][row_i][col_j] != targets[row_i,col_j,z]:
                    print("ERROR: Input image arrays do not match converted vector arrays in SequentialModel for i = ", row_i, " j = ", col_j, " z = ", z)
                    print("Vector element value = ", checkArray[z][row_i][col_j])
                    print("Input image array value = ", targets[row_i,col_j,z])
                    return False

    return True
