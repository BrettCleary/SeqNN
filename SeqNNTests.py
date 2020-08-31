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

def convertCsvToNpyFile(fileName):
    pdArray = pd.read_csv(fileName + ".csv")
    npArray = pdArray.to_numpy()
    np.save(fileName + ".npy", npArray)

def convertAllCsvToNpy():
    csvFileNames = ["Data//TestA_fourClassEightBinaryPixels", "Data//TestB_fourClassSixteenBinaryPixels",
    "Data//TestC_fourClassEightBinarySparsePixels", "Data//TestD_MNIST_tenDigits", "Data//TestE_MNIST_tenDigits_normalized",
    "Data//TestF_MNIST_fourDigits_normalized"]
    #csvFileNames = ["Data//MNIST_train"]
    numRowsInput = [2, 4, 2, 28, 28, 28]
    numColsInput = [4, 4, 4, 28, 28, 28]
    numOutputClasses = [4, 4, 5, 10, 10, 4]
    for index in range(len(csvFileNames)):
        [trainData, trainTargets] = createTrainArrays(csvFileNames[index] + ".csv", numRowsInput[index], numColsInput[index], numOutputClasses[index])
        np.save(csvFileNames[index] + "_trainData.npy", trainData)
        np.save(csvFileNames[index] + "_trainTargets.npy", trainTargets)
        #convertCsvToNpyFile(csvFileNames[index])

def convertMNIST_Arrays():
    #[trainData, trainTargets] = createTrainArrays("Data//MNIST_train.csv", 28, 28, 10)
    #np.save("Data//MNIST_trainData.npy", trainData)
    #np.save("Data//MNIST_trainTargets.npy", trainTargets)
    testData = createNpImgArray("Data//MNIST_test.csv", False, 28, 28)
    np.save("Data//MNIST_testData.npy", testData)

def createNpImgArray(csvFileName, ignoreFirstColumn, inputRows, inputCols):
    dfTrainRaw = pd.read_csv(csvFileName)

    dfTrain = dfTrainRaw
    if ignoreFirstColumn:
        dfTrain = dfTrainRaw.iloc[:, 1:]

    #if len(dfTrain.iterrows()) != self.__inputRows:
    #    print("for file ", csvFileName, " num input rows provided in constructor = \t", len(dfTrain.iterrows()), " does not equal the numRows = \t", len(dfTrain.columns), " and input")

    imagesArray = np.zeros((inputRows, inputCols))
    for rowIndex, row in dfTrain.iterrows():
        if rowIndex % 100 == 0:
            print("rowIndex for train data", rowIndex)
        numpyArray = np.zeros((inputRows, inputCols))
        for colIndex in range(len(dfTrain.columns)):
            #read pixels into 2d numpy array
            npRow = math.floor(colIndex / inputCols)
            npCol = colIndex % inputCols
            temp = dfTrain.iloc[rowIndex, colIndex]
            numpyArray[npRow, npCol] = temp
        imagesArray = np.dstack((imagesArray, numpyArray))
    returnArray = imagesArray[:,:,1:]
    #print("img array shape:")
    #print (returnArray.shape)
    np.ascontiguousarray(returnArray)
    #print (returnArray.shape)
    return returnArray

def createNpImgTargetArray(csvFileName, numOutputClasses):
    dfTrainRaw = pd.read_csv(csvFileName)

    dfTarget = dfTrainRaw.iloc[:, 0]

    targetArrayNp = np.zeros((1, numOutputClasses))
    for rowIndex, row in dfTarget.iteritems():
        #print("rowIndex for train targets", rowIndex)
        targetArray = np.zeros((1, numOutputClasses))
        targetArray[0, dfTarget.iloc[rowIndex]] = 1
        targetArrayNp = np.dstack((targetArrayNp, targetArray))
    #print ("target array shape: ")
    #print (targetArrayNp.shape)
    returnArray = targetArrayNp[:,:,1:]
    np.ascontiguousarray(returnArray)
    return returnArray

def createTrainArrays(csvFileName, inputRows, inputCols, numOutputClasses):
    dfTrainRaw = pd.read_csv(csvFileName)

    dfTrain = dfTrainRaw
    dfTrain = dfTrainRaw.iloc[:, 1:]
    dfTarget = dfTrainRaw.iloc[:, 0]
    
    imagesArray = np.zeros((inputRows, inputCols))
    targetArrayNp = np.zeros((1, numOutputClasses))
    for rowIndex, row in dfTrain.iterrows():
        if rowIndex % 100 == 0:
            print("rowIndex for train arrays", rowIndex)
        numpyArray = np.zeros((inputRows, inputCols))
        for colIndex in range(len(dfTrain.columns)):
            #read pixels into 2d numpy array
            npRow = math.floor(colIndex / inputCols)
            npCol = colIndex % inputCols
            temp = dfTrain.iloc[rowIndex, colIndex]
            numpyArray[npRow, npCol] = temp
            
        targetArray = np.zeros((1, numOutputClasses))
        targetArray[0, dfTarget.iloc[rowIndex]] = 1
        targetArrayNp = np.dstack((targetArrayNp, targetArray))
        imagesArray = np.dstack((imagesArray, numpyArray))
    dataArray = imagesArray[:,:,1:]
    trainData = np.ascontiguousarray(dataArray)
    targetArray = targetArrayNp[:,:,1:]
    trainTargets = np.ascontiguousarray(targetArray)
    return [trainData, trainTargets]

def createTestArrays(csvFileName, inputRows, inputCols, numOutputClasses):
    dfTrainRaw = pd.read_csv(csvFileName)

    dfTrain = dfTrainRaw
    dfTrain = dfTrainRaw.iloc[:, 1:]
    dfTarget = dfTrainRaw.iloc[:, 0]
    
    imagesArray = np.zeros((inputRows, inputCols))
    targetArrayNp = np.zeros((1, numOutputClasses))
    for rowIndex, row in dfTrain.iterrows():
        if rowIndex % 100 == 0:
            print("rowIndex for test arrays", rowIndex)
        numpyArray = np.zeros((inputRows, inputCols))
        for colIndex in range(len(dfTrain.columns)):
            #read pixels into 2d numpy array
            npRow = math.floor(colIndex / inputCols)
            npCol = colIndex % inputCols
            temp = dfTrain.iloc[rowIndex, colIndex]
            numpyArray[npRow, npCol] = temp
        
        targetArray = np.zeros((1, numOutputClasses))
        targetArray[0, dfTarget.iloc[rowIndex]] = 1
        targetArrayNp = np.dstack((targetArrayNp, targetArray))
        imagesArray = np.dstack((imagesArray, numpyArray))
    dataArray = imagesArray[:,:,1:]
    testData = np.ascontiguousarray(dataArray)
    targetArray = targetArrayNp[:,:,1:]
    testTargets = np.ascontiguousarray(targetArray)
    return [testData, testTargets]

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

def DataConvertTest(csvFileName, inputData, inputTargets):
    print("\nImage array convert test for csv file: \t", csvFileName, " is successful: ", imageArrayConvertTest(inputData))
    print("Target vector convert test for csv file:", csvFileName, " is successful: ", targetVectorConvertTest(inputTargets))
    print()

def SingleDenseLayerTest(csvFileName, trainData, trainTargets, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, stepSize, 
useEarlyStopping = False, stopPercentThreshold = 0.1, numEpochsBetweenChecks = 1, numFailsToStop = 10):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.setTrainData(trainData, trainTargets)
    denseLayer = CNN.DenseLayer(stepSize, 1, numOutputClasses)
    model.addLayer(denseLayer)
    #print(model.__model.CheckGradientNumerically())
    model.trainNN(batchSize, numEpochs, useEarlyStopping, stopPercentThreshold, numEpochsBetweenChecks, numFailsToStop)

    errorRate = model.calcTestErrorRate(trainData, trainTargets)
    print("\nThe error rate for ", csvFileName, " is ", errorRate, "\n")

def Simple2DCnnTest(csvFileName, trainData, trainTargets, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, cnnStepSize, denseLayerStepSize):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.setTrainData(trainData, trainTargets)
    #print(model.getTrainData().shape)
    #print(model.getTrainTargets().shape)
    cnnLayer = CNN.Conv2DLayer(2, 2, 2, 2, 0, cnnStepSize)
    #cnnLayer = CNN.Conv2DLayer(1, 1, 1, 1, 0)
    model.addLayer(cnnLayer)
    denseLayer = CNN.DenseLayer(denseLayerStepSize, 1, numOutputClasses)
    model.addLayer(denseLayer)
    print("Initial numerical gradient agrees with initial backprop gradient :", model.checkGradientNumerically())
    model.trainNN(batchSize, numEpochs)
    print("Final Numerical gradient agrees with final backprop gradient", model.checkGradientNumerically())

    errorRate = model.calcTestErrorRate(trainData, trainTargets)
    print("\nThe error rate for ", csvFileName, " is ", errorRate, "\n")

def Simple2DCnnPoolTest(csvFileName, trainData, trainTargets, inputRows, inputCols, numOutputClasses, batchSize, numEpochs, cnnStepSize, denseLayerStepSize):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.setTrainData(trainData, trainTargets)
    #print(model.getTrainData().shape)
    #print(model.getTrainTargets().shape)
    cnnLayer = CNN.Conv2DLayer(2, 2, 2, 2, 0, cnnStepSize)
    #cnnLayer = CNN.Conv2DLayer(1, 1, 1, 1, 0)
    model.addLayer(cnnLayer)
    poolLayer = CNN.Pool2DLayer(True, 2, 2)
    model.addLayer(poolLayer)
    denseLayer = CNN.DenseLayer(denseLayerStepSize, 1, numOutputClasses)
    model.addLayer(denseLayer)
    print("Initial numerical gradient agrees with initial backprop gradient :", model.checkGradientNumerically())
    model.trainNN(batchSize, numEpochs)
    print("Final Numerical gradient agrees with final backprop gradient", model.checkGradientNumerically())

    #model.calcTestErrorRate(csvFileName)

    #output = model.predict(csvFileName, True)
    #errorRate = model.__Calc2DErrorRate__(model.getTrainTargets(), output)
    print("\nThe error rate for ", csvFileName, " is ", model.calcTestErrorRate(trainData, trainTargets), "\n")

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