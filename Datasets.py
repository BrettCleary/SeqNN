import numpy as np
import pandas as pd
import math

#[row, col, imageIndex]
mnistTrainData = np.load('Data//MNIST_trainData.npy')

mnistTrainTargets = np.load('Data//MNIST_trainTargets.npy')

#testD_trainData = np.load("Data//TestD_MNIST_tenDigits" + "_trainData.npy")
#testD_trainTargets = np.load("Data//TestD_MNIST_tenDigits" + "_trainTargets.npy")

def convertCsvToNpyFile(fileName):
    pdArray = pd.read_csv(fileName + ".csv")
    npArray = pdArray.to_numpy()
    np.save(fileName + ".npy", npArray)

def convertAllCsvToNpy():
    csvFileNames = ["Data//TestA_fourClassEightBinaryPixels", "Data//TestB_fourClassSixteenBinaryPixels",
    "Data//TestC_fourClassEightBinarySparsePixels", "Data//TestD_MNIST_tenDigits", "Data//TestE_MNIST_tenDigits_normalized",
    "Data//TestF_MNIST_fourDigits_normalized"]
    numRowsInput = [2, 4, 2, 28, 28, 28]
    numColsInput = [4, 4, 4, 28, 28, 28]
    numOutputClasses = [4, 4, 5, 10, 10, 4]
    for index in range(len(csvFileNames)):
        [trainData, trainTargets] = createTrainArrays(csvFileNames[index] + ".csv", numRowsInput[index], numColsInput[index], numOutputClasses[index])
        np.save(csvFileNames[index] + "_trainData.npy", trainData)
        np.save(csvFileNames[index] + "_trainTargets.npy", trainTargets)
        #convertCsvToNpyFile(csvFileNames[index])

def convertMnistCsvToNpy():
    csvFileNames = ["Data//MNIST"]
    numRowsInput = [28]
    numColsInput = [28]
    numOutputClasses = [10]
    for index in range(len(csvFileNames)):
        [trainData, trainTargets] = createTrainArrays(csvFileNames[index] + ".csv", numRowsInput[index], numColsInput[index], numOutputClasses[index])
        np.save(csvFileNames[index] + "_trainData.npy", trainData)
        np.save(csvFileNames[index] + "_trainTargets.npy", trainTargets)


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
        #if rowIndex % 100 == 0:
        #    print("rowIndex for train data", rowIndex)
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
        #if rowIndex % 100 == 0:
        #    print("rowIndex for train arrays", rowIndex)
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
        #if rowIndex % 100 == 0:
        #    print("rowIndex for test arrays", rowIndex)
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
