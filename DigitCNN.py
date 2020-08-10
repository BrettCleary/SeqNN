import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import CNNSlow
import cProfile
import re
import CNN

import sys
print(sys.executable)


def cnnTest():
    imgWidth = 28
    imgHeight = 28
    strideHor = 4
    strideVert = 4
    fieldWidth = 4
    fieldHeight = 4
    poolWidth = 2
    poolHeight = 2
    step = 0.01

    #model = CNNSlow.Cnn(imgWidth, imgHeight, strideHor, strideVert, fieldWidth, fieldHeight, poolWidth, poolHeight, step)

    def createNpImgArray(csvFileName, isTrain, allImagesArray, targetArrayNp):
        dfTrainRaw = pd.read_csv(csvFileName)

        dfTrain = dfTrainRaw
        dfTarget = dfTrainRaw.iloc[:, 0]

        if isTrain:
            dfTrain = dfTrainRaw.iloc[:, 1:]


        for rowIndex, row in dfTrain.iterrows():
            numpyArray = np.zeros((imgWidth, imgHeight))
            for colIndex in range(len(dfTrain.columns)):
                #read pixels into 2d numpy array
                npRow = math.floor(colIndex / imgWidth)
                npCol = colIndex % imgWidth
                temp = dfTrain.iloc[rowIndex, colIndex]
                numpyArray[npRow, npCol] = temp
            allImagesArray.append(numpyArray)
            if isTrain:
                targetArray = np.zeros(10)
                targetArray[dfTarget.iloc[rowIndex]] = 1
                targetArrayNp.append(targetArray)
        return


    allImagesArrayTrain = [] #list of 2d numpy arrays representing pixel values
    targetArrayTrain = []

    allImagesArrayTest = [] 
    dummyTarget = []

    createNpImgArray('Data\\trainScratch.csv', True, allImagesArrayTrain, targetArrayTrain)
    #createNpImgArray('Data\\test.csv', False, allImagesArrayTest, dummyTarget)


    #cross validate
    n_foldsBoundaries = 6
    n_folds = n_foldsBoundaries - 1
    n_ImagesPerFold = math.floor(len(allImagesArrayTrain) / n_folds)
    for i in range(n_folds):
        holdOutSet = allImagesArrayTrain[i * n_ImagesPerFold : (i + 1) * n_ImagesPerFold]
        holdOutTargets = targetArrayTrain[i * n_ImagesPerFold : (i + 1) * n_ImagesPerFold]
        #holdOutSet = allImagesArrayTrain
        #holdOutTargets = targetArrayTrain
        aTrainSet = []
        aTargetTrain = []
        bTrainSet = []
        bTargetTrain = []

        if i != 0:
            aTrainSet = allImagesArrayTrain[: i * n_ImagesPerFold]
            aTargetTrain = targetArrayTrain[: i * n_ImagesPerFold]
        if i != range(n_folds - 1):
            bTrainSet = allImagesArrayTrain[(i + 1) * n_ImagesPerFold :]
            bTargetTrain = targetArrayTrain[(i + 1) * n_ImagesPerFold :]

        trainingSet = aTrainSet + bTrainSet
        trainingTargets = aTargetTrain + bTargetTrain
        #trainingSet = allImagesArrayTrain
        #trainingTargets = targetArrayTrain

        cvModel = CNNSlow.Cnn(imgWidth, imgHeight, strideHor, strideVert, fieldWidth, fieldHeight, poolWidth, poolHeight, step)
        for iteration in range(1):
            #if iteration % 50 == 0:
            #print(iteration)
            cvModel.sequentialTrain(trainingSet, trainingTargets)

        #test accuracy on holdOutSet
        numWrong = 0
        for imgIndex in range(len(holdOutSet)):
            img = holdOutSet[imgIndex]
            predictedNum = cvModel.predictNumber(img)
            if holdOutTargets[imgIndex][predictedNum] != 1:
                numWrong += 1

        #print('Error Rate: ', numWrong / n_ImagesPerFold * 100, '% on validation set ', i)


#cnnTest()
    #model.sequentialTrain(allImagesArrayTrain, targetArrayNp)
#cProfile.run('cnnTest()')

