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


#def cnnTest():
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

def createNpImgArray(csvFileName, isTrain):
    dfTrainRaw = pd.read_csv(csvFileName)

    dfTrain = dfTrainRaw
    if isTrain:
        dfTrain = dfTrainRaw.iloc[:, 1:]

    imagesArray = np.zeros((imgHeight, imgWidth))
    for rowIndex, row in dfTrain.iterrows():
        numpyArray = np.zeros((imgHeight, imgWidth))
        for colIndex in range(len(dfTrain.columns)):
            #read pixels into 2d numpy array
            npRow = math.floor(colIndex / imgWidth)
            npCol = colIndex % imgWidth
            temp = dfTrain.iloc[rowIndex, colIndex]
            numpyArray[npRow, npCol] = temp
        imagesArray = np.dstack((imagesArray, numpyArray))
    return imagesArray[:,:,1:]

def createNpImgTargetArray(csvFileName):
    dfTrainRaw = pd.read_csv(csvFileName)

    dfTarget = dfTrainRaw.iloc[:, 0]

    targetArrayNp = np.zeros((1, 10))
    for rowIndex, row in dfTarget.iteritems():
        targetArray = np.zeros((1, 10))
        targetArray[0, dfTarget.iloc[rowIndex]] = 1
        targetArrayNp = np.dstack((targetArrayNp, targetArray))
    return targetArrayNp[:,:,1:]


#allImagesArrayTrain = [] #list of 2d numpy arrays representing pixel values
#targetArrayTrain = []

allImagesArrayTest = [] 
dummyTarget = []

allImagesArrayTrain = createNpImgArray('Data\\trainScratch.csv', True)
targetArrayTrain = createNpImgTargetArray('Data\\trainScratch.csv')

model = CNN.SequentialModel()

#print(allImagesArrayTrain.shape)
#print(allImagesArrayTrain.dtype)
#print(targetArrayTrain.shape)
model.AddInputDataPoints(allImagesArrayTrain)
model.AddTargetVectors(targetArrayTrain)

#print("made it here")
batchSize = 10
numEpochs = 2

model.SetBatchSize(batchSize)
model.SetNumEpochs(numEpochs)
print("")
print("set batch size and num epochs")

model.Train()
#model.CheckGradientNumerically()
#model.Train()

print("trained")

print("")

#print(model.CheckGradientNumerically())

output = model.Predict(allImagesArrayTrain)
print(output)
print("printed")

#cross validate
#n_foldsBoundaries = 6
#n_folds = n_foldsBoundaries - 1
#n_ImagesPerFold = math.floor(len(allImagesArrayTrain) / n_folds)
#for i in range(n_folds):
#    holdOutSet = allImagesArrayTrain[i * n_ImagesPerFold : (i + 1) * n_ImagesPerFold]
#    holdOutTargets = targetArrayTrain[i * n_ImagesPerFold : (i + 1) * n_ImagesPerFold]
#    #holdOutSet = allImagesArrayTrain
#    #holdOutTargets = targetArrayTrain
#    aTrainSet = []
#    aTargetTrain = []
#    bTrainSet = []
#    bTargetTrain = []

    #if i != 0:
    #    aTrainSet = allImagesArrayTrain[: i * n_ImagesPerFold]
    #    aTargetTrain = targetArrayTrain[: i * n_ImagesPerFold]
    #if i != range(n_folds - 1):
    #    bTrainSet = allImagesArrayTrain[(i + 1) * n_ImagesPerFold :]
    #    bTargetTrain = targetArrayTrain[(i + 1) * n_ImagesPerFold :]
#
 #   trainingSet = aTrainSet + bTrainSet
 #   trainingTargets = aTargetTrain + bTargetTrain
    #trainingSet = allImagesArrayTrain
    #trainingTargets = targetArrayTrain

    #cvModel = CNNSlow.Cnn(imgWidth, imgHeight, strideHor, strideVert, fieldWidth, fieldHeight, poolWidth, poolHeight, step)
    #for iteration in range(1):
        #if iteration % 50 == 0:
        #print(iteration)
    #    cvModel.sequentialTrain(trainingSet, trainingTargets)

    #test accuracy on holdOutSet
    #numWrong = 0
    #for imgIndex in range(len(holdOutSet)):
   #    img = holdOutSet[imgIndex]
    #    predictedNum = cvModel.predictNumber(img)
    #    if holdOutTargets[imgIndex][predictedNum] != 1:
    #        numWrong += 1

    #print('Error Rate: ', numWrong / n_ImagesPerFold * 100, '% on validation set ', i)


#cnnTest()
    #model.sequentialTrain(allImagesArrayTrain, targetArrayNp)
#cProfile.run('cnnTest()')

