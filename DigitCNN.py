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
import SeqNN
import SeqNNTests

#def cnnTest():
imgWidth = 8
imgHeight = 2
strideHor = 4
strideVert = 4
fieldWidth = 4
fieldHeight = 4
poolWidth = 2
poolHeight = 2
step = 0.01
numOutputClasses = 4
padding = 0

#model = CNNSlow.Cnn(imgWidth, imgHeight, strideHor, strideVert, fieldWidth, fieldHeight, poolWidth, poolHeight, step)


csvFileName = "Data//trainScratchSimple3.csv"

SeqNNTests.RunAllTests(csvFileName)

model = SeqNN.SeqNN(2, 4, 4, csvFileName)
layer = CNN.Conv2DLayer(fieldHeight, fieldWidth, strideHor, strideVert, padding)
model.__addLayer__(layer)





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

