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
import Datasets as ds

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


csvFileNameTrain = "Data//MNIST_trainShort.csv"
#csvFileNameTest = "Data//MNIST_testShort.csv"
csvFileNameTest = "Data//MNIST_trainShort.csv"

mnistTrain = np.true_divide(ds.mnistTrainData, 255.0)
mnistTargets = ds.mnistTrainTargets

testDData = np.true_divide(ds.testD_trainData, 255.0)
testDTargets = ds.testD_trainTargets

#print(testDData[5,:,0])
#print(testDTargets)

#SeqNNTests.convertAllCsvToNpy()

#SeqNNTests.RunAllTests()
#SeqNNTests.RunAllTests(True, 3)
#SeqNNTests.RunOneCnnTest(1)
#SeqNNTests.RunMNISTTest()

print("tests finished")

def CNNPoolTest(trainData, trainTargets, inputRows, inputCols, numOutputClasses, 
batchSize, numEpochs, cnnStepSize, denseLayerStepSize, testData, testTargets):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.setTrainData(trainData, trainTargets)
    print("starting to read data")
    cnnLayer = CNN.Conv2DLayer(2, 2, 2, 2, 0, cnnStepSize)
    model.addLayer(cnnLayer)
    poolLayer = CNN.Pool2DLayer(True, 2, 2)
    model.addLayer(poolLayer)
    denseLayer = CNN.DenseLayer(denseLayerStepSize, 1, numOutputClasses)
    model.addLayer(denseLayer)
    print("starting to train nn")
    model.setValidationDataAndTargets(trainData, trainTargets)
    model.trainNN(batchSize, numEpochs, True, 0.1, 1, 5)

    #output = model.predict(csvFileNameTest, True)
    print("starting to calc test error rate")
    errorRate = model.calcTestErrorRate(testData, testTargets)
    print("\nThe error rate for test dataset is ", errorRate, "\n")

def CNNDenseTest(trainData, trainTargets, inputRows, inputCols, numOutputClasses, 
batchSize, numEpochs, cnnStepSize, denseLayerStepSize, testData, testTargets):
    model = SeqNN.SeqNN(inputRows, inputCols, numOutputClasses)
    model.setTrainData(trainData, trainTargets)
    print("starting to read data")
    denseLayer = CNN.DenseLayer(denseLayerStepSize, 1, 128)
    model.addLayer(denseLayer)
    #poolLayer = CNN.Pool2DLayer(True, 2, 2)
    #model.addLayer(poolLayer)
    denseLayer = CNN.DenseLayer(denseLayerStepSize, 1, numOutputClasses)
    model.addLayer(denseLayer)
    print("starting to train nn")
    model.setValidationDataAndTargets(trainData, trainTargets)
    model.trainNN(batchSize, numEpochs, True, 0.1, 1, 100)

    #output = model.predict(csvFileNameTest, True)
    print("starting to calc test error rate")
    errorRate = model.calcTestErrorRate(testData, testTargets)
    print("\nThe error rate for test dataset is ", errorRate, "\n")
#def MNIST_Test()

CNNPoolTest(mnistTrain, mnistTargets, 28, 28, 10, 1, 10, 1, 0.04, mnistTrain, mnistTargets)
#CNNDenseTest(mnistTrain, mnistTargets, 28, 28, 10, 1, 5, 1, 0.04, mnistTrain, mnistTargets)
#CNNPoolTest(testDData, testDTargets, 28, 28, 10, 1, 100, 1, 0.04, testDData, testDTargets)

#SeqNNTests.Simple2DCnnPoolTest(csvFileNameTrain, 28, 28, 10, 1, 10000, 1, 0.04)

#SeqNNTests.convertAllCsvToNpy()
#SeqNNTests.convertMNIST_Arrays()




#model = SeqNN.SeqNN(2, 4, 4, csvFileName)
#layer = CNN.Conv2DLayer(fieldHeight, fieldWidth, strideHor, strideVert, padding)
#model.__addLayer__(layer)





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

