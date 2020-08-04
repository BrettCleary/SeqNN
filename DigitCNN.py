import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import CNN

imgWidth = 28
imgHeight = 28
strideHor = 4
strideVert = 4
fieldWidth = 4
fieldHeight = 4
poolWidth = 2
poolHeight = 2
step = 0.01

model = CNN.Cnn(imgWidth, imgHeight, strideHor, strideVert, fieldWidth, fieldHeight, poolWidth, poolHeight, step)

#imA = Image.open("zero.jpg").convert('L')
#data = np.array( imA, dtype='uint8' )

#target = np.zeros(10)
#target[1] = 1

#imgArray = [data]
#targetArray = [target]

def createNpImgArray(csvFileName, isTrain, allImagesArray, targetArrayNp):
    dfTrain = pd.read_csv(csvFileName)
    for rowIndex, row in dfTrain.iterrows():
        numpyArray = np.zeros((imgWidth, imgHeight))
        for colIndex in range(len(dfTrain.columns)):
            #read pixels into 2d numpy array
            if isTrain:
                npRow = (colIndex - 1) / imgWidth
                npCol = (colIndex - 1) % imgWidth
            else:
                npRow = colIndex / imgWidth
                npCol = colIndex % imgWidth
            numpyArray[npRow, npCol] = dfTrain[rowIndex, colIndex]
        allImagesArray.append(numpyArray)
        targetArray = np.zeros(10)
        if isTrain:
            targetArray[dfTrain[rowIndex, 0]] = 1
            targetArrayNp.append(targetArray)
    return


allImagesArrayTrain = [] #list of 2d numpy arrays representing pixel values
targetArrayTrain = []

allImagesArrayTest = [] 
dummyTarget = []

createNpImgArray('Data\\train.csv', True, allImagesArrayTrain, targetArrayTrain)
createNpImgArray('Data\\test.csv', False, allImagesArrayTest, dummyTarget)


#cross validate
n_folds = 5
n_ImagesPerFold = math.floor(len(allImagesArrayTrain) / n_folds)
for i in range(n_folds - 1):
    holdOutSet = allImagesArrayTrain[i * n_ImagesPerFold : (i + 1) * n_ImagesPerFold]
    holdOutTargets = targetArrayTrain[i * n_ImagesPerFold : (i + 1) * n_ImagesPerFold]
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
    
    cvModel = CNN.Cnn(imgWidth, imgHeight, strideHor, strideVert, fieldWidth, fieldHeight, poolWidth, poolHeight, step)
    cvModel.sequentialTrain(trainingSet, trainingTargets)

    #test accuracy on holdOutSet
    numWrong = 0
    for imgIndex in range(len(holdOutSet)):
        img = holdOutSet[imgIndex]
        if holdOutTargets[imgIndex, cvModel.predictNumber(img)] != 1:
            numWrong += 1

    print('Error Rate: ', numWrong / n_ImagesPerFold, ' on validation set ', i)



#model.sequentialTrain(allImagesArrayTrain, targetArrayNp)

