import numpy as np
import math

class Cnn:
 
    __imgWidth = 0
    __imgHeight = 0
    __strideHor = 0
    __strideVert = 0
    __fieldWidth = 0
    __fieldHeight = 0
    __firstLayerCol = 0
    __firstLayerRows = 0
    __inputWeights = None
    __firstLayerOutput = None
    __firstLayerAdjList = None
    __poolWidth = 0
    __poolHeight = 0
    __firstPoolRows = 0
    __firstPoolCols = 0
    __firstPoolLayer = None
    __step = 0.0
    __outputWeights = None
    __outputClasses = None
    __outputErrors = np.zeros(10)
    __outputWeightDer = None
    __firstLayerWeightDer = None
    __data = None

    def __init__(self, imgW, imgH, strideH, strideV, fieldW, fieldH, poolW, poolH, gradStep):
        self.__imgWidth = imgW
        self.__imgHeight = imgH
        self.__strideHor = strideH
        self.__strideVert = strideV
        self.__fieldWidth = fieldW
        self.__fieldHeight = fieldH
        self.__firstLayerCol = math.floor((self.__imgWidth - self.__fieldWidth) / self.__strideHor + 1)
        self.__firstLayerRows = math.floor((self.__imgHeight - self.__fieldHeight) / self.__strideVert + 1)
        self.__inputWeights = np.ones((self.__firstLayerCol*self.__firstLayerRows, self.__fieldWidth*self.__fieldHeight + 1))
        self.__firstLayerOutput = np.zeros((self.__firstLayerRows, self.__firstLayerCol))
        self.__firstLayerAdjList = [[[] for j in range(self.__firstLayerCol)] for j in range(self.__firstLayerRows)]
        self.__poolWidth = poolW
        self.__poolHeight = poolH
        self.__firstPoolRows = math.floor(self.__firstLayerRows / poolH)
        self.__firstPoolCols = math.floor(self.__firstLayerCol / poolW)
        self.__firstPoolLayer = np.zeros((self.__firstPoolRows, self.__firstPoolCols))
        self.__step = gradStep
        self.__outputWeights = np.zeros((10, self.__firstPoolRows*self.__firstPoolCols + 1))
        self.__outputClasses = np.zeros(10)
        self.__outputWeightDer = np.zeros((10, self.__firstPoolRows*self.__firstPoolCols + 1))
        self.__firstLayerWeightDer = np.zeros((self.__firstLayerCol*self.__firstLayerRows, fieldW*fieldH + 1))

    def __fwdPropConv__(self):
        outputIndex = 0
        for x in range(self.__firstLayerCol):
            for y in range(self.__firstLayerRows):
                fieldArray2D = np.zeros((self.__fieldWidth, self.__fieldHeight))
                for zX in range(self.__fieldWidth):
                    for zY in range(self.__fieldHeight):
                        fieldArray2D[zX, zY] = self.__data[x*self.__fieldWidth + zX, y*self.__fieldHeight + zY]
                fieldArray1D = fieldArray2D.flatten()
                self.__firstLayerOutput[x, y] = 1 / (1 + math.exp(-1 * np.dot(self.__inputWeights[outputIndex, 0:(self.__fieldWidth*self.__fieldHeight)], fieldArray1D.T) - self.__inputWeights[outputIndex, self.__fieldWidth*self.__fieldHeight]))
                outputIndex += 1
        return

    def __pool__(self):
        for x in range(self.__firstPoolCols):
            for y in range(self.__firstPoolRows):
                minVal = 255
                for poolX in range(self.__poolWidth):
                    for poolY in range(self.__poolHeight):
                        if self.__firstLayerOutput[x*self.__poolWidth + poolX, y*self.__poolHeight + poolY] < minVal:
                            minVal = self.__firstLayerOutput[x*self.__poolWidth + poolX, y*self.__poolHeight + poolY]
                            #add [x,y] to the adj list of [x*poolW + poolX, y*poolH + poolY]
                            self.__firstLayerAdjList[x*self.__poolWidth + poolX][y*self.__poolHeight + poolY].append([x,y])
                self.__firstPoolLayer[x,y] = minVal
        return

    def __outputLayer__(self):
        self.__outputClasses = np.zeros(10)

        for num in range(10):
            poolLayer1D = self.__firstPoolLayer.flatten()
            a = -1 * np.dot(self.__outputWeights[num, 0:self.__firstPoolRows*self.__firstPoolCols], poolLayer1D) - self.__outputWeights[num, self.__firstPoolRows*self.__firstPoolCols]
            self.__outputClasses[num] = 1 / (1 + math.exp(a))
        
        return

    def __gradientDescent__(self, weights, weightDer, step):
        if np.size(weights, 0) != np.size(weightDer, 0) or np.size(weights, 1) != np.size(weightDer, 1):
            print('ERROR: __gradientDescent__ weights and weightDer dimensions do not match')
            return
        
        for i in range(np.size(weights,0)):
            for j in range(np.size(weights,1)):
                weights[i, j] -= step * weightDer[i,j]
        return

    def __fwdProp__(self, data):
        self.__data = data
        self.__fwdPropConv__()
        self.__pool__()
        self.__outputLayer__()
        return
    
    def __backprop__(self, target):
        self.__outputLayerBackprop__(target)
        self.__convLayerBackprop__()
        return
    
    def __outputLayerBackprop__(self, target):
        for num in range(10):
            self.__outputErrors[num] = self.__outputClasses[num] - target[num]
            poolLayer1D = self.__firstPoolLayer.flatten()
            for nodei in range(self.__firstPoolRows*self.__firstPoolCols):
                self.__outputWeightDer[num, nodei] = self.__outputErrors[num] * poolLayer1D[nodei]
        return

    def __convLayerBackprop__(self):
        firstLayerErrors = np.zeros((self.__firstLayerRows, self.__firstLayerCol))
        for first_i in range(self.__firstLayerRows):
            for first_j in range(self.__firstLayerCol):
                totalErrorSum = 0
                for k in range(10):
                    errorSum = 0
                    for adj_xy in self.__firstLayerAdjList[first_j][first_i]:
                        errorSum += self.__outputWeights[k, adj_xy[1]*self.__firstPoolRows + adj_xy[0]]
                    errorSum = errorSum * self.__outputErrors[k]
                    totalErrorSum += errorSum
                firstLayerErrors[first_i][first_j] = totalErrorSum * self.__firstLayerOutput[first_i][first_j] * (1 - self.__firstLayerOutput[first_i][first_j])
                for field_i in range(self.__fieldWidth):
                    for field_j in range(self.__fieldHeight):
                        imageVal = self.__data[first_j*self.__fieldWidth + field_i, first_i*self.__fieldHeight + field_j]
                        self.__firstLayerWeightDer[first_i*self.__firstLayerRows + first_j, field_i*self.__fieldWidth + field_j] = imageVal * firstLayerErrors[first_i][first_j]
        return
    
    def __adjustWeights__(self):
        self.__gradientDescent__(self.__outputWeights, self.__outputWeightDer, self.__step)
        self.__gradientDescent__(self.__inputWeights, self.__firstLayerWeightDer, self.__step)
        return

    def sequentialTrain(self, grayscaleImageArray, targetLabelArray):
        numImages = np.size(targetLabelArray, 0)
        for i in range(numImages):
            self.__fwdProp__(grayscaleImageArray[i])
            self.__backprop__(targetLabelArray[i])
            self.__adjustWeights__()
            #print(self.__firstLayerWeightDer)
            #print(self.__outputWeightDer)
            #print('input weights')
            #for i in self.__inputWeights:
            #    print(i)
        return

    def predictNumber(self, img):
        self.__fwdProp__(img)
        maxNum = 0
        maxProb = 0.0
        for num in range(9):
            if self.__outputClasses[num] > maxProb:
                maxProb = self.__outputClasses[num]
                maxNum = num
        return maxNum