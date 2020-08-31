import numpy as np

#[row, col, imageIndex]
mnistTrainData = np.load('Data//MNIST_trainData.npy')

mnistTrainTargets = np.load('Data//MNIST_trainTargets.npy')

#[row, col, imageIndex]
mnistTestData = np.load('Data//MNIST_testData.npy')


testD_trainData = np.load("Data//TestD_MNIST_tenDigits" + "_trainData.npy")
testD_trainTargets = np.load("Data//TestD_MNIST_tenDigits" + "_trainTargets.npy")