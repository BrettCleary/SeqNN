import numpy as np
import pandas as pd
import SeqNN
import SeqNNTests
import Datasets as ds

mnistData = np.true_divide(ds.mnistTrainData, 255.0)
mnistTargets = ds.mnistTrainTargets

mnistTrainData = mnistData[:,:,:-500]
mnistTrainTargets = mnistTargets[:,:,:-500]

mnistValidationData = mnistData[:,:,-500:-250]
mnistValidationTargets = mnistTargets[:,:,-500:-250]

mnistTestData = mnistData[:,:,-250:]
mnistTestTargets = mnistTargets[:,:,-250:]

#MNIST Model
model = SeqNN.SeqNN([
    SeqNN.Conv2DLayer(8, 8, 1, 1, 0, 0.5, 0.9, 
    SeqNN.Regularizer.SOFTWEIGHTSHARING, 0.001, 2, 0.1, 0.001, 0.03),
    SeqNN.Pool2DLayer(True, 2, 2),
    SeqNN.DenseLayer(0.02, 1, 10, 0.9, SeqNN.ActFxn.SOFTMAX, SeqNN.Regularizer.NONE, 0.01)
])

model.trainNN(1, 20, mnistTrainData, mnistTrainTargets, mnistValidationData, mnistValidationTargets, True, 0.1, 2, 5)

errorRate = model.calcTestErrorRate(mnistTestData, mnistTestTargets)
print("\nThe error rate for test dataset is ", "%.1f" % round(errorRate, 2), "\n")