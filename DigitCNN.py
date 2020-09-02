import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import cProfile
import re
import SeqNN
import SeqNNTests
import Datasets as ds
import time

mnistData = np.true_divide(ds.mnistTrainData, 255.0)
mnistTargets = ds.mnistTrainTargets

mnistTrainData = mnistData[:,:,:-7500]
mnistTrainTargets = mnistTargets[:,:,:-7500]

mnistValidationData = mnistData[:,:,-7500:-5000]
mnistValidationTargets = mnistTargets[:,:,-7500:-5000]

mnistTestData = mnistData[:,:,-5000:]
mnistTestTargets = mnistTargets[:,:,-5000:]


t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

#MNIST Model
model = SeqNN.SeqNN([
    SeqNN.Conv2DLayer(7, 7, 1, 1, 0, 0.5, 0.9, 
    SeqNN.Regularizer.SOFTWEIGHTSHARING, 0.001, 2, 0.1, 0.001, 0.03),
    SeqNN.Pool2DLayer(True, 2, 2),
    SeqNN.DenseLayer(0.02, 1, 10, 0.9, SeqNN.ActFxn.SOFTMAX, SeqNN.Regularizer.NONE, 0.01)
])

model.trainNN(1, 10, mnistTrainData, mnistTrainTargets, mnistValidationData, mnistValidationTargets, True, 0.1, 1, 5)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

errorRate = model.calcTestErrorRate(mnistTestData, mnistTestTargets)
print("\nThe error rate for test dataset is ", errorRate, "\n")

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)