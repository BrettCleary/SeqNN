import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import cProfile
import re
import CNN
import SeqNN
import SeqNNTests
import Datasets as ds

mnistData = np.true_divide(ds.mnistTrainData, 255.0)
mnistTargets = ds.mnistTrainTargets

mnistTrainData = mnistData[:,:,:-7500]
mnistTrainTargets = mnistTargets[:,:,:-7500]

mnistValidationData = mnistData[:,:,-7500:-5000]
mnistValidationTargets = mnistTargets[:,:,-7500:-5000]

mnistTestData = mnistData[:,:,-5000:]
mnistTestTargets = mnistTargets[:,:,-5000:]

#MNIST Model
model = SeqNN.SeqNN([
    CNN.Conv2DLayer(7, 7, 1, 1, 0, 0.5, 0.9),
    CNN.Pool2DLayer(True, 2, 2),
    CNN.DenseLayer(0.02, 1, 10, 0.9, 1)
])

model.trainNN(10, 10, mnistTrainData, mnistTrainTargets, mnistValidationData, mnistValidationTargets, True, 0.1, 1, 5)

errorRate = model.calcTestErrorRate(mnistTestData, mnistTestTargets)
print("\nThe error rate for test dataset is ", errorRate, "\n")