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

mnistTrain = np.true_divide(ds.mnistTrainData, 255.0)
mnistTargets = ds.mnistTrainTargets

#MNIST Model
model = SeqNN.SeqNN([
    CNN.Conv2DLayer(2, 2, 2, 2, 0, 0.5, 0.9),
    CNN.Pool2DLayer(True, 2, 2),
    CNN.DenseLayer(0.02, 1, 10, 0.9)
])

model.trainNN(10, 10, mnistTrain, mnistTargets, mnistTrain, mnistTargets, True, 0.1, 1, 5)

errorRate = model.calcTestErrorRate(mnistTrain, mnistTargets)
print("\nThe error rate for test dataset is ", errorRate, "\n")