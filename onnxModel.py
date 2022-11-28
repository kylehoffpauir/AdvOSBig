from __future__ import print_function
import argparse
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import IntTensor
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import onnxruntime as rt
import math
import warnings
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import mxnet as mx
import onnx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.contrib import onnx as onnx_mxnet

import tflite


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def run(dataset2):
    results = []
    model = Net()
    state_dict = torch.load("mnist_cnn.pt")
    model.load_state_dict(state_dict)
    ##<All keys matched successfully>
    correct = 0
    #image, label = dataset2[random.randint(0, 9999)]
    for image, label in dataset2:
        # image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        output = model(image)
        output = torch.argmax(output)
        #print(output, label, output == label)
        results.append(output==label)
    return len(results) / len(dataset2)

def novelRun(imgPath):
    image = keras.preprocessing.image.load_img(
        imgPath,
        color_mode = 'grayscale',
        target_size=(28, 28),
        interpolation='bilinear'
    )

    input_image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, 0)
    # Show the pre-processed input image
    # using the show_sample from tflite
    tflite.show_sample(input_image, ['Input Image'], 1)
    results = []
    model = Net()
    state_dict = torch.load("mnist_cnn.pt")
    model.load_state_dict(state_dict)
    correct = 0
    input_image = torch.tensor(input_image)
    image = input_image.unsqueeze(0)
    output = model(image)
    conf_score = torch.nn.functional.softmax(output, dim=1)
    output = IntTensor.item(torch.argmax(output))
    return output, conf_score
