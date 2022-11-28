import os

import psutil
import onnxModel as onnx
import tflite
from torchvision import datasets, transforms
import numpy as np
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import time


# gives a single float value
psutil.cpu_percent()
# gives an object with many fields
psutil.virtual_memory()
# you can convert that object to a dictionary
dict(psutil.virtual_memory()._asdict())

# make a random dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
print(len(dataset2))
print("TFLITE")
tflite_beginVM = psutil.cpu_percent(3)
print("begin time: ")
print(time.time())
tflite_result, tflite_avgconf = tflite.run(dataset2)
print("calculating cpu percentage")
print(psutil.cpu_percent(3))
print("end time: ")
print(time.time())

print("ONNX")
onnx_beginVM = psutil.cpu_percent(3)
print("begin time")
print(time.time())
onnx_result = onnx.run(dataset2)
print("calculating cpu percentage")
onnx_endVM = dict(psutil.virtual_memory()._asdict())
print("end time")
print(time.time())


print("tflite accuracy: " + str(tflite_result))
print("tflite conf: " + str(tflite_avgconf))
print("tflite begin " + str(onnx_beginVM))
print("tflite end " + str(onnx_endVM))

print("onnx accuracy: " + str(onnx_result))
print("onnx begin " + str(onnx_beginVM))
print("onnx end " + str(onnx_endVM))


print("Novel small test set: ")
# new data
tf_score = 0.0
onnx_score = 0.0
total = 0
for root, dir, imgPath in os.walk("novelset/"):
    for image in imgPath:
        actual = str(image)[0]
        print("on image: " + image)
        tflite_result, tflite_conf = tflite.novelRun(root+image)
        if str(tflite_result) == str(actual):
            tf_score += 1.0
        onnx_result, onnx_conf = onnx.novelRun(root+image)
        if str(onnx_result) == str(actual):
            onnx_score += 1.0
        print("TFLITE\nresult = " + str(tflite_result) + "\n confidence: " + str(tflite_conf))
        print("\nONNX\nresult= " + str(onnx_result) + "\n")
        total += 1.0
print("\n\n\nTF SCORE: " + str((tf_score / total) * 100))
print("ONNX SCORE: " + str((onnx_score / total) * 100))