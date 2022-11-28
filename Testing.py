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
tflite_beginVM = psutil.cpu_percent(3)
print(time.time())
tflite_result, tflite_avgconf = tflite.run(dataset2)
print(psutil.cpu_percent(3))
print(time.time())

onnx_beginVM = psutil.cpu_percent(3)
print(time.time())
onnx_result = onnx.run(dataset2)
onnx_endVM = dict(psutil.virtual_memory()._asdict())
print(time.time())


print("tflite accuracy: " + str(tflite_result))
print("tflite conf: " + str(tflite_avgconf))
print("tflite begin " + str(onnx_beginVM))
print("tflite end " + str(onnx_endVM))

print("onnx accuract: " + str(onnx_result))
print("onnx begin " + str(onnx_beginVM))
print("onnx end " + str(onnx_endVM))



# new data
# for filepath in os.walk("newImages/")

#novelData=imageProcess.imageprepare('newImages/seven.png')#file path here
tflite_result, tflite_conf = tflite.novelRun('newImages/twoblack.png')
onnx_result = onnx.novelRun('newImages/twoblack.png')
#     print(len(x))# mnist IMAGES are 28x28=784 pixels