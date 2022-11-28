import psutil
import onnxModel as onnx
import tflite
from torchvision import datasets, transforms
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
tflite_beginVM = dict(psutil.virtual_memory()._asdict())
tflite_result = tflite.run(dataset2)
tflite_endVM = dict(psutil.virtual_memory()._asdict())

onnx_beginVM = dict(psutil.virtual_memory()._asdict())
onnx_result = onnx.run(dataset2)
onnx_endVM = dict(psutil.virtual_memory()._asdict())

print("tflite accuracy: " + str(tflite_result))
print("tflite begin " + str(onnx_beginVM))
print("tflite end " + str(onnx_endVM))

print("onnx accuract: " + str(onnx_result))
print("onnx begin " + str(onnx_beginVM))
print("onnx end " + str(onnx_endVM))

# new data
# for filepath in os.walk("newImages/")
#     x=imageprepare('newImages/seven.png')#file path here
#     print(len(x))# mnist IMAGES are 28x28=784 pixels