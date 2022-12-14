import math

import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageFilter

# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from tensorflow.lite.python.lite import TFLiteConverter

# Helper function to display digit images


def show_sample(images, labels, sample_count=25):
    # Create a square with can fit {sample_count} images
    grid_count = math.ceil(math.ceil(math.sqrt(sample_count)))
    grid_count = min(grid_count, len(images), len(labels))

    plt.figure(figsize=(2*grid_count, 2*grid_count))
    for i in range(sample_count):
        plt.subplot(grid_count, grid_count, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.xlabel(labels[i])
    plt.show()

def run(dataset2):
    results = []
    confidence = []
    for image, label in dataset2:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="model_quant_tl.tflite")
        #allocate the tensors
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
        interpreter.invoke()
        output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
        interpreter.get_output_details()
        # Print the model's classification result
        digit = np.argmax(output)
        #print('Predicted Digit: %d\nConfidence: %f' % (digit, output[digit]))
        confidence.append(output[digit])
        results.append(output==label)
    return (len(results) / len(dataset2)), (sum(confidence) / len(confidence))


def novelRun(imgPath):
    image = keras.preprocessing.image.load_img(
        imgPath,
        color_mode = 'grayscale',
        target_size=(28, 28),
        interpolation='bilinear'
    )
    input_image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, 0)
    show_sample(input_image, ['Input Image'], 1)
    interpreter = tf.lite.Interpreter(model_path="model_quant_tl.tflite")
    interpreter.allocate_tensors()
    #Run the inference
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
    interpreter.invoke()
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
    # Print the model's classification result
    digit = np.argmax(output)
    # print('Predicted Digit: %d\nConfidence: %f' % (digit, output[digit]))
    return digit, output[digit]

