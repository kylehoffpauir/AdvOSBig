import tensorflow as tf
from tensorflow import keras

# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

train_df = pd.read_csv('IOT-temp.csv')
train_df.columns = [column.lower() for column in train_df.columns]
print(train_df.columns)

test_df = pd.read_csv('IOT-temp.csv')
test_df.columns = [column.lower() for column in test_df.columns]
print(test_df.columns)

data = pd.read_csv('IOT-temp.csv')
print(data.head())

print(train_df.head())


# https://www.tensorflow.org/lite/models/convert