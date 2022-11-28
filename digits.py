from sklearn.datasets import load_digits
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Load Data
digits = load_digits()

# Create features
X = digits.data

# Create target
y = digits.target

# Check shape of X
X.shape

(1797, 64)

# Each image is 8px * 8px that's why 64 pixels
print(digits.images[4])

# array([[  0.,   0.,   0.,   1.,  11.,   0.,   0.,   0.],
#        [  0.,   0.,   0.,   7.,   8.,   0.,   0.,   0.],
#        [  0.,   0.,   1.,  13.,   6.,   2.,   2.,   0.],
#        [  0.,   0.,   7.,  15.,   0.,   9.,   8.,   0.],
#        [  0.,   5.,  16.,  10.,   0.,  16.,   6.,   0.],
#        [  0.,   4.,  15.,  16.,  13.,  16.,   1.,   0.],
#        [  0.,   0.,   0.,   3.,  15.,  10.,   0.,   0.],
#        [  0.,   0.,   0.,   2.,  16.,   4.,   0.,   0.]])

# Target Value
y[59]

3

# Let's see how the image looks like
plt.gray()
plt.matshow(digits.images[59])
plt.show()

