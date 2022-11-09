import theano
import theano.tensor as T
import lasagne
import numpy as np
import sklearn.datasets
#from __future__ import print_function
import os
import matplotlib.pyplot as plt
import IPython.display
#IPython.display.Image("http://static-vegetariantimes.s3.amazonaws.com/wp-content/uploads/2009/03/10851medium.jpg")

import warnings
warnings.filterwarnings('ignore', module='lasagne')

# Generate synthetic data
N_CLASSES = 4
X, y = sklearn.datasets.make_classification(n_features=2, n_redundant=0,
 n_classes=N_CLASSES, n_clusters_per_class=1)
# Convert to theano floatX
X = X.astype(theano.config.floatX)
# Labels should be ints
y = y.astype('int32')
# Make a scatter plot where color encodes class
plt.scatter(X[:, 0], X[:, 1], c=y)

# First, construct an input layer.
# The shape parameter defines the expected input shape, which is just the shape of our data matrix X.
l_in = lasagne.layers.InputLayer(shape=X.shape)

# We'll create a network with two dense layers: A tanh hidden layer and a softmax output layer.
l_hidden = lasagne.layers.DenseLayer(
 # The first argument is the input layer
 l_in,
 # This defines the layer's output dimensionality
 num_units=10,
 # Various nonlinearities are available
 nonlinearity=lasagne.nonlinearities.tanh)
# For our output layer, we'll use a dense layer with a softmax nonlinearity.
l_output = lasagne.layers.DenseLayer(
 l_hidden, num_units=N_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)

net_output = lasagne.layers.get_output(l_output)

# As a loss function, we'll use Lasagne's categorical_crossentropy function.
# This allows for the network output to be class probabilities,
# but the target output to be integers denoting the class.
true_output = T.ivector('true_output')
loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output, true_output))

# Retrieving all parameters of the network is done using get_all_params,
# which recursively collects the parameters of all layers connected to the provided layer.
all_params = lasagne.layers.get_all_params(l_output)
# Now, we'll generate updates using Lasagne's SGD function
updates = lasagne.updates.sgd(loss, all_params, learning_rate=1)
# Finally, we can compile Theano functions for training and computing the output.
# Note that because loss depends on the input variable of our input layer,
# we need to retrieve it and tell Theano to use it.
train = theano.function([l_in.input_var, true_output], loss, updates=updates)
get_output = theano.function([l_in.input_var], net_output)

# Train (bake?) for 100 epochs
for n in range(100):
    train(X, y)

# Compute the predicted label of the training data.
# The argmax converts the class probability output to class label
y_predicted = np.argmax(get_output(X), axis=1)
# Plot incorrectly classified points as black dots
plt.scatter(X[:, 0], X[:, 1], c=(y != y_predicted), cmap=plt.cm.gray_r)
# Compute and display the accuracy
plt.title("Accuracy: {}%".format(100*np.mean(y == y_predicted)))
plt.show()




