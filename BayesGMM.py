# ROFFO GIORGIO

from __future__ import print_function, division
from builtins import range, input

import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Classifier import BayesGMM
from utils import get_mnist


# Load a dataset: in this case I'm working on MNIST
X, Y = get_mnist()

# Initialize the Bayesian Classifier wit Gaussian Mixture Models
clf = BayesGMM()

# Train the classifier
clf.fit(X, Y)


# show one sample for each class
# also show the mean image learned
for k in range(clf.K):

  sample, mean = clf.sample_given_y(k)

  plt.subplot(1,2,1)
  plt.imshow(sample, cmap='gray')
  plt.title("Sample")
  plt.subplot(1,2,2)
  plt.imshow(mean, cmap='gray')
  plt.title("Mean")
  plt.show()

# generate a random sample
sample, mean = clf.sample()
plt.subplot(1,2,1)
plt.imshow(sample, cmap='gray')
plt.title("Random Sample from Random Class")
plt.subplot(1,2,2)
plt.imshow(mean, cmap='gray')
plt.title("Corresponding Cluster Mean")
plt.show()
