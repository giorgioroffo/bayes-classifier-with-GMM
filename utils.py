# ROFFO GIORGIO

import pandas as pd

from sklearn.utils import shuffle

def get_mnist(limit=None):

  print("Reading MNIST and scaling data...")
  df = pd.read_csv('dataset/train.csv')
  data = df.values
  X = data[:, 1:] / 255.0 # data is from 0..255
  Y = data[:, 0]
  X, Y = shuffle(X, Y)
  if limit is not None:
    X, Y = X[:limit], Y[:limit]
  return X, Y
