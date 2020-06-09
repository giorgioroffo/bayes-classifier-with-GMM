# ROFFO GIORGIO

from sklearn.mixture import BayesianGaussianMixture
import numpy as np


def clamp_sample(x):
  print('clamp_sample(x): ')
  print (x.shape)
  x = np.minimum(x, 1)
  print(x.shape)
  x = np.maximum(x, 0)
  print(x.shape)
  return x


class BayesGMM:
  def fit(self, X, Y):
    # assume classes are numbered 0...K-1
    self.K = len(set(Y))

    self.gaussians = []
    self.p_y = np.zeros(self.K)
    for k in range(self.K):
      print("Fitting gmm", k)
      Xk = X[Y == k]
      self.p_y[k] = len(Xk)
      gmm = BayesianGaussianMixture(10)
      gmm.fit(Xk)
      self.gaussians.append(gmm)
    # normalize p(y)
    self.p_y /= self.p_y.sum()

  def sample_given_y(self, y):
    gmm = self.gaussians[y]
    sample = gmm.sample()
    # sample[0] the sample
    # sample[1] which cluster sample[0] came from
    mean = gmm.means_[sample[1]]
    return clamp_sample(sample[0].reshape(28, 28)), mean.reshape(28, 28)

  def sample(self):
    y = np.random.choice(self.K, p=self.p_y)
    return clamp_sample(self.sample_given_y(y))

