from .param import Param

import tensorflow as tf
import numpy as np
from gpflow import transforms
from gpflow import settings

float_type = settings.float_type
jitter_level = settings.jitter

class Kernel:
    def __init__(self,input_dim, variance, lengthscales, ARD, name="kernel"):
        self.input_dim = input_dim
        self.ARD = ARD
        if self.ARD is True:
            lengthscales *= np.ones([1,self.input_dim])
        with tf.name_scope(name):
            lengthscales = Param(lengthscales,
                                      transform=transforms.positive,
                                      name=name+"_lengthscale")
            variance     = Param(variance,
                                      transform=transforms.positive,
                                      name=name+"_variance")
        self.lengthscales = lengthscales()
        self.variance = variance()

    def square_dist(self,X, X2=None):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), -1)
        if X2 is None:
            return -2. * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2. * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

class RBF(Kernel):
    def __init__(self,input_dim, variance=1., lengthscales=1., ARD=False, name="rbf"):
        super().__init__(input_dim = input_dim,
                         variance = variance,
                         lengthscales = lengthscales,
                         ARD = ARD,
                         name = name)

    def K(self,X,X2=None):
        if X2 is None:
            r2 = self.square_dist(X)
        else:
            r2 = self.square_dist(X, X2)
        return self.variance * tf.exp(-r2/2.)

    def Ksymm(self,X):
        r2 = self.square_dist(X)
        return self.variance * tf.exp(-r2/2.)

    def Kdiag(self,X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class Matern52(Kernel):
    def __init__(self,input_dim, variance=1., lengthscales=1., ARD=False, name="matern52"):
        super().__init__(input_dim = input_dim,
                         variance = variance,
                         lengthscales = lengthscales,
                         ARD = ARD,
                         name = name)

    def K(self,X,X2=None):
        sqrt5 = np.sqrt(5.)
        if X2 is None:
            # clipping around the (single) float precision which is ~1e-45
            r = tf.sqrt(tf.maximum(self.square_dist(X), 1e-40))
        else:
            r = tf.sqrt(tf.maximum(self.square_dist(X, X2), 1e-40))
        return self.variance * (1. + sqrt5 * r + 5./3.*tf.square(r)) * tf.exp(-sqrt5 * r)

    def Ksymm(self,X):
        sqrt5 = np.sqrt(5.)
        r = tf.sqrt(tf.maximum(self.square_dist(X), 1e-40))
        return self.variance * (1. + sqrt5 * r + 5./3.*tf.square(r)) * tf.exp(-sqrt5 * r)

    def Kdiag(self,X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class Matern32(Kernel):
   def __init__(self,input_dim, variance=1., lengthscales=1., ARD=False, name="matern32"):
       super().__init__(input_dim = input_dim,
                        variance = variance,
                        lengthscales = lengthscales,
                        ARD = ARD,
                        name = name)

   def K(self,X,X2=None):
       sqrt3 = np.sqrt(3.)
       if X2 is None:
           # clipping around the (single) float precision which is ~1e-45
           r = tf.sqrt(tf.maximum(self.square_dist(X), 1e-40))
       else:
           r = tf.sqrt(tf.maximum(self.square_dist(X, X2), 1e-40))
       return self.variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)

   def Ksymm(self,X):
       sqrt3 = np.sqrt(3.)
       r = tf.sqrt(tf.maximum(self.square_dist(X), 1e-40))
       return self.variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)

   def Kdiag(self,X):
       return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class Matern12(Kernel):
   def __init__(self,input_dim, variance=1., lengthscales=1., ARD=False, name="matern52"):
       super().__init__(input_dim = input_dim,
                        variance = variance,
                        lengthscales = lengthscales,
                        ARD = ARD,
                        name = name)

   def K(self,X,X2=None):
       if X2 is None:
           # clipping around the (single) float precision which is ~1e-45
           r = tf.sqrt(tf.maximum(self.square_dist(X), 1e-40))
       else:
           r = tf.sqrt(tf.maximum(self.square_dist(X, X2), 1e-40))
       return self.variance * tf.exp(-r)

   def Ksymm(self,X):
       r = tf.sqrt(tf.maximum(self.square_dist(X), 1e-40))
       return self.variance * tf.exp(-r)

   def Kdiag(self,X):
       return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

