from .param import Param
from .utils import reparameterize

import tensorflow as tf
import numpy as np
from gpflow.conditionals import conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow import transforms
from gpflow import settings
from .settings import Settings
float_type = settings.float_type
SEED = Settings().tf_op_random_seed


class Layer:
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        raise NotImplementedError

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,U,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,U,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean is (S,U,N,D_out) var is (S,N,D_out)

        :param X:  The input locations (S,U,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,U,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]

        f = lambda X: self.conditional_ND(X, full_cov=full_cov)
        mean, var = tf.map_fn(f,X, dtype=(settings.float_type,settings.float_type))

        if full_cov is True:
            return tf.reshape(tf.stack(mean),[S,N,self.num_outputs]), tf.reshape(tf.stack(var),[S,N,N,self.num_outputs])
        else:
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whiteed sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whiteed representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = self.num_outputs

        if z is None:
            #z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
            z = tf.truncated_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var

class SVGP_Layer(Layer):
    def __init__(self, kern, Z, num_inducing, num_outputs, mean_function=None, white=True):
        self.white = white
        self.kern = kern
        self.num_inputs = kern.input_dim
        self.num_outputs = num_outputs
        self.num_inducing = num_inducing
        self.q_diag = False
        Um = np.zeros((self.num_inducing, self.num_outputs))
        Us_sqrt = np.ones((self.num_inducing, self.num_outputs)) if self.q_diag else np.array([np.eye(self.num_inducing) for _ in range(self.num_outputs)])
        with tf.name_scope("inducing"):
            self.Z  = Param(Z, name="z")()
            self.Um  = Param(Um, name="u")()
            if self.q_diag:
                self.Us_sqrt = Param(Us_sqrt, transforms.positive, name="u_variance")()
            else:
                self.Us_sqrt = Param(Us_sqrt, transforms.LowerTriangular(self.num_inducing,self.num_outputs), name="u_variance")()

        self.Ku = self.kern.Ksymm(self.Z) + tf.eye(tf.shape(self.Z)[0],dtype=self.Z.dtype)*settings.jitter
        self.Lu = tf.cholesky(self.Ku)
        self.mean_function = mean_function

    def base_conditional(self, Kmn, Kmm, Knn, f, full_cov=False, q_sqrt=None, white=True):
        # compute kernel stuff
        num_func = tf.shape(f)[1]  # R
        Lm = tf.cholesky(Kmm)

        # Compute the projection matrix A
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

        # compute the covariance due to the conditioning
        if full_cov:
            fvar = Knn - tf.matmul(A, A, transpose_a=True)
            fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # R x N x N
        else:
            fvar = Knn - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(fvar[None, :], [num_func, 1])  # R x N

        # another backsubstitution in the unwhitened case
        if not white:
            A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

        # construct the conditional mean
        fmean = tf.matmul(A, f, transpose_a=True)

        if q_sqrt is not None:
            if q_sqrt.get_shape().ndims == 2:
                LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
            elif q_sqrt.get_shape().ndims == 3:
                L = q_sqrt
                A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
                LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
            else:  # pragma: no cover
                raise ValueError("Bad dimension for q_sqrt: %s" %
                                 str(q_sqrt.get_shape().ndims))
            if full_cov:
                fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
            else:
                fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N

        if not full_cov:
            fvar = tf.transpose(fvar)  # N x R

        return fmean, fvar  # N x R, R x N x N or N x R

    def conditional_ND(self, X, full_cov=False):
        Kuf = self.kern.K(self.Z, X)
        Kff = self.kern.Ksymm(X) if full_cov else self.kern.Kdiag(X)
        mean, var = self.base_conditional(Kmn=Kuf, Kmm=self.Ku, Knn=Kff, f=self.Um,
                                          full_cov=full_cov, q_sqrt=self.Us_sqrt, white=self.white)
        if self.mean_function is not None:
            mean += self.mean_function(X)
        return mean, var

    def KL(self):
        Ku = None if self.white else self.Ku
        return gauss_kl(self.Um, self.Us_sqrt, Ku)


class SVGP_Z_Layer(SVGP_Layer):
    def __init__(self, kern, num_inducing, num_outputs, mean_function=None, white=True):
        self.white = white
        self.kern = kern
        self.num_inputs = kern.input_dim
        self.num_outputs = num_outputs
        self.num_inducing = num_inducing
        self.q_diag = False
        Um = np.zeros((self.num_inducing, self.num_outputs))
        Us_sqrt = np.ones((self.num_inducing, self.num_outputs)) if self.q_diag else np.array([np.eye(self.num_inducing) for _ in range(self.num_outputs)])
        with tf.name_scope("inducing"):
            self.Um  = Param(Um, name="u")()
            if self.q_diag:
                self.Us_sqrt = Param(Us_sqrt, transforms.positive,name="u_variance")()
            else:
                self.Us_sqrt = Param(Us_sqrt, transforms.LowerTriangular(self.num_inducing,self.num_outputs),name="u_variance")()
        self.mean_function = mean_function

    def initialize_Z(self, Z_trans):
        self.num_inducing = tf.shape(Z_trans)[0]
        self.Z = Z_trans
        self.Ku = self.kern.Ksymm(self.Z) + tf.eye(tf.shape(self.Z)[0],dtype=self.Z.dtype)*settings.jitter
        self.Lu = tf.cholesky(self.Ku)

