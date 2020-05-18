# Borrowed and modified from: https://github.com/GPflow/GPflow/

from .param import Param

import numpy as np
import tensorflow as tf

from gpflow import logdensities
from gpflow import priors
from gpflow import settings
from gpflow import transforms
from gpflow.quadrature import hermgauss
from gpflow.quadrature import ndiagquad, ndiag_mc


class Likelihood:
    def __init__(self, *args, **kwargs):
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                              self.num_gauss_hermite_points,
                              Fmu, Fvar)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            \log \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, logspace=True, Y=Y)

    def variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, Y=Y)


class Gaussian(Likelihood):
    def __init__(self, variance=1e-0, D=None, **kwargs):
        super().__init__(**kwargs)
        if D is not None: # allow different noises for outputs
            variance = variance * np.ones((1,D))
        self.variance = Param(variance, transform=transforms.Log1pe(),name = "noise_variance")()

    def logp(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return logdensities.gaussian(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


def log_bernoulli(Y_true, p):
    return Y_true * tf.log(p + 1e-6) + (1. - Y_true) * tf.log(1. - p + 1e-6)


class Bernoulli(Likelihood):
    def __init__(self, invlink=inv_probit, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink

    def logp(self, F, Y):
        return logdensities.bernoulli(Y, self.invlink(F))

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return super().predict_mean_and_var(Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return logdensities.bernoulli(Y, p)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)


class RobustMax:
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.
    """

    def __init__(self, num_classes, epsilon=1e-3, **kwargs):
        self.epsilon = tf.cast(epsilon, tf.float64)
        self.num_classes = num_classes

    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, tf.squeeze(1. - self.epsilon), tf.squeeze(self._eps_K1))

    def _eps_K1(self):
        return self.epsilon / (self.num_classes - 1.)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = tf.cast(Y, tf.int64)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.float_type) 
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1) # size (N,)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1)) # N x S, where S is the size of gh_x

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2) # N x P x S
        cdfs = 0.5 * (1.0 + tf.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.float_type)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2) # N x P x S

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, reduction_indices=[1]), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1))) # N x 1


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None, **kwargs):
        """
        A likelihood that can do multi-way classification.
        Currently the only valid choice
        of inverse-link function (invlink) is an instance of RobustMax.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        if invlink is None:
            invlink = RobustMax(self.num_classes)
        elif not isinstance(invlink, RobustMax):
            raise NotImplementedError
        self.invlink = invlink

    def logp(self, F, Y):
        if isinstance(self.invlink, RobustMax):
            hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
            yes = tf.ones(tf.shape(Y), dtype=settings.float_type) - self.invlink.epsilon
            no = tf.zeros(tf.shape(Y), dtype=settings.float_type) + self.invlink._eps_K1()
            p = tf.where(hits, yes, no)
            return tf.log(p)
        else:
            raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            ve = p * tf.log(1. - self.invlink.epsilon) + (1. - p) * tf.log(self.invlink._eps_K1())
            return ve
        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        if isinstance(self.invlink, RobustMax):
            # To compute this, we'll compute the density for each possible output
            possible_outputs = [tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in
                                range(self.num_classes)]
            ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
            ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
            return ps, ps - tf.square(ps)
        else:
            raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        return tf.log(self._predict_non_logged_density(Fmu, Fvar, Y))

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            den = p * (1. - self.invlink.epsilon) + (1. - p) * (self.invlink._eps_K1())
            return den
        else:
            raise NotImplementedError

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)


class MultiClass_SoftMax(Likelihood):
    def __init__(self, num_classes, **kwargs):
        """
        Multi-class classification using augmented softmax, see
        2019-Scalable Gaussian Process Classification with Additive Noise for Various Likelihoods
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def variational_expectations(self, Fmu, Fvar, Y):
        Y = tf.cast(Y, tf.int64)
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.float_type) 
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.float_type)
        Fmu_selected = tf.reduce_sum(oh_on * Fmu, 1) 
        Fvar_selected = tf.reduce_sum(oh_on * Fvar, 1)

        P = tf.exp(0.5*Fvar_selected - Fmu_selected) * tf.reduce_sum(tf.exp(0.5*Fvar + Fmu)*oh_off, 1) # (N,)
        ve = - tf.log(1. + P[:,None])

        return ve

    def predict_mean_and_var(self, Fmu, Fvar):
        N_sample = 20
        u = np.random.randn(N_sample, self.num_classes) 
        u_3D = tf.tile(tf.expand_dims(u, 1), [1, tf.shape(Fmu)[0], 1]) 
        Fmu_3D = tf.tile(tf.expand_dims(Fmu, 0), [N_sample, 1, 1]) 
        Fvar_3D = tf.tile(tf.expand_dims(Fvar, 0), [N_sample, 1, 1])
        exp_term = tf.exp(Fmu_3D + tf.sqrt(Fvar) * u_3D) 
        exp_sum_term = tf.tile(tf.expand_dims(tf.reduce_sum(exp_term, -1), 2), [1, 1, self.num_classes])
        ps = tf.reduce_sum(exp_term / exp_sum_term, 0) / N_sample
        vs = tf.reduce_sum(tf.square(exp_term / exp_sum_term), 0) / N_sample - tf.square(ps)

        return ps, vs
