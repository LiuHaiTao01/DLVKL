from .param import Param
from .utils import reparameterize
from .broadcasting_lik import BroadcastingLikelihood
from .integrators import EulerMaruyama
from .nn import mlp_share, mlp_share_t
from .utils import pca

import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow import transforms
from .settings import Settings
float_type = settings.float_type
jitter_level = settings.jitter


class GP:
    """
    Stochastic variational Gaussian process (SVGP)
    """
    def __init__(self, likelihood, pred_layer, num_samples=1, num_data=None):
        self.num_samples = num_samples
        self.num_data = num_data
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.pred_layer = pred_layer

    def integrate(self, X, S=1): # transform inputs
        return tf.tile(X[None, :, :], [S,1,1]), None # S,N,D

    def propagate(self, Xt, full_cov=False): # F ~ q(f) = N(Fmean, Fvar)
        F, Fmean, Fvar = self.pred_layer.sample_from_conditional(Xt, full_cov=full_cov)
        return F, Fmean, Fvar # S,N,D

    def _build_predict(self, Xt, full_cov=False): # q(f)
        _, Fmeans, Fvars = self.propagate(Xt, full_cov=full_cov)
        return Fmeans, Fvars # S,N,D

    def E_log_p_Y(self, Xt, Y): # E_{q(f)}[logp(y|f)]
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        return tf.reduce_mean(var_exp, 0)  # N, D

    def _build_likelihood(self,X,Y): # ELBO
        Xt = self.integrate(X, self.num_samples)[0] # S,N,D
        # E_{q(f)}[logp(y|f)]
        L = tf.reduce_sum(self.E_log_p_Y(Xt,Y))
        # KL[q(u)||p(u)]
        KL_pred = self.pred_layer.KL()
        # ELBO
        scale = tf.cast(self.num_data / tf.shape(Y)[0], float_type)
        return L * scale - KL_pred # scalar

    def predict_f(self, Xnew, S=1): # q(f*)
        Xnewt = self.integrate(Xnew, S)[0]
        return self._build_predict(Xnewt, full_cov=False)

    def predict_y(self, Xnew, S=1): # q(y*)
        Xnewt = self.integrate(Xnew, S)[0]
        Fmean, Fvar = self._build_predict(Xnewt, full_cov=False)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)


class DLVKL(GP):
    """
    Deep latent-variable kernel learning
    """
    def __init__(self, likelihood, pred_layer, dimX, latent_dim, Z, num_samples, num_data=None):
        GP.__init__(self, likelihood, pred_layer, num_samples, num_data)
        self.dimX, self.latent_dim = dimX, latent_dim
        self.prior_noise = Param(1e-1, transform=transforms.Log1pe(), name="prior_var")()
        self.nn_encoder = mlp_share(self.dimX, self.latent_dim*2, var=self.prior_noise)
        # reasign Z
        self.Z = Param(Z, name="z")()
        Z_mean = self.nn_encoder.forward(self.Z)[0]
        self.pred_layer.initialize_Z(Z_mean)

    def integrate(self, X, S=1):
        Xmean_t, Xvar_t = self.nn_encoder.forward(X)
        sXmean_t, sXvar_t = tf.tile(Xmean_t[None,:,:], [S, 1, 1]), tf.tile(Xvar_t[None,:,:], [S, 1, 1])
        z = tf.random_normal(tf.shape(sXmean_t), dtype=float_type)
        Xt = reparameterize(sXmean_t, sXvar_t, z)
        return Xt, Xmean_t, Xvar_t

    def _build_likelihood(self,X,Y,beta=1e-2):
        Xt, Z_mean, Z_var = self.integrate(X, self.num_samples) # S,N,D
        # E_{q(f)}[logp(y|f)]
        L = tf.reduce_sum(self.E_log_p_Y(Xt,Y))
        # KL[q(u)||p(u)]
        KL_pred = self.pred_layer.KL()
        # KL(q(z)||p(z)]
        if self.dimX == self.latent_dim:
            Z_prior = tf.identity(X)
        elif self.dimX > self.latent_dim: # PCA projection
            Z_prior = pca(X, dim=self.latent_dim)
        else: # zero-padding
            Z_prior = tf.concat([X, tf.zeros([tf.shape(X)[0], self.latent_dim - self.dimX])], axis=-1)
        p_Z = tf.distributions.Normal(loc = Z_prior, scale = tf.ones_like(Z_prior) * self.prior_noise) # informative prior       
        q_Z = tf.distributions.Normal(loc = Z_mean, scale = tf.sqrt(Z_var + 1e-6))
        KL_Z = tf.reduce_sum(q_Z.kl_divergence(p_Z))
        # ELBO
        scale = tf.cast(self.num_data, float_type) / tf.cast(tf.shape(Y)[0], float_type)
        return (L - beta * KL_Z) * scale - KL_pred

    def get_latent_variable(self, X, S=1):
        X_mean, X_var = self.nn_encoder.forward(X)
        return X_mean, X_var


class DLVKL_NSDE_base(GP):
    """
    Deep latent-variable kernel learning using neural SDE and hybrid prior
    """
    def __init__(self, likelihood, pred_layer, latent_dim, num_samples=1,
                 flow_time = 1.0, flow_nsteps = 20, num_data=None):
        GP.__init__(self, likelihood, pred_layer, num_samples, num_data)
        self.latent_dim = latent_dim
        self.flow_time, self.flow_nsteps = flow_time, flow_nsteps
        self.prior_noise = Param(1e-2 / self.flow_time, transform=transforms.Log1pe(), name="prior_var")()
        self.nn_diff = mlp_share_t(self.latent_dim+1, self.latent_dim * 2, var=self.prior_noise)
        self.sde_solver = EulerMaruyama(self.nn_diff.forward, self.flow_time, self.flow_nsteps)
        
    def integrate(self, X_latent, S=1):
        Xt, Xt_prev, mu_t_prev, var_t_prev = self.sde_solver.forward(X_latent, S) # SxNxD
        return Xt, Xt_prev, X_latent, mu_t_prev, var_t_prev

    def KL_X(self, Xt, Xt_prev, mu_t_prev, var_t_prev, X_prior_mean, X_prior_var): 
        p_X = tf.distributions.Normal(loc = X_prior_mean, scale = tf.sqrt(X_prior_var + jitter_level))
        logp = p_X.log_prob(Xt) # S,N,D

        q_X = tf.distributions.Normal(loc = mu_t_prev, scale = tf.sqrt(var_t_prev + jitter_level))
        if self.num_samples == 1: # single-sample approx.
            logq = q_X.log_prob(Xt) # S,N,D 
        else: # MC approx.
            flogpdf = lambda x: tf.log(tf.reduce_mean(q_X.prob(tf.tile(x[None,:,:],[self.num_samples,1,1])), 0)) # N,D
            logq = tf.map_fn(flogpdf, Xt, dtype=float_type) # S,N,D

        kl = tf.reduce_sum(tf.reduce_mean(logq - logp, axis=0))
        return kl, logq

    def _build_likelihood(self, X, Y, beta=1e-2):
        Xt, Xt_prev, X_latent, mu_t_prev, var_t_prev = self.integrate(X, self.num_samples) # S,N,D
        L = tf.reduce_sum(self.E_log_p_Y(Xt,Y))

        X_prior_mean = tf.tile(X_latent[None,:,:], [self.num_samples,1,1]) # S,N,D
        X_prior_var = tf.ones_like(Xt) * self.flow_time * self.prior_noise # S,N,D
        KL_X, logq = self.KL_X(Xt, Xt_prev, mu_t_prev, var_t_prev, X_prior_mean, X_prior_var)

        KL_pred = self.pred_layer.KL()
        
        scale = tf.cast(self.num_data / tf.shape(Y)[0], dtype=float_type)
        return (L - beta * KL_X) * scale - KL_pred


class DLVKL_NSDE(DLVKL_NSDE_base):
    """
    DLVKL using encoded inducing strategy
    """
    def __init__(self, likelihood, pred_layer, latent_dim, Z, num_samples=1,
                 flow_time = 1.0, flow_nsteps = 20, num_data=None):
        DLVKL_NSDE_base.__init__(self,likelihood,pred_layer,latent_dim,num_samples,flow_time,flow_nsteps,num_data)
        # reasign Z
        self.Z_ind = Param(Z, name="z_ind")()
        Z_trans = self.sde_solver.forward(self.Z_ind, self.num_samples)[0] # S, N, D
        Z_trans = tf.reduce_mean(Z_trans, axis=0) # N, D
        self.pred_layer.initialize_Z(Z_trans)
    



