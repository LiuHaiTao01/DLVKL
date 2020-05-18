from .likelihoods import Gaussian
import tensorflow as tf

class BroadcastingLikelihood:
    """
    A wrapper for the likelihood to broadcast over the samples dimension. The Gaussian doesn't
    need this, but for the others we can apply reshaping and tiling.

    With this wrapper all likelihood functions behave correctly with inputs of shape S,N,D,
    but with Y still of shape N,D
    """
    def __init__(self, likelihood):
        self.likelihood = likelihood

        if isinstance(likelihood, Gaussian):
            self.needs_broadcasting = False
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if self.needs_broadcasting is False:
            return f(vars_SND, [tf.expand_dims(v, 0) for v in vars_ND])

        else:
            S, N, D = [tf.shape(vars_SND[0])[i] for i in range(3)]
            vars_tiled = [tf.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

            flattened_SND = [tf.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [tf.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [tf.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return tf.reshape(flattened_result, [S, N, -1])

    def variational_expectations(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                                vars_SND[1],
                                                                                vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])

    def logp(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    def conditional_mean(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
         return self._broadcast(f,[F], [])

    def conditional_variance(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
         return self._broadcast(f,[F], [])

    def predict_mean_and_var(self, Fmu, Fvar):
         f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0],
                                                                             vars_SND[1])
         return self._broadcast(f,[Fmu, Fvar], [])

    def predict_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_density(vars_SND[0],
                                                                       vars_SND[1],
                                                                       vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])