import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from gpflow import settings
from .settings import Settings

float_type = settings.float_type
jitter_level = settings.jitter
SEED = Settings().tf_op_random_seed

class EulerMaruyama:
    def __init__(self,f,total_time,nsteps):
        self.total_time = total_time
        self.nsteps = nsteps
        self.ts = np.linspace(0, total_time,nsteps)
        self.f = f

    def forward(self, y0, S):
        N, D = tf.shape(y0)[0], tf.shape(y0)[1]
        y0_tile = tf.tile(y0[None,:,:], [S, 1, 1]) # S, N, D
        y0_flat = tf.reshape(y0_tile, [S * N, D]) # SxN, D
        time_delta = self.ts[1:] - self.ts[:-1]
        time_grid = self.ts[:-1]
        time_combined = tf.concat([time_grid[:,None],time_delta[:,None]],axis=1)

        y_t, y_t_prev, mu_t_prev, var_t_prev = self._scan_loop(self.f, time_combined, y0_flat) # S*N, D
        return tf.reshape(y_t, [S, N, D]), tf.reshape(y_t_prev, [S, N, D]), tf.reshape(mu_t_prev, [S, N, D]), tf.reshape(var_t_prev, [S, N, D])

    def _scan_loop(self, evol_func, time_combined, y):
        step = len(self.ts) - 1
        assert step >= 1
        for i in range(step):
            t = self.ts[i]; dt = time_combined[i, 1]
            mu, var = evol_func(y, t) # S*N, D
            z = tf.truncated_normal(shape=tf.shape(y), dtype=float_type)
            dy = mu * dt + tf.sqrt(var * dt + jitter_level) * z # S*N, D
            if step >= 2:
                if i == step - 2:
                    y_prev = y + dy
            else:
                y_prev = tf.identity(y)
            if i == step - 1:
                mu_t_prev = y_prev + mu * dt; var_t_prev = var * dt
            y += dy
        return y, y_prev, mu_t_prev, var_t_prev
