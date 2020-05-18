import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from scipy.cluster.vq import kmeans, kmeans2

from DLVKL.likelihoods import Gaussian
from DLVKL.models import GP, DLVKL, DLVKL_NSDE
from DLVKL.layers import SVGP_Layer, SVGP_Z_Layer
from DLVKL.kernels import RBF
from DLVKL.settings import Settings

import pickle 

from gpflow import settings
float_type = settings.float_type

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=5e-3,help='learning_rate')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--num_iter',type=int,default=5000,help='train iterations')
parser.add_argument('--inducing_size',type=int,default=20,help='inducing size')
parser.add_argument('--beta',type=float,default=1e-2,help='trade-off para')
parser.add_argument('--flow_time',type=float,default=1.0,help='flow time')
parser.add_argument('--flow_step',type=int,default=10,help='flow steps')
parser.add_argument('--model',type=str,default='DLVKL',help='gp model')
args = parser.parse_args()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

#***************************************
# Load data
#***************************************
f = lambda X: np.cos(5*X)*np.exp(-X/2)+1. if X<0. else np.cos(5*X)*np.exp(-X/2)-1.
N, Ns = 50, 1000
Xtrain = np.vstack([np.linspace(-1.5,0.,int(N/2))[:,None], np.linspace(0,1.5,int(N/2))[:,None]])
noise = 0.
Ytrain = np.reshape([f(x) for x in Xtrain], Xtrain.shape) + np.random.randn(*Xtrain.shape)*noise
Xtest = np.linspace(-1.5,1.5,Ns)[:,None]

# Normalization
Ymean, Ystd = np.mean(Ytrain), np.std(Ytrain)
Ytrain_norm = (Ytrain - Ymean) / Ystd
Xmean, Xstd = np.mean(Xtrain, axis=0, keepdims=True), np.std(Xtrain, axis=0, keepdims=True)
Xtrain_norm = (Xtrain - Xmean) / Xstd
Xtest_norm = (Xtest - Xmean) / Xstd

#***************************************
# Model and training settings
#***************************************
m_GP          = args.model             # model to be fitted
num_iter      = args.num_iter          # optimization iterations
lr            = args.lr                # learning rate for Adam optimizer
num_minibatch = min(args.batch_size,N) # batch size for SGD
num_samples   = 1                      # number of MC samples of SDE solutions
num_data      = Xtrain.shape[0]        # number of training data
num_dimX      = Xtrain.shape[1]        # input dimensions
latent_dim    = Xtrain.shape[1]        # latent dimensions
num_output    = 1                      # binary classification output dimensions
num_ind       = args.inducing_size     # number of inducing variables
beta          = args.beta              # trade-off parameter
flow_time     = args.flow_time         # SDE integration time
flow_nsteps   = args.flow_step         # number of discretizations in Euler Maruyama solver


sess = tf.InteractiveSession()

X_placeholder = tf.placeholder(dtype=float_type,shape=[None,num_dimX])
Y_placeholder = tf.placeholder(dtype=float_type,shape=[None,1])

train_dataset  = tf.data.Dataset.from_tensor_slices((X_placeholder,Y_placeholder))
train_dataset  = train_dataset.shuffle(buffer_size=Xtrain.shape[0], seed=seed)
train_dataset  = train_dataset.batch(num_minibatch)
train_dataset  = train_dataset.repeat()
train_iterator = train_dataset.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
X,Y = iterator.get_next()

#***************************************
# Model
#***************************************
class KERNEL:
    kern = RBF
    lengthscales = 1.
    sf2 = 1.
    ARD = True

# GP layer
pred_layer_kernel = KERNEL.kern(input_dim=latent_dim, lengthscales=KERNEL.lengthscales, variance=KERNEL.sf2, ARD=KERNEL.ARD)
Z = kmeans(Xtrain_norm,num_ind)[0]
if m_GP == 'DLVKL_NSDE' or m_GP == 'DLVKL': # encoded inducing
    pred_layer = SVGP_Z_Layer(kern=pred_layer_kernel, num_inducing = num_ind, num_outputs=num_output)
else:
    pred_layer = SVGP_Layer(kern=pred_layer_kernel, Z=Z, num_inducing = num_ind, num_outputs=num_output)

# model definition
lik = Gaussian()
if m_GP == 'GP':
    model = GP(likelihood=lik, pred_layer=pred_layer, num_data=num_data)
elif m_GP == 'DLVKL':
    model = DLVKL(likelihood=lik, pred_layer=pred_layer, dimX=num_dimX, 
                  Z=Z,
                  latent_dim=latent_dim, num_samples = num_samples, num_data=num_data)
elif m_GP == 'DLVKL_NSDE':
    model = DLVKL_NSDE(likelihood = lik, pred_layer = pred_layer, latent_dim = latent_dim,
                       Z = Z, flow_time = flow_time, flow_nsteps = flow_nsteps, 
                       num_samples = num_samples, num_data = num_data)
else:
    raise KeyError

#***************************************
# Optimization objective and summary statistics
#***************************************
lowerbound = model._build_likelihood(X,Y,beta=beta) if m_GP == 'DLVKL_NSDE' or m_GP == 'DLVKL' else model._build_likelihood(X,Y)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train_op = optimizer.minimize(-1.*lowerbound)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_handle  = sess.run(train_iterator.string_handle())
sess.run(train_iterator.initializer,{X_placeholder:Xtrain_norm, Y_placeholder:Ytrain_norm})

#***************************************
# Model training
#***************************************
print('{:>5s}'.format("iter") + '{:>24s}'.format("ELBO:"))

for i in range(1,num_iter+1):
    try:
        sess.run(train_op,feed_dict={handle:train_handle})
        if i % 50 == 0:
            elbo = sess.run(lowerbound,{handle:train_handle})
            print('{:>5d}'.format(i)  + '{:>24.6f}'.format(elbo))
    except KeyboardInterrupt as e:
        print("stopping training")
        break

#***************************************
# Prediction and Plot
#***************************************
S = 10
m, v = model.predict_y(X, S)

m, v = np.average(m.eval({X:Xtest_norm}),0), np.average(v.eval({X:Xtest_norm}),0)
var = v * Ystd**2
mu = m * Ystd + Ymean # recover predictions

f, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(Xtest.flatten(), mu.flatten(), color='r',linewidth=2)
l = (mu.flatten() - 2*var.flatten()**0.5)
u = (mu.flatten() + 2*var.flatten()**0.5)
ax.fill_between(Xtest.flatten(), l, u, color='r', alpha=0.3)
ax.scatter(Xtrain, Ytrain, marker='o', s=50, color='gray', edgecolors='k', alpha=0.6)
ax.legend(['prediction mean', 'prediction variance', 'data'], fontsize=15)
ax.set_title(m_GP)
ax.tick_params(labelsize=16)
ax.set_xlabel(r'Input', fontsize=20)
ax.set_ylabel(r'Output', fontsize=20)

plt.savefig('figs/'+'regression_'+m_GP+'.png')
plt.show()





