import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from scipy.cluster.vq import kmeans, kmeans2

from DLVKL.likelihoods import Bernoulli
from DLVKL.models import GP, DLVKL, DLVKL_NSDE
from DLVKL.layers import SVGP_Layer, SVGP_Z_Layer
from DLVKL.kernels import RBF
from DLVKL.settings import Settings

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
# Load banana data
#***************************************
def gridParams(mins, maxs):
    nGrid = 50
    xspaced = np.linspace(mins[0], maxs[0], nGrid)
    yspaced = np.linspace(mins[1], maxs[1], nGrid)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(),yy.flatten())).T
    return mins, maxs, xx, yy, Xplot

Xtrain = np.loadtxt('data/banana_X_train', delimiter=',')
Ytrain = np.loadtxt('data/banana_Y_train', delimiter=',').reshape(-1,1)
# test set
mins, maxs, xx, yy, Xtest = gridParams([-3,-2.5],[ 3, 3.0])

# normalization
Xmean, Xstd = np.mean(Xtrain, axis=0, keepdims=True), np.std(Xtrain, axis=0, keepdims=True)
Xtrain_norm = (Xtrain - Xmean) / Xstd 
Xtest_norm = (Xtest - Xmean) / Xstd

#***************************************
# Model and training settings
#***************************************
m_GP          = args.model         # model to be fitted
num_iter      = args.num_iter      # optimization iterations
lr            = args.lr            # learning rate for Adam optimizer
num_minibatch = args.batch_size    # batch size for SGD
num_samples   = 1                  # number of MC samples of SDE solutions
num_data      = Xtrain.shape[0]    # number of training data
num_dimX      = Xtrain.shape[1]    # input dimensions
latent_dim    = Xtrain.shape[1]    # latent dimensions
num_output    = 1                  # binary classification output dimensions
num_ind       = args.inducing_size # number of inducing variables
beta          = args.beta          # trade-off parameter
flow_time     = args.flow_time     # SDE integration time
flow_nsteps   = args.flow_step     # number of discretizations in Euler Maruyama solver

X_placeholder = tf.placeholder(dtype = float_type,shape=[None,None])
Y_placeholder = tf.placeholder(dtype = float_type,shape=[None,1])

train_dataset  = tf.data.Dataset.from_tensor_slices((X_placeholder,Y_placeholder))
train_dataset  = train_dataset.shuffle(buffer_size=min(num_minibatch*10,num_data), seed=seed).batch(num_minibatch).repeat()
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
pred_layer_kernel = KERNEL.kern(input_dim=latent_dim, lengthscales=KERNEL.lengthscales, variance=1e-0, ARD=KERNEL.ARD)
Z = kmeans(Xtrain_norm,num_ind)[0]
if m_GP == 'DLVKL_NSDE' or m_GP == 'DLVKL': # encoded inducing
    pred_layer = SVGP_Z_Layer(kern=pred_layer_kernel, num_inducing = num_ind, num_outputs=num_output)
else:
    pred_layer = SVGP_Layer(kern=pred_layer_kernel, Z=Z, num_inducing = num_ind, num_outputs=num_output)

# model definition
lik = Bernoulli()
if m_GP == 'GP':
    model = GP(likelihood = lik, pred_layer = pred_layer, num_data = num_data)
elif m_GP == 'DLVKL':
    model = DLVKL(likelihood=lik, pred_layer=pred_layer, dimX=num_dimX, latent_dim=latent_dim, 
                  Z=Z, 
                  num_samples = num_samples, num_data=num_data)
elif m_GP == 'DLVKL_NSDE':
    model = DLVKL_NSDE(likelihood  = lik, pred_layer  = pred_layer, latent_dim = latent_dim,
                       Z = Z, flow_time = flow_time, flow_nsteps = flow_nsteps,
                       num_samples = num_samples, num_data = num_data)
else:
    raise KeyError

#***************************************
# Optimization objective and summary statistics
#***************************************
lowerbound = model._build_likelihood(X,Y,beta=beta) if m_GP == 'DLVKL_NSDE' or m_GP == 'DLVKL' else model._build_likelihood(X,Y)
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(-1.*lowerbound)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_handle  = sess.run(train_iterator.string_handle())
sess.run(train_iterator.initializer,{X_placeholder:Xtrain_norm,Y_placeholder:Ytrain})

#***************************************
# Model training
#***************************************
print('{:>5s}'.format("iter") + '{:>24s}'.format("ELBO:"))

for i in range(1,num_iter+1):
    try:
        sess.run(train_op,feed_dict={handle:train_handle})
        if i % 50 == 0:
            elbo = sess.run(lowerbound,{handle:train_handle})
            print('{:>5d}'.format(i) + '{:>24.6f}'.format(elbo))
    except KeyboardInterrupt as e:
        print("stopping training")
        break

#***************************************
# Prediction
#***************************************
col1 = '#0172B2'
col2 = '#CC6600'
col3 = '#CC0000'

S = 10
ps = model.predict_y(X_placeholder,S)[0]
ps = sess.run(ps,{X_placeholder:Xtest_norm}) # S, N, 1
p = np.average(ps,0)

f, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(Xtrain[:,0][Ytrain[:,0]==0], Xtrain[:,1][Ytrain[:,0]==0], 'o', color=col1, ms=12, alpha=0.5)
ax.plot(Xtrain[:,0][Ytrain[:,0]==1], Xtrain[:,1][Ytrain[:,0]==1], 'o', color=col2, ms=12, alpha=0.5)
ax.contour(xx, yy, p.reshape(*xx.shape), [0.5], colors='k', linewidths=4.,zorder=100)
ax.set_title(m_GP)
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$x_1$', fontsize=20)
ax.set_ylabel(r'$x_2$', fontsize=20)

plt.savefig('figs/'+'binary_'+m_GP+'.png')
plt.show()
