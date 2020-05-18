import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from scipy.cluster.vq import kmeans2

from DLVKL.likelihoods import Gaussian
from DLVKL.models import DLVKL
from DLVKL.layers import SVGP_Z_Layer
from DLVKL.kernels import RBF
from DLVKL.settings import Settings

import gpflow
from gpflow import transforms
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
parser.add_argument('--model',type=str,default='DLVKL',help='gp model')
args = parser.parse_args()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

#***************************************
# Load data
#***************************************
data = np.load('./data/three_phase_oil_flow.npz')
Ytrain = data['Y']  # following the GPflow notation we assume this dataset has size [num_data, output_dim]
labels = data['labels']  # integer in [0, 2] indicating to which class the datapoint belongs [num_data,]. Not used for model fitting, only for plotting afterwards.

# normalize
Ytrain = (Ytrain - np.mean(Ytrain, 0, keepdims=True)) / np.std(Ytrain, 0, keepdims=True)

print('Number of points: {} and number of dimensions: {}'.format(Ytrain.shape[0], Ytrain.shape[1]))

#***************************************
# Model and training settings
#***************************************
m_GP          = args.model             # model to be fitted
num_iter      = args.num_iter          # optimization iterations
lr            = args.lr                # learning rate for Adam optimizer
num_minibatch = args.batch_size        # batch size for SGD
num_samples   = 1                      # number of MC samples of SDE solutions
num_data      = Ytrain.shape[0]        # number of training data
latent_dim    = 2                      # latent dimensions
num_output    = Ytrain.shape[1]        # binary classification output dimensions
num_ind       = args.inducing_size     # number of inducing variables
beta          = args.beta              # trade-off parameter

sess = tf.InteractiveSession()
Y_placeholder = tf.placeholder(dtype=float_type,shape=[None,None])

train_dataset  = tf.data.Dataset.from_tensor_slices((Y_placeholder))
train_dataset  = train_dataset.shuffle(buffer_size=num_data, seed=seed)
train_dataset  = train_dataset.batch(num_minibatch)
train_dataset  = train_dataset.repeat()
train_iterator = train_dataset.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
Y = iterator.get_next()

#***************************************
# Model
#***************************************
class KERNEL:
    kern = RBF
    lengthscales = 1.
    sf2 = 1.
    ARD = True

# GP layer
pred_kernel = KERNEL.kern(input_dim=latent_dim, lengthscales=KERNEL.lengthscales, variance=KERNEL.sf2, ARD=KERNEL.ARD)
Z = kmeans2(Ytrain, num_ind, minit='points')[0]
pred_layer = SVGP_Z_Layer(kern=pred_kernel, num_inducing=num_ind, num_outputs=num_output)

# model definition
lik = Gaussian()
if m_GP == 'DLVKL':
    model = DLVKL(likelihood=lik, pred_layer=pred_layer, dimX=num_output,
                      latent_dim=latent_dim, Z=Z, num_samples = num_samples, num_data=num_data)
else:
    raise KeyError

#***************************************
# Optimization objective and summary statistics
#***************************************
lowerbound = model._build_likelihood(Y,Y,beta=beta)
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(-1.*lowerbound)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_handle  = sess.run(train_iterator.string_handle())
sess.run(train_iterator.initializer,{Y_placeholder:Ytrain})

#***************************************
# Model training
#***************************************
print('{:>5s}'.format("iter")        + '{:>24s}'.format("ELBO:"))
for i in range(1,num_iter+1):
    try:
        sess.run(train_op,feed_dict={handle:train_handle})
        if i % 50 == 0:
            elbo = sess.run(lowerbound,{handle:train_handle})
            print('{:>5d}'.format(i) + '{:>24.4f}'.format(elbo))
    except KeyboardInterrupt as e:
        print("stopping training")
        break

#***************************************
# Prediction
#***************************************
X_mean, X_var = model.get_latent_variable(Y)
X_mean = X_mean.eval({Y:Ytrain})
X_var = X_var.eval({Y:Ytrain})

f, ax = plt.subplots(1, 1, figsize=(10, 6))
for i in np.unique(labels):
    ax.plot(X_mean[labels==i, 0], X_mean[labels==i, 1], 'o', label=i, ms=10, alpha=0.5)
ax.set_title(m_GP)
ax.tick_params(labelsize=16)
ax.set_xlabel(r'$x_1$', fontsize=20)
ax.set_ylabel(r'$x_2$', fontsize=20)

plt.savefig('figs/'+'dim_reduction_'+m_GP+'.png')
plt.show()








