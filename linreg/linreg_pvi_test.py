from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import pdb
import os
import numpy as np

import linreg_models
import data
import tensorflow as tf
import glob
import matplotlib.pylab as plt

parser = argparse.ArgumentParser(
    description="synchronous distributed variational training: compute test statistics based on saved params.")
parser.add_argument("--param-file", type=str,
                    help="parameter file")
parser.add_argument("--data", default='toy_1d', type=str,
                    help="Data set: toy_1d.")
parser.add_argument("--seed", default=42, type=int,
                    help="Random seed.")


if __name__ == "__main__":
    args = parser.parse_args()
    fname = args.param_file
    dataset = args.data
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset == 'toy_1d':
        data_func = data.get_toy_1d_shard

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)
    n_train_master = x_train.shape[0]
    in_dim = x_train.shape[1]
    
    tf.reset_default_graph()
    # Initialize the model
    net = linreg_models.LinReg_MFVI_analytic(
        in_dim, n_train_master, single_thread=False)
    keys = net.get_params()[0]
    params = np.load(fname)
    net.set_params(keys, [params['n1'], params['n2']], False)
    
    nplot = 400
    xplot = np.linspace(-4, 4, nplot).reshape([nplot, 1])
    m, v = net.prediction(xplot)
    plt.figure()
    plt.plot(x_train[:, 0], y_train, 'k+')
    plt.plot(xplot[:, 0], m, 'b-')
    plt.fill_between(xplot[:, 0], m+2*np.sqrt(v), m-2*np.sqrt(v), color='b', alpha=0.5)
    plt.show()

