from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import pdb
import numpy as np
import os

import ray
import linreg_models
import data
import tensorflow as tf

parser = argparse.ArgumentParser(description="synchronous distributed variational training.")
parser.add_argument("--data", default='toy_1d', type=str,
                    help="Data set: toy_1d.")
parser.add_argument("--seed", default=42, type=int,
                    help="Random seed.")
parser.add_argument("--no-intervals", default=20, type=int,
                    help="Number of measurements/parameter savings.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--data-type", default='homous', type=str,
                    help="Data distribution: homous (homogeneous) vs inhomous (inhomogeneous).")
parser.add_argument("--damping", default=0.0, type=float,
                    help="damping rate, new = damping * old + (1-damping) * new.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")

@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.params = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.params[key] += value

    def pull(self, keys):
        return [self.params[key] for key in keys]

@ray.remote
class Worker(object):
    def __init__(
            self, worker_index, no_workers, din, data_func,
            data_type='homous', seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # get data for this worker
        self.x_train, self.y_train, _, _ = data_func(
            worker_index, no_workers, data_type)
        # Initialize the model
        n_train_worker = x_train.shape[0]
        self.net = linreg_models.LinReg_MFVI_DPSGD(
            din, n_train_worker, 
            init_seed=seed, no_workers=no_workers)
        self.keys = self.net.get_params()[0]

    def get_delta(self, params, damping=0.5):
        # apply params
        self.net.set_params(self.keys, params)
        # train the network
        self.net.train(self.x_train, self.y_train)
        # get local delta and push to server
        delta = self.net.get_param_diff(damping=damping)
        return delta

def compute_update(keys, deltas, method='sum'):
    mean_delta = []
    for i, key in enumerate(keys):
        no_deltas = len(deltas)
        for j in range(no_deltas):
            if j == 0:
                sum_delta = np.copy(deltas[j][i])
            else:
                sum_delta += deltas[j][i]
        if method == 'sum':
            mean_delta.append(sum_delta)
        elif method == 'mean':
            mean_delta.append(sum_delta / no_deltas)
    return mean_delta

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address, redirect_worker_output=False, redirect_output=False)
    damping = args.damping
    data_type = args.data_type
    seed = args.seed
    no_intervals = args.no_intervals
    no_workers = args.num_workers
    dataset = args.data

    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset == 'toy_1d':
        data_func = data.get_toy_1d_shard

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)
    n_train_master = x_train.shape[0]
    in_dim = x_train.shape[1]
    net = linreg_models.LinReg_MFVI_DPSGD(in_dim, n_train_master)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)
    
    # create workers
    workers = [
        Worker.remote(i, no_workers, in_dim, data_func,
            data_type=data_type, seed=seed) 
        for i in range(no_workers)]
    i = 0
    current_params = ray.get(ps.pull.remote(all_keys))

    path_prefix = '/tmp/distributed_training/'
    path = path_prefix + 'pvi_sync_%s_data_%s_seed_%d_no_workers_%d_damping_%.3f/' % (
        dataset, data_type, seed, no_workers, damping)
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez_compressed(
        path + 'params_interval_%d.npz' % i, 
        n1=current_params[0], n2=current_params[1])
    time_fname = path + 'train_time.txt'
    time_file = open(time_fname, 'w', 0)
    time_file.write('%.4f\n' % 0)

    while i < no_intervals:
        start_time = time.time()
        ###########
        deltas = [
            worker.get_delta.remote(
                current_params, damping=damping)
            for worker in workers]
        mean_delta = compute_update(all_keys, ray.get(deltas))
        ps.push.remote(all_keys, mean_delta)
        current_params = ray.get(ps.pull.remote(all_keys))
        ###########

        end_time = time.time()
        train_time = end_time - start_time
        #train_times.append(train_time)
        np.savez_compressed(
            path + 'params_interval_%d.npz' % (i+1),
            n1=current_params[0], n2=current_params[1])
        print("Interval {} done".format(i))
        i += 1
        time_file.write('%.4f\n' % train_time)
        print(current_params)
        
    time_file.close()

