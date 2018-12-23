from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import pdb
import os
import numpy as np

import ray
import linreg_models
import data
import tensorflow as tf

parser = argparse.ArgumentParser(description="asynchronous distributed variational training")
parser.add_argument("--data", default='toy_1d', type=str,
                    help="Data set: toy_1d.")
parser.add_argument("--seed", default=42, type=int,
                    help="Random seed.")
parser.add_argument("--interval-time", default=2, type=float,
                    help="Interval time in second between measurements/parameter savings.")
parser.add_argument("--no-intervals", default=20, type=int,
                    help="Number of measurements/parameter savings.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--data-type", default='homous', type=str,
                    help="Data distribution: homous (homogeneous) vs inhomous (inhomogeneous).")
parser.add_argument("--damping", default=0.0, type=float,
                    help="outer loop damping rate, new = damping * old + (1-damping) * new.")
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
def worker_task(ps, worker_index, no_workers, din, data_func,
    damping=0.5, seed=0, data_type='homous'):
    
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # get data for this worker
    x_train, y_train, _, _ = data_func(worker_index, no_workers, data_type)

    # Initialize the model
    n_train_worker = x_train.shape[0]
    net = linreg_models.LinReg_MFVI_analytic(
        din, n_train_worker, init_seed=seed, 
        no_workers=no_workers)
    keys = net.get_params()[0]

    while True:
        # Get the current params from the parameter server.
        # print('get params from server and set...')
        params = ray.get(ps.pull.remote(keys))
        net.set_params(keys, params)

        # Compute an update and push it to the parameter server.
        # print('train the worker\'s local network')
        net.train(x_train, y_train)

        # todo: damping should relate to the number of workers
        # print('get local delta and push to server')
        delta = net.get_param_diff(damping=damping)
        ps.push.remote(keys, delta)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address)
    damping = args.damping
    data_type = args.data_type
    seed = args.seed
    interval_time = args.interval_time
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
    net = linreg_models.LinReg_MFVI_analytic(in_dim, n_train_master)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)
    
    # Start some training tasks.
    worker_tasks = [
        worker_task.remote(ps, i, no_workers, in_dim, data_func,
            damping=damping, data_type=data_type, seed=seed) 
        for i in range(no_workers)]

    path_prefix = '/tmp/distributed_training/'
    path = path_prefix + 'pvi_async_%s_data_%s_seed_%d_no_workers_%d_interval_time_%.2f_damping_%.3f/' % (
        dataset, data_type, seed, no_workers, interval_time, damping)

    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    while i < no_intervals:
        current_params = ray.get(ps.pull.remote(all_keys))
        print(current_params)
        np.savez_compressed(
            path + 'params_interval_%d.npz' % i, 
            n1=current_params[0], n2=current_params[1])
        time.sleep(interval_time)
        print("Interval {} done".format(i))
        i += 1
