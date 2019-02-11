from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import pdb
import numpy as np
import os

import ray
import linreg.linreg_models as linreg_models
import linreg.data as data
import tensorflow as tf
from linreg.moments_accountant import MomentsAccountantPolicy, MomentsAccountant
from linreg.inference_utils import save_predictive_plot, exact_inference

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
parser.add_argument("--mean", default=2, type=float,
                    help="Linear Regression Slope")
parser.add_argument("--noise-std", default=1, type=float,
                    help="Noise Standard Deviation")


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
            self, worker_index, no_workers, din, data_func, log_path, noise_var=1, seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # get data for this worker
        self.x_train, self.y_train, _, _ = data_func(
            worker_index, no_workers)
        # Initialize the model
        n_train_worker = x_train.shape[0]
        self.n_train = n_train_worker
        self.accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA, 1e-5, 2000000, 32)
        self.net = linreg_models.LinReg_MFVI_DPSGD(
            din, n_train_worker, self.accountant, noise_var=noise_var, init_seed=seed,
            no_workers=no_workers)
        # self.accountant.log_moments_increment = self.net.generate_log_moments(n_train_master, 32)
        self.accountant.log_moments_increment = np.ones(32);
        self.keys = self.net.get_params()[0]
        np.savetxt(log_path + "data/worker_{}_x.txt".format(worker_index), self.x_train)
        np.savetxt(log_path + "data/worker_{}_y.txt".format(worker_index), self.y_train)

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
    noise_std = args.noise_std
    mean = args.mean

    noise_var = np.square(noise_std)

    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, data_type, mean, noise_std)

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)
    n_train_master = x_train.shape[0]


    in_dim = x_train.shape[1]
    accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA, 1e-5, 200, 32)
    net = linreg_models.LinReg_MFVI_DPSGD(in_dim, n_train_master, accountant, noise_var=noise_var)
    accountant.log_moments_increment = np.ones(32);
    # accountant.log_moments_increment = net.generate_log_moments(n_train_master, 32)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)

    timestr = time.strftime("%m-%d;%H:%M:%S")
    path = 'logs/dpsgd_sync_pvi/' + timestr + '/'
    os.makedirs(path + "data/")
    # create workers
    workers = [
        Worker.remote(i, no_workers, in_dim, data_func, path, noise_var, seed=seed)
        for i in range(no_workers)]
    i = 0
    current_params = ray.get(ps.pull.remote(all_keys))

    if not os.path.exists(path):
        os.makedirs(path)

    N_train_worker = data_func(0, no_workers)[0].shape[0]
    # print("N Train Worker: {}".format(N_train_worker))
    names = ["c", "learning_rate_mean", "learning_rate_var", "noise_scale", "num_iterations", "L", "N_train_worker",
             "Num_workers", "mean", "noise_var"]
    params_save = net.get_params_for_logging()
    params_save.append(N_train_worker)
    params_save.append(no_workers)
    params_save.append(mean)
    params_save.append(noise_var)
    param_file = path + 'settings.txt'
    text_file = open(param_file, "w")
    text_file.write('DPSGD Parameters\n')
    for i in range(len(names)):
        text_file.write('{} : {:.2e} \n'.format(names[i], params_save[i]))
    text_file.close()

    tracker_file = path + 'params.txt'

    while i < no_intervals:
        start_time = time.time()
        deltas = [
            worker.get_delta.remote(
                current_params, damping=damping)
            for worker in workers]
        mean_delta = compute_update(all_keys, ray.get(deltas))
        ps.push.remote(all_keys, mean_delta)
        current_params = ray.get(ps.pull.remote(all_keys))
        print("Interval {} done".format(i))
        i += 1
        print(current_params)

        # save to file, tracking stuff
        with open(tracker_file, 'a') as file:
            file.write("{} {}\n".format(current_params[0], current_params[1]))

    meanpres = current_params[0]
    pres = current_params[1]
    var = 1 / pres
    mean = meanpres / pres
    save_predictive_plot(path + 'pred.png', x_train, y_train, mean, var, 1)
