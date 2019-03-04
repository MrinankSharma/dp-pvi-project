from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import pdb
import numpy as np
import os
import json

import ray
import linreg.linreg_models as linreg_models
import linreg.data as data
import tensorflow as tf
from linreg.moments_accountant import MomentsAccountantPolicy, MomentsAccountant
from linreg.inference_utils import save_predictive_plot, exact_inference, KL_Gaussians

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
            self, worker_index, no_workers, din, data_func, log_path, max_eps, dp_noise, c, L_in, noise_var=1, seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # get data for this worker
        self.x_train, self.y_train, _, _ = data_func(
            worker_index, no_workers)
        # Initialize the model
        n_train_worker = self.x_train.shape[0]
        self.n_train = n_train_worker
        self.accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS, 1e-5, max_eps, 32)
        self.net = linreg_models.LinReg_MFVI_DP_analytic(
            din, n_train_worker, self.accountant, noise_var=noise_var, init_seed=seed,
            no_workers=no_workers, clipping_bound=c,
            dp_noise_scale=dp_noise, L=L_in)
        # self.accountant.log_moments_increment = np.ones(32);
        self.accountant.log_moments_increment = self.net.generate_log_moments(n_train_worker, 32)
        print(self.accountant.log_moments_increment)
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

    def get_privacy_spent(self):
        return self.accountant.current_tracked_val


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


def run_dp_analytical_pvi_sync(mean, seed, max_eps, x_train, y_train, model_noise_std, data_func,
                               dp_noise_scale, no_workers, damping, no_intervals, clipping_bound, L):
    # update seeds
    np.random.seed(seed)
    tf.set_random_seed(seed)
    n_train_master = x_train.shape[0]

    in_dim = x_train.shape[1]
    accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA, 1e-5, 200, 32)
    net = linreg_models.LinReg_MFVI_DP_analytic(in_dim, n_train_master, accountant, noise_var=model_noise_std ** 2)

    _, _, exact_mean_pres, exact_pres = exact_inference(x_train, y_train, net.prior_var_num, model_noise_std ** 2 ** 2)
    print("Exact Inference Params: {}, {}".format(exact_mean_pres, exact_pres))

    # accountant is not important here...
    accountant.log_moments_increment = np.ones(32);
    # accountant.log_moments_increment = net.generate_log_moments(n_train_master, 32)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)

    timestr = time.strftime("%m-%d;%H:%M:%S")
    path = 'logs/dp_analytical_sync_pvi/' + timestr + '/'
    os.makedirs(path + "data/")
    # create workers
    workers = [
        Worker.remote(i, no_workers, in_dim, data_func, path, max_eps, dp_noise_scale, clipping_bound, L,
                      model_noise_std ** 2, seed=seed)
        for i in range(no_workers)]
    i = 0
    current_params = ray.get(ps.pull.remote(all_keys))

    if not os.path.exists(path):
        os.makedirs(path)

    N_train_worker = data_func(0, no_workers)[0].shape[0]
    # print("N Train Worker: {}".format(N_train_worker))
    names = ["c", "dp_noise_scale", "L", "N_train_worker",
             "Num_workers", "mean", "noise_var"]
    params_save = net.get_params_for_logging()
    params_save.append(N_train_worker)
    params_save.append(no_workers)
    params_save.append(mean)
    params_save.append(model_noise_std ** 2)
    param_file = path + 'settings.txt'
    text_file = open(param_file, "w")
    text_file.write('DP Parameters\n')
    for i in range(len(names)):
        text_file.write('{} : {:.2e} \n'.format(names[i], params_save[i]))
    text_file.write('Exact Inference (Non-PVI)\n')
    text_file.write("Params: {}, {}".format(exact_mean_pres, exact_pres))
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
    eps = workers[0].get_privacy_spent.remote()
    eps = ray.get(eps)
    plot_title = "({:.3e}, {:.3e})-DP".format(eps, 1e-5)

    save_predictive_plot(path + 'pred.png', x_train, y_train, mean, var, model_noise_std**2, plot_title)
    # report the privacy cost and plot onto the graph...
    print(plot_title)
    KL_loss = KL_Gaussians(exact_mean_pres, exact_pres, current_params[0], current_params[1])
    print(KL_loss)
    return eps, KL_loss


if __name__ == "__main__":
    args = parser.parse_args()

    # load required arguments
    damping_args = args.damping
    data_type_args = args.data_type
    seed_args = args.seed
    no_intervals_args = args.no_intervals
    no_workers_args = args.num_workers
    dataset_args = args.data
    noise_std_args = args.noise_std
    mean_args = args.mean

    # param_file = args.param_file
    #
    # with open(param_file) as f:
    #     dpsgd_params = json.load(f)['dpsgd_params']
    #
    # dpsgd_C = dpsgd_params['C']
    # dpsgd_L = dpsgd_params['L']
    # dpsgd_sigma = dpsgd_params['sigma']

    ray.init(redis_address=args.redis_address, redirect_worker_output=False, redirect_output=False)

    np.random.seed(seed_args)
    tf.set_random_seed(seed_args)

    if dataset_args == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, data_type_args, mean_args, noise_std_args)

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)
    L = 1000
    run_dp_analytical_pvi_sync(mean_args, seed_args, 1, x_train, y_train, noise_std_args, data_func,
                               1, no_workers_args, damping_args, no_intervals_args,
                               10, L)