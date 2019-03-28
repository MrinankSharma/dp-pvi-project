from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import os
import json

import copy
import ray
import linreg.linreg_models as linreg_models
import tensorflow as tf
from linreg.moments_accountant import MomentsAccountantPolicy, MomentsAccountant
from linreg.inference_utils import KL_Gaussians
from linreg.log_moment_utils import generate_log_moments

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
parser.add_argument("--dp-noise-scale", default=1, type=float,
                    help="DP Noise")
parser.add_argument("--clipping-bound", default=10000, type=float,
                    help="DP clipping")


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values, conv_thres=0.01 * 1e-2):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.params = dict(zip(keys, values))
        self.should_stop = False
        self.conv_thres = conv_thres
        self.param_it_count = 0.0

    def push(self, keys, values):
        updates = {}
        orig_vals = {}
        pres_key = 'variational_nat/precision'

        for key, val in self.params.iteritems():
            orig_vals[key] = copy.deepcopy(val)
            updates[key] = 0
            # print(" first step {}: {} orig: {}".format(key, self.params[key], orig_vals[key]))

        for key, value in zip(keys, values):
            self.params[key] += value
            updates[key] += value

        # reject value if it results in a negative variance.
        if self.params[pres_key] < 0:
            for key, value in self.params.iteritems():
                # print("before update {}: {} orig: {}".format(key, self.params[key], orig_vals[key]))
                self.params[key] = orig_vals[key]
                # print("{}: {} orig: {}".format(key, self.params[key], orig_vals[key]))

            print('Rejected negative precision')

        if not self.should_stop:
            self.should_stop = True
            for k in keys:
                val = np.abs(updates[k] / orig_vals[k])
                if val > self.conv_thres:
                    self.should_stop = False

    def pull(self, keys):
        return [self.params[key] for key in keys]

    def get_should_stop(self):
        return self.should_stop


@ray.remote
class Worker(object):
    def __init__(
            self, no_workers, din, worker_data, prior_var, noise_var=1):
        # get data for this worker
        self.x_train = worker_data[0]
        self.y_train = worker_data[1]

        # Initialize the model
        n_train_worker = self.x_train.shape[0]
        self.n_train = n_train_worker
        self.net = linreg_models.LinReg_MFVI_analytic(
            din, n_train_worker, noise_var=noise_var,
            no_workers=no_workers)

        self.keys = self.net.get_params()[0]

    def get_delta(self, params, damping=0.5):
        # apply params
        self.net.set_params(self.keys, params)
        # train the network
        self.net.train(self.x_train, self.y_train)
        # get local delta and push to server
        delta = self.net.get_param_diff(damping=damping)

        return delta


def add_noise_clip_delta(deltas, clipping_bound, noise_scale):
    no_delta = len(deltas)
    new_deltas = []
    for i in range(no_delta):
        norm = (deltas[i][0] ** 2 + deltas[i][1] ** 2) ** 0.5
        if clipping_bound < norm:
            scaling = clipping_bound / norm
        else:
            scaling = 1
        new_delta = [deltas[i][0] * scaling, deltas[i][1] * scaling]
        new_deltas.append(new_delta)

    noise = np.random.normal(0, noise_scale * clipping_bound, [2])
    # include the noise as part of the deltas
    new_deltas[0][0] = new_deltas[0][0] + noise[0]
    new_deltas[0][1] = new_deltas[0][1] + noise[1]
    return new_deltas


def compute_true_update(keys, deltas, method='sum'):
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


def compute_update(keys, deltas, clipping_bound, noise_scale, method='sum'):
    mean_delta = []
    updated_deltas = add_noise_clip_delta(deltas, clipping_bound, noise_scale)
    for i, key in enumerate(keys):
        no_deltas = len(updated_deltas)
        for j in range(no_deltas):
            if j == 0:
                sum_delta = np.copy(updated_deltas[j][i])
            else:
                sum_delta += updated_deltas[j][i]
        if method == 'sum':
            mean_delta.append(sum_delta)
        elif method == 'mean':
            mean_delta.append(sum_delta / no_deltas)
    return mean_delta


@ray.remote
def run_global_dp_analytical_pvi_sync(experiment_setup, mean, seed, max_eps, all_workers_data, dp_noise_scale, damping,
                                      clipping_bound, log_moments=None):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    n_train_master = experiment_setup['num_workers'] * experiment_setup['points_per_worker']
    in_dim = 1

    accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS, 1e-5, max_eps, 32)
    net = linreg_models.LinReg_MFVI_analytic(in_dim, n_train_master, noise_var=experiment_setup['model_noise_std'] ** 2,
                                             prior_var=experiment_setup['prior_std'] ** 2)

    if log_moments is None:
        # accountant is not important here...
        # print('calculating log moments')
        accountant.log_moments_increment = generate_log_moments(1, 32, dp_noise_scale, 1)
    else:
        # print('reusing log moments')
        accountant.log_moments_increment = log_moments

    # accountant.log_moments_increment = net.generate_log_moments(n_train_master, 32)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)

    path = experiment_setup['output_base_dir'] + 'logs/global_dp_analytical_sync_pvi/' + time.strftime(
        "%m-%d;%H:%M:%S") + "-s-" + str(seed) + '/'
    # create workers
    workers = [
        Worker.remote(i, experiment_setup['num_workers'], in_dim, all_workers_data[i],
                      experiment_setup['prior_std'] ** 2, experiment_setup['model_noise_std'] ** 2)
        for i in range(experiment_setup['num_workers'])]
    i = 0
    current_params = ray.get(ps.pull.remote(all_keys))

    if not os.path.exists(path):
        os.makedirs(path)

    run_data = {
        "c": clipping_bound,
        "dp_noise_scale": dp_noise_scale,
        "damping": damping,
        "max_eps": max_eps,
    }
    logging_dict = copy.deepcopy(experiment_setup)
    logging_dict.update(run_data)
    setup_file = path + 'run-params.json'
    with open(setup_file, 'w') as outfile:
        json.dump(logging_dict, outfile)

    tracker_vals = []

    while i < experiment_setup['num_intervals']:
        deltas = [
            worker.get_delta.remote(
                current_params, damping=damping)
            for worker in workers]
        true_sum_delta = compute_true_update(all_keys, ray.get(deltas))
        sum_delta = compute_update(all_keys, ray.get(deltas), clipping_bound, dp_noise_scale)
        should_stop_priv = accountant.update_privacy_budget()
        current_eps = accountant.current_tracked_val
        ps.push.remote(all_keys, sum_delta)
        current_params = ray.get(ps.pull.remote(all_keys))
        KL_loss = KL_Gaussians(current_params[0], current_params[1], experiment_setup['exact_mean_pres'],
                               experiment_setup['exact_pres'])
        tracker_i = [sum_delta[0], sum_delta[1], current_params[0], current_params[1], KL_loss, current_eps,
                     true_sum_delta[0], true_sum_delta[1]]
        tracker_vals.append(tracker_i)
        print("Interval {} done: {}".format(i, current_params))
        i += 1

        if ray.get(ps.get_should_stop.remote()):
            # break from the while loop if we should stop, convergence wise.
            print("Converged - stop training")
            break

        if should_stop_priv:
            print("Exceeded Privacy Budget - stop training")
            break

    eps = accountant.current_tracked_val

    KL_loss = KL_Gaussians(current_params[0], current_params[1], experiment_setup['exact_mean_pres'],
                           experiment_setup['exact_pres'])

    tracker_array = np.array(tracker_vals)
    np.savetxt(path + 'tracker.csv', tracker_array, delimiter=',')
    return eps, KL_loss, tracker_array
