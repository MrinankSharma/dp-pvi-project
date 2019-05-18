from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import copy
import pdb
import numpy as np
import os
import json

import ray
import linreg.linreg_models as linreg_models
import linreg.data as data
import tensorflow as tf
from linreg.moments_accountant import MomentsAccountantPolicy, MomentsAccountant
from linreg.inference_utils import KL_Gaussians


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values, conv_thres=-1):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.params = dict(zip(keys, values))
        self.should_stop = False
        self.conv_thres = conv_thres
        self.param_it_count = 0.0

    def push(self, keys, values):
        orig_vals = {}
        updates = {}
        for key, val in self.params.iteritems():
            orig_vals[key] = val
            updates[key] = 0

        for key, value in zip(keys, values):
            self.params[key] += value
            updates[key] += value

        if not self.should_stop:
            self.should_stop = True
            for k in keys:
                val = np.abs(updates[k] / orig_vals[k])
                if val > self.conv_thres:
                    self.should_stop = False

        pres_key = 'variational_nat/precision'
        # reject value if it results in a negative variance.
        if self.params[pres_key] < 0:
            for key, value in self.params.iteritems():
                self.params[key] = orig_vals[key]
            print('Rejected negative precision')

    def pull(self, keys):
        return [self.params[key] for key in keys]

    def get_should_stop(self):
        return self.should_stop


@ray.remote
class Worker(object):
    def __init__(
            self, no_workers, din, worker_data, max_eps, dp_noise, c, noise_var=1, prior_var=1,
            model_config="clipped_noisy", log_moments=None):
        # get data for this worker
        self.x_train = worker_data[0]
        self.y_train = worker_data[1]
        # Initialize the model
        n_train_worker = self.x_train.shape[0]
        self.n_train = n_train_worker
        self.accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS, 1e-5, max_eps, 32)
        self.net = linreg_models.LinReg_MFVI_DP_analytic(
            din, n_train_worker, self.accountant, noise_var=noise_var, no_workers=no_workers, clipping_bound=c,
            dp_noise_scale=dp_noise, prior_var=prior_var, model_config=model_config)

        if log_moments is None:
            self.accountant.log_moments_increment = self.net.generate_log_moments(n_train_worker, 32)
        else:
            self.accountant.log_moments_increment = log_moments

        self.keys = self.net.get_params()[0]

    def get_delta(self, params, damping=0):
        # apply params
        self.net.set_params(self.keys, params)
        # train the network
        self.net.train(self.x_train, self.y_train)
        # get local delta and push to server
        delta = self.net.get_param_diff(damping=damping)
        return delta

    def get_privacy_spent(self):
        return self.accountant.current_tracked_val

    def get_should_stop(self):
        return self.accountant.should_stop


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


@ray.remote
def run_dp_analytical_pvi_sync(experiment_setup, seed, all_workers_data, log_moments=None):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    n_train_master = experiment_setup['num_workers'] * experiment_setup['dataset']['points_per_worker']

    in_dim = 1
    accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS, 1e-5, experiment_setup['max_eps'], 32)
    net = linreg_models.LinReg_MFVI_DP_analytic(in_dim, n_train_master, accountant,
                                                prior_var=experiment_setup['prior_std'] ** 2,
                                                noise_var=experiment_setup['dataset']['model_noise_std'] ** 2)

    # accountant is not important here...
    accountant.log_moments_increment = np.ones(32)
    # accountant.log_moments_increment = net.generate_log_moments(n_train_master, 32)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)

    # create workers
    workers = [
        Worker.remote(experiment_setup['num_workers'], in_dim, all_workers_data[i], experiment_setup['max_eps'],
                      experiment_setup['dp_noise_scale'],
                      experiment_setup['clipping_bound'], experiment_setup['dataset']['model_noise_std'] ** 2,
                      experiment_setup['prior_std'] ** 2,
                      experiment_setup['model_config'], log_moments)
        for i in range(experiment_setup['num_workers'])]
    i = 0
    current_params = ray.get(ps.pull.remote(all_keys))

    # logging stuff
    path = experiment_setup['output_base_dir'] + 'logs/dp_ana_sync_pvi/' + time.strftime(
        "%m-%d;%H:%M:%S") + "-s-" + str(seed) + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    run_data = {
        "c": experiment_setup['clipping_bound'],
        "dp_noise_scale": experiment_setup['dp_noise_scale'],
        "learning_rate": experiment_setup['learning_rate'],
        "max_eps": experiment_setup['max_eps'],
        "model_config": experiment_setup['model_config']
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
                current_params, damping=(1 - experiment_setup['learning_rate']))
            for worker in workers]
        sum_delta = compute_update(all_keys, ray.get(deltas))

        current_eps = ray.get(workers[0].get_privacy_spent.remote())
        ps.push.remote(all_keys, sum_delta)
        current_params = ray.get(ps.pull.remote(all_keys))
        print("Interval {} done: {}".format(i, current_params))
        i += 1

        KL_loss = KL_Gaussians(current_params[0], current_params[1], experiment_setup['exact_mean_pres'],
                               experiment_setup['exact_pres'])

        tracker_i = [sum_delta[0], sum_delta[1], current_params[0], current_params[1], KL_loss, current_eps]
        tracker_vals.append(tracker_i)

        if ray.get(ps.get_should_stop.remote()):
            # break from the while loop if we should stop, convergence wise.
            print("Converged - stop training")
            break

        if ray.get(workers[0].get_should_stop.remote()):
            print("Exceded Privacy Budget - Stop")
            break

    eps = workers[0].get_privacy_spent.remote()
    eps = ray.get(eps)
    # compute KL(q||p)
    KL_loss = KL_Gaussians(current_params[0], current_params[1], experiment_setup['exact_mean_pres'],
                           experiment_setup['exact_pres'])

    print("KL: {}".format(KL_loss))
    tracker_array = np.array(tracker_vals)
    np.savetxt(path + 'tracker.csv', tracker_array, delimiter=',')

    return eps, KL_loss, tracker_array
