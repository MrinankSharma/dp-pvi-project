from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import os

import ray
import linreg.linreg_models as linreg_models
import json
import linreg.data as data
import tensorflow as tf
from linreg.moments_accountant import MomentsAccountantPolicy, MomentsAccountant
from linreg.inference_utils import exact_inference, KL_Gaussians
from linreg.log_moment_utils import generate_log_moments


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

    def pull(self, keys):
        return [self.params[key] for key in keys]

    def get_should_stop(self):
        return self.should_stop


@ray.remote
class Worker(object):
    def __init__(self, din, no_workers, worker_data, max_eps, dp_noise, c, noise_var=1, prior_var=1,
                 learning_rate=1e-3, num_iterations=100, lot_size=10, log_moments=None):
        # get data for this worker
        self.x_train = worker_data[0]
        self.y_train = worker_data[1]
        # Initialize the model
        n_train_worker = self.x_train.shape[0]
        self.n_train = n_train_worker
        self.accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS, 1e-2, max_eps, 32)

        self.net = linreg_models.LinReg_MFVI_DPSGD(
            din, n_train_worker, self.accountant, noise_var=noise_var, no_workers=no_workers, gradient_bound=c,
            prior_var=prior_var, dpsgd_noise_scale=dp_noise, lot_size=lot_size,
            num_iterations=num_iterations, learning_rate=learning_rate)

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
def run_dpsgd_pvi_sync(experiment_setup, seed, max_eps, all_workers_data, exact_params=None,
                       log_moments=None):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    n_train_master = experiment_setup['num_workers'] * experiment_setup['dataset']['points_per_worker']

    in_dim = 1
    accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA, 1e-5, max_eps, 32)
    net = linreg_models.LinReg_MFVI_DPSGD(in_dim, n_train_master, accountant,
                                          prior_var=experiment_setup['prior_std'] ** 2,
                                          noise_var=experiment_setup['dataset']['model_noise_std'] ** 2)

    # accountant is not important here...
    accountant.log_moments_increment = np.ones(32)
    # accountant.log_moments_increment = net.generate_log_moments(n_train_master, 32)
    all_keys, all_values = net.get_params()
    ps = ParameterServer.remote(all_keys, all_values)

    # create workers
    workers = [
        Worker.remote(in_dim, experiment_setup['num_workers'], all_workers_data[i], max_eps,
                      experiment_setup['dp_noise_scale'],
                      experiment_setup['clipping_bound'], experiment_setup['dataset']['model_noise_std'] ** 2,
                      experiment_setup['prior_std'] ** 2,
                      experiment_setup['learning_rate'], experiment_setup['local_num_iterations'],
                      experiment_setup['lot_size'], log_moments)
        for i in range(experiment_setup['num_workers'])]
    i = 0
    current_params = ray.get(ps.pull.remote(all_keys))

    # logging stuff
    path = experiment_setup['output_base_dir'] + 'logs/dpsgd_sync_pvi/' + time.strftime(
        "%m-%d;%H:%M:%S") + "-s-" + str(seed) + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    setup_file = path + 'run-params.json'
    with open(setup_file, 'w') as outfile:
        json.dump(experiment_setup, outfile)

    tracker_vals = []

    while i < experiment_setup['num_intervals']:
        deltas = [
            worker.get_delta.remote(
                current_params, damping=experiment_setup["global_damping"])
            for worker in workers]
        sum_delta = compute_update(all_keys, ray.get(deltas))

        current_eps = ray.get(workers[0].get_privacy_spent.remote())
        ps.push.remote(all_keys, sum_delta)
        current_params = ray.get(ps.pull.remote(all_keys))
        print("Interval {} done: {} eps:{} ".format(i, current_params, current_eps))
        i += 1

        if exact_params == None:
            KL_loss = KL_Gaussians(current_params[0], current_params[1], experiment_setup['exact_mean_pres'],
                                   experiment_setup['exact_pres'])
        else:
            KL_loss = KL_Gaussians(current_params[0], current_params[1], exact_params[0], exact_params[1])

        tracker_i = [sum_delta[0], sum_delta[1], current_params[0], current_params[1], KL_loss, current_eps]
        tracker_vals.append(tracker_i)

        if ray.get(ps.get_should_stop.remote()):
            # break from the while loop if we should stop, convergence wise.
            print("Converged - stop training")
            break

    eps = workers[0].get_privacy_spent.remote()
    eps = ray.get(eps)
    # compute KL(q||p)
    if exact_params == None:
        KL_loss = KL_Gaussians(current_params[0], current_params[1], experiment_setup['exact_mean_pres'],
                               experiment_setup['exact_pres'])
    else:
        KL_loss = KL_Gaussians(current_params[0], current_params[1], exact_params[0], exact_params[1])

    print("KL: {}".format(KL_loss))
    tracker_array = np.array(tracker_vals)
    np.savetxt(path + 'tracker.csv', tracker_array, delimiter=',')

    return eps, KL_loss, tracker_array


if __name__ == "__main__":
    experiment_setup = {
        "seed": 42,
        "dataset": {
            "dataset": 'toy_1d',
            "data_type": 'homous',
            "mean": 2,
            "model_noise_std": 0.5,
            "points_per_worker": 10,
        },
        "prior_std": 5,
        "num_workers": 5,
        "num_intervals": 1000,
        "output_base_dir": '',
        "dp_noise_scale": 1,
        "clipping_bound": 10,
        "local_damping": 0,
        "global_damping": 0,
        "max_eps": 1e50,
        "learning_rate": 1e-5,
        "local_num_iterations": 50,
        "lot_size": 5,
    }

    dataset_setup = experiment_setup["dataset"]
    if dataset_setup['dataset'] == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, dataset_setup['data_type'],
                                                         dataset_setup['mean'],
                                                         dataset_setup['model_noise_std'],
                                                         experiment_setup['num_workers'] * dataset_setup[
                                                             'points_per_worker'])

    workers_data = [data_func(w_i, experiment_setup['num_workers']) for w_i in range(experiment_setup['num_workers'])]
    x_train = np.array([[]])
    y_train = np.array([])
    for worker_data in workers_data:
        x_train = np.append(x_train, worker_data[0])
        y_train = np.append(y_train, worker_data[1])

    _, _, exact_mean_pres, exact_pres = exact_inference(x_train, y_train, experiment_setup['prior_std'],
                                                        dataset_setup['model_noise_std'] ** 2)

    log_moments = generate_log_moments(experiment_setup["dataset"]["points_per_worker"], 32, experiment_setup["dp_noise_scale"],
                         experiment_setup["lot_size"])
    print("Exact Params: {}, {}".format(exact_mean_pres, exact_pres))
    ray.init()
    results = run_dpsgd_pvi_sync.remote(experiment_setup, 1, 1e50, workers_data, [exact_mean_pres, exact_pres], log_moments)
    ray.get(results)
