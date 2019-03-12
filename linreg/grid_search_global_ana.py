import numpy as np
import tensorflow as tf
import traceback

import argparse

import linreg.data as data
from linreg_global_dp_ana_pvi_sync import run_global_dp_analytical_pvi_sync
from log_moment_utils import generate_log_moments
from linreg.file_utils import get_params_from_csv
import itertools
import time
import os
import ray

parser = argparse.ArgumentParser(description="grid search for whole client level dp")
parser.add_argument("--output-base-dir", default='', type=str,
                    help="output base folder.")
parser.add_argument("--tag", default='default', type=str)
parser.add_argument("--overwrite", dest='overwrite', action='store_true')
parser.add_argument("--testing", dest='testing', action='store_true')


if __name__ == "__main__":
    args = parser.parse_args()
    output_base_dir = args.output_base_dir
    should_overwrite = args.overwrite
    testing = args.testing
    tag = args.tag

    # really, we should average over multiple seeds
    seed = 42
    dataset = 'toy_1d'
    data_type = "homous"
    mean = 2
    model_noise_std = 0.5
    no_workers = 5
    damping = 0
    # will stop when the privacy budget is reached!
    no_intervals = 500
    N_dp_seeds = 10

    max_eps_values = [np.inf]
    dp_noise_scales = [1e-6, 1e-4, 1e-2, 1]
    clipping_bounds = [1, 1e2, 1e4, 1e6]

    if testing:
        max_eps_values = [1]
        dp_noise_scales = [1e-3]
        clipping_bounds = [1e-3]
        L_values = [1000]
        N_dp_seeds = 4
        tag = 'testing'
        should_overwrite = True

    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, data_type, mean, model_noise_std)

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)

    param_combinations = list(itertools.product(max_eps_values, dp_noise_scales, clipping_bounds))
    timestr = time.strftime("%m-%d;%H:%M:%S")
    path = output_base_dir
    path = path + 'logs/gs_global_ana/' + tag + '/'
    try:
        os.makedirs(path)
    except OSError:
        print('Duplicate tag being used')
    log_file_path = path + 'results.txt'
    csv_file_path = path + 'results.csv'

    searched_params = []
    # if some results have already been processed with this tag
    if os.path.exists(csv_file_path):
        # if overwriting, delete the results files
        if should_overwrite:
            os.remove(csv_file_path)
            os.remove(log_file_path)
        else:
            # do not duplicate experiments
            searched_params = get_params_from_csv(csv_file_path)

    min_kl = 10000
    ray.init()

    dp_seeds = np.arange(1, N_dp_seeds + 1)

    experiment_counter = 1
    for param_combination in param_combinations:
        try:
            if param_combination in searched_params:
                experiment_counter += 1
                # skip, but dont resave...
                print('Skipping: ' + str(param_combination))
                continue
            print('Running for: ' + str(param_combination))
            max_eps = param_combination[0]
            dp_noise_scale = param_combination[1]
            clipping_bound = param_combination[2]

            eps_i = np.zeros(N_dp_seeds)
            kl_i = np.zeros(N_dp_seeds)

            results_objects = []
            log_moments = generate_log_moments(5, 32, dp_noise_scale, 5)

            # this code parallises the executtion of multiple seeds at once, then collects results for them.

            # start everything running...
            for ind, seed in enumerate(dp_seeds):
                # hack to cache results

                results = run_global_dp_analytical_pvi_sync.remote(None, mean, seed, max_eps, x_train, y_train,
                                                                   model_noise_std,
                                                                   data_func,
                                                                   dp_noise_scale, no_workers, damping, no_intervals,
                                                                   clipping_bound, output_base_dir, log_moments)
                results_objects.append((results, ind))

            # fetch one by one
            for results_tup in results_objects:
                [results_obj, ind] = results_tup
                results = ray.get(results_obj)
                eps = results[0]
                kl = results[1]
                eps_i[ind] = eps
                kl_i[ind] = kl
                tracker_array = results[2]
                fname = path + 'e{}s{}.csv'.format(experiment_counter, ind)
                np.savetxt(fname, tracker_array, delimiter=',')

            eps = np.mean(eps_i)
            kl = np.mean(kl_i)
            eps_var = np.var(eps_i)
            kl_var = np.var(kl_i)

            if kl < min_kl:
                print('New Min KL: {}'.format(kl))
                print(param_combination)
                min_kl = kl

            text_file = open(log_file_path, "a")
            text_file.write(
                "max eps: {} eps: {} eps_var: {:.4e} dp_noise: {} c: {} kl: {} kl_var: {:.4e} experiment_counter: {}\n".format(max_eps, eps,
                                                                                                        eps_var,
                                                                                                        dp_noise_scale,
                                                                                                        clipping_bound,
                                                                                                        kl, kl_var, experiment_counter))
            text_file.close()
            csv_file = open(csv_file_path, "a")
            csv_file.write(
                "{},{},{:.4e},{},{},{},{:.4e},{}\n".format(max_eps, eps, eps_var, dp_noise_scale, clipping_bound, kl,
                                                        kl_var, experiment_counter))
            csv_file.close()
            experiment_counter += 1
        except Exception, e:
            traceback.print_exc()
            continue
