import numpy as np
import tensorflow as tf
import traceback

import argparse

import linreg.data as data
from linreg.linreg_dp_ana_pvi_sync import run_dp_analytical_pvi_sync
from linreg.file_utils import get_params_from_csv
import itertools
import time
import os
import ray

from linreg.log_moment_utils import generate_log_moments

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
    N_dp_seeds = 10
    # will stop when the privacy budget is reached!
    no_intervals = 5000

    max_eps_values = [1, 100, 10000]
    dp_noise_scales = [1e-3, 1e-2, 0.1, 1, 10]
    clipping_bounds = [1e-3, 1e-1, 1, 1e1, 1e2]
    L_values = [1000]

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

    param_combinations = list(itertools.product(max_eps_values, dp_noise_scales, clipping_bounds, L_values))

    timestr = time.strftime("%m-%d;%H:%M:%S")
    path = output_base_dir
    path = output_base_dir + 'logs/gs_local_ana/' + tag + '/'
    try:
        os.makedirs(path)
    except OSError:
        print('Duplicate tag being used')
    log_file_path = path + 'results.txt'
    csv_file_path = path + 'results.csv'

    dp_seeds = np.arange(1, N_dp_seeds + 1)

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

    experiment_counter = 0
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
            L = param_combination[3]

            eps_i = np.zeros(N_dp_seeds)
            kl_i = np.zeros(N_dp_seeds)

            results_objects = []
            log_moments = generate_log_moments(1000, 32, dp_noise_scale, L)

            # start everything running...
            for ind, seed in enumerate(dp_seeds):
                results = run_dp_analytical_pvi_sync.remote(mean, seed, max_eps, x_train, y_train,
                                                            model_noise_std,
                                                            data_func,
                                                            dp_noise_scale, no_workers, damping, no_intervals,
                                                            clipping_bound, L, output_base_dir, log_moments)
                results_objects.append((results, ind))

            # fetch one by one
            for results_tup in results_objects:
                [results_obj, ind] = results_tup
                results = ray.get(results_obj)
                eps = results[0]
                kl = results[1]
                eps_i[ind] = eps
                kl_i[ind] = kl
                # save the tracker_array
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

            # print(log_file)
            # print('logging!')
            text_file = open(log_file_path, "a")
            text_file.write(
                "max_eps: {} eps: {} eps_var: {:.4e} dp_noise: {} c: {} kl: {} kl_var: {:.4e} experiment_counter:{} L:{}\n".format(
                    max_eps,
                    eps,
                    eps_var,
                    dp_noise_scale,
                    clipping_bound,
                    kl,
                    kl_var,
                    experiment_counter,
                    L))
            text_file.close()
            csv_file = open(csv_file_path, "a")
            csv_file.write(
                "{},{},{:.4e},{},{},{},{:.4e},{},{}\n".format(max_eps, eps, eps_var, dp_noise_scale, clipping_bound,
                                                              kl,
                                                              kl_var, experiment_counter, L))
            csv_file.close()
            experiment_counter += 1
        except Exception, e:
            traceback.print_exc()
            continue
