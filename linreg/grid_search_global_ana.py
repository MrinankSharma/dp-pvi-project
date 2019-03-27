import numpy as np
import tensorflow as tf
import traceback

import argparse

import linreg.data as data
from linreg_global_dp_ana_pvi_sync import run_global_dp_analytical_pvi_sync
from log_moment_utils import generate_log_moments
from linreg.file_utils import get_params_from_csv
from linreg.inference_utils import exact_inference
import itertools
import time
import os
import json
import ray

parser = argparse.ArgumentParser(description="grid search for whole client level dp")
parser.add_argument("--output-base-dir", default='', type=str,
                    help="output base folder.")
parser.add_argument("--tag", default='default', type=str)
parser.add_argument("--overwrite", dest='overwrite', action='store_true')
parser.add_argument("--testing", dest='testing', action='store_true')
parser.add_argument("--average", dest='average', action='store_true')
parser.add_argument("--no-workers", default=20, type=int,
                    help="output base folder.")

if __name__ == "__main__":
    args = parser.parse_args()
    output_base_dir = args.output_base_dir
    should_overwrite = args.overwrite
    testing = args.testing
    average = args.average
    tag = args.tag
    no_workers = args.no_workers

    experiment_setup = {
        "seed": 42,
        "dataset": 'toy_1d',
        "data_type": 'homous',
        "mean": 2,
        "model_noise_std": 0.5,
        "prior_std": 5,
        "N_dp_seeds": 10,
        "points_per_worker": 10,
        "tag": tag,
        "num_workers": no_workers,
        "num_intervals": 250,
        "output_base_dir": output_base_dir
    }

    max_eps_values = [np.inf]
    dp_noise_scales = [1]
    damping_vals = [0.25, 0.5, 0.9]
    clipping_bounds = [10, 20, 40, 60, 100]

    if testing:
        max_eps_values = [np.inf]
        dp_noise_scales = [1]
        clipping_bounds = [10]
        damping_vals = [0.9]
        experiment_setup["N_dp_seeds"] = 1
        tag = 'testing'
        should_overwrite = True

    np.random.seed(experiment_setup['seed'])
    tf.set_random_seed(experiment_setup['seed'])

    if experiment_setup['dataset'] == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, experiment_setup['data_type'],
                                                         experiment_setup['mean'], experiment_setup['model_noise_std'],
                                                         experiment_setup['points_per_worker'] * experiment_setup[
                                                             'num_workers'])

    workers_data = [data_func(w_i, no_workers) for w_i in range(no_workers)]
    x_train = np.array([[]])
    y_train = np.array([])
    for worker_data in workers_data:
        x_train = np.append(x_train, worker_data[0])
        y_train = np.append(y_train, worker_data[1])

    param_combinations = list(itertools.product(max_eps_values, dp_noise_scales, clipping_bounds, damping_vals))

    _, _, exact_mean_pres, exact_pres = exact_inference(x_train, y_train, experiment_setup['prior_std'] ** 2,
                                                        experiment_setup['model_noise_std'] ** 2)

    experiment_setup['exact_mean_pres'] = exact_mean_pres
    experiment_setup['exact_pres'] = exact_pres

    path = output_base_dir + 'logs/gs_global_ana/' + tag + '/'
    try:
        os.makedirs(path)
    except OSError:
        print('Duplicate tag being used')
    log_file_path = path + 'results.txt'
    csv_file_path = path + 'results.csv'

    setup_file = path + 'setup.json'
    with open(setup_file, 'w') as outfile:
        json.dump(experiment_setup, outfile)

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

    dp_seeds = np.arange(1, experiment_setup["N_dp_seeds"] + 1)

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
            damping_val = param_combination[3]

            eps_i = np.zeros(experiment_setup["N_dp_seeds"])
            kl_i = np.zeros(experiment_setup["N_dp_seeds"])

            results_objects = []
            log_moments = generate_log_moments(no_workers, 32, dp_noise_scale, no_workers)

            # this code parallises the executtion of multiple seeds at once, then collects results for them.
            # start everything running...
            for ind, seed in enumerate(dp_seeds):
                results = run_global_dp_analytical_pvi_sync.remote(experiment_setup, seed, max_eps, workers_data,
                                                                   dp_noise_scale, no_workers, damping_val,
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

            text_file = open(log_file_path, "a")
            results_array = [max_eps, eps, eps_var, dp_noise_scale, clipping_bound, kl, kl_var, experiment_counter,
                             damping_val]
            text_file.write(
                """max eps: {} eps: {} eps_var: {:.4e} dp_noise: {} c: {} kl: {} 
                kl_var: {:.4e} experiment_counter: {} damping: {}\n""".format(*results_array))
            text_file.close()
            csv_file = open(csv_file_path, "a")
            csv_file.write(
                "{},{},{:.4e},{},{},{},{:.4e},{},{}\n".format(*results_array))
            csv_file.close()

            experiment_counter += 1
        except Exception, e:
            traceback.print_exc()
            continue
