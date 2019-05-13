import numpy as np
import tensorflow as tf
import traceback
import pprint
import argparse
import copy
import hashlib

import linreg.data as data
from linreg.linreg_global_dp_ana_pvi_sync import run_global_dp_analytical_pvi_sync
from linreg.inference_utils import exact_inference, generateDictCombinations
from linreg.file_utils import get_experiment_tags_from_csv, get_experiment_tag_params
import os
import ray
import json

from linreg.log_moment_utils import generate_log_moments

parser = argparse.ArgumentParser(description="grid search for whole client level dp")
parser.add_argument("--output-base-dir", default='', type=str,
                    help="output base folder.")
parser.add_argument("--tag", default='default', type=str)
parser.add_argument("--overwrite", dest='overwrite', action='store_true')
parser.add_argument("--testing", dest='testing', action='store_true')
parser.add_argument("--no-workers", default=20, type=int,
                    help="num_workers.")
parser.add_argument("--N-dp-seeds", default=5, type=int,
                    help="output base folder.")


def generate_datasets(experiment_setup):
    dataset_setup = experiment_setup["dataset"]
    datasets = []
    exact_params = []
    combinations = []
    dataset_setups = []
    for mean_val in dataset_setup["mean"]:
        if dataset_setup['dataset'] == 'toy_1d':
            data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, dataset_setup['data_type'],
                                                             mean_val,
                                                             dataset_setup['model_noise_std'],
                                                             experiment_setup['num_workers'] * dataset_setup[
                                                                 'points_per_worker'])

        workers_data = [data_func(w_i, no_workers) for w_i in range(no_workers)]
        x_train = np.array([[]])
        y_train = np.array([])
        for worker_data in workers_data:
            x_train = np.append(x_train, worker_data[0])
            y_train = np.append(y_train, worker_data[1])

        _, _, exact_mean_pres, exact_pres = exact_inference(x_train, y_train, experiment_setup['prior_std'],
                                                            dataset_setup['model_noise_std'] ** 2)
        specific_setup = copy.deepcopy(dataset_setup)
        specific_setup["mean"] = mean_val
        exact_params.append([exact_mean_pres, exact_pres])
        datasets.append(workers_data)
        combinations.append(mean_val)
        dataset_setups.append(specific_setup)

    return datasets, exact_params, combinations, dataset_setups


def clippingConfigToInt(cfg):
    if cfg == "not_clipped":
        return 0
    elif cfg == "clipped_worker":
        return 1
    elif cfg == "clipped_server":
        return 2


def noiseConfigToInt(cfg):
    if cfg == "not_noisy":
        return 0
    elif cfg == "noisy":
        return 1
    elif cfg == "noisy_worker":
        return 2


def learningRateToInts(cfg):
    scheme = cfg["scheme"]
    start_value = cfg["start_value"]
    if scheme == "constant":
        return [1, start_value, 0, 0]
    elif scheme == "step":
        return [2, start_value, cfg["factor"], cfg["interval"]]
    elif scheme == "exponential":
        return [3, start_value, cfg["alpha"], 0]


if __name__ == "__main__":
    args = parser.parse_args()
    output_base_dir = args.output_base_dir
    should_overwrite = args.overwrite
    testing = args.testing
    tag = args.tag
    no_workers = args.no_workers

    experiment_setup = {
        "seed": 42,
        "dataset": {
            "dataset": 'toy_1d',
            "data_type": 'homous',
            "mean": [2],
            "model_noise_std": 0.5,
            "points_per_worker": 10,
        },
        "clipping_config": ["clipped_worker"],
        "noise_config": ["noisy_worker"],
        "N_dp_seeds": args.N_dp_seeds,
        "prior_std": 5,
        "tag": tag,
        "num_workers": no_workers,
        "num_intervals": 250,
        "output_base_dir": output_base_dir,
        "dp_noise_scale": [1, 4, 8],
        "clipping_bound": [0.1, 1, 10],
        "local_damping": 0,
        "learning_rate": [
            {
                "scheme": "constant",
                "start_value": [0.15, 0.5, 0.85],
            },
            {
                "scheme": "step",
                "start_value": [0.15, 0.5, 0.85],
                "factor": 0.5,
                "interval": 20,
            },
            {
                "scheme": "step",
                "start_value": [0.15, 0.5, 0.85],
                "factor": 0.8,
                "interval": 10,
            },
            {
                "scheme": "exponential",
                "start_value": [0.15, 0.5, 0.85],
                "alpha": [0.05, 0.03],
            }
        ],
        "max_eps": [10, 50, 100],
        "convergence_threshold": "disabled",
        "convergence_length": 50
    }

    if testing:
        experiment_setup["N_dp_seeds"] = 1
        experiment_setup["dataset"]["mean"] = [2]
        experiment_setup["num_intervals"] = 100
        experiment_setup["dp_noise_scale"] = 8
        experiment_setup["clipping_bound"] = 5
        experiment_setup["num_workers"] = 1
        experiment_setup["learning_rate"] = [
            {
                "scheme": "step",
                "start_value": [0.15],
                "factor": 0.5,
                "interval": 20,
            }
        ]
        experiment_setup["max_eps"] = np.inf

        tag = 'testing'
        should_overwrite = True

    np.random.seed(experiment_setup['seed'])
    tf.set_random_seed(experiment_setup['seed'])

    path = output_base_dir
    path = output_base_dir + 'logs/gs_global_bias_ana/' + tag + '/'

    try:
        os.makedirs(path)
    except OSError:
        print('Duplicate tag being used')
    log_file_path = path + 'results.txt'
    csv_file_path = path + 'results.csv'

    setup_file = path + 'setup.json'
    with open(setup_file, 'w') as outfile:
        json.dump(experiment_setup, outfile)

    dp_seeds = np.arange(1, experiment_setup['N_dp_seeds'] + 1)
    ray.init()

    datasets, exact_params, combinations, specific_setups = generate_datasets(experiment_setup)
    experiment_setup.pop("dataset")
    experiment_list = generateDictCombinations(experiment_setup)

    alreadyRunExperiments = []
    if os.path.exists(csv_file_path):
        # if overwriting, delete the results files
        if should_overwrite:
            os.remove(csv_file_path)
            os.remove(log_file_path)
        else:
            # do not duplicate experiments
            alreadyRunExperiments = get_experiment_tags_from_csv(csv_file_path)

    experiment_counter = 0
    for dataset_indx, dataset in enumerate(datasets):
        for experiment_setup in experiment_list:
            experiment_code = hashlib.sha1(json.dumps(experiment_setup, sort_keys=True)).hexdigest()
            full_setup = copy.deepcopy(experiment_setup)
            full_setup["dataset"] = specific_setups[dataset_indx]
            full_setup["exact_mean_pres"] = exact_params[dataset_indx][0]
            full_setup["exact_pres"] = exact_params[dataset_indx][1]

            print(experiment_code)

            if experiment_code in alreadyRunExperiments and not should_overwrite:
                # csv_fixed_file_path = path + 'results_fixed.csv'
                # exp_params = get_experiment_tag_params(csv_file_path, experiment_code)
                # exp_params.insert(1, experiment_setup["max_eps"])
                # csv_fixed_file = open(csv_fixed_file_path, "a")
                # csv_fixed_file.write(
                #     "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*exp_params))
                # csv_fixed_file.close()
                print("Skipping Experiment")
                pprint.pprint(full_setup, width=1)
                print("Experiment Skipped \n\n")
                continue

            try:
                pprint.pprint(full_setup, width=1)

                eps_i = np.zeros(experiment_setup['N_dp_seeds'])
                kl_i = np.zeros(experiment_setup['N_dp_seeds'])

                results_objects = []
                log_moments = generate_log_moments(no_workers, 32, experiment_setup['dp_noise_scale'], no_workers)

                # start everything running...
                for ind, seed in enumerate(dp_seeds):
                    results = run_global_dp_analytical_pvi_sync.remote(full_setup, seed, dataset, log_moments=None)
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

                text_file = open(log_file_path, "a")
                results_array_txt = [eps, full_setup["max_eps"], eps_var, kl, kl_var, experiment_counter,
                                     full_setup["dataset"]["mean"],
                                     full_setup["clipping_config"], full_setup["noise_config"],
                                     full_setup["dp_noise_scale"],
                                     full_setup["clipping_bound"], full_setup["local_damping"],
                                     exact_params[dataset_indx][0], exact_params[dataset_indx][1]]

                results_array_csv = [eps, full_setup["max_eps"], eps_var, kl, kl_var, experiment_counter,
                                     full_setup["dataset"]["mean"],
                                     clippingConfigToInt(full_setup["clipping_config"]),
                                     noiseConfigToInt(full_setup["noise_config"]),
                                     full_setup["dp_noise_scale"],
                                     full_setup["clipping_bound"], full_setup["local_damping"],
                                     exact_params[dataset_indx][0], exact_params[dataset_indx][1], experiment_code]
                results_array_txt.extend(learningRateToInts(full_setup["learning_rate"]))
                results_array_csv.extend(learningRateToInts(full_setup["learning_rate"]))
                text_file.write(
                    """eps: {} max_eps: {} eps_var: {:.4e} kl: {} kl_var: {:.4e} experiment_counter:{}
                     mean: {}, clipping_config: {}, noise_config: {}, dp_noise_scale: {}, 
                      clipping_bound: {}, local_damping: {}, 
                      exact_mean_pres: {:.4e}, exact_pres: {:.4e}, learning_rate_type: {},
                      learning_rate_start: {}, learning_rate_param1: {}, 
                      learning_rate_param2: {} \n\n""".format(
                        *results_array_txt))
                text_file.close()
                csv_file = open(csv_file_path, "a")
                csv_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*results_array_csv))
                csv_file.close()
                experiment_counter += 1
            except Exception, e:
                traceback.print_exc()
                continue
