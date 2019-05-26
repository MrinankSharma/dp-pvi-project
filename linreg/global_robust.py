import numpy as np
import tensorflow as tf
import traceback
import pprint
import argparse
import copy
import hashlib

import linreg.data as data
from linreg.linreg_global_dp_ana_pvi_sync import run_global_dp_analytical_pvi_sync
from linreg.inference_utils import generateDictCombinations
from linreg.file_utils import get_experiment_tags_from_csv
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
parser.add_argument("--N-seeds", default=50, type=int,
                    help="output base folder.")
parser.add_argument("--ppw", dest="ppw", action='store_true')
parser.add_argument("--exp", dest="exp", default=0, type=int)


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
    np.random.seed(42)
    tf.set_random_seed(42)

    args = parser.parse_args()
    output_base_dir = args.output_base_dir
    should_overwrite = args.overwrite
    testing = args.testing
    tag = args.tag

    if args.ppw:
        full_experiment_setup = {
            "dataset": {
                "dataset": 'toy_1d',
                "data_type": 'homous',
                "mean": "sample",
                "model_noise_std": "sample",
                "points_per_worker": [10, 30, 50, 70, 90],
            },
            "clipping_config": ["clipped_worker"],
            "noise_config": ["noisy_worker"],
            "N_seeds": args.N_seeds,
            "prior_std": 5,
            "tag": tag,
            "num_workers": 20,
            "num_intervals": 250,
            "output_base_dir": output_base_dir,
            "dp_noise_scale": 5,
            "clipping_bound": {
                "type": "scaled",
                "value": 0.5,
            },
            "local_damping": 0,
            "learning_rate": [{
                "scheme": "constant",
                "start_value": 0.1,
            }],
            "max_eps": 10,
            "convergence_threshold": "disabled",
            "convergence_length": 50
        }
    else:
        full_experiment_setup = {
            "dataset": {
                "dataset": 'toy_1d',
                "data_type": 'homous',
                "mean": "sample",
                "model_noise_std": "sample",
                "points_per_worker": 10,
            },
            "clipping_config": ["clipped_worker"],
            "noise_config": ["noisy_worker"],
            "N_seeds": args.N_seeds,
            "prior_std": 5,
            "tag": tag,
            "num_workers": [10, 20, 30, 40, 50],
            "num_intervals": 250,
            "output_base_dir": output_base_dir,
            "dp_noise_scale": 5,
            "clipping_bound": {
                "type": "scaled",
                "value": 0.5,
            },
            "local_damping": 0,
            "learning_rate": [{
                "scheme": "constant",
                "start_value": 0.1,
            }],
            "max_eps": 10,
            "convergence_threshold": "disabled",
            "convergence_length": 50
        }

    if args.exp == 3:
        full_experiment_setup = {
            "dataset": {
                "dataset": 'toy_1d',
                "data_type": 'homous',
                "mean": "sample",
                "model_noise_std": "sample",
                "points_per_worker": 10,
            },
            "clipping_config": "clipped_worker",
            "noise_config": ["noisy_worker"],
            "N_seeds": args.N_seeds,
            "prior_std": 5,
            "tag": tag,
            "num_workers": 20,
            "num_intervals": 250,
            "output_base_dir": output_base_dir,
            "dp_noise_scale": 5,
            "clipping_bound": [
                {
                    "type": "scaled",
                    "value": 0.1,
                },
                {
                    "type": "scaled",
                    "value": 0.5,
                },
                {
                    "type": "scaled",
                    "value": 1,
                },
                {
                    "type": "scaled",
                    "value": 5,
                },
                {
                    "type": "scaled",
                    "value": 10,
                },
            ],
            "local_damping": 0,
            "learning_rate": [
                {
                    "scheme": "constant",
                    "start_value": 0.1,
                },
                {
                    "scheme": "constant",
                    "start_value": 0.2,
                },
                {
                    "scheme": "constant",
                    "start_value": 0.5,
                }
            ],
            "max_eps": 10,
            "convergence_threshold": "disabled",
            "convergence_length": 50
        }
    elif args.exp == 4:
        full_experiment_setup = {
            "dataset": {
                "dataset": 'toy_1d',
                "data_type": 'homous',
                "mean": "sample",
                "model_noise_std": "sample",
                "points_per_worker": 10,
            },
            "clipping_config": ["clipped_worker"],
            "noise_config": ["not_noisy"],
            "N_seeds": args.N_seeds,
            "prior_std": 5,
            "tag": tag,
            "num_workers": 20,
            "num_intervals": 450,
            "output_base_dir": output_base_dir,
            "dp_noise_scale": 5,
            "clipping_bound": [
                {
                    "type": "scaled",
                    "value": 0.1,
                },
                {
                    "type": "scaled",
                    "value": 0.5,
                },
                {
                    "type": "scaled",
                    "value": 1,
                },
                {
                    "type": "scaled",
                    "value": 5,
                },
                {
                    "type": "scaled",
                    "value": 10,
                },
            ],
            "local_damping": 0,
            "learning_rate": [{
                "scheme": "constant",
                "start_value": 0.5,
            }],
            "max_eps": 10,
            "convergence_threshold": "disabled",
            "convergence_length": 50
        }

    if testing:
        full_experiment_setup["N_seeds"] = 2
        full_experiment_setup["dataset"]["mean"] = "sample"
        full_experiment_setup["num_intervals"] = 250
        full_experiment_setup["dp_noise_scale"] = 10
        full_experiment_setup["clipping_bound"] = {
            "type": "scaled",
            "value": 0.5,
        }
        full_experiment_setup["num_workers"] = [2, 4]
        full_experiment_setup["learning_rate"] = [{
            "scheme": "constant",
            "start_value": [0.1],
        }]
        full_experiment_setup["max_eps"] = 10

        tag = 'testing'
        # should_overwrite = True

    path = output_base_dir + 'logs/gs_global_robust_ana/' + tag + '/'

    try:
        os.makedirs(path)
    except OSError:
        print('Duplicate tag being used')
    log_file_path = path + 'results.txt'
    csv_file_path = path + 'results.csv'

    setup_file = path + 'setup.json'
    with open(setup_file, 'w') as outfile:
        json.dump(full_experiment_setup, outfile)

    dp_seeds = np.arange(1, full_experiment_setup['N_seeds'] + 1)
    ray.init()

    experiment_list = generateDictCombinations(full_experiment_setup)

    alreadyRunExperiments = []
    if os.path.exists(csv_file_path):
        # if overwriting, delete the results files
        if should_overwrite:
            os.remove(csv_file_path)
            os.remove(log_file_path)
        else:
            # do not duplicate experiments
            alreadyRunExperiments = get_experiment_tags_from_csv(csv_file_path, offset_from_end=0)
            print(alreadyRunExperiments)

    experiment_counter = 0
    all_results_objects = []
    for experiment_setup in experiment_list:
        for ind, seed in enumerate(dp_seeds):
            full_setup = copy.deepcopy(experiment_setup)
            dataset, exact_params, sampled_params = data.generate_random_dataset(full_setup)
            full_setup["seed_used"] = seed
            full_setup["dataset"]["mean_val"] = sampled_params[0]
            full_setup["dataset"]["model_noise_std"] = sampled_params[1]
            full_setup["exact_mean_pres"] = exact_params[0]
            full_setup["exact_pres"] = exact_params[1]
            full_setup["clipping_bound"] = full_setup["clipping_bound"]["value"] * full_setup["dataset"][
                "points_per_worker"]
            np.random.seed(seed)
            tf.set_random_seed(seed)
            experiment_code = hashlib.sha1(json.dumps(full_setup, sort_keys=True)).hexdigest()
            print(experiment_code)

            print("Sampled Parameters: {}".format(sampled_params))
            print("Exact Inference Params: {}".format(exact_params))

            if experiment_code in alreadyRunExperiments and not should_overwrite:
                print("Skipping Experiment")
                pprint.pprint(full_setup, width=1)
                experiment_counter += 1
                print("Experiment Skipped \n\n")
                continue

            try:
                pprint.pprint(full_setup, width=1)
                # print("Calc Log Moments")
                # only thing that matters is q
                log_moments = generate_log_moments(1, 32, full_experiment_setup['dp_noise_scale'], 1)
                # print("Calc Log Moments - Done \n\n")

                # set things running so we can get the results later - this ought to boost performance.
                results = run_global_dp_analytical_pvi_sync.remote(full_setup, seed, dataset, log_moments=None)
                results_array = [experiment_counter, -1, -1, sampled_params[0], sampled_params[1], exact_params[0],
                                 exact_params[1], full_setup['clipping_bound'],
                                 full_setup['learning_rate']['start_value'],
                                 full_setup["num_workers"], full_setup["dataset"]["points_per_worker"], experiment_code]
                results_obj = [results_array, results]
                experiment_counter += 1
                all_results_objects.append(results_obj)

                # flush every five experiments.
                if seed % 5 == 4:
                    for results_obj_i in all_results_objects:
                        [eps, kl, tracker] = ray.get(results_obj_i[1])
                        results_array = results_obj_i[0]
                        results_array[1] = eps
                        results_array[2] = kl
                        experiment_counter = results_array[0]
                        fname = path + 'e{}.csv'.format(experiment_counter)
                        np.savetxt(fname, tracker, delimiter=',')

                        text_file = open(log_file_path, "a")
                        text_file.write(
                            "experiment_counter:{} eps: {} kl: {} mean: {} noise_e: {} exact_mean_pres: {:.4e} \
                             exact_pres: {:.4e} c: {} eta: {} num_workers: {} points_per_worker:{} code: {} \n\n".format(
                                *results_array))
                        text_file.close()
                        csv_file = open(csv_file_path, "a")
                        csv_file.write(
                            "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*results_array))
                        csv_file.close()
                        experiment_counter += 1
                    all_results_objects = []

            except Exception, e:
                traceback.print_exc()
                continue

        for results_obj_i in all_results_objects:
            [eps, kl, tracker] = ray.get(results_obj_i[1])
            results_array = results_obj_i[0]
            results_array[1] = eps
            results_array[2] = kl
            experiment_counter = results_array[0]
            fname = path + 'e{}.csv'.format(experiment_counter)
            np.savetxt(fname, tracker, delimiter=',')

            text_file = open(log_file_path, "a")
            text_file.write(
                "experiment_counter:{} eps: {} kl: {} mean: {} noise_e: {} exact_mean_pres: {:.4e} \
                 exact_pres: {:.4e} c: {} eta: {} num_workers: {} points_per_worker:{} code: {} \n\n".format(
                    *results_array))
            text_file.close()
            csv_file = open(csv_file_path, "a")
            csv_file.write(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*results_array))
            csv_file.close()
            experiment_counter += 1
        all_results_objects = []
