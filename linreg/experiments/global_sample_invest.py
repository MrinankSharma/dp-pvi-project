import argparse
import copy
import hashlib
import json
import os
import pprint
import traceback

import numpy as np
import ray
import tensorflow as tf

import linreg.data as data
from linreg.experiments.linreg_global_dp_ana_pvi_sync import run_global_dp_analytical_pvi_sync
from linreg.file_utils import get_experiment_tags_from_csv
from linreg.inference_utils import generateDictCombinations
from linreg.log_moment_utils import generate_log_moments

parser = argparse.ArgumentParser(description="grid search for whole client level dp")
parser.add_argument("--output-base-dir", default='', type=str,
                    help="output base folder.")
parser.add_argument("--tag", default='default', type=str)
parser.add_argument("--overwrite", dest='overwrite', action='store_true')
parser.add_argument("--testing", dest='testing', action='store_true')
parser.add_argument("--no-workers", default=20, type=int,
                    help="num_workers.")
parser.add_argument("--N-seeds", default=5, type=int,
                    help="output base folder.")
parser.add_argument("--not-noisy", dest='not_noisy', action='store_true')

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

    if args.not_noisy:
        config = "not_noisy"
    else:
        config = "noisy_worker"

    full_experiment_setup = {
        "dataset": {
            "dataset": 'toy_1d',
            "data_type": 'homous',
            "mean": "sample",
            "model_noise_std": "sample",
            "points_per_worker": 10,
        },
        "clipping_config": ["clipped_worker"],
        "noise_config": config,
        "N_seeds": args.N_seeds,
        "prior_std": 5,
        "tag": tag,
        "num_workers": no_workers,
        "num_intervals": 500,
        "output_base_dir": output_base_dir,
        "dp_noise_scale": 5,
        "clipping_bound": [1, 5, 10, 50, 100],
        "local_damping": 0,
        "learning_rate": [{
            "scheme": "constant",
            "start_value": [0.1, 0.2, 0.5],
        }],
        "max_eps": np.inf,
        "convergence_threshold": "disabled",
        "convergence_length": 50
    }

    if testing:
        full_experiment_setup["N_seeds"] = 2
        full_experiment_setup["dataset"]["mean"] = "sample"
        full_experiment_setup["num_intervals"] = 250
        full_experiment_setup["dp_noise_scale"] = 10
        full_experiment_setup["clipping_bound"] = [100]
        full_experiment_setup["num_workers"] = 1
        full_experiment_setup["learning_rate"] = [{
            "scheme": "constant",
            "start_value": [0.5],
        }]
        full_experiment_setup["max_eps"] = 10

        tag = 'testing'
        should_overwrite = True

    path = output_base_dir + 'logs/gs_global_sampled_ana/' + tag + '/'

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

    experiment_counter = 0
    for ind, seed in enumerate(dp_seeds):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        dataset, exact_params, sampled_params = data.generate_random_dataset(full_experiment_setup)

        for experiment_setup in experiment_list:
            full_setup = copy.deepcopy(experiment_setup)
            full_setup["seed_used"] = seed
            full_setup["dataset"]["mean_val"] = sampled_params[0]
            full_setup["dataset"]["model_noise_std"] = sampled_params[1]
            full_setup["exact_mean_pres"] = exact_params[0]
            full_setup["exact_pres"] = exact_params[1]
            experiment_code = hashlib.sha1(json.dumps(full_setup, sort_keys=True)).hexdigest()
            print(experiment_code)

            print("Sampled Parameters: {}".format(sampled_params))
            print("Exact Inference Params: {}".format(exact_params))

            if experiment_code in alreadyRunExperiments and not should_overwrite:
                print("Skipping Experiment")
                pprint.pprint(full_setup, width=1)
                print("Experiment Skipped \n\n")
                continue

            try:
                pprint.pprint(full_setup, width=1)
                print("Calc Log Moments")
                log_moments = generate_log_moments(no_workers, 32, full_experiment_setup['dp_noise_scale'], no_workers)
                print("Calc Log Moments - Done \n\n")
                results = run_global_dp_analytical_pvi_sync.remote(full_setup, seed, dataset, log_moments=None)
                [eps, kl, tracker] = ray.get(results)
                fname = path + 'e{}.csv'.format(experiment_counter, ind)
                np.savetxt(fname, tracker, delimiter=',')

                text_file = open(log_file_path, "a")
                results_array = [experiment_counter, eps, kl, sampled_params[0], sampled_params[1], exact_params[0],
                                 exact_params[1], full_setup['clipping_bound'], full_setup['learning_rate']['start_value'],
                                 full_setup["num_workers"], full_setup["dataset"]["points_per_worker"], experiment_code]
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
            except Exception, e:
                traceback.print_exc()
                continue
