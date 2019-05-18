import numpy as np
import tensorflow as tf
import traceback
import hashlib
import argparse
import copy
import os
import ray
import json

from linreg.linreg_dp_ana_pvi_sync import run_dp_analytical_pvi_sync
from linreg.data import generate_random_dataset
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
parser.add_argument("--N-seeds", default=30, type=int,
                    help="output base folder.")

if __name__ == "__main__":
    args = parser.parse_args()
    output_base_dir = args.output_base_dir
    should_overwrite = args.overwrite
    testing = args.testing
    tag = args.tag
    no_workers = args.no_workers

    full_experiment_setup = {
        "dataset": {
            "dataset": 'toy_1d',
            "data_type": 'homous',
            "mean_vals": "sample",
            "model_noise_std": "sample",
            "points_per_worker": 10,
        },
        "model_config": "clipped_noisy",
        "N_seeds": args.N_seeds,
        "prior_std": 5,
        "tag": tag,
        "num_workers": no_workers,
        "num_intervals": 250,
        "output_base_dir": output_base_dir,
        "max_eps": 10,
        "dp_noise_scale": 5,
        "clipping_bound": [0.25, 0.5, 1, 2, 4, 8],
        "learning_rate": 0.5,
    }

    if testing:
        full_experiment_setup["model_config"] = ["clipped_noisy"]
        full_experiment_setup["N_seeds"] = 2
        full_experiment_setup["clipping_bound"] = [1, 2]
        full_experiment_setup["num_workers"] = 2
        tag = 'testing'
        should_overwrite = True

    path = output_base_dir
    path = output_base_dir + 'logs/gs_local_final_ana/' + tag + '/'

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
    experiment_counter = 0

    log_moments = generate_log_moments(full_experiment_setup['num_workers'], 32,
                                       full_experiment_setup['dp_noise_scale'], full_experiment_setup['num_workers'])

    for ind, seed in enumerate(dp_seeds):
        # generate dataset for this seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        dataset, exact_params, sampled_params = generate_random_dataset(full_experiment_setup)
        for experiment_setup in experiment_list:
            try:
                print("Sampled Parameters: {}".format(sampled_params))
                print("Exact Inference Params: {}".format(exact_params))

                full_setup = copy.deepcopy(experiment_setup)
                full_setup["dataset"]["mean_val"] = sampled_params[0]
                full_setup["dataset"]["model_noise_std"] = sampled_params[1]
                full_setup["exact_mean_pres"] = exact_params[0]
                full_setup["exact_pres"] = exact_params[1]

                results = run_dp_analytical_pvi_sync.remote(full_setup, seed, dataset, log_moments)
                [eps, kl, tracker] = ray.get(results)

                fname = path + 'e{}.csv'.format(experiment_counter, ind)
                np.savetxt(fname, tracker, delimiter=',')

                text_file = open(log_file_path, "a")
                results_array = [experiment_counter, eps, kl, sampled_params[0], sampled_params[1], exact_params[0], exact_params[1],
                                 full_setup['clipping_bound']]
                text_file.write(
                    """experiment_counter:{} eps: {} kl: {} mean: {} noise_e: {} exact_mean_pres: {:.4e}
                     exact_pres: {:.4e} c: {}\n\n""".format(
                        *results_array))
                text_file.close()
                csv_file = open(csv_file_path, "a")
                csv_file.write(
                    "{},{},{},{},{},{},{},{}\n".format(*results_array))
                csv_file.close()
                experiment_counter += 1
            except Exception, e:
                traceback.print_exc()
                continue
