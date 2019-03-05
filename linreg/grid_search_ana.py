import numpy as np
import tensorflow as tf

import linreg.data as data
from linreg.linreg_dp_ana_pvi_sync import run_dp_analytical_pvi_sync
import itertools
import time
import os
import ray

if __name__ == "__main__":
    # really, we should average over multiple seeds
    seed = 42
    dataset = 'toy_1d'
    data_type = "homous"
    mean = 2
    model_noise_std = 0.5
    no_workers = 5
    damping = 0
    N_dp_seeds = 5
    # will stop when the privacy budget is reached!
    no_intervals = 100

    dp_seeds = np.arange(1, N_dp_seeds)

    max_eps_values = [1, 10, 30, 100]
    dp_noise_scales = [1e-3, 1e-2, 0.1, 1, 10]
    clipping_bounds = [1e-5, 1e-2, 1, 1e2, 1e5]
    L_values = [10, 100, 500]

    max_eps_values = [1]
    dp_noise_scales = [1e-3]
    clipping_bounds = [1]
    L_values = [10, 1000]

    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, data_type, mean, model_noise_std)

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)

    param_combinations = list(itertools.product(max_eps_values, dp_noise_scales, clipping_bounds, L_values))

    timestr = time.strftime("%m-%d;%H:%M:%S")
    path = 'logs/gs_local_linreg_dp_ana/' + timestr + '/'
    os.makedirs(path)
    log_file = path + 'results.txt'
    min_kl = 10000
    ray.init()
    for param_combination in param_combinations:

        print('Running for: ' + str(param_combination))
        max_eps = param_combination[0]
        dp_noise_scale = param_combination[1]
        clipping_bound = param_combination[2]
        L = param_combination[3]

        eps_i = np.zeros(N_dp_seeds)
        kl_i = np.zeros(N_dp_seeds)

        for ind, seed in enumerate(dp_seeds):
            results = run_dp_analytical_pvi_sync(mean, seed, max_eps, x_train, y_train, model_noise_std, data_func,
                                                 dp_noise_scale, no_workers, damping, no_intervals, clipping_bound, L)
            eps = results[0]
            kl = results[1]
            eps_i[ind] = eps
            kl_i[ind] = kl

        eps = np.mean(eps_i)
        kl = np.mean(kl_i)
        eps_var = np.var(eps_i)
        kl_var = np.var(kl_i)

        if kl < min_kl:
            print('New Min KL: {}'.format(kl))
            print(param_combination)
            min_kl = kl

        text_file = open(log_file, "a")
        text_file.write(
            "max eps: {} eps: {} eps_var: {:.4e} dp_noise: {} c: {} kl: {} kl_var: {:.4e}  L:{}\n".format(max_eps, eps,
                                                                                                          eps_var,
                                                                                                          dp_noise_scale,
                                                                                                          clipping_bound,
                                                                                                          kl, kl_var,
                                                                                                          L))
        text_file.close()
