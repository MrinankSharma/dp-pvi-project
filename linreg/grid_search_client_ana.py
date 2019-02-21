import numpy as np
import tensorflow as tf

import linreg.data as data
from linreg_client_dp_ana_pvi_sync import run_client_dp_analytical_pvi_sync

if __name__ == "__main__":
    # really, we should average over multiple seeds
    seed = 42
    dataset = 'toy_1d'
    data_type = "homous"
    mean = 2
    model_noise_std = 0.5
    no_workers = 5
    damping = 0
    # will stop when the privacy budget is reached!
    no_intervals = 1000

    max_eps = 30
    dp_noise_scale = 1
    clipping_bound = 5

    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset == 'toy_1d':
        data_func = lambda idx, N: data.get_toy_1d_shard(idx, N, data_type, mean, model_noise_std)

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)


    run_client_dp_analytical_pvi_sync(None, mean, seed, max_eps, x_train, y_train, model_noise_std, data_func,
                                      dp_noise_scale, no_workers, damping, no_intervals, clipping_bound)