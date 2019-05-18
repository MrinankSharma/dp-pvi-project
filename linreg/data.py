import numpy as np
import pdb
import copy

from inference_utils import exact_inference


def get_toy_1d(mean, noise_std, n_train):
    a = mean
    # n_train = 1000
    n_test = 200
    xtrain = np.random.randn(n_train, 1)
    ytrain = a * xtrain + noise_std * np.random.randn(n_train, 1)
    xtest = np.random.randn(n_test, 1)
    ind = np.argsort(xtest[:, 0])
    xtest = xtest[ind, :]
    ytest = a * xtest + 0.5 * np.random.randn(n_test, 1)
    return xtrain, ytrain.reshape([n_train]), xtest, ytest.reshape([n_test])

# simulate data in multiple shards
def get_toy_1d_shard_homogeneous(shard_idx, total_shards, mean, noise_std, n_train):
    X_train, y_train, X_test, y_test = get_toy_1d(mean, noise_std, n_train)
    N_train = X_train.shape[0]
    N_per_shard = int(N_train / total_shards)
    start_ind = shard_idx * N_per_shard
    end_ind = (shard_idx + 1) * N_per_shard
    X_i = X_train[start_ind:end_ind, :]
    y_i = y_train[start_ind:end_ind]
    return X_i, y_i, X_test, y_test

def get_toy_1d_shard_inhomogeneous(shard_idx, total_shards, mean, noise_std, n_train):
    X_train, y_train, X_test, y_test = get_toy_1d(mean, noise_std, n_train)
    ind = np.argsort(X_train[:, 0])
    X_train = X_train[ind, :]
    y_train = y_train[ind]
    N_train = X_train.shape[0]
    N_per_shard = int(N_train / total_shards)
    start_ind = shard_idx * N_per_shard
    end_ind = (shard_idx + 1) * N_per_shard
    X_i = X_train[start_ind:end_ind, :]
    y_i = y_train[start_ind:end_ind]
    return X_i, y_i, X_test, y_test

# simulate data in multiple shards with homogeneous and inhomogeneous data
def get_toy_1d_shard(shard_idx, total_shards, option='homous', mean=2, noise_std=1, n_train=50):
    if option == 'homous':
        return get_toy_1d_shard_homogeneous(shard_idx, total_shards, mean, noise_std, n_train)
    elif option == 'inhomous':
        return get_toy_1d_shard_inhomogeneous(shard_idx, total_shards, mean, noise_std, n_train)
    else:
        print 'unknown option, returning nothing!'

def generate_datasets(experiment_setup):
    dataset_setup = experiment_setup["dataset"]
    datasets = []
    exact_params = []
    combinations = []
    dataset_setups = []
    for mean_val in dataset_setup["mean"]:
        if dataset_setup['dataset'] == 'toy_1d':
            data_func = lambda idx, N: get_toy_1d_shard(idx, N, dataset_setup['data_type'],
                                                             mean_val,
                                                             dataset_setup['model_noise_std'],
                                                             experiment_setup['num_workers'] * dataset_setup[
                                                                 'points_per_worker'])

        workers_data = [data_func(w_i, experiment_setup["num_workers"]) for w_i in range(experiment_setup["num_workers"])]
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

def generate_random_dataset(experiment_setup):
    dataset_setup = experiment_setup["dataset"]
    mean_val = np.random.normal(loc=0, scale=experiment_setup["prior_std"])
    model_noise_std = np.random.uniform(low=0.25, high=2)

    if dataset_setup['dataset'] == 'toy_1d':
        data_func = lambda idx, N: get_toy_1d_shard(idx, N, dataset_setup['data_type'],
                                                    mean_val,
                                                    model_noise_std,
                                                    experiment_setup['num_workers'] * dataset_setup[
                                                        'points_per_worker'])

    workers_data = [data_func(w_i, experiment_setup["num_workers"]) for w_i in
                    range(experiment_setup["num_workers"])]
    x_train = np.array([[]])
    y_train = np.array([])
    for worker_data in workers_data:
        x_train = np.append(x_train, worker_data[0])
        y_train = np.append(y_train, worker_data[1])

    _, _, exact_mean_pres, exact_pres = exact_inference(x_train, y_train, experiment_setup['prior_std'],
                                                        model_noise_std ** 2)

    sampled_params = [mean_val, model_noise_std]
    exact_params = [exact_mean_pres, exact_pres]

    return workers_data, exact_params, sampled_params


if __name__ == '__main__':
    res = get_toy_1d_shard(0, 1)
    pdb.set_trace()
    res = get_toy_1d_shard(2, 5, 'homous')
    pdb.set_trace()
    res = get_toy_1d_shard(4, 10, 'inhomous')
    pdb.set_trace()
    # pdb.set_trace()