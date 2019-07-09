import numpy as np

import logging

logger = logging.getLogger(__name__)

def generate_mean(dataset_options):
    mean_type = dataset_options["mean"]["type"]
    try:
        if mean_type == "sample":
            mean_val =  np.random.normal(loc=0, scale=dataset_options["prior_std"])
        elif mean_type == "value":
            mean_val = dataset_options["mean"]["value"]
        else:
            raise ValueError()
    except (ValueError, KeyError):
        logger.error("Invalid Mean Options")

    logger.info("Mean Val Generated: {}".format(mean_val))
    return mean_val

def generate_noise_std_mean(dataset_options):
    noise_type = dataset_options["model_noise_std"]["type"]
    try:
        if noise_type == "sample":
            noise_val =  np.random.uniform(dataset_options["model_noise_std"]["min_val"],
                                     dataset_options["model_noise_std"]["max_val"])
        elif noise_type == "value":
            noise_val = dataset_options["model_noise_std"]["value"]
        else:
            raise ValueError()
    except (ValueError, KeyError):
        logger.error("Invalid Noise STD Options")

    logger.info("Noise Val Generated: {}".format(noise_val))
    return noise_val


def getPointsPerWorker(dataset_options, worker_indx, M):
    try:
        if dataset_options["points_per_worker"]["type"] == "uniform":
            return dataset_options["points_per_worker"]["value"]
        else:
            raise ValueError()
    except (ValueError, KeyError):
        logger.error("Invalid Points per Worker Options")

def whitenFeature(data_feature):
    feature = data_feature
    feature = feature - np.mean(feature)
    feature = feature / np.std(feature)
    return feature

def get_workers_data(mean_val, model_noise_std, dataset_options, experiment_options):
    M = experiment_options["num_workers"]
    # points to generate per worker
    N_m = np.array([getPointsPerWorker(dataset_options, i, M) for i in range(M)])
    N_total = np.sum(N_m)
    data_indices = np.cumsum(N_m)
    data_indices = np.insert(data_indices, 0, 0)
    xtrain_full = np.random.randn(N_total, 1)

    try:
        if dataset_options["data_type"] == "homous":
            # currently homogenous
            pass
        elif dataset_options["data_type"] == "inhomous":
            xtrain_full = np.sort(xtrain_full)
        else:
            raise ValueError()
    except (ValueError, KeyError):
        logger.error("Invalid Data Type Options")

    ytrain_full = mean_val * xtrain_full + model_noise_std * np.random.randn(N_total, 1)

    if dataset_options["whitened"]:
        logger.info("Data Whitening Applied")
        xtrain_full = whitenFeature(xtrain_full)
        ytrain_full = whitenFeature(ytrain_full)

    workers_data = []

    for i in range(M):
        xtrain_i = xtrain_full[data_indices[i]:data_indices[i+1]]
        ytrain_i = ytrain_full[data_indices[i]:data_indices[i + 1]]
        workers_data.append((i, xtrain_i, ytrain_i))


def generate_random_dataset(dataset_options, experiment_options):
    mean_val = generate_mean(dataset_options)
    model_noise_std = generate_noise_std_mean(dataset_options)
    workers_data = get_workers_data(mean_val, model_noise_std, dataset_options, experiment_options)

        #
        # workers_data = [data_func(w_i, experiment_setup["num_workers"]) for w_i in
        #                 range(experiment_setup["num_workers"])]
        # x_train = np.array([[]])
        # y_train = np.array([])
        # for worker_data in workers_data:
        #     x_train = np.append(x_train, worker_data[0])
        #     y_train = np.append(y_train, worker_data[1])
        #
        # _, _, exact_mean_pres, exact_pres = exact_inference(x_train, y_train, experiment_setup['prior_std'],
        #                                                     model_noise_std ** 2)
        #
        # sampled_params = [mean_val, model_noise_std]
        # exact_params = [exact_mean_pres, exact_pres]
        #
        # return workers_data, exact_params, sampled_params
