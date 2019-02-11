import numpy as np
import pdb

def get_toy_1d(mean, noise_std):
    a = mean
    n_train = 5000
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
def get_toy_1d_shard_homogeneous(shard_idx, total_shards, mean, noise_std):
    X_train, y_train, X_test, y_test = get_toy_1d(mean, noise_std)
    N_train = X_train.shape[0]
    N_per_shard = int(N_train / total_shards)
    start_ind = shard_idx * N_per_shard
    end_ind = (shard_idx + 1) * N_per_shard
    X_i = X_train[start_ind:end_ind, :]
    y_i = y_train[start_ind:end_ind]
    return X_i, y_i, X_test, y_test

def get_toy_1d_shard_inhomogeneous(shard_idx, total_shards, mean, noise_std):
    X_train, y_train, X_test, y_test = get_toy_1d(mean, noise_std)
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
def get_toy_1d_shard(shard_idx, total_shards, option='homous', mean=2, noise_std=1):
    if option == 'homous':
        return get_toy_1d_shard_homogeneous(shard_idx, total_shards, mean, noise_std)
    elif option == 'inhomous':
        return get_toy_1d_shard_inhomogeneous(shard_idx, total_shards, mean, noise_std)
    else:
        print 'unknown option, returning nothing!'

if __name__ == '__main__':
    res = get_toy_1d_shard(0, 1)
    pdb.set_trace()
    res = get_toy_1d_shard(2, 5, 'homous')
    pdb.set_trace()
    res = get_toy_1d_shard(4, 10, 'inhomous')
    pdb.set_trace()
    # pdb.set_trace()