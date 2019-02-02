import numpy as np
import pdb

def get_toy_1d():
    a = 2
    n_train = 10000
    # n_train = 1000
    n_test = 2000
    xtrain = np.random.randn(n_train, 1)
    ytrain = a * xtrain + 0.5 * np.random.randn(n_train, 1)
    xtest = np.random.randn(n_test, 1)
    ind = np.argsort(xtest[:, 0])
    xtest = xtest[ind, :]
    ytest = a * xtest + 0.5 * np.random.randn(n_test, 1)
    return xtrain, ytrain.reshape([n_train]), xtest, ytest.reshape([n_test])

# simulate data in multiple shards
def get_toy_1d_shard_homogeneous(shard_idx, total_shards):
    X_train, y_train, X_test, y_test = get_toy_1d()
    N_train = X_train.shape[0]
    N_per_shard = int(N_train / total_shards)
    start_ind = shard_idx * N_per_shard
    end_ind = (shard_idx + 1) * N_per_shard
    X_i = X_train[start_ind:end_ind, :]
    y_i = y_train[start_ind:end_ind]
    return X_i, y_i, X_test, y_test

def get_toy_1d_shard_inhomogeneous(shard_idx, total_shards):
    X_train, y_train, X_test, y_test = get_toy_1d()
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
def get_toy_1d_shard(shard_idx, total_shards, option='homous'):
    if option == 'homous':
        return get_toy_1d_shard_homogeneous(shard_idx, total_shards)
    elif option == 'inhomous':
        return get_toy_1d_shard_inhomogeneous(shard_idx, total_shards)
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