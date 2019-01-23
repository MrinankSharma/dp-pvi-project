import linreg_models
import data
import numpy as np

if __name__ == "__main__":
    data_func = data.get_toy_1d_shard
    delta = 1e-5

    # Create a parameter server with some random params.
    x_train, y_train, x_test, y_test = data_func(0, 1)
    n_train_master = x_train.shape[0]
    in_dim = x_train.shape[1]
    print(n_train_master)
    net = linreg_models.LinReg_MFVI_DPSGD(in_dim, n_train_master, dpsgd_noise_scale=1, lot_size=50,
                                          num_iterations=100)
    [best_eps_MA, epsilons_MA, num_evals , epoch_sf] = net.track_privacy_moments_accountant_fixed_delta(n_train_master, 32, 10,
                                                                                                delta)

    [best_eps_SC, epsilons_SC, num_evals, epoch_sf] = net.track_privacy_adv_composition_fixed_delta(n_train_master, 10, delta)

    np.savetxt('plot_files/sc_vals.txt', epsilons_SC)
    np.savetxt('plot_files/ma_vals.txt', epsilons_MA)
    np.savetxt('plot_files/epoch_conv.txt', [epoch_sf])
    print(best_eps_MA)
    print(best_eps_SC)
