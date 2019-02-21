import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def save_predictive_plot(filepath, x_train, y_train, pred_mean, pred_var, noise_var, title, max_val=5, grid_step = 0.01):
    grid_vals = np.arange(start=-max_val, stop=max_val, step=grid_step)
    mean_pred = grid_vals * pred_mean
    # plot +-3 standard deviations
    std_pred = 3*np.sqrt(noise_var + pred_var * np.square(grid_vals))
    upper_pred = mean_pred + std_pred
    lower_pred = mean_pred - std_pred
    plt.plot(grid_vals, mean_pred, 'r')
    plt.scatter(x_train, y_train, marker='x')
    plt.fill_between(grid_vals, upper_pred, lower_pred, alpha=0.5)
    plt.title(title);
    plt.savefig(filepath)

def exact_inference(x_train, y_train, prior_var, noise_var):
    # prior is assumed to be zero mean
    xtx = np.einsum('na, na->a', x_train, x_train)
    xty = np.einsum('na, n->a', x_train, y_train)
    post_var = 1/(noise_var**-1 * xtx + prior_var**-1)
    post_mean = post_var * (noise_var**-1)*xty
    post_pres = 1/post_var
    return post_mean, post_var, post_mean*post_pres, post_pres

def KL_Gaussians(nat11, nat12, nat21, nat22):
    v1 = 1/nat12
    v2 = 1/nat22
    m1 = nat11/nat12
    m2 = nat21/nat22
    KL = np.log(np.sqrt(v2/v1)) + (v1 + (m1-m2)**2)/(2*v2)
    return KL


