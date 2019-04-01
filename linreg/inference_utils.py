import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import copy

def save_predictive_plot(filepath, x_train, y_train, pred_mean, pred_var, noise_var, title, max_val=5, grid_step = 0.01):
    grid_vals = np.arange(start=-max_val, stop=max_val, step=grid_step)
    mean_pred = grid_vals * pred_mean
    # plot +-3 standard deviations
    std_pred = 3*np.sqrt(pred_var * np.square(grid_vals))
    upper_pred = mean_pred + std_pred
    lower_pred = mean_pred - std_pred
    plt.plot(grid_vals, mean_pred, 'r')
    plt.scatter(x_train, y_train, marker='x')
    plt.fill_between(grid_vals, upper_pred, lower_pred, alpha=0.5)
    plt.title(title);
    plt.savefig(filepath)

def exact_inference(x_train, y_train, prior_var, noise_var):
    # prior is assumed to be zero mean
    xtx = np.dot(x_train, x_train)
    xty = np.dot(x_train, y_train)
    post_var = 1/(noise_var**-1 * xtx + prior_var**-1)
    post_mean = post_var * (noise_var**-1) * xty
    post_pres = 1/post_var
    return post_mean, post_var, post_mean*post_pres, post_pres

def KL_Gaussians(nat11, nat12, nat21, nat22):
    v1 = 1/nat12
    v2 = 1/nat22
    m1 = nat11/nat12
    m2 = nat21/nat22
    KL = np.log(np.sqrt(v2/v1)) + (v1 + (m1-m2)**2)/(2*v2) - 0.5
    return KL


def generateDictCombinations(setup):
    keys_of_lists = []
    keys_of_dicts = []
    for key, value in setup.iteritems():
        if isinstance(value, (list,)):
            keys_of_lists.append(key)
        elif isinstance(value, (dict,)):
            keys_of_dicts.append(key)

    list_combinations = [setup[k] for k in keys_of_lists]
    sub_dict_combinations = []
    for ind, dict_key in enumerate(keys_of_dicts):
        sub_dict_combination = generateDictCombinations(setup[dict_key])
        sub_dict_combinations.append(sub_dict_combination)

    keys_of_lists.extend(keys_of_dicts);
    list_combinations.extend(sub_dict_combinations)
    all_keys = keys_of_lists
    all_combinations = list_combinations
    combinations_expanded = list(itertools.product(*all_combinations))
    output_dictionaries = []

    for ind, val in enumerate(combinations_expanded):
        new_dict = copy.deepcopy(setup)
        for key_ind, key_value in enumerate(all_keys):
            new_dict[key_value] = val[key_ind]
        output_dictionaries.append(new_dict)

    return output_dictionaries



