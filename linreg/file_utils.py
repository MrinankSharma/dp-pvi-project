import csv


def get_params_from_csv(csv_file_path):
    params_searched = []
    max_eps_ind = 0
    dp_noise_ind = 3
    clipping_bound_ind = 4
    L_ind = 8
    with open(csv_file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) == 8:
                params_searched.append(
                    (float(row[max_eps_ind]), float(row[dp_noise_ind]), float(row[clipping_bound_ind])))
            elif len(row) == 9:
                params_searched.append((float(row[max_eps_ind]), float(row[dp_noise_ind]),
                                        float(row[clipping_bound_ind]), float(row[L_ind])))

    return params_searched

def get_experiment_tags_from_csv(csv_file_path, offset_from_end = 4):
    codes = []
    with open(csv_file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        try:
            for row in reader:
                code_index = len(row) - 1 - offset_from_end
                codes.append(row[code_index])
        except IndexError:
            pass

    return codes

def get_experiment_tag_params(csv_file_path, exp_str):
    with open(csv_file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        try:
            for row in reader:
                code_index = len(row) - 1 - 4;
                if row[code_index] == exp_str:
                    return row
        except IndexError:
            pass
