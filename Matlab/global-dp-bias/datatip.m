function output_txt = datatip(obj,event_obj, results_mat, exp_folder, setup)
    pos = get(event_obj, 'Position');
    x_vals = results_mat(:, 1);
    y_vals = results_mat(:, 3);
    labels = results_mat(:, 5);
    points = [x_vals'; y_vals'];
    point = [pos(1); pos(2)];
    indices = find(sum(points == point) == 2);
    label = num2str(labels(indices)');
    output_txt = {...
    ['X: ', num2str(pos(1),4)]...
    ['Y: ', num2str(pos(2),4)] ...
    ['Experiment[s]: ', label]...
};