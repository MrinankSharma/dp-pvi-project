function plotExperiment(exp_ind, exp_folder, results_mat)
    exp_ind_str = num2str(exp_ind);
    files = dir(strcat(exp_folder, 'e', num2str(exp_ind_str), 's*.csv'));
    strip_func = @(file) strcat(file.folder, '/', file.name);
    files = arrayfun(strip_func, files, 'UniformOutput', false);
    
    exp_ind = exp_ind + 1;
    
    eps = results_mat(exp_ind, 1)
    kl = results_mat(exp_ind, 3)
    % 0 - not clipped, 1 - clipped at worker, 2 - clipped at server
    clipping_config = results_mat(exp_ind, 7);
    % 0 - not noisy, 1 - noisy
    noise_config = results_mat(exp_ind, 8);
    dp_noise= results_mat(exp_ind, 9);
    c = results_mat(exp_ind, 10);
    local_damping = results_mat(exp_ind, 11);
    global_damping = results_mat(exp_ind, 12);
    exact_mean_pres = results_mat(exp_ind, 13)
    exact_mean = results_mat(exp_ind, 6)
    exact_pres = exact_mean_pres / exact_mean;
    
    title_str = sprintf("Global DP Experiment:%d")
    
    delta_fig = figure('pos', [10 10 1000 800]);
    df_ax1 = subplot(211);
    xlabel('Iteration');
    ylabel('$\Delta_1$');
    title(title_str)
    hold on;
    
    df_ax2 = subplot(212);
    xlabel('Iteration');
    ylabel('$\Delta_2$');
    hold on;

    param_fig = figure('pos', [10 10 1000 800]);
    pf_ax1 = subplot(311);
    xlabel('Iteration');
    ylabel('$\mu \lambda$');
    hold on;
    
    pf_ax2 = subplot(312);
    xlabel('Iteration');
    ylabel('$\lambda$');
    hold on;
    
    pf_ax3 = subplot(313);
    xlabel('Iteration');
    ylabel('$\mu$');
    hold on;
    
    extra_fig = figure('pos', [10 10 1000 800]);
    subplot(211)
    ef_ax1 = subplot(211);
    xlabel('Iteration');
    ylabel('$\mathcal{KL}(q(\theta)||p(\theta| \mathcal{D}))$')
    hold on;
    
    ef_ax2 = subplot(212);
    xlabel('Iteration');
    xlabel('$\epsilon, \delta=10^{-5}$')
    hold on;
    
    tags = [];
    max_R = 0;
        
    for file_ind = 1:length(files)
        file_path = files{file_ind};
        [s, f] = regexp(file_path, '\ds\w*.');
        tag = file_path((s+1):(f-1));
        tags = [tags string(tag)];
%         [mean_delta[0], mean_delta[1], current_params[0], current_params[1], KL_loss, current_eps]
        data_mat = csvread(file_path);
        [R, ~] = size(data_mat);
        plot(df_ax1, 1:R, data_mat(:, 1));
        plot(df_ax2, 1:R, data_mat(:, 2));
        plot(pf_ax1, 1:R, data_mat(:, 3));
        plot(pf_ax2, 1:R, data_mat(:, 4));
        plot(pf_ax3, 1:R, data_mat(:, 3)./data_mat(:, 4));
        plot(ef_ax1, 1:R, data_mat(:, 5));
        plot(ef_ax2, 1:R, data_mat(:, 6));
    end
    plot(pf_ax1, pf_ax1.XLim, [exact_mean_pres exact_mean_pres], '--')
    plot(pf_ax2, pf_ax2.XLim, [exact_pres exact_pres], '--') 
    plot(pf_ax3, pf_ax3.XLim, [exact_mean exact_mean], '--') 
    
    legend(df_ax1, tags, 'location', 'northeastoutside');
    legend(pf_ax1, [tags, 'true value'], 'location','northeastoutside');
    legend(ef_ax1, tags, 'location', 'northeastoutside');
    saveas(delta_fig, strcat(exp_folder, 'delta-experiment', exp_ind_str, '.png'))
    saveas(param_fig, strcat(exp_folder, 'param-experiment', exp_ind_str, '.png'))
    saveas(extra_fig, strcat(exp_folder, 'extra-experiment', exp_ind_str, '.png'))
end