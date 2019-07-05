function plotExperiment(exp_ind, exp_folder, results_mat)
    file_path = strcat(exp_folder, '/', num2str(exp_ind), '.csv')
    exp_ind = exp_ind + 1;
    
    exact_mean = results_mat(exp_ind, 6);
    exact_pres = results_mat(exp_ind, 7);
    
    title_str = sprintf("Datapoint Level DP Experiment:%d");
    
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
    max_R = 0;
       
    data_mat = csvread(file_path);
    [R, ~] = size(data_mat);
    plot(df_ax1, 1:R, data_mat(:, 1));
    plot(df_ax2, 1:R, data_mat(:, 2));
    plot(pf_ax1, 1:R, data_mat(:, 3));
    plot(pf_ax2, 1:R, data_mat(:, 4));
    plot(pf_ax3, 1:R, data_mat(:, 3)./data_mat(:, 4));
    plot(ef_ax1, 1:R, data_mat(:, 5));
    plot(ef_ax2, 1:R, data_mat(:, 6));
    
    plot(pf_ax1, pf_ax1.XLim, [exact_mean_pres exact_mean_pres], '--')
    plot(pf_ax2, pf_ax2.XLim, [exact_pres exact_pres], '--') 
    plot(pf_ax3, pf_ax3.XLim, [exact_mean exact_mean], '--') 
end