function plotSingleSeed(exp_ind, exp_folder, seed_ind, results_mat)
    file = strcat(exp_folder, 'e', num2str(exp_ind), 's', num2str(seed_ind), '.csv')
    data_mat = csvread(file); 
    
    eps = results_mat(exp_ind, 2);
    kl = results_mat(exp_ind, 6);
    c = results_mat(exp_ind, 5);
    noise = results_mat(exp_ind, 4);
    counter = results_mat(exp_ind, 8);
    damping = results_mat(exp_ind, 9);
    title_str = sprintf("averages exp: %d eps: %.4e kl: %.4e c: %.4e noise: %.4e damping: %.4e ",  ...
        counter, eps, kl, c, noise, damping)
    
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
    
    tags = []
    tags_delta = []
        
    [R, ~] = size(data_mat);
    std(20*data_mat(100:end, 1))
    std(20*data_mat(100:end, 2))
    plot(df_ax1, 1:R, 20*data_mat(:, 1));
    plot(df_ax2, 1:R, 20*data_mat(:, 2));
    plot(pf_ax1, 1:R, data_mat(:, 3));
    plot(df_ax1, 1:R, data_mat(:, 7), '--');
    plot(pf_ax2, 1:R, data_mat(:, 4));
    plot(df_ax2, 1:R, data_mat(:, 8), '--');
    plot(pf_ax3, 1:R, data_mat(:, 3)./data_mat(:, 4));
    plot(ef_ax1, 1:R, data_mat(:, 5));
    plot(ef_ax2, 1:R, data_mat(:, 6));
end