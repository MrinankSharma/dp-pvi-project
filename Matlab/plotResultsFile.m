% function plotResultsFile(filename)
    filename = '/Users/msharma/workspace/IIB/dp-pvi-project/linreg/logs/gs_global_ana/default/results.csv';
    exp_folder = '/Users/msharma/workspace/IIB/dp-pvi-project/linreg/logs/gs_global_ana/default/';
    
    exp_folder = '/Users/msharma/workspace/IIB/dp-pvi-project/remote-results/gs_local_ana/first_pass/'
    exp_folder = '/Users/msharma/workspace/IIB/dp-pvi-project/remote-results/gs_global_ana/first_pass/'
    filename = strcat(exp_folder, 'results.csv')
    filename = strcat(exp_folder, 'results.csv')
    results_mat = csvread(filename);
%     max eps: 1 eps: 2 eps_var: 3 dp_noise: 4 c: 5 kl: 6 kl_var: 7
%     experiment_counter: 8 ignore L for the time being
    eps_vals = results_mat(:, 2);
    kl_vals = results_mat(:, 6);
    c_vals = results_mat(:, 5);
    noise_vals = results_mat(:, 4);
    counter_vals = results_mat(:, 8);
    
    f = figure('pos', [10 10 1200 800]);
    scatter(eps_vals, kl_vals, 100*log10(noise_vals*10^5), log10(c_vals), 'o', 'filled');
    
    colormap parula
    c = colorbar;
    c.Label.String = 'Log10 Clipping Bound';
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    ylabel('$\mathcal{KL}(q(\theta)||p(\theta| \mathcal{D}))$')
    xlabel('$\epsilon, \delta=10^{-5}$')
    title('Local DP: Color is Clipping, Size is Noise')
    
    datacursormode on
    dcm = datacursormode(f);
    mydatatip = @(a, b) experimentcounterdt(a, b, eps_vals, kl_vals, counter_vals, exp_folder);
    set(dcm,'UpdateFcn', mydatatip);
    
% end
