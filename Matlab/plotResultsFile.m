% function plotResultsFile(filename)
    filename = 'temp_results_local.csv'
    results_mat = csvread(filename)
    % max_eps, eps, eps_var, dp_noise_scale, clipping_bound, kl, kl_var, L
    eps_vals = results_mat(:, 2);
    kl_vals = results_mat(:, 6);
    L_vals = results_mat(:, 8);
    c_vals = results_mat(:, 5);
    noise_vals = results_mat(:, 4);
    
    figure('pos', [10 10 1200 800]);
    scatter(eps_vals(L_vals==10), kl_vals(L_vals==10), 100*log10(noise_vals(L_vals==10)*10^5), log10(c_vals(L_vals==10)), 'o', 'filled');
    hold on;
    scatter(eps_vals(L_vals==100), kl_vals(L_vals==100), 100*log10(noise_vals(L_vals==100)*10^5), log10(c_vals(L_vals==100)), 'd', 'filled');
    scatter(eps_vals(L_vals==500), kl_vals(L_vals==500), 100*log10(noise_vals(L_vals==500)*10^5), log10(c_vals(L_vals==500)), 'h', 'filled');
    colormap parula
    c = colorbar;
    c.Label.String = 'Log10 Clipping Bound';
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    ylabel('$\mathcal{KL}(p(\theta| \mathcal{D})||q(\theta))$')
    xlabel('$\epsilon, \delta=10^{-5}$')
    title('Local DP: Color is Clipping, Size is Noise, Circle (L=10), Diamond (L=100), Star (L=500) ')
% end
