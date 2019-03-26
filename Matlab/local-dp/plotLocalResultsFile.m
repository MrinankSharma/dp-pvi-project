setGraphDefaults;

exp_folder = '/Users/msharma/workspace/IIB/dp-pvi-project/hinton-scratch/logs/gs_local_ana/damped_temp/';
filename = strcat(exp_folder, 'results.csv');
setup_name = strcat(exp_folder, 'setup.json');

setup = jsondecode(fileread(setup_name));

results_mat = csvread(filename);

eps_vals = results_mat(:, 2);
kl_vals = results_mat(:, 6);
c_vals = results_mat(:, 5);
noise_vals = results_mat(:, 4);
counter_vals = results_mat(:, 8);
damping_vals = results_mat(:, 9);

f = figure('pos', [10 10 1200 800]);
scatter(eps_vals, kl_vals, damping_vals*600, log10(c_vals), 'o', 'filled');

colormap parula
c = colorbar;
c.Label.String = 'Log10 Clipping Bound';
set(gca,'xscale','log')
set(gca,'yscale','log')
ylabel('$\mathcal{KL}(q(\theta)||p(\theta| \mathcal{D}))$')
xlabel('$\epsilon, \delta=10^{-5}$')
title(sprintf('Local Noisy Updates, Num Workers: %d, size is amount of damping.', ...
    setup.num_workers))
text(160, max(kl_vals) - mean(kl_vals), sprintf(' c %.4e',unique(c_vals)))
text(160, max(kl_vals) - 2*mean(kl_vals), sprintf(' noise %.4e',unique(noise_vals)))
text(160, max(kl_vals) - 3*mean(kl_vals), sprintf(' damping %.4e',unique(damping_vals)))

saveas(f, strcat(exp_folder,'overviewgraph.png'))
datacursormode on
dcm = datacursormode(f);
mydatatip = @(a, b) datatip(a, b, results_mat, exp_folder, setup);
set(dcm,'UpdateFcn', mydatatip);


