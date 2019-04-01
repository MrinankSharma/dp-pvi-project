setGraphDefaults;

exp_folder = '/Users/msharma/workspace/IIB/dp-pvi-project/hinton-scratch/logs/gs_global_bias_ana/check-bias-clip/';
filename = strcat(exp_folder, 'results.csv');
setup_name = strcat(exp_folder, 'setup.json');

setup = jsondecode(fileread(setup_name));

%%
results_mat = csvread(filename);

eps_vals = results_mat(:, 1);
kl_vals = results_mat(:, 3);
counter_vals = results_mat(:, 5);
% 0 - not clipped, 1 - clipped at worker, 2 - clipped at server
clipping_config_vals = results_mat(:, 7);
% 0 - not noisy, 1 - noisy
noise_config_vals = results_mat(:, 8);
dp_noise_vals = results_mat(:, 9);
c_vals = results_mat(:, 10);
local_damping_vals = results_mat(:, 11);
global_damping_vals = results_mat(:, 12);
exact_mean_pres_vals = results_mat(:, 13);
exact_pres_vals = results_mat(:, 14);

unclipped_not_noisy = find(((clipping_config_vals == 0) + (noise_config_vals == 0)) == 2);
unclipped_noisy = find(((clipping_config_vals == 0) + (noise_config_vals == 1)) == 2);

worker_clipped_not_noisy = find(((clipping_config_vals == 1) + (noise_config_vals == 0)) == 2);
worker_clipped_noisy = find(((clipping_config_vals == 1) + (noise_config_vals == 1)) == 2);

server_clipped_not_noisy = find(((clipping_config_vals == 2) + (noise_config_vals == 0)) == 2);
server_clipped_noisy = find(((clipping_config_vals == 2) + (noise_config_vals == 1)) == 2);

%%
f = figure('pos', [10 10 1200 800]);
marker_size = 150;
scatter(eps_vals(unclipped_not_noisy), kl_vals(unclipped_not_noisy), ...
    marker_size, log10(c_vals(unclipped_not_noisy)), 'o', 'filled');
hold on;
scatter(eps_vals(unclipped_noisy), kl_vals(unclipped_noisy), ...
    marker_size, log10(c_vals(unclipped_noisy)), '+');

scatter(eps_vals(worker_clipped_not_noisy), kl_vals(worker_clipped_not_noisy), ...
    marker_size, log10(c_vals(worker_clipped_not_noisy)), 's', 'filled');
scatter(eps_vals(worker_clipped_noisy), kl_vals(worker_clipped_noisy), ...
    marker_size, log10(c_vals(worker_clipped_noisy)), 'd', 'filled');

scatter(eps_vals(server_clipped_not_noisy), kl_vals(server_clipped_not_noisy), ...
    marker_size, log10(c_vals(server_clipped_not_noisy)), '*');
scatter(eps_vals(server_clipped_noisy), kl_vals(server_clipped_noisy), ...
    marker_size, log10(c_vals(server_clipped_noisy)), 'p', 'filled');

legend('Unclipped, Noiseless', 'Unclipped, Noisy', ...
    'Clipped at Worker, Noiseless', ...
    'Clipped at Worker, Noisy', ...
    'Clipped at Server, Noiseless', ...
    'Clipped at Server, Noisy', 'Location', 'northwest')

colormap parula
c = colorbar;
c.Label.String = 'Log10 Clipping Bound';
set(gca,'xscale','log')
set(gca,'yscale','log')
ylabel('$\mathcal{KL}(q(\theta)||p(\theta| \mathcal{D}))$')
xlabel('$\epsilon, \delta=10^{-5}$')
title(sprintf('Global DP, Num Workers: %d', ...
    setup.num_workers))

saveas(f, strcat(exp_folder,'overviewgraph.png'))
datacursormode on
dcm = datacursormode(f);
mydatatip = @(a, b) datatip(a, b, results_mat, exp_folder, setup);
set(dcm,'UpdateFcn', mydatatip);


