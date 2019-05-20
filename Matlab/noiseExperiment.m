a = -2;
sigma_noise = 0.5;
N = 10;

prior_mean = 0;
prior_sigma = 5;

x = linspace(-1, 1, N);
e = sigma_noise * randn(1, N);
y = a * x + e; 

xTx = x*x';
xTy = x*y'; 
var_post = (prior_sigma^2 * sigma_noise^2)*(sigma_noise^2 + prior_sigma^2*xTx)^-1;
mean_post = (xTy * prior_sigma^2 + sigma_noise^2 * prior_mean) * (sigma_noise^2 + prior_sigma^2*xTx)^-1;

%% compute noisy estimates:
N_test = 10000;
var_post_estimates = zeros(1, N_test);
mean_post_estimates = zeros(1, N_test);
nat1_estimates = zeros(1, N_test);
nat2_estimates = zeros(1, N_test);

for i=1:N_test
    xTx_noisy = sum(x.^2) + randn;
    xTy_noisy = sum(x.*y) + randn;
    var_post_i = (prior_sigma^2 * sigma_noise^2)*(sigma_noise^2 + prior_sigma^2*xTx_noisy)^-1;
    mean_post_i = (xTy_noisy * prior_sigma^2 + sigma_noise^2 * prior_mean) * ...
        (sigma_noise^2 + prior_sigma^2*xTx_noisy)^-1;
    var_post_estimates(i) = var_post_i - var_post;
    mean_post_estimates(i) = mean_post_i - mean_post;
    nat1_post_estimates(i) = 1/var_post_i - 1/var_post;
    nat2_post_estimates(i) = mean_post_i/var_post_i - mean_post/var_post;
end


figure('pos', [10 10 1000 1000]);
subplot(221)
h = histogram(mean_post_estimates, 'BinEdges', -2:0.04:2, 'Normalization', 'pdf')
xlabel('$\mu - \mu^*$')
ylabel('$p$')
title('Empirical Distribution of Mean Error; $\theta=2$')
xlim([-2 2.5])
subplot(222)
h2 = histogram(var_post_estimates,'BinEdges', -3:0.0009:3, 'Normalization', 'pdf')
xlim([-0.04 0.1])
xlabel('$\sigma^2 - \sigma^{2,*}$')
ylabel('$p$')
title('Empirical Distribution of Variance Error; $\theta=2$')

a = 2;
sigma_noise = 0.5;
N = 10;

prior_mean = 0;
prior_sigma = 5;

x = linspace(-1, 1, N);
e = sigma_noise * randn(1, N);
y = a * x + e; 

xTx = x*x';
xTy = x*y'; 
var_post = (prior_sigma^2 * sigma_noise^2)*(sigma_noise^2 + prior_sigma^2*xTx)^-1;
mean_post = (xTy * prior_sigma^2 + sigma_noise^2 * prior_mean) * (sigma_noise^2 + prior_sigma^2*xTx)^-1;

%% compute noisy estimates:
N_test = 10000;
var_post_estimates = zeros(1, N_test);
mean_post_estimates = zeros(1, N_test);
nat1_estimates = zeros(1, N_test);
nat2_estimates = zeros(1, N_test);

for i=1:N_test
    xTx_noisy = sum(x.^2) + randn;
    xTy_noisy = sum(x.*y) + randn;
    var_post_i = (prior_sigma^2 * sigma_noise^2)*(sigma_noise^2 + prior_sigma^2*xTx_noisy)^-1;
    mean_post_i = (xTy_noisy * prior_sigma^2 + sigma_noise^2 * prior_mean) * ...
        (sigma_noise^2 + prior_sigma^2*xTx_noisy)^-1;
    var_post_estimates(i) = var_post_i - var_post;
    mean_post_estimates(i) = mean_post_i - mean_post;
    nat1_post_estimates(i) = 1/var_post_i - 1/var_post;
    nat2_post_estimates(i) = mean_post_i/var_post_i - mean_post/var_post;
end
subplot(223)
h = histogram(mean_post_estimates, 'BinEdges', -2:0.04:2, 'Normalization', 'pdf')
xlabel('$\mu - \mu^*$')
ylabel('$p$')
title('Empirical Distribution of Mean Error; $\theta=-2$')
xlim([-2 2.5])
subplot(224)
h2 = histogram(var_post_estimates,'BinEdges', -3:0.0009:3, 'Normalization', 'pdf')
xlim([-0.04 0.1])
xlabel('$\sigma^2 - \sigma^{2,*}$')
ylabel('$p$')
title('Empirical Distribution of Variance Error; $\theta=-2$')

% subplot(413)
% h3 = histogram(nat1_post_estimates,'Normalization', 'pdf')
% xlabel('Precision Prediction Error, $e$')
% ylabel('$p_e$')
% title(sprintf('Mean Error: %.3e', mean(nat1_post_estimates)))
% subplot(414)
% h4 = histogram(nat2_post_estimates, 'Normalization', 'pdf')
% xlabel('Mean time Precision Prediction Error, $e$')
% ylabel('$p_e$')
% title(sprintf('Mean Error: %.3e', mean(nat2_post_estimates)))

% %
% N_test = 10000;
% var_post_estimates = zeros(1, N_test);
% mean_post_estimates = zeros(1, N_test);
% nat1_estimates = zeros(1, N_test);
% nat2_estimates = zeros(1, N_test);
% 
% for i=1:N_test
%     x_noisy = x + 0.4*randn(1, N);
%     y_noisy = y + 0.4*randn(1, N);
%     xTx_noisy = x_noisy * x_noisy';
%     xTy_noisy = x_noisy * y_noisy';
%     var_post_i = (prior_sigma^2 * sigma_noise^2)*(sigma_noise^2 + prior_sigma^2*xTx_noisy)^-1;
%     mean_post_i = (xTy_noisy * prior_sigma^2 + sigma_noise^2 * prior_mean) * ...
%         (sigma_noise^2 + prior_sigma^2*xTx_noisy)^-1;
%     var_post_estimates(i) = var_post_i - var_post;
%     mean_post_estimates(i) = mean_post_i - mean_post;
%     nat1_post_estimate(i) = 1/var_post_i - 1/var_post;
%     nat2_post_estimate(i) = mean_post_i/var_post_i - mean_post/var_post;
% end
% 
% 
% figure('pos', [10 10 1000 800]);
% subplot(411)
% h = histogram(mean_post_estimates, 'BinEdges', -2:0.04:2, 'Normalization', 'pdf')
% xlabel('Mean Prediction Error, $e$')
% ylabel('$p_e$')
% title(sprintf('Indiv Noise: Mean Error: %.3f', mean(mean_post_estimates)))
% xlim([-2 2])
% subplot(412)
% h2 = histogram(var_post_estimates,'BinEdges', -3:0.0009:3, 'Normalization', 'pdf')
% xlim([-0.1 0.1])
% xlabel('Variance Prediction Error, $e$')
% ylabel('$p_e$')
% title(sprintf('Mean Error: %.3e', mean(var_post_estimates)))
% subplot(413)
% h3 = histogram(nat1_post_estimate,'Normalization', 'pdf')
% xlabel('Precision Prediction Error, $e$')
% ylabel('$p_e$')
% title(sprintf('Mean Error: %.3e', mean(nat1_post_estimates)))
% subplot(414)
% h4 = histogram(nat2_post_estimate, 'Normalization', 'pdf')
% xlabel('Mean time Precision Prediction Error, $e$')
% ylabel('$p_e$')
% title(sprintf('Mean Error: %.3e', mean(nat2_post_estimates)))
%     