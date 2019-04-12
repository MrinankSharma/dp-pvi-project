N = 100;
x = randn(N, 1);
bound = 1;

xs_c = x.^2 ./ max(1, x.^2/bound); 
true_val = sum(x.^2)
est = sum(xs_c);

est_vals = zeros(100, 1);

for j=1:100
    est_vals(j) = est;
    signal = (x.^2 - est/N);
    sum(signal);
    est+sum(signal);
    clipped_signal = signal./(max(1, signal));
    sum(clipped_signal);
    est = est + sum(clipped_signal);
end
plot(est_vals)
hold on;
plot([0 100], [true_val true_val], '--')
plot([0 100], [sum(xs_c) sum(xs_c)], '-.')
legend('Iterative Estimate', 'True Value', 'One-Shot Estimate')