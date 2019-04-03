N = 100;
x = randn(N, 1);
bound = 1;

xs_c = x.^2 ./ max(1, x.^2/bound); 
true_val = sum(x.^2)
est = sum(xs_c);

for j=1:100
    signal = (x.^2 - est/N);
    sum(signal)
    est+sum(signal)
    clipped_signal = signal./(max(1, signal));
    sum(clipped_signal)
    est = est + sum(clipped_signal)
end