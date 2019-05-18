from linreg.log_moment_utils import generate_log_moments
from linreg.moments_accountant import MomentsAccountant, MomentsAccountantPolicy
import numpy as np

if __name__ == "__main__":
    delta = 1e-5
    N = 1000
    L_vals = [1, 10, 50, 100, 500]
    N_epochs = 1000
    eps_L_values = []

    for L in L_vals:
        log_moments = generate_log_moments(N, 32, 1, L)
        accountant = MomentsAccountant(MomentsAccountantPolicy.FIXED_DELTA, 1e-5, np.inf, 32)
        accountant.log_moments_increment = log_moments
        eps_vals = []
        it_per_epoch = int(np.ceil(N / L))
        print(it_per_epoch)
        for i in range(N_epochs):
            for j in range(it_per_epoch):
                accountant.update_privacy_budget()
            eps_vals.append(accountant.current_tracked_val)

        eps_L_values.append(eps_vals)

    results = np.array(eps_L_values)
    np.savetxt('q_exp.csv', results, delimiter=',')



