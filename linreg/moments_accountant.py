# object of this class to be passed into each worker object, tracks privacy locally.
from enum import Enum
import copy
import numpy as np


# create enum to store policy 'type' for the moments accountant attached to each worker
class MomentsAccountantPolicy(Enum):
    # these policies will simply track and not cause termination of the network
    FIXED_EPS = 1
    FIXED_DELTA = 2
    # these policies will track and return values if termination should occur
    FIXED_EPS_MAX_DELTA = 3
    FIXED_DELTA_MAX_EPS = 4


class MomentsAccountant(object):
    def __init__(self, policy, fixed_val, maximum_val, max_lambda):
        self.policy = policy
        self.fixed_val = fixed_val
        self.maximum_val = maximum_val
        self.tracked_val_history = [0]
        self.current_tracked_val = 0
        self.max_lambda = max_lambda
        # current values of log moments - to be set by whichever class is calling this. This is the key thing which
        # will depend on the specific algorithm settings being used, and so they are simply set here.
        self.log_moments = None
        self.log_moments_increment = None
        self.should_stop = False

    def update_privacy_budget(self, new_log_moments_increment=None):
        # update log moments if something specific has changed
        if new_log_moments_increment is not None:
            self.log_moments_increment = new_log_moments_increment
        elif self.log_moments_increment is None:
            raise ValueError('Log moments increment has not been set yet!')

        if self.log_moments is not None:
            self.log_moments = self.log_moments + self.log_moments_increment
        else:
            # to be safe, take value only
            self.log_moments = copy.deepcopy(self.log_moments_increment)

        if (self.policy is MomentsAccountantPolicy.FIXED_DELTA) or (
                    self.policy is MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS):
            self.fixed_delta_update_eps()
        elif (self.policy is MomentsAccountantPolicy.FIXED_EPS) or (
                    self.policy is MomentsAccountantPolicy.FIXED_EPS_MAX_DELTA):
            self.fixed_eps_update_delta()

        self.should_stop = False

        if ((self.policy is MomentsAccountantPolicy.FIXED_EPS_MAX_DELTA) or (
                    self.policy is MomentsAccountantPolicy.FIXED_DELTA_MAX_EPS)) and (
            self.current_tracked_val > self.maximum_val):
            # if we are tracking against a maximum, and if the current tracked value exceeds the inputted maximum, we
            # need the algorithm to stop
            self.should_stop = True

        return self.should_stop


    def fixed_eps_update_delta(self):
        eps = self.fixed_val
        delta = np.inf
        max_lambda = self.max_lambda
        for lambda_val in range(1, max_lambda + 1):
            current_delta_bound = np.exp(self.log_moments[lambda_val - 1] - lambda_val * eps)
            # use the smallest upper bound for the tightest guarantee
            if current_delta_bound < delta:
                delta = current_delta_bound

        self.tracked_val_history.append(delta)
        self.current_tracked_val = delta

    def fixed_delta_update_eps(self):
        delta = self.fixed_val
        # set to infinity - we are dealing with bounds ...
        eps = np.inf
        max_lambda = self.max_lambda
        for lambda_val in range(1, max_lambda + 1):
            current_eps_bound = (1.0 / lambda_val) * (self.log_moments[lambda_val - 1] - np.log(delta))
            # use the smallest upper bound for the tightest guarantee
            if current_eps_bound < eps:
                eps = current_eps_bound
        self.tracked_val_history.append(eps)
        self.current_tracked_val = eps
