from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
from linreg.tfutils import TensorFlowVariablesWithScope
from linreg.log_moment_utils import generate_log_moments

from linreg.inference_utils import exact_inference

float_type = tf.float32
int_type = tf.int32

import mpmath as mp


class LinReg_MFVI_analytic():
    """
    Stochastic global variational inference using Adam optimiser
    fully factorised Gaussian approximation
    variational parameters: mu and log(var)
    """

    def __init__(self, din, n_train, noise_var=1,
                 prior_mean=0.0, prior_var=1.0, no_workers=1,
                 clipping_bound=10, model_config=None, global_damping=0,
                 dp_noise_level=10, single_thread=True):
        self.din = din
        # input and output placeholders
        self.xtrain = tf.placeholder(float_type, [None, din], 'input')
        self.xtest = tf.placeholder(float_type, [None, din], 'test_input')
        self.ytrain = tf.placeholder(float_type, [None], 'target')
        self.n_train = n_train
        self.no_workers = no_workers
        self.noise_var = noise_var
        self.prior_var_num = prior_var

        if model_config is None:
            self.model_config = {
                "clipping": "worker",
                "noisy": "noisy_worker"
            }

        # these settings affect the deltas which are sent back to the central parameter server
        self.clipping_bound = clipping_bound
        self.model_config = model_config

        self.noise_scale = clipping_bound * dp_noise_level / no_workers ** 0.5;

        # create parameters for the model
        #  [no_params, w_mean, w_var, w_n1, w_n2,
        #        prior_mean, prior_var, local_n1, local_n2]
        res = self._create_params(prior_mean, prior_var)
        self.no_weights = res[0]
        self.w_mean, self.w_var = res[1], res[2]
        self.w_n1, self.w_n2 = res[3], res[4]
        self.prior_mean, self.prior_var = res[5], res[6]
        self.local_n1, self.local_n2 = res[7], res[8]
        self.global_damping = global_damping

        # create helper assignment ops
        self._create_assignment_ops()

        # build objective and prediction functions
        self.energy_fn, self.kl_term, self.expected_lik = self._build_energy()
        self.predict_m, self.predict_v = self._build_prediction()

        self.train_updates = self._create_train_step()
        # launch a session
        if single_thread:
            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
        else:
            self.sess = tf.Session()

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.params = TensorFlowVariablesWithScope(
            self.energy_fn, self.sess,
            scope='variational_nat', input_variables=[self.w_n1, self.w_n2])

    def _create_train_step(self):
        xtx = tf.einsum('na,nb->ab', self.xtrain, self.xtrain)
        xty = tf.einsum('na,n->a', self.xtrain, self.ytrain)
        Voinv = tf.diag(1.0 / self.prior_var)
        Voinvmo = self.prior_mean / self.prior_var
        Vinv = Voinv + xtx / self.noise_var
        Vinvm = Voinvmo + xty / self.noise_var
        m = tf.reduce_sum(tf.linalg.solve(Vinv, tf.expand_dims(Vinvm, 1)), axis=1)
        v = 1.0 / tf.diag_part(Vinv)
        update_m = self.w_mean.assign(m)
        update_v = self.w_var.assign(v)
        return update_m, update_v

    def train(self, x_train, y_train):
        N = x_train.shape[0]
        sess = self.sess
        _, c, c_kl, c_lik = sess.run(
            [self.train_updates, self.energy_fn, self.kl_term, self.expected_lik],
            feed_dict={self.xtrain: x_train, self.ytrain: y_train})
        return np.array([c, c_kl, c_lik])

    def get_weights(self):
        w_mean, w_var = self.sess.run([self.w_mean, self.w_log_var])
        return w_mean, w_var

    def _build_energy(self):
        # compute the expected log likelihood
        no_train = tf.cast(self.n_train, float_type)
        w_mean, w_var = self.w_mean, self.w_var
        const_term = -0.5 * no_train * np.log(2 * np.pi * self.noise_var)
        pred = tf.einsum('nd,d->n', self.xtrain, w_mean)
        ydiff = self.ytrain - pred
        quad_term = -0.5 / self.noise_var * tf.reduce_sum(ydiff ** 2)
        xxT = tf.reduce_sum(self.xtrain ** 2, axis=0)
        trace_term = -0.5 / self.noise_var * tf.reduce_sum(xxT * w_var)
        expected_lik = const_term + quad_term + trace_term

        # compute the kl term analytically
        const_term = -0.5 * self.no_weights
        log_std_diff = tf.log(self.prior_var) - tf.log(w_var)
        log_std_diff = 0.5 * tf.reduce_sum(log_std_diff)
        mu_diff_term = (w_var + (self.prior_mean - w_mean) ** 2) / self.prior_var
        mu_diff_term = 0.5 * tf.reduce_sum(mu_diff_term)
        kl = const_term + log_std_diff + mu_diff_term

        kl_term = kl / no_train
        expected_lik_term = - expected_lik
        return kl_term + expected_lik_term, kl_term, expected_lik_term

    def _create_params(self, prior_mean, prior_var):
        no_params = self.din
        init_var = 1 * np.ones([no_params])
        init_mean = np.zeros([no_params])
        init_n2 = 1.0 / init_var
        init_n1 = init_mean / init_var

        with tf.name_scope('variational_cov'):
            w_var = tf.Variable(
                tf.constant(init_var, dtype=float_type),
                name='variance')
            w_mean = tf.Variable(
                tf.constant(init_mean, dtype=float_type),
                name='mean')

        with tf.name_scope('variational_nat'):
            w_n1 = tf.Variable(
                tf.constant(init_n1, dtype=float_type),
                name='precision_x_mean')
            w_n2 = tf.Variable(
                tf.constant(init_n2, dtype=float_type),
                name='precision')

        prior_mean_val = prior_mean * np.ones([no_params])
        prior_var_val = prior_var * np.ones([no_params])
        with tf.name_scope('prior'):
            prior_mean = tf.Variable(
                tf.constant(prior_mean_val, dtype=float_type),
                trainable=False,
                name='mean')
            prior_var = tf.Variable(
                tf.constant(prior_var_val, dtype=float_type),
                trainable=False,
                name='variance')

        no_workers = self.no_workers
        prior_n1 = prior_mean_val / prior_var_val
        prior_n2 = 1.0 / prior_var_val
        data_n1 = init_n1 - prior_n1
        data_n2 = init_n2 - prior_n2
        local_n1 = data_n1 / self.no_workers
        local_n2 = data_n2 / self.no_workers

        res = [no_params, w_mean, w_var, w_n1, w_n2,
               prior_mean, prior_var, local_n1, local_n2]
        return res

    def _create_assignment_ops(self):
        # create helper assignment ops
        self.prior_mean_val = tf.placeholder(dtype=float_type)
        self.prior_var_val = tf.placeholder(dtype=float_type)
        self.post_mean_val = tf.placeholder(dtype=float_type)
        self.post_var_val = tf.placeholder(dtype=float_type)
        self.post_n1_val = tf.placeholder(dtype=float_type)
        self.post_n2_val = tf.placeholder(dtype=float_type)

        self.prior_mean_op = self.prior_mean.assign(self.prior_mean_val)
        self.prior_var_op = self.prior_var.assign(self.prior_var_val)
        self.post_mean_op = self.w_mean.assign(self.post_mean_val)
        self.post_var_op = self.w_var.assign(self.post_var_val)
        self.post_n1_op = self.w_n1.assign(self.post_n1_val)
        self.post_n2_op = self.w_n2.assign(self.post_n2_val)

    def set_params(self, variable_names, new_params, update_prior=True):
        # set natural parameters
        self.params.set_weights(dict(zip(variable_names, new_params)))
        # compute and set covariance parameters
        post_n1, post_n2 = self.sess.run([self.w_n1, self.w_n2])
        # print([post_n1, post_n2])
        post_var = 1.0 / post_n2
        post_mean = post_n1 / post_n2
        self.sess.run(
            [self.post_var_op, self.post_mean_op],
            feed_dict={self.post_var_val: post_var,
                       self.post_mean_val: post_mean})
        if update_prior:
            # compute and set prior
            prior_n1 = post_n1 - self.local_n1
            prior_n2 = post_n2 - self.local_n2
            prior_var = 1.0 / prior_n2
            prior_mean = prior_n1 / prior_n2
            self.sess.run([self.prior_mean_op, self.prior_var_op], feed_dict={
                self.prior_mean_val: prior_mean, self.prior_var_val: prior_var})

    def set_nat_params(self):
        # compute and set natural parameters
        post_mean, post_var = self.sess.run([self.w_mean, self.w_var])
        post_n2 = 1.0 / post_var
        post_n1 = post_mean / post_var
        self.sess.run(
            [self.post_n1_op, self.post_n2_op],
            feed_dict={self.post_n1_val: post_n1,
                       self.post_n2_val: post_n2})

    def get_nat_params(self):
        self.set_nat_params()
        return self.sess.run([self.w_n1, self.w_n2])

    def get_params(self):
        self.set_nat_params()
        # get natural params and return
        params = self.params.get_weights()
        return list(params.keys()), list(params.values())

    def get_param_diff(self, damping=0.5):
        self.set_nat_params()
        # update local factor
        post_n1, post_n2, prior_mean, prior_var = self.sess.run([
            self.w_n1, self.w_n2, self.prior_mean, self.prior_var])
        prior_n2 = 1.0 / prior_var
        prior_n1 = prior_mean / prior_var
        # print("Worker Prior:{}, {}".format(prior_n1, prior_n2))
        # print("Worker Post:{}, {}".format(post_n1, post_n2))
        new_local_n1 = post_n1 - prior_n1
        new_local_n2 = post_n2 - prior_n2
        old_local_n1 = self.local_n1
        old_local_n2 = self.local_n2
        new_local_n1 = damping * old_local_n1 + (1.0 - damping) * new_local_n1
        new_local_n2 = damping * old_local_n2 + (1.0 - damping) * new_local_n2
        new_local_n2[np.where(new_local_n2 < 0)] = 0
        delta_n1 = new_local_n1 - old_local_n1
        delta_n2 = new_local_n2 - old_local_n2
        param_deltas = [delta_n1, delta_n2]
        # first, compute parameter deltas as usual.

        if self.model_config["clipping"] == "clipped":
            norm = (param_deltas[0] ** 2 + param_deltas[1] ** 2) ** 0.5
            if self.clipping_bound < norm:
                scaling = self.clipping_bound / norm
            else:
                scaling = 1
            param_deltas = [scaling * param_deltas[0], scaling * param_deltas[1]]

        if self.model_config["noise"] == "noisy_worker":
            # print("adding noise at the worker")
            noise = np.random.normal(0, self.noise_scale, [2])
            param_deltas = [param_deltas[0] + noise[0], param_deltas[1] + noise[1]]

        self.local_n1 = old_local_n1 + (1 - self.global_damping) * param_deltas[0]
        self.local_n2 = old_local_n2 + (1 - self.global_damping) * param_deltas[1]

        # we need to take the effects of teh global damping into account for the local factors.

        return param_deltas

    def _build_prediction(self):
        x = self.xtest
        w_mean, w_var = self.w_mean, self.w_var
        mean = tf.einsum('nd,d->n', x, w_mean)
        var = tf.reduce_sum(x * w_var * x, axis=1)
        return mean, var

    def close_session(self):
        self.sess.close()

    def compute_energy(self, x_train, y_train, batch_size=None):
        sess = self.sess
        c = sess.run(
            [self.energy_fn],
            feed_dict={self.xtrain: x_train, self.ytrain: y_train})[0]
        return c

    def prediction(self, x_test, batch_size=500):
        # Test model
        N = x_test.shape[0]
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            m, v = self.sess.run(
                [self.predict_m, self.predict_v],
                feed_dict={self.xtest: batch_x})
            if i == 0:
                ms, vs = m, v
            else:
                if len(m.shape) == 3:
                    concat_axis = 1
                else:
                    concat_axis = 0
                ms = np.concatenate(
                    (ms, m), axis=concat_axis)
                vs = np.concatenate(
                    (vs, v), axis=concat_axis)
        return ms, vs


class LinReg_MFVI_GRA():
    """
    Stochastic global variational inference using Adam optimiser
    fully factorised Gaussian approximation
    variational parameters: mu and log(var)
    """

    def __init__(self, din, n_train, noise_var=0.25,
                 prior_mean=0.0, prior_var=1, no_workers=1,
                 single_thread=True):
        self.din = din
        # input and output placeholders
        self.xtrain = tf.placeholder(float_type, [None, din], 'input')
        self.xtest = tf.placeholder(float_type, [None, din], 'test_input')
        self.ytrain = tf.placeholder(float_type, [None], 'target')
        self.n_train = n_train
        self.no_workers = no_workers
        self.noise_var = noise_var

        # create parameters for the model
        #  [no_params, w_mean, w_var, w_n1, w_n2,
        #        prior_mean, prior_var, local_n1, local_n2]
        res = self._create_params(prior_mean, prior_var)
        self.no_weights = res[0]
        self.w_mean, self.w_log_var = res[1], res[2]
        self.w_n1, self.w_n2 = res[3], res[4]
        self.prior_mean, self.prior_var = res[5], res[6]
        self.local_n1, self.local_n2 = res[7], res[8]

        # create helper assignment ops
        self._create_assignment_ops()

        # build objective and prediction functions
        self.energy_fn, self.kl_term, self.expected_lik = self._build_energy()
        self.predict_m, self.predict_v = self._build_prediction()

        self.train_updates = self._create_train_step()
        # launch a session
        if single_thread:
            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
        else:
            self.sess = tf.Session()

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.params = TensorFlowVariablesWithScope(
            self.energy_fn, self.sess,
            scope='variational_nat', input_variables=[self.w_n1, self.w_n2])

    def train(self, x_train, y_train):
        N = x_train.shape[0]
        sess = self.sess
        tracker_file = "logs/gra_indiv_terms.txt"
        for x in range(1000000):
            a, c, c_kl, c_lik = sess.run(
                [self.train_updates, self.energy_fn, self.kl_term, self.expected_lik],
                feed_dict={self.xtrain: x_train, self.ytrain: y_train})

            tracking_grads = a[2:]
            with open(tracker_file, 'a') as file:
                tracking_grads = tracking_grads[0]
                file.write("{} {} {} {}\n".format(np.asscalar(tracking_grads[0]), np.asscalar(tracking_grads[1]),
                                                  np.asscalar(tracking_grads[2]), np.asscalar(tracking_grads[3])))

        return np.array([c, c_kl, c_lik])

    def get_weights(self):
        w_mean, w_log_var = self.sess.run([self.w_mean, self.w_log_var])
        return w_mean, np.exp(w_log_var)

    def _build_energy(self):
        # compute the expected log likelihood
        no_train = tf.cast(self.n_train, float_type)
        w_mean, w_log_var = self.w_mean, self.w_log_var
        w_var = tf.exp(w_log_var)
        const_term = -0.5 * no_train * np.log(2 * np.pi * self.noise_var)
        pred = tf.einsum('nd,d->n', self.xtrain, w_mean)
        ydiff = self.ytrain - pred
        quad_term = -0.5 / self.noise_var * tf.reduce_sum(ydiff ** 2)
        xxT = tf.reduce_sum(self.xtrain ** 2, axis=0)
        trace_term = -0.5 / self.noise_var * tf.reduce_sum(xxT * w_var)
        expected_lik = const_term + quad_term + trace_term

        # compute the kl term analytically
        const_term = -0.5 * self.no_weights
        log_std_diff = tf.log(self.prior_var) - tf.log(w_var)
        log_std_diff = 0.5 * tf.reduce_sum(log_std_diff)
        mu_diff_term = (w_var + (self.prior_mean - w_mean) ** 2) / self.prior_var
        mu_diff_term = 0.5 * tf.reduce_sum(mu_diff_term)
        kl = const_term + log_std_diff + mu_diff_term

        kl_term = kl / no_train
        expected_lik_term = - expected_lik
        return kl_term + expected_lik_term, kl_term, expected_lik_term

    def _create_params(self, prior_mean, prior_var):
        no_params = self.din
        # initialise worker to prior mean and variance
        init_var = prior_var * np.ones([no_params])
        init_mean = prior_mean * np.zeros([no_params])
        init_n2 = 1.0 / init_var
        init_n1 = init_mean / init_var

        with tf.name_scope('variational_cov'):
            w_log_var = tf.Variable(
                tf.constant(np.log(init_var), dtype=float_type),
                name='log_variance')
            w_mean = tf.Variable(
                tf.constant(init_mean, dtype=float_type),
                name='mean')

        with tf.name_scope('variational_nat'):
            w_n1 = tf.Variable(
                tf.constant(init_n1, dtype=float_type),
                name='precision_x_mean')
            w_n2 = tf.Variable(
                tf.constant(init_n2, dtype=float_type),
                name='precision')

        prior_mean_val = prior_mean * np.ones([no_params])
        prior_var_val = prior_var * np.ones([no_params])
        with tf.name_scope('prior'):
            prior_mean = tf.Variable(
                tf.constant(prior_mean_val, dtype=float_type),
                trainable=False,
                name='mean')
            prior_var = tf.Variable(
                tf.constant(prior_var_val, dtype=float_type),
                trainable=False,
                name='variance')

        no_workers = self.no_workers
        prior_n1 = prior_mean_val / prior_var_val
        prior_n2 = 1.0 / prior_var_val
        data_n1 = init_n1 - prior_n1
        data_n2 = init_n2 - prior_n2
        local_n1 = data_n1 / self.no_workers
        local_n2 = data_n2 / self.no_workers

        res = [no_params, w_mean, w_log_var, w_n1, w_n2,
               prior_mean, prior_var, local_n1, local_n2]
        return res

    def _create_assignment_ops(self):
        # create helper assignment ops
        self.prior_mean_val = tf.placeholder(dtype=float_type)
        self.prior_var_val = tf.placeholder(dtype=float_type)
        self.post_mean_val = tf.placeholder(dtype=float_type)
        self.post_var_val = tf.placeholder(dtype=float_type)
        self.post_n1_val = tf.placeholder(dtype=float_type)
        self.post_n2_val = tf.placeholder(dtype=float_type)

        self.prior_mean_op = self.prior_mean.assign(self.prior_mean_val)
        self.prior_var_op = self.prior_var.assign(self.prior_var_val)
        self.post_mean_op = self.w_mean.assign(self.post_mean_val)
        self.post_var_op = self.w_log_var.assign(self.post_var_val)
        self.post_n1_op = self.w_n1.assign(self.post_n1_val)
        self.post_n2_op = self.w_n2.assign(self.post_n2_val)

    def set_params(self, variable_names, new_params, update_prior=True):
        # set natural parameters
        self.params.set_weights(dict(zip(variable_names, new_params)))
        # compute and set covariance parameters
        post_n1, post_n2 = self.sess.run([self.w_n1, self.w_n2])
        # print([post_n1, post_n2])
        post_var = 1.0 / post_n2
        post_mean = post_n1 / post_n2
        self.sess.run(
            [self.post_var_op, self.post_mean_op],
            feed_dict={self.post_var_val: post_var,
                       self.post_mean_val: post_mean})
        if update_prior:
            # compute and set prior
            prior_n1 = post_n1 - self.local_n1
            prior_n2 = post_n2 - self.local_n2
            prior_var = 1.0 / prior_n2
            prior_mean = prior_n1 / prior_n2
            self.sess.run([self.prior_mean_op, self.prior_var_op], feed_dict={
                self.prior_mean_val: prior_mean, self.prior_var_val: prior_var})

    def set_nat_params(self):
        # compute and set natural parameters
        post_mean, post_log_var = self.sess.run([self.w_mean, self.w_log_var])
        post_var = np.exp(post_log_var)
        post_n2 = 1.0 / post_var
        post_n1 = post_mean / post_var
        self.sess.run(
            [self.post_n1_op, self.post_n2_op],
            feed_dict={self.post_n1_val: post_n1,
                       self.post_n2_val: post_n2})

    def get_nat_params(self):
        self.set_nat_params()
        return self.sess.run([self.w_n1, self.w_n2])

    def get_params(self):
        self.set_nat_params()
        # get natural params and return
        params = self.params.get_weights()
        return list(params.keys()), list(params.values())

    def get_param_diff(self, damping=0.5):
        self.set_nat_params()
        # update local factor
        post_n1, post_n2, prior_mean, prior_var = self.sess.run([
            self.w_n1, self.w_n2, self.prior_mean, self.prior_var])
        prior_n2 = 1.0 / prior_var
        prior_n1 = prior_mean / prior_var
        new_local_n1 = post_n1 - prior_n1
        new_local_n2 = post_n2 - prior_n2
        old_local_n1 = self.local_n1
        old_local_n2 = self.local_n2
        new_local_n1 = damping * old_local_n1 + (1.0 - damping) * new_local_n1
        new_local_n2 = damping * old_local_n2 + (1.0 - damping) * new_local_n2
        new_local_n2[np.where(new_local_n2 < 0)] = 0
        delta_n1 = new_local_n1 - old_local_n1
        delta_n2 = new_local_n2 - old_local_n2
        self.local_n1 = new_local_n1
        self.local_n2 = new_local_n2
        param_deltas = [delta_n1, delta_n2]
        return param_deltas

    def _build_prediction(self):
        x = self.xtest
        w_mean, w_log_var = self.w_mean, self.w_log_var
        w_var = tf.exp(w_log_var)
        mean = tf.einsum('nd,d->n', x, w_mean)
        var = tf.reduce_sum(x * w_var * x, axis=1)
        return mean, var

    def close_session(self):
        self.sess.close()

    def compute_energy(self, x_train, y_train, batch_size=None):
        sess = self.sess
        c = sess.run(
            [self.energy_fn],
            feed_dict={self.xtrain: x_train, self.ytrain: y_train})[0]
        return c

    def prediction(self, x_test, batch_size=500):
        # Test model
        N = x_test.shape[0]
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            m, v = self.sess.run(
                [self.predict_m, self.predict_v],
                feed_dict={self.xtest: batch_x})
            if i == 0:
                ms, vs = m, v
            else:
                if len(m.shape) == 3:
                    concat_axis = 1
                else:
                    concat_axis = 0
                ms = np.concatenate(
                    (ms, m), axis=concat_axis)
                vs = np.concatenate(
                    (vs, v), axis=concat_axis)
        return ms, vs

    def _create_train_step(self):
        mean_grad, log_var_grad, tracking_terms = self.build_gradient()
        # perform incredibly simple gradient ascent - we are trying to maximise the free energy
        learning_rate = tf.constant([0.000001])
        update_m = self.w_mean.assign_add(learning_rate * mean_grad)
        update_v = self.w_log_var.assign_add(learning_rate * 10.85 * log_var_grad)
        return update_m, update_v, tracking_terms, tf.print(1 / tf.exp(self.w_log_var))

    def build_gradient(self):
        # build the gradient of the free energy over all datapoints
        # note that the input to this function are values which are being investigated.
        xTx = tf.einsum('na,nb->ab', self.xtrain, self.xtrain)
        xTy = tf.einsum('na,n->a', self.xtrain, self.ytrain)

        w_var = tf.exp(self.w_log_var)
        prior_var = self.prior_var

        entropy_mean_term = 0
        entropy_var_term = 1 / (2 * w_var)
        lik_mean_term = (-0.5 / (self.noise_var)) * (2 * self.w_mean * xTx - 2 * xTy)
        lik_var_term = (-0.5 / self.noise_var) * xTx
        prior_mean_term = (-0.5 / prior_var) * (2 * self.w_mean - 2 * self.prior_mean)
        prior_var_term = (-0.5 / prior_var)

        non_data_mean_term = entropy_mean_term + prior_mean_term
        non_data_var_term = entropy_var_term + prior_var_term

        tracking_terms = [non_data_mean_term, w_var * non_data_var_term, lik_mean_term, w_var * lik_var_term]

        # sum and reshape into vectors
        mean_term = tf.reshape(entropy_mean_term + lik_mean_term + prior_mean_term, [1])
        var_term = tf.reshape(entropy_var_term + lik_var_term + prior_var_term, [1])
        return mean_term, w_var * var_term, tracking_terms


class LinReg_MFVI_DPSGD():
    """
    Stochastic global variational inference using Differentially
    Private Stochastic Gradient Ascent
    fully factorised Gaussian approximation
    variational parameters: mu and log(var)
    """

    def __init__(self, din, n_train, accountant, noise_var=1,
                 prior_mean=0.0, prior_var=1,
                 no_workers=1, gradient_bound=5,
                 learning_rate=1e-6, dpsgd_noise_scale=5, lot_size=1000,
                 num_iterations=10000, single_thread=True):
        self.din = din
        # input and output placeholders
        self.xtrain = tf.placeholder(float_type, [None, din], 'input')
        self.xtest = tf.placeholder(float_type, [None, din], 'test_input')
        self.ytrain = tf.placeholder(float_type, [None], 'target')
        self.n_train = n_train
        self.no_workers = no_workers
        self.noise_var = noise_var

        self.accountant = accountant

        # DP-SGD Parameters
        self.gradient_bound = gradient_bound
        self.learning_rate_mean = tf.constant(learning_rate, dtype=float_type)
        self.learning_rate_var = tf.constant(learning_rate * 1e3, dtype=float_type)
        self.dpsgd_noise_scale = dpsgd_noise_scale
        self.num_iterations = num_iterations
        self.lot_size = lot_size

        res = self._create_params(prior_mean, prior_var)
        self.no_weights = res[0]
        self.w_mean, self.w_log_var = res[1], res[2]
        self.w_n1, self.w_n2 = res[3], res[4]
        self.prior_mean, self.prior_var = res[5], res[6]
        self.local_n1, self.local_n2 = res[7], res[8]

        self.prior_var_num = prior_var

        noise_var_dpsgd = dpsgd_noise_scale * gradient_bound
        # create noise distribution for convience
        # multiple dimension by 2 since there is a mean and variance for each dimension
        self.noise_dist = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(2 * din),
                                                                   scale_diag=noise_var_dpsgd * np.ones(2 * din))
        # create helper assignment ops
        self._create_assignment_ops()

        # build objective and prediction functions
        self.energy_fn, self.kl_term, self.expected_lik = self._build_energy()
        self.predict_m, self.predict_v = self._build_prediction()

        self.train_updates = self._create_train_step()
        # launch a session
        if single_thread:
            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
        else:
            self.sess = tf.Session()

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.params = TensorFlowVariablesWithScope(
            self.energy_fn, self.sess,
            scope='variational_nat', input_variables=[self.w_n1, self.w_n2])

    def get_params_for_logging(self):
        # we want to save the dpsgd parameters we are gonna use
        learning_rate_mean, learning_rate_var = self.sess.run([self.learning_rate_mean, self.learning_rate_var])
        return [self.gradient_bound, learning_rate_mean, learning_rate_var,
                self.dpsgd_noise_scale, self.num_iterations, self.lot_size]

    def train(self, x_train, y_train):
        N = x_train.shape[0]
        sess = self.sess
        # really, this should be updated at every single point. However, it takes ages to run....
        # self.accountant.log_moments_increment = self.generate_log_moments(N, self.accountant.max_lambda)
        for x in range(self.num_iterations):
            if not self.accountant.should_stop:
                # print("Local Iteration {}".format(x))
                _, c, c_kl, c_lik = sess.run(
                    [self.train_updates, self.energy_fn, self.kl_term, self.expected_lik],
                    feed_dict={self.xtrain: x_train, self.ytrain: y_train})
                self.accountant.update_privacy_budget()

        if not self.accountant.should_stop:
            # print("Privacy Cost: " + str(self.accountant.current_tracked_val))
            return np.array([c, c_kl, c_lik])

    def get_weights(self):
        w_mean, w_log_var = self.sess.run([self.w_mean, self.w_log_var])
        return w_mean, np.exp(w_log_var)

    def _build_energy(self):
        # compute the expected log likelihood
        no_train = tf.cast(self.n_train, float_type)
        w_mean, w_log_var = self.w_mean, self.w_log_var
        w_var = tf.exp(w_log_var)
        const_term = -0.5 * no_train * np.log(2 * np.pi * self.noise_var)
        pred = tf.einsum('nd,d->n', self.xtrain, w_mean)
        ydiff = self.ytrain - pred
        quad_term = -0.5 / self.noise_var * tf.reduce_sum(ydiff ** 2)
        xxT = tf.reduce_sum(self.xtrain ** 2, axis=0)
        trace_term = -0.5 / self.noise_var * tf.reduce_sum(xxT * w_var)
        expected_lik = const_term + quad_term + trace_term

        # compute the kl term analytically
        const_term = -0.5 * self.no_weights
        log_std_diff = tf.log(self.prior_var) - tf.log(w_var)
        log_std_diff = 0.5 * tf.reduce_sum(log_std_diff)
        mu_diff_term = (w_var + (self.prior_mean - w_mean) ** 2) / self.prior_var
        mu_diff_term = 0.5 * tf.reduce_sum(mu_diff_term)
        kl = const_term + log_std_diff + mu_diff_term

        kl_term = kl / no_train
        expected_lik_term = - expected_lik
        return kl_term + expected_lik_term, kl_term, expected_lik_term

    def _create_params(self, prior_mean, prior_var):
        no_params = self.din
        # initialise worker to prior mean and variance
        init_var = prior_var * np.ones([no_params])
        init_mean = prior_mean * np.zeros([no_params])
        init_n2 = 1.0 / init_var
        init_n1 = init_mean / init_var

        with tf.name_scope('variational_cov'):
            w_log_var = tf.Variable(
                tf.constant(np.log(init_var), dtype=float_type),
                name='log_variance')
            w_mean = tf.Variable(
                tf.constant(init_mean, dtype=float_type),
                name='mean')

        with tf.name_scope('variational_nat'):
            w_n1 = tf.Variable(
                tf.constant(init_n1, dtype=float_type),
                name='precision_x_mean')
            w_n2 = tf.Variable(
                tf.constant(init_n2, dtype=float_type),
                name='precision')

        prior_mean_val = prior_mean * np.ones([no_params])
        prior_var_val = prior_var * np.ones([no_params])
        with tf.name_scope('prior'):
            prior_mean = tf.Variable(
                tf.constant(prior_mean_val, dtype=float_type),
                trainable=False,
                name='mean')
            prior_var = tf.Variable(
                tf.constant(prior_var_val, dtype=float_type),
                trainable=False,
                name='variance')

        no_workers = self.no_workers
        prior_n1 = prior_mean_val / prior_var_val
        prior_n2 = 1.0 / prior_var_val
        data_n1 = init_n1 - prior_n1
        data_n2 = init_n2 - prior_n2
        local_n1 = data_n1 / self.no_workers
        local_n2 = data_n2 / self.no_workers

        res = [no_params, w_mean, w_log_var, w_n1, w_n2,
               prior_mean, prior_var, local_n1, local_n2]
        return res

    def _create_assignment_ops(self):
        # create helper assignment ops
        self.prior_mean_val = tf.placeholder(dtype=float_type)
        self.prior_var_val = tf.placeholder(dtype=float_type)
        self.post_mean_val = tf.placeholder(dtype=float_type)
        self.post_var_val = tf.placeholder(dtype=float_type)
        self.post_n1_val = tf.placeholder(dtype=float_type)
        self.post_n2_val = tf.placeholder(dtype=float_type)

        self.prior_mean_op = self.prior_mean.assign(self.prior_mean_val)
        self.prior_var_op = self.prior_var.assign(self.prior_var_val)
        self.post_mean_op = self.w_mean.assign(self.post_mean_val)
        self.post_var_op = self.w_log_var.assign(self.post_var_val)
        self.post_n1_op = self.w_n1.assign(self.post_n1_val)
        self.post_n2_op = self.w_n2.assign(self.post_n2_val)

    def set_params(self, variable_names, new_params, update_prior=True):
        # set natural parameters
        self.params.set_weights(dict(zip(variable_names, new_params)))
        # compute and set covariance parameters
        post_n1, post_n2 = self.sess.run([self.w_n1, self.w_n2])
        # print([post_n1, post_n2])
        post_var = 1.0 / post_n2
        post_mean = post_n1 / post_n2
        self.sess.run(
            [self.post_var_op, self.post_mean_op],
            feed_dict={self.post_var_val: np.log(post_var),
                       self.post_mean_val: post_mean})
        if update_prior:
            # compute and set prior
            prior_n1 = post_n1 - self.local_n1
            prior_n2 = post_n2 - self.local_n2
            prior_var = 1.0 / prior_n2
            prior_mean = prior_n1 / prior_n2
            self.sess.run([self.prior_mean_op, self.prior_var_op], feed_dict={
                self.prior_mean_val: prior_mean, self.prior_var_val: prior_var})

    def set_nat_params(self):
        # compute and set natural parameters
        post_mean, post_log_var = self.sess.run([self.w_mean, self.w_log_var])
        post_var = np.exp(post_log_var)
        post_n2 = 1.0 / post_var
        post_n1 = post_mean / post_var
        self.sess.run(
            [self.post_n1_op, self.post_n2_op],
            feed_dict={self.post_n1_val: post_n1,
                       self.post_n2_val: post_n2})

    def get_nat_params(self):
        self.set_nat_params()
        return self.sess.run([self.w_n1, self.w_n2])

    def get_params(self):
        self.set_nat_params()
        # get natural params and return
        params = self.params.get_weights()
        return list(params.keys()), list(params.values())

    def get_param_diff(self, damping=0.5):
        self.set_nat_params()
        # update local factor
        post_n1, post_n2, prior_mean, prior_var = self.sess.run([
            self.w_n1, self.w_n2, self.prior_mean, self.prior_var])
        prior_n2 = 1.0 / prior_var
        prior_n1 = prior_mean / prior_var
        new_local_n1 = post_n1 - prior_n1
        new_local_n2 = post_n2 - prior_n2
        old_local_n1 = self.local_n1
        old_local_n2 = self.local_n2
        new_local_n1 = damping * old_local_n1 + (1.0 - damping) * new_local_n1
        new_local_n2 = damping * old_local_n2 + (1.0 - damping) * new_local_n2
        new_local_n2[np.where(new_local_n2 < 0)] = 0
        delta_n1 = new_local_n1 - old_local_n1
        delta_n2 = new_local_n2 - old_local_n2
        self.local_n1 = new_local_n1
        self.local_n2 = new_local_n2
        param_deltas = [delta_n1, delta_n2]
        return param_deltas

    def _build_prediction(self):
        x = self.xtest
        w_mean, w_log_var = self.w_mean, self.w_log_var
        w_var = tf.exp(w_log_var)
        mean = tf.einsum('nd,d->n', x, w_mean)
        var = tf.reduce_sum(x * w_var * x, axis=1)
        return mean, var

    def close_session(self):
        self.sess.close()

    def compute_energy(self, x_train, y_train, batch_size=None):
        sess = self.sess
        c = sess.run(
            [self.energy_fn],
            feed_dict={self.xtrain: x_train, self.ytrain: y_train})[0]
        return c

    def prediction(self, x_test, batch_size=500):
        # Test model
        N = x_test.shape[0]
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            m, v = self.sess.run(
                [self.predict_m, self.predict_v],
                feed_dict={self.xtest: batch_x})
            if i == 0:
                ms, vs = m, v
            else:
                if len(m.shape) == 3:
                    concat_axis = 1
                else:
                    concat_axis = 0
                ms = np.concatenate(
                    (ms, m), axis=concat_axis)
                vs = np.concatenate(
                    (vs, v), axis=concat_axis)
        return ms, vs

    def _create_train_step(self):
        # mean_grad, log_var_grad = self.build_gradient()
        # perform incredibly simple gradient ascent - we are trying to maximise the free energy
        mean_noisy_grad, noisy_log_var_grad = self.build_noisy_partial_gradient()
        update_m = self.w_mean.assign_add(self.learning_rate_mean * mean_noisy_grad)
        update_lv = self.w_log_var.assign_add(self.learning_rate_var * noisy_log_var_grad)
        # approximately true optimal parameters - to check if gradient is small enough
        # update_m = self.w_mean.assign([2])
        # update_lv = self.w_log_var.assign([tf.log(1/95000)])
        return update_m, update_lv

    def build_gradient(self):
        # build the gradient of the free energy over all datapoints
        # note that the input to this function are values which are being investigated.
        xTx = tf.einsum('na,nb->ab', self.xtrain, self.xtrain)
        xTy = tf.einsum('na,n->a', self.xtrain, self.ytrain)

        w_var = tf.exp(self.w_log_var)
        prior_var = self.prior_var

        entropy_mean_term = 0
        entropy_var_term = 1 / (2 * w_var)
        lik_mean_term = (-0.5 / (self.noise_var)) * (2 * self.w_mean * xTx - 2 * xTy)
        lik_var_term = (-0.5 / self.noise_var) * xTx
        prior_mean_term = (-0.5 / prior_var) * (2 * self.w_mean - 2 * self.prior_mean)
        prior_var_term = (-0.5 / prior_var)

        # sum and reshape into vectors
        mean_term = tf.reshape(entropy_mean_term + lik_mean_term + prior_mean_term, [1])
        var_term = tf.reshape(entropy_var_term + lik_var_term + prior_var_term, [1])
        return mean_term, w_var * var_term

    def select_lot(self, lot_size, x, y):
        # select lot to perform SGD on
        N = x.shape[0]
        indices = np.random.choice(np.arange(0, N, 1), lot_size, replace=False)
        x_red = x[indices]
        y_red = y[indices]
        return [x_red, y_red]

    def clip_sum_gradients(self, mean_indiv_term, var_indiv_term):
        c_t = self.gradient_bound
        grads = np.sqrt(np.square(mean_indiv_term) + np.square(var_indiv_term))
        factors = np.max(np.stack((grads / c_t, np.ones(grads.size))), axis=0)
        # factors = np.ones(mean_indiv_term.size)
        clipped_mean_grads = sum(mean_indiv_term / factors)
        clipped_var_grad = sum(var_indiv_term / factors)
        return clipped_mean_grads.astype(np.float32), clipped_var_grad.astype(np.float32)

    def build_noisy_partial_gradient(self):
        # build gradient of the free energy over a lot_size datapoints for stochastic
        # gradient descent
        # reduced versions of the training data
        red = tf.py_func(self.select_lot, [self.lot_size, self.xtrain, self.ytrain], (float_type, float_type))
        L = self.lot_size
        # use shape operator, because the shape is able to vary at runtime
        N = tf.shape(self.xtrain)[0]

        x_red = red[0]
        y_red = red[1]

        # set shapes - we know that the reduced input versions will be of this shape (by definition)
        x_red.set_shape([L, 1])
        y_red.set_shape([L])

        # now this needs to be stored per individual. create 1d vectors for each results
        xTx_i = tf.einsum('na,na->n', x_red, x_red)
        xTy_i = tf.einsum('na,n->n', x_red, y_red)

        w_var = tf.exp(self.w_log_var)
        prior_var = self.prior_var

        # ensure it is float 32
        prefactor = tf.cast(1 / N, dtype=float_type)

        # these are now individual terms
        lik_mean_term_i = (-0.5 / self.noise_var) * (2 * self.w_mean * xTx_i - 2 * xTy_i)
        lik_var_term_i = (-0.5 / self.noise_var) * xTx_i

        # these terms needs to be rescaling by 1/N (summed later on)
        entropy_mean_term = 0
        entropy_var_term = prefactor * 1 / (2 * w_var)
        prior_mean_term = prefactor * (-0.5 / prior_var) * (2 * self.w_mean - 2 * self.prior_mean)
        prior_var_term = prefactor * (-0.5 / prior_var)

        # sum and reshape into vectors
        mean_term_i = entropy_mean_term + lik_mean_term_i + prior_mean_term
        var_term_i = entropy_var_term + lik_var_term_i + prior_var_term
        mean_term, var_term = tf.py_func(self.clip_sum_gradients, [mean_term_i, var_term_i], (float_type, float_type))
        # now add noise to the term - sample from noise term and then add to the mean and variance terms to create a
        # noisy gradient
        noise = self.noise_dist.sample()
        noisy_mean_term = tf.reshape(mean_term + tf.cast(noise[0], dtype=float_type), [1])
        # rescale as parameterised in terms of the log of the variance since the variance must remain positive
        noisy_log_var_term = tf.reshape(w_var * (var_term + tf.cast(noise[1], dtype=float_type)), [1])
        return noisy_mean_term, noisy_log_var_term

    @staticmethod
    def to_np_float_64(v):
        if math.isnan(v) or math.isinf(v):
            return np.inf
        else:
            return np.float64(v)

    @staticmethod
    def pdf_gauss(x, sigma, mean):
        return mp.mpf(1.) / mp.sqrt(mp.mpf("2.") * sigma ** 2 * mp.pi) * mp.exp(
            - (x - mean) ** 2 / (mp.mpf("2.") * sigma ** 2))

    @staticmethod
    def get_I1_I2_lambda(lambda_val, pdf1, pdf2):
        I1_func = lambda x: pdf1(x) * (pdf1(x) / pdf2(x)) ** lambda_val
        I2_func = lambda x: pdf2(x) * (pdf2(x) / pdf1(x)) ** lambda_val
        return I1_func, I2_func

    def generate_log_moments(self, N, max_lambda):
        L = self.lot_size
        q = L / N

        # generate pdfs which are to be integrated numerically
        pdf1 = lambda x: LinReg_MFVI_DPSGD.pdf_gauss(x, self.dpsgd_noise_scale, mp.mpf(0))
        pdf2 = lambda x: (1 - q) * LinReg_MFVI_DPSGD.pdf_gauss(x, self.dpsgd_noise_scale, mp.mpf(0)) + \
                         q * LinReg_MFVI_DPSGD.pdf_gauss(x, self.dpsgd_noise_scale, mp.mpf(1))

        # placeholder for alpha_M(lambda) for each iteration
        alpha_M_lambda = np.zeros(max_lambda)

        for lambda_val in range(1, max_lambda + 1):
            # it isn't defined which dataset is D and which is D' - thus consider both and take the maximum
            I1_func, I2_func = self.get_I1_I2_lambda(lambda_val, pdf1, pdf2)
            I1_val, _ = mp.quad(I1_func, [-mp.inf, mp.inf], error=True)
            I2_val, _ = mp.quad(I2_func, [-mp.inf, mp.inf], error=True)

            I1_val = LinReg_MFVI_DPSGD.to_np_float_64(I1_val)
            I2_val = LinReg_MFVI_DPSGD.to_np_float_64(I2_val)

            if I1_val > I2_val:
                alpha_M_lambda[lambda_val - 1] = np.log(I1_val)
            else:
                alpha_M_lambda[lambda_val - 1] = np.log(I2_val)

        return alpha_M_lambda

    def track_privacy_moments_accountant_fixed_delta(self, N, max_lambda_in, num_intervals, delta):
        # note that in this class, the settings used are the same for every iteration. Therefore, the numerical
        # integration required can be performed once and then multiplied

        max_lambda = int(np.ceil(max_lambda_in))

        # generate log moments
        num_iterations = self.num_iterations
        log_moments = self.generate_log_moments(N, max_lambda)

        # total evals
        total_evals = num_intervals * num_iterations
        epsilons = np.zeros(total_evals)

        # convert to deltas given fixed epsilon for each iteration
        for i in range(1, total_evals + 1):
            eps = np.inf
            for lambda_val in range(1, max_lambda + 1):
                current_eps_bound = (1 / lambda_val) * ((i * log_moments[lambda_val - 1]) - np.log(delta))
                # use the smallest upper bound for the tightest guarantee
                if current_eps_bound < eps:
                    eps = current_eps_bound
            # store running track...
            epsilons[i - 1] = eps

        epoch_sf = self.lot_size / N

        return epsilons[-1], epsilons, total_evals, epoch_sf

    def track_privacy_adv_composition_fixed_delta(self, N, num_intervals, delta):
        L = self.lot_size
        q = L / N
        num_iterations = self.num_iterations
        total_evals = num_iterations * num_intervals
        epsilons = np.zeros(total_evals)

        delta_prime = 0.1 * delta

        for k in range(1, total_evals + 1):
            amp_delta_i = (delta - delta_prime) / k
            delta_i = amp_delta_i / q

            beta = np.sqrt(2 * np.log(1.25 / delta_i))
            eps_i = beta / self.dpsgd_noise_scale

            # try amplification
            eps_i_amp = np.log10(1 + (q * (np.exp(eps_i) - 1)))
            if eps_i_amp < eps_i:
                eps_i = eps_i_amp

            # compose back
            eps_tot = (np.sqrt(2 * k * np.log(1 / delta_prime)) * eps_i) + (k * eps_i * (np.exp(eps_i) - 1))
            epsilons[k - 1] = eps_tot

        return epsilons[-1], epsilons, total_evals, q


class LinReg_MFVI_DP_analytic():
    """
    Stochastic global variational inference using Adam optimiser
    fully factorised Gaussian approximation
    variational parameters: mu and log(var)
    """

    def __init__(self, din, n_train, accountant, noise_var=1,
                 prior_mean=0.0, prior_var=1.0,
                 no_workers=1, clipping_bound=10,
                 dp_noise_scale=0.01, L=1000, model_config="clipped_noisy", single_thread=True):
        self.din = din
        # input and output placeholders
        self.xtrain = tf.placeholder(float_type, [None, din], 'input')
        self.xtest = tf.placeholder(float_type, [None, din], 'test_input')
        self.ytrain = tf.placeholder(float_type, [None], 'target')
        self.n_train = n_train
        self.no_workers = no_workers
        self.noise_var = noise_var
        self.model_config = model_config

        self.accountant = accountant

        res = self._create_params(prior_mean, prior_var)
        self.no_weights = res[0]
        self.w_mean, self.w_var = res[1], res[2]
        self.w_n1, self.w_n2 = res[3], res[4]
        self.prior_mean, self.prior_var = res[5], res[6]
        self.local_n1, self.local_n2 = res[7], res[8]

        self.prior_var_num = prior_var

        self.clipping_bound = clipping_bound
        self.dp_noise_scale = dp_noise_scale
        self.lot_size = self.n_train

        # create helper assignment ops
        self._create_assignment_ops()

        # build objective and prediction functions
        self.energy_fn, self.kl_term, self.expected_lik = self._build_energy()
        self.predict_m, self.predict_v = self._build_prediction()

        noise_var_dpsgd = dp_noise_scale * clipping_bound
        # create noise distribution for convience
        # multiple dimension by 2 since there is a mean and variance for each dimension
        self.noise_dist = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(2 * din),
                                                                   scale_diag=noise_var_dpsgd * np.ones(2 * din))

        self.train_updates = self._create_train_step()
        # launch a session
        if single_thread:
            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
        else:
            self.sess = tf.Session()

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.params = TensorFlowVariablesWithScope(
            self.energy_fn, self.sess,
            scope='variational_nat', input_variables=[self.w_n1, self.w_n2])

    def get_params_for_logging(self):
        # we want to save the dpsgd parameters we are gonna use
        return [self.clipping_bound, self.dp_noise_scale, self.lot_size]

    def select_lot(self, lot_size, x, y):
        # select lot to perform SGD on
        N = x.shape[0]
        indices = np.random.choice(np.arange(0, N, 1), lot_size, replace=False)
        x_red = x[indices]
        y_red = y[indices]
        return [x_red, y_red]

    def clip_sum_values(self, mean_vals, var_vals):
        c_t = self.clipping_bound
        grads = np.sqrt(np.square(mean_vals) + np.square(var_vals))
        factors = np.max(np.stack((grads / c_t, np.ones(grads.size))), axis=0)
        clipped_mean_vals = sum(mean_vals / factors)
        clipped_var_vals = sum(var_vals / factors)
        return clipped_mean_vals.astype(np.float32), clipped_var_vals.astype(np.float32)

    def _create_train_step(self):
        L = self.lot_size
        # use shape operator, because the shape is able to vary at runtime

        xTx_i = tf.einsum('na,na->n', self.xtrain, self.xtrain)
        xTy_i = tf.einsum('na,n->n', self.xtrain, self.ytrain)

        if self.model_config == "not_clipped_not_noisy":
            xTx_noisy = tf.math.reduce_sum(xTx_i)
            xTy_noisy = tf.math.reduce_sum(xTy_i)
        elif self.model_config == "clipped_not_noisy":
            xTx_noisy, xTy_noisy = tf.py_func(self.clip_sum_values, [xTx_i, xTy_i], (float_type, float_type))
        elif self.model_config == "not_clipped_noisy":
            xTx = tf.math.reduce_sum(xTx_i)
            xTy = tf.math.reduce_sum(xTy_i)
            noise = self.noise_dist.sample()
            xTx_noisy = tf.reshape(xTx + tf.cast(noise[0], dtype=float_type), [1])
            xTy_noisy = tf.reshape((xTy + tf.cast(noise[1], dtype=float_type)), [1])
        elif self.model_config == "clipped_noisy":
            xTx_cs, xTy_cs = tf.py_func(self.clip_sum_values, [xTx_i, xTy_i], (float_type, float_type))
            noise = self.noise_dist.sample()
            xTx_noisy = tf.reshape(xTx_cs + tf.cast(noise[0], dtype=float_type), [1])
            xTy_noisy = tf.reshape((xTy_cs + tf.cast(noise[1], dtype=float_type)), [1])
        elif self.model_config == "updated_clipped_not_noisy":
            xTx_i = xTx_i / self.noise_var + (1 / self.prior_var - 1 / self.w_var) * np.float(1.0 / 10);
            xTy_i = xTy_i / self.noise_var + (
                                                 self.prior_mean / self.prior_var - self.w_mean / self.w_var) * np.float(
                1.0 / 10);
            pres_update, mean_update = tf.py_func(self.clip_sum_values, [xTx_i, xTy_i], (float_type, float_type))
            pres = pres_update + 1 / self.w_var;
            mean_pres = mean_update + self.w_mean / self.w_var;
            update_m = self.w_mean.assign(mean_pres / pres);
            update_v = self.w_var.assign(1 / pres)
            return update_m, update_v, tf.print(pres), tf.print(mean_pres)

        Voinv = tf.diag(1.0 / self.prior_var)
        Voinvmo = self.prior_mean / self.prior_var
        Vinv = Voinv + xTx_noisy / self.noise_var
        Vinvm = Voinvmo + xTy_noisy / self.noise_var
        m = tf.reduce_sum(tf.linalg.solve(Vinv, tf.expand_dims(Vinvm, 1)), axis=1)
        v = 1.0 / tf.diag_part(Vinv)
        update_m = self.w_mean.assign(m)
        update_v = self.w_var.assign(v)
        return update_m, update_v

    def train(self, x_train, y_train):
        N = x_train.shape[0]
        sess = self.sess
        if not self.accountant.should_stop:
            _, c, c_kl, c_lik = sess.run([self.train_updates, self.energy_fn, self.kl_term, self.expected_lik],
                                         feed_dict={self.xtrain: x_train, self.ytrain: y_train})
            self.accountant.update_privacy_budget()
            return np.array([c, c_kl, c_lik])
        else:
            return np.array([0, 0, 0])

    def get_weights(self):
        w_mean, w_var = self.sess.run([self.w_mean, self.w_log_var])
        return w_mean, w_var

    def _build_energy(self):
        # compute the expected log likelihood
        no_train = tf.cast(self.n_train, float_type)
        w_mean, w_var = self.w_mean, self.w_var
        const_term = -0.5 * no_train * np.log(2 * np.pi * self.noise_var)
        pred = tf.einsum('nd,d->n', self.xtrain, w_mean)
        ydiff = self.ytrain - pred
        quad_term = -0.5 / self.noise_var * tf.reduce_sum(ydiff ** 2)
        xxT = tf.reduce_sum(self.xtrain ** 2, axis=0)
        trace_term = -0.5 / self.noise_var * tf.reduce_sum(xxT * w_var)
        expected_lik = const_term + quad_term + trace_term

        # compute the kl term analytically
        const_term = -0.5 * self.no_weights
        log_std_diff = tf.log(self.prior_var) - tf.log(w_var)
        log_std_diff = 0.5 * tf.reduce_sum(log_std_diff)
        mu_diff_term = (w_var + (self.prior_mean - w_mean) ** 2) / self.prior_var
        mu_diff_term = 0.5 * tf.reduce_sum(mu_diff_term)
        kl = const_term + log_std_diff + mu_diff_term

        kl_term = kl / no_train
        expected_lik_term = - expected_lik
        return kl_term + expected_lik_term, kl_term, expected_lik_term

    def _create_params(self, prior_mean, prior_var):
        no_params = self.din
        init_var = 2 * np.ones([no_params])
        init_mean = np.zeros([no_params])
        init_n2 = 1.0 / init_var
        init_n1 = init_mean / init_var

        with tf.name_scope('variational_cov'):
            w_var = tf.Variable(
                tf.constant(init_var, dtype=float_type),
                name='variance')
            w_mean = tf.Variable(
                tf.constant(init_mean, dtype=float_type),
                name='mean')

        with tf.name_scope('variational_nat'):
            w_n1 = tf.Variable(
                tf.constant(init_n1, dtype=float_type),
                name='precision_x_mean')
            w_n2 = tf.Variable(
                tf.constant(init_n2, dtype=float_type),
                name='precision')

        prior_mean_val = prior_mean * np.ones([no_params])
        prior_var_val = prior_var * np.ones([no_params])
        with tf.name_scope('prior'):
            prior_mean = tf.Variable(
                tf.constant(prior_mean_val, dtype=float_type),
                trainable=False,
                name='mean')
            prior_var = tf.Variable(
                tf.constant(prior_var_val, dtype=float_type),
                trainable=False,
                name='variance')

        no_workers = self.no_workers
        prior_n1 = prior_mean_val / prior_var_val
        prior_n2 = 1.0 / prior_var_val
        data_n1 = init_n1 - prior_n1
        data_n2 = init_n2 - prior_n2
        local_n1 = data_n1 / self.no_workers
        local_n2 = data_n2 / self.no_workers

        res = [no_params, w_mean, w_var, w_n1, w_n2,
               prior_mean, prior_var, local_n1, local_n2]
        return res

    def _create_assignment_ops(self):
        # create helper assignment ops
        self.prior_mean_val = tf.placeholder(dtype=float_type)
        self.prior_var_val = tf.placeholder(dtype=float_type)
        self.post_mean_val = tf.placeholder(dtype=float_type)
        self.post_var_val = tf.placeholder(dtype=float_type)
        self.post_n1_val = tf.placeholder(dtype=float_type)
        self.post_n2_val = tf.placeholder(dtype=float_type)

        self.prior_mean_op = self.prior_mean.assign(self.prior_mean_val)
        self.prior_var_op = self.prior_var.assign(self.prior_var_val)
        self.post_mean_op = self.w_mean.assign(self.post_mean_val)
        self.post_var_op = self.w_var.assign(self.post_var_val)
        self.post_n1_op = self.w_n1.assign(self.post_n1_val)
        self.post_n2_op = self.w_n2.assign(self.post_n2_val)

    def set_params(self, variable_names, new_params, update_prior=True):
        # set natural parameters
        self.params.set_weights(dict(zip(variable_names, new_params)))
        # compute and set covariance parameters
        post_n1, post_n2 = self.sess.run([self.w_n1, self.w_n2])
        # print([post_n1, post_n2])
        post_var = 1.0 / post_n2
        post_mean = post_n1 / post_n2
        self.sess.run(
            [self.post_var_op, self.post_mean_op],
            feed_dict={self.post_var_val: post_var,
                       self.post_mean_val: post_mean})
        if update_prior:
            # compute and set prior
            prior_n1 = post_n1 - self.local_n1
            prior_n2 = post_n2 - self.local_n2
            prior_var = 1.0 / prior_n2
            prior_mean = prior_n1 / prior_n2
            self.sess.run([self.prior_mean_op, self.prior_var_op], feed_dict={
                self.prior_mean_val: prior_mean, self.prior_var_val: prior_var})

    def set_nat_params(self):
        # compute and set natural parameters
        post_mean, post_var = self.sess.run([self.w_mean, self.w_var])
        post_n2 = 1.0 / post_var
        post_n1 = post_mean / post_var
        self.sess.run(
            [self.post_n1_op, self.post_n2_op],
            feed_dict={self.post_n1_val: post_n1,
                       self.post_n2_val: post_n2})

    def get_nat_params(self):
        self.set_nat_params()
        return self.sess.run([self.w_n1, self.w_n2])

    def get_params(self):
        self.set_nat_params()
        # get natural params and return
        params = self.params.get_weights()
        return list(params.keys()), list(params.values())

    def get_param_diff(self, damping=0.5):
        self.set_nat_params()
        # update local factor
        post_n1, post_n2, prior_mean, prior_var = self.sess.run([
            self.w_n1, self.w_n2, self.prior_mean, self.prior_var])
        prior_n2 = 1.0 / prior_var
        prior_n1 = prior_mean / prior_var
        new_local_n1 = post_n1 - prior_n1
        new_local_n2 = post_n2 - prior_n2
        old_local_n1 = self.local_n1
        old_local_n2 = self.local_n2
        new_local_n1 = damping * old_local_n1 + (1.0 - damping) * new_local_n1
        new_local_n2 = damping * old_local_n2 + (1.0 - damping) * new_local_n2
        new_local_n2[np.where(new_local_n2 < 0)] = 0
        delta_n1 = new_local_n1 - old_local_n1
        delta_n2 = new_local_n2 - old_local_n2
        self.local_n1 = new_local_n1
        self.local_n2 = new_local_n2
        param_deltas = [delta_n1, delta_n2]
        return param_deltas

    def _build_prediction(self):
        x = self.xtest
        w_mean, w_var = self.w_mean, self.w_var
        mean = tf.einsum('nd,d->n', x, w_mean)
        var = tf.reduce_sum(x * w_var * x, axis=1)
        return mean, var

    def close_session(self):
        self.sess.close()

    def compute_energy(self, x_train, y_train, batch_size=None):
        sess = self.sess
        c = sess.run(
            [self.energy_fn],
            feed_dict={self.xtrain: x_train, self.ytrain: y_train})[0]
        return c

    def prediction(self, x_test, batch_size=500):
        # Test model
        N = x_test.shape[0]
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            m, v = self.sess.run(
                [self.predict_m, self.predict_v],
                feed_dict={self.xtest: batch_x})
            if i == 0:
                ms, vs = m, v
            else:
                if len(m.shape) == 3:
                    concat_axis = 1
                else:
                    concat_axis = 0
                ms = np.concatenate(
                    (ms, m), axis=concat_axis)
                vs = np.concatenate(
                    (vs, v), axis=concat_axis)
        return ms, vs

    def generate_log_moments(self, N, max_lambda):
        return generate_log_moments(N, max_lambda, self.dp_noise_scale, self.lot_size)
