from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import tensorflow as tf
import numpy as np
import math
import time
from tfutils import TensorFlowVariablesWithScope
import pdb

float_type = tf.float32
int_type = tf.int32


class LinReg_MFVI_analytic():
    """
    Stochastic global variational inference using Adam optimiser
    fully factorised Gaussian approximation
    variational parameters: mu and log(var)
    """

    def __init__(self, din, n_train, noise_var=0.01,
        prior_mean=0.0, prior_var=1.0, 
        init_seed=0, no_workers=1,
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
        res = self._create_params(init_seed, prior_mean, prior_var)
        self.no_weights = res[0]
        self.w_mean, self.w_var = res[1], res[2]
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
        const_term = -0.5 * no_train * np.log(2*np.pi*self.noise_var)
        pred = tf.einsum('nd,d->n', self.xtrain, w_mean)
        ydiff = self.ytrain - pred
        quad_term = -0.5 / self.noise_var * tf.reduce_sum(ydiff**2)
        xxT = tf.reduce_sum(self.xtrain**2, axis=0)
        trace_term = -0.5 / self.noise_var * tf.reduce_sum(xxT * w_var)
        expected_lik = const_term + quad_term + trace_term

        # compute the kl term analytically
        const_term = -0.5 * self.no_weights
        log_std_diff = tf.log(self.prior_var) - tf.log(w_var)
        log_std_diff = 0.5 * tf.reduce_sum(log_std_diff)
        mu_diff_term = (w_var + (self.prior_mean - w_mean)**2) / self.prior_var
        mu_diff_term = 0.5 * tf.reduce_sum(mu_diff_term)
        kl = const_term + log_std_diff + mu_diff_term

        kl_term = kl / no_train
        expected_lik_term = - expected_lik
        return kl_term + expected_lik_term, kl_term, expected_lik_term
    
    def _create_params(self, init_seed, prior_mean, prior_var):
        no_params = self.din
        init_var = 1e6*np.ones([no_params])
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
        new_local_n1 = damping * old_local_n1 + (1.0-damping) * new_local_n1
        new_local_n2 = damping * old_local_n2 + (1.0-damping) * new_local_n2
        new_local_n2[np.where(new_local_n2<0)] = 0
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

