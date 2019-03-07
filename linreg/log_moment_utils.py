import tensorflow as tf
import numpy as np
import math

float_type = tf.float32
int_type = tf.int32

import mpmath as mp


def to_np_float_64(v):
    if math.isnan(v) or math.isinf(v):
        return np.inf
    else:
        return np.float64(v)


def pdf_gauss(x, sigma, mean):
    return mp.mpf(1.) / mp.sqrt(mp.mpf("2.") * sigma ** 2 * mp.pi) * mp.exp(
        - (x - mean) ** 2 / (mp.mpf("2.") * sigma ** 2))


def get_I1_I2_lambda(lambda_val, pdf1, pdf2):
    I1_func = lambda x: pdf1(x) * (pdf1(x) / pdf2(x)) ** lambda_val
    I2_func = lambda x: pdf2(x) * (pdf2(x) / pdf1(x)) ** lambda_val
    return I1_func, I2_func


def integral_inf_mp(fn):
    integral, _ = mp.quad(fn, [-mp.inf, mp.inf], error=True)
    return integral


def generate_log_moments(N, max_lambda, noise_scale, lot_size):
    # print('generating log moments')
    L = lot_size
    q = 1.0 * L / N

    # these moments are a function of q, noise_scale and max_lambda

    # generate pdfs which are to be integrated numerically
    pdf1 = lambda x: pdf_gauss(x, noise_scale, mp.mpf(0))
    pdf2 = lambda x: (1 - q) * pdf_gauss(x, noise_scale, mp.mpf(0)) + \
                     q * pdf_gauss(x, noise_scale, mp.mpf(1))

    # placeholder for alpha_M(lambda) for each iteration
    alpha_M_lambda = np.zeros(max_lambda)

    for lambda_val in range(1, max_lambda + 1):
        # it isn't defined which dataset is D and which is D' - thus consider both and take the maximum
        I1_func, I2_func = get_I1_I2_lambda(lambda_val, pdf1, pdf2)
        I1_val = integral_inf_mp(I1_func)
        I2_val = integral_inf_mp(I2_func)

        if I1_val > I2_val:
            alpha_M_lambda[lambda_val - 1] = to_np_float_64(mp.log(I1_val))
        else:
            alpha_M_lambda[lambda_val - 1] = to_np_float_64(mp.log(I2_val))

    return alpha_M_lambda
