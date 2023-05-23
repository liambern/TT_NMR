import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from keras import backend as K
# import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

K.set_floatx('float64')
float_type = tf.float64
import scipy.ndimage as spn
import time
import argparse
import pickle as pkl

parser = argparse.ArgumentParser(description='run')
parser.add_argument(
    '-d',
    '--dim',
    type=int,
    help='number of dimensions',
    required=True)
parser.add_argument(
    '-sd',
    '--size_dim',
    type=int,
    help='size of dimensions',
    required=True)
parser.add_argument(
    '-half',
    '--half',
    type=bool,
    help='whether to use half search',
    required=False,
    default=False)
parser.add_argument(
    '-tensor',
    '--tensor',
    type=str,
    help='random/hilbert',
    required=False,
    default="hilbert")
parser.add_argument(
    '-g',
    '--graph',
    type=bool,
    help='whether to plot progress',
    required=False,
    default=True)
parser.add_argument(
    '-test',
    '--test',
    type=bool,
    help='whether it is a part of the run file',
    required=False,
    default=False)
parser.add_argument(
    '-max',
    '--x_max',
    type=float,
    help='Border of the coordinates',
    required=True)
parser.add_argument(
    '-min',
    '--x_min',
    type=float,
    help='Border of the coordinates',
    required=True)

args = parser.parse_args()
half = args.half
xmin = args.x_min
xmax = args.x_max
d = args.dim
sd = args.size_dim
# h = 2.e0/sd
h = 2e-2
err_wanted = 1.e-12
A_shape = tf.constant([sd] * d)
test_n = 10000

@tf.function(experimental_relax_shapes=True)
def evaluate(A, indices):
    size = tf.shape(indices)[-1]
    final_shape = tf.shape(indices)[:-1]
    flattened = tf.reshape(indices, [tf.size(indices) // size, size])
    values = A(shift(flattened))
    return tf.reshape(values, final_shape)

# @tf.function(jit_compile=True, experimental_relax_shapes=True)
def matrix_k(left, right, A_shape, k):
    if tf.shape(left)[1] == 0:
        left = left[0:1, :]
    if tf.shape(right)[1] == 0:
        right = right[0:1, :]
    l_s = tf.shape(left)
    r_s = tf.shape(right)
    all_k = tf.transpose([tf.range(A_shape[k])])
    first = tf.tile(right, [A_shape[k], 1])
    second = tf.concat([tf.repeat(all_k, r_s[0], axis=0), first], axis=1)
    batched_Tk = tf.concat([tf.repeat(left, r_s[0] * A_shape[k], axis=0), tf.tile(second, [l_s[0], 1])], axis=1)
    Tk = tf.reshape(batched_Tk, [l_s[0], A_shape[k], r_s[0], tf.size(A_shape)])
    return Tk


# @tf.function(experimental_relax_shapes=True)
def add_pivot(Tk, Tk_plus, I, J_plus, k, pivot):
    piv_left = pivot[:, :k + 1]
    piv_right = pivot[:, k + 1:]
    new_Tk = tf.scatter_nd(tf.cast(tf.where(tf.ones_like(Tk)), tf.int32), tf.reshape(Tk, [-1]),
                           tf.cast(tf.shape(Tk) + tf.constant([0, 0, 1]), tf.int32))
    first_k = tf.tile(piv_right, [A_shape[k], 1])
    second_k = tf.concat([tf.transpose([tf.range(A_shape[k])]), first_k], axis=1)
    new_Tk_part = tf.concat([tf.repeat(I, A_shape[k], axis=0), tf.tile(second_k, [tf.shape(I)[0], 1])], axis=1)
    new_Tk_part_values = evaluate(A, new_Tk_part)
    ##
    left_ind = tf.transpose([tf.repeat(tf.range(tf.shape(Tk)[0]), sd)])
    mid_ind = tf.transpose([tf.tile(tf.range(sd), [tf.shape(Tk)[0]])])
    right_ind = tf.ones([sd*tf.shape(Tk)[0], 1], dtype=tf.int32)*tf.shape(Tk)[2]
    indices_to_update = tf.concat([left_ind, mid_ind, right_ind], axis=1)
    ##
    # indices_to_update = tf.boolean_mask(tf.cast(tf.where(tf.ones_like(new_Tk)), tf.int32),
    #                                     tf.logical_not(tf.reduce_any(tf.reduce_all(
    #                                         tf.expand_dims(tf.where(tf.ones_like(new_Tk)), 0) == tf.expand_dims(
    #                                             tf.where(tf.ones_like(Tk)), 1), axis=2), axis=0)))
    new_Tk = tf.tensor_scatter_nd_update(new_Tk, indices_to_update, new_Tk_part_values)

    new_Tk_plus = tf.scatter_nd(tf.cast(tf.where(tf.ones_like(Tk_plus)), tf.int32), tf.reshape(Tk_plus, [-1]),
                                tf.cast(tf.shape(Tk_plus) + tf.constant([1, 0, 0]), tf.int32))
    first_k_plus = tf.tile(J_plus, [A_shape[k], 1])
    second_k_plus = tf.concat(
        [tf.repeat(tf.transpose([tf.range(A_shape[k])]), tf.shape(J_plus)[0], axis=0), first_k_plus], axis=1)
    new_Tk_part_plus = tf.concat([tf.repeat(piv_left, A_shape[k] * tf.shape(J_plus)[0], axis=0), second_k_plus],
                                 axis=1)
    new_Tk_part_values_plus = evaluate(A, new_Tk_part_plus)
    ##
    left_ind_plus = tf.ones([sd*tf.shape(Tk_plus)[2], 1], dtype=tf.int32)*tf.shape(Tk_plus)[0]
    mid_ind_plus = tf.transpose([tf.repeat(tf.range(sd), tf.shape(Tk_plus)[2])])
    right_ind_plus = tf.transpose([tf.tile(tf.range(tf.shape(Tk_plus)[2]), [sd])])
    indices_to_update_plus = tf.concat([left_ind_plus, mid_ind_plus, right_ind_plus], axis=1)
    ##
    # indices_to_update_plus = tf.boolean_mask(tf.cast(tf.where(tf.ones_like(new_Tk_plus)), tf.int32),
    #                                          tf.logical_not(tf.reduce_any(tf.reduce_all(
    #                                              tf.expand_dims(tf.where(tf.ones_like(new_Tk_plus)),
    #                                                             0) == tf.expand_dims(tf.where(tf.ones_like(Tk_plus)),
    #                                                                                  1), axis=2), axis=0)))
    new_Tk_plus = tf.tensor_scatter_nd_update(new_Tk_plus, indices_to_update_plus, new_Tk_part_values_plus)
    return new_Tk, new_Tk_plus


# @tf.function(experimental_relax_shapes=True)
def full_pi_alpha_exact(I, J_plus, A_shape, k):
    all_k = tf.transpose([tf.range(A_shape[k])])
    all_k_plus = tf.transpose([tf.range(A_shape[k + 1])])
    first = tf.concat([tf.repeat(all_k_plus, tf.shape(J_plus)[0], axis=0), tf.tile(J_plus, [A_shape[k + 1], 1])],
                      axis=1)
    second = tf.concat([tf.repeat(all_k, tf.shape(first)[0], axis=0), tf.tile(first, [A_shape[k], 1])], axis=1)
    batched_PIk = tf.concat([tf.repeat(I, tf.shape(second)[0], axis=0), tf.tile(second, [tf.shape(I)[0], 1])],
                            axis=1)
    PIk = tf.reshape(batched_PIk,
                     [tf.shape(I)[0], A_shape[k], A_shape[k + 1], tf.shape(J_plus)[0], tf.size(A_shape)])
    return PIk


# @tf.function(experimental_relax_shapes=True)
def half_pi_alpha_exact(I_used, J_used, all_k, all_k_plus, A_shape):
    first = tf.concat([tf.repeat(all_k_plus, tf.shape(J_used)[0], axis=0), tf.tile(J_used, [tf.size(all_k_plus), 1])],
                      axis=1)
    second = tf.concat([tf.repeat(all_k, tf.shape(first)[0], axis=0), tf.tile(first, [tf.size(all_k), 1])], axis=1)
    batched_PIk = tf.concat([tf.repeat(I_used, tf.shape(second)[0], axis=0), tf.tile(second, [tf.shape(I_used)[0], 1])],
                            axis=1)
    PIk = tf.reshape(batched_PIk,
                     [tf.shape(I_used)[0], tf.size(all_k), tf.size(all_k_plus), tf.shape(J_used)[0], tf.size(A_shape)])
    return PIk


# @tf.function(experimental_relax_shapes=True)
def find_pivot(A, parts, fullI, Tk, Tk_plus, I, I_plus, J_plus, k, half):
    if half:
        I_used = I
        all_k = tf.transpose([tf.range(A_shape[k])])
        all_k_plus = tf.transpose([tf.range(A_shape[k + 1])])
        J_used = J_plus
        if tf.cast(tf.random.uniform([1], maxval=2, dtype=tf.int32), tf.bool):
            r1 = tf.random.uniform([1], maxval=tf.shape(I_used)[0], dtype=tf.int32)[0]
            I_used = I_used[r1:r1 + 1]
            r2 = tf.random.uniform([1], maxval=tf.shape(all_k)[0], dtype=tf.int32)[0]
            all_k = all_k[r2:r2 + 1]
            PI_approx = tf.einsum('ijk,kb...->ijb...', QR(Tk, I_plus, I)[r1:r1 + 1, r2:r2 + 1, :], Tk_plus)
        else:
            r1 = tf.random.uniform([1], maxval=tf.shape(all_k_plus)[0], dtype=tf.int32)[0]
            all_k_plus = all_k_plus[r1:r1 + 1]
            r2 = tf.random.uniform([1], maxval=tf.shape(J_used)[0], dtype=tf.int32)[0]
            J_used = J_used[r2:r2 + 1]
            PI_approx = tf.einsum('ijk,kb...->ijb...', QR(Tk, I_plus, I), Tk_plus[:, r1:r1 + 1, r2:r2 + 1])

        PI_ind = half_pi_alpha_exact(I_used, J_used, all_k, all_k_plus, A_shape)
        PI = evaluate(A, PI_ind)

    else:
        PI_ind = full_pi_alpha_exact(I, J_plus, A_shape, k)
        PI = evaluate(A, PI_ind)
        PI_approx = tf.einsum('ijk,kb...->ijb...', QR(Tk, I_plus, I), Tk_plus)
        integral_weight = 1.#env_err(parts, fullI, k)

    diff = integral_weight * tf.abs(PI - PI_approx) / (tf.abs(PI) + err_wanted)
    # shift_diff = diff - tf.reduce_max(diff)
    # new_pivot = tf.gather_nd(PI_ind, tf.where(shift_diff == 0))
    # return new_pivot[0:1, :], tf.reduce_max(diff)

    # diff = integral_weight * tf.abs((PI - PI_approx)) / (tf.abs(PI) + err_wanted)
    err, new_pivot = exact_search(diff, PI_ind)
    return new_pivot, err

@tf.function(jit_compile=True)
def approx_search(A, ind):
    err, pos = tf.math.approx_max_k(tf.reshape(tf.cast(A, tf.float32), [-1]), 1, reduction_dimension=0)
    return tf.cast(err[0], float_type), tf.expand_dims(tf.reshape(ind, [-1, d])[pos[0]], 0)

# @tf.function(experimental_relax_shapes=True)
def exact_search(A, ind):
    max_err = tf.reduce_max(A)
    shift_diff = A - tf.reduce_max(A)
    new_pivot = tf.gather_nd(ind, tf.where(shift_diff == 0))
    return max_err, new_pivot[0:1, :]

def inital_Tk(A, I, J, A_shape):
    parts = []
    for k in range(len(A_shape)):
        left = I[k]
        right = J[k]
        Tk = matrix_k(left, right, A_shape, k)
        parts.append(evaluate(A, Tk))
    return parts


@tf.function(experimental_relax_shapes=True)
def evaluate_full(parts, I):
    r = tf.squeeze(parts[len(parts) - 1], axis=2)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = QR(parts[i], I[i + 1], I[i])
        r = tf.einsum('ijk,kb...->ijb...', TP, r)
    TP = tf.squeeze(QR(parts[0], I[1], I[0]), axis=0)
    r = tf.einsum('ij,jb...->ib...', TP, r)
    return r

@tf.function(experimental_relax_shapes=True)
def evaluate_point(parts, I, point):
    r = tf.squeeze(parts[len(parts) - 1], axis=2)[:, point[-1]:point[-1]+1]
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = QR(parts[i], I[i + 1], I[i])[:, point[i]:point[i]+1, :]
        r = tf.einsum('ijk,kb...->ijb...', TP, r)
    TP = tf.squeeze(QR(parts[0], I[1], I[0]), axis=0)[point[0]:point[0]+1, :]
    r = tf.einsum('ij,jb...->ib...', TP, r)
    return tf.reshape(r, [-1])

@tf.function(experimental_relax_shapes=True)
def evaluate_points(parts, I, points):
    r = tf.gather(tf.squeeze(parts[len(parts) - 1], axis=2), points[:, -1], axis=1)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = tf.gather(QR(parts[i], I[i + 1], I[i]), points[:, i], axis=1)
        r = tf.einsum('ijk,kj->ij', TP, r)
    TP = tf.gather(QR(parts[0], I[1], I[0]), points[:, 0], axis=1)
    r = tf.einsum('ijk,kj->ij', TP, r)
    return tf.transpose(r)


@tf.function(experimental_relax_shapes=True)
def test(A, parts, I, n):
    random_points = tf.random.uniform([n, d], maxval=sd, dtype=tf.int32)
    approx = evaluate_points(parts, I, random_points)
    exact = A(shift(random_points))
    err = tf.reduce_max(tf.abs(exact-approx)/(tf.abs(exact)+err_wanted))
    return err


@tf.function(experimental_relax_shapes=True)
def evaluate_integral(parts, I):
    wk, shift_size = weights()
    r = tf.reduce_sum(tf.squeeze(wk*parts[len(parts) - 1], axis=2), axis=1, keepdims=True)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = tf.reduce_sum(wk*QR(parts[i], I[i + 1], I[i]), axis=1, keepdims=True)
        r = tf.einsum('ijk,kb...->ijb...', TP, r)
    TP = tf.reduce_sum(tf.squeeze(wk*QR(parts[0], I[1], I[0]), axis=0), axis=0, keepdims=True)
    r = tf.einsum('ij,jb...->ib...', TP, r)
    return tf.reshape(r, [-1]) * tf.cast(shift_size**d, float_type)


@tf.function(experimental_relax_shapes=True)
def env_err(parts, I, k):
    wk, shift_size = weights()
    if k == 0:
        L = tf.ones([1], dtype=float_type)
    else:
        L = tf.reduce_sum(wk * QR(parts[k-1], I[k], I[k-1]), axis=1, keepdims=True)
        for i in range(k - 1)[::-1]:
            TP = tf.reduce_sum(wk * QR(parts[i], I[i + 1], I[i]), axis=1, keepdims=True)
            L = tf.einsum('ijk,kb...->ijb...', TP, L) * tf.cast(shift_size**k, float_type)
    if k > d - 3:
        R = tf.ones([1], dtype=float_type)
    else:
        R = tf.reduce_sum(wk*parts[len(parts) - 1], axis=1, keepdims=True)
        for i in range(len(parts) - 1)[k+2:][::-1]:
            TP = tf.reduce_sum(wk*QR(parts[i], I[i + 1], I[i]), axis=1, keepdims=True)
            R = tf.einsum('ijk,kb...->ijb...', TP, R)
        Tk = parts[k+1]
        size = tf.shape(Tk)
        I_plus = I[k+2]
        I_k = I[k+1]
        Pk_inxs_right = I_plus[:, -1]
        Pk_inxs_left = I_plus[:, :-1]
        Tk_changed = tf.reshape(Tk, [size[0] * size[1], size[2]])
        first_inx = tf.cast(
            tf.where(tf.reduce_all(tf.expand_dims(I_k, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2))[:, 1],
            tf.int32)
        Pk_changed_inxs = (first_inx * size[1] + Pk_inxs_right)
        P = tf.gather(Tk_changed, Pk_changed_inxs)
        P_inv = tf.linalg.inv(P)
        R = tf.einsum('ij,jb...->ib...', P_inv, R) * tf.cast(shift_size**(d-k-2), float_type)
    L = tf.reshape(L, [-1])
    R = tf.reshape(R, [-1])
    matrix = tf.abs(tf.einsum('i,j->ij', L, R))
    return tf.expand_dims(tf.expand_dims(matrix, 1), 1)


@tf.function(experimental_relax_shapes=True)
def QR(Tk, I_plus, I):
    size = tf.shape(Tk)
    Pk_inxs_right = I_plus[:, -1]
    Pk_inxs_left = I_plus[:, :-1]
    Tk_changed = tf.reshape(Tk, [size[0] * size[1], size[2]])
    full_inxs = tf.transpose([tf.range(size[0] * size[1])])
    first_inx = tf.cast(tf.where(tf.reduce_all(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2))[:, 1],
                        tf.int32)
    Pk_changed_inxs = (first_inx * size[1] + Pk_inxs_right)
    all_but_Pk = tf.reshape(tf.where(tf.logical_not(tf.reduce_any(full_inxs == Pk_changed_inxs, axis=1))), [-1])
    full_Q = tf.linalg.qr(Tk_changed)[0]
    Q = tf.gather(full_Q, Pk_changed_inxs)
    Q_tag = tf.gather(full_Q, all_but_Pk)
    Q_inv = tf.linalg.inv(Q)
    bottom = tf.matmul(Q_tag, Q_inv)
    top = tf.eye(tf.shape(Pk_inxs_right)[0], dtype=float_type)
    r = tf.scatter_nd(tf.transpose([all_but_Pk]), bottom, [size[0] * size[1], size[2]])
    r = tf.tensor_scatter_nd_add(r, tf.transpose([Pk_changed_inxs]), top)
    return tf.reshape(r, tf.shape(Tk))


# @tf.function(experimental_relax_shapes=True)
def one_step(part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k, pivot):
    piv_left = pivot[:, :k + 1]
    piv_right = pivot[:, k + 1:]
    J_k_update = tf.concat([Jk, piv_right], axis=0)
    I_k_plus_update = tf.concat([Ik_plus, piv_left], axis=0)
    pparts, pparts_plus = add_pivot(part_k, part_k_plus, Ik, Jk_plus, k, pivot)
    pivots = tf.concat([pivots_k, pivot], axis=0)
    return pparts, pparts_plus, J_k_update, I_k_plus_update, pivots


def run(A, integral, A_shape, initial_pivots, half=False, graph=False):
    A_int = integral()
    I = []
    J = []
    for i in range(len(A_shape)):
        I.append(initial_pivots[0:1, :i])
        J.append(initial_pivots[0:1, i + 1:])
    parts = inital_Tk(A, I, J, A_shape)
    err_max_list = [1.]
    err_int_list = []
    mean_chi_list = []
    pivots = [initial_pivots] * (len(A_shape) - 1)
    #ev = tf.abs((evaluate_full(parts, I) - A) / A)
    # max_err = test(A, parts, I, test_n)#tf.reduce_max(ev)
    #err_max_list.append(max_err)
    err_int_list.append(tf.abs(A_int-evaluate_integral(parts, I)))#/(tf.abs(A_int)+err_wanted))
    # err_int_list.append(evaluate_integral(parts, I))
    all_pivots = tf.concat(pivots, axis=0)
    mean_chi_list.append(tf.shape(all_pivots)[0] / (tf.shape(all_pivots)[1] - 1))
    sweeps = 0
    try:
        while err_max_list[-1] > err_wanted:
            # print(parts)
            max_err_i = 0
            # print(I)
            for k in (list(range(len(A_shape) - 1)) + list(range(len(A_shape) - 1)[::-1])):
                part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k = parts[k], parts[k + 1], I[k], I[k + 1], J[k], \
                                                                             J[k + 1], pivots[k], k
                pivot, max_err_pi = find_pivot(A, parts, I, part_k, part_k_plus, Ik, Ik_plus, Jk_plus, k, half)
                if half:
                    for halfi in range(5):
                        test_pivot, test_max_err_pi = find_pivot(A, parts, I, part_k, part_k_plus, Ik, Ik_plus, Jk_plus, k, half)
                        if test_max_err_pi > max_err_pi:
                            pivot = test_pivot
                            max_err_pi = test_max_err_pi
                if max_err_pi > max_err_i:
                    max_err_i = max_err_pi
                if max_err_pi > err_wanted:
                    pparts, pparts_plus, J_k_update, I_k_plus_update, pivots_new = one_step(part_k, part_k_plus, Ik, Ik_plus,
                                                                                        Jk,
                                                                                        Jk_plus, pivots_k, k, pivot)
                    add_condJ = tf.logical_not(tf.reduce_any(
                        tf.reduce_all(tf.expand_dims(Jk, axis=0) == tf.expand_dims(J_k_update[-1:], axis=1), axis=2),
                        axis=1))
                    add_condI = tf.logical_not(tf.reduce_any(
                        tf.reduce_all(tf.expand_dims(Ik_plus, axis=0) == tf.expand_dims(I_k_plus_update[-1:], axis=1),
                                      axis=2),
                        axis=1))
                    if add_condJ and add_condI: #just in case
                        parts[k] = pparts
                        parts[k + 1] = pparts_plus
                        J[k] = J_k_update
                        I[k + 1] = I_k_plus_update
                        pivots[k] = pivots_new
            max_err = max_err_i

            if graph:
                # ev = tf.abs((evaluate_full(parts, I) - A) / A)
                # max_err = test(A, parts, I, test_n)#tf.reduce_max(ev)
                err_max_list.append(max_err.numpy())
                err_int_list.append(tf.abs(A_int - evaluate_integral(parts, I)))# / (tf.abs(A_int)+err_wanted).numpy())
                # err_int_list.append(evaluate_integral(parts, I))
                all_pivots = tf.concat(pivots, axis=0)
                mean_chi_list.append(tf.shape(all_pivots)[0] / (tf.shape(all_pivots)[1] - 1))
            sweeps += 1
    except KeyboardInterrupt:
        pass
    max_err = test(A, parts, I, test_n)
    # err_max_list.append(max_err)
    return err_max_list[1:], mean_chi_list[:-1], err_int_list[:-1], max_err



@tf.function(experimental_relax_shapes=True)
def no_shift(indices):
    return tf.cast(indices, float_type)

@tf.function(experimental_relax_shapes=True)
def quad_weights():
    return tf.cast((xmax - xmin) / (sd-1), float_type) * tf.expand_dims(tf.expand_dims(tf.concat([[0.5], tf.ones(sd-2, float_type), [0.5]], axis=0), axis=0), axis=2), tf.cast(1., float_type)

@tf.function(experimental_relax_shapes=True)
def uniform_shift(indices, xmin=xmin, xmax=xmax):
    return (indices / (sd - 1)) * (xmax - xmin) + xmin

@tf.function(experimental_relax_shapes=True)
def tanh_sinh_shift(indices, h=h, xmin=xmin, xmax=xmax):
    k = tf.cast(indices - sd//2, float_type)
    xk = tf.math.tanh(0.5*np.pi*tf.math.sinh(k*h))
    xk = (xk + 1.) / 2. #scale to [0,1]
    xk = xk * (xmax-xmin) + xmin #scale to [xmin,xmax]
    return xk

@tf.function(experimental_relax_shapes=True)
def tanh_sinh_weights(h=h):
    k = tf.cast(tf.range(sd) - sd//2, float_type)
    # tf.print(tf.math.tanh(0.5*np.pi*tf.math.sinh(k*h)))
    wk = 0.5*h*np.pi*tf.math.cosh(k*h) / (tf.math.cosh(0.5*np.pi*tf.math.sinh(k*h)))**2.
    return tf.expand_dims(tf.expand_dims(wk, axis=0), axis=2), (xmax-xmin)/2.

n_gaus = 2
@tf.function(experimental_relax_shapes=True)
def gaussian(x, n=n_gaus, sigma=0.05):
    result = 0.
    for i in range(n):
        result += tf.math.reduce_prod(tf.exp(-0.5 * (x - (i+1.)/(n+1)) ** 2 / sigma ** 2) / (10.*sigma * (2. * np.pi) ** 0.5), axis=1, keepdims=True)
    return result

@tf.function(experimental_relax_shapes=True)
def gaussian_integral(n=n_gaus):
    return tf.cast(n, float_type) / 10.

@tf.function(experimental_relax_shapes=True)
def sinus(x):
    return tf.sin(tf.reduce_sum(x, axis=1, keepdims=True))

@tf.function(experimental_relax_shapes=True)
def sinus_integral(d=d):
    return tf.math.imag(((tf.exp(1.j)-1)/1.j)**d)

@tf.function(experimental_relax_shapes=True, jit_compile=True)
def hilbert_function(x):
    return 1./tf.reduce_sum(x+1., axis=1, keepdims=True)

A = gaussian
shift = uniform_shift
weights = quad_weights
integral = gaussian_integral

if args.test:
    initial_pivots = tf.zeros(tf.shape([A_shape]), dtype=tf.int32)  # so it would be deterministic
    # tracing...
    err_max, chi_mean, int_list, max_err = run(A, integral, A_shape, initial_pivots, half=half,
                                               graph=True)  # fixme for some reason doesn't work with graph=False
else:
    initial_pivots = tf.random.uniform([1, d], maxval=A_shape[0], dtype=tf.int32)


print("starting...")
t = time.time()
err_max, chi_mean, int_list, max_err = run(A, integral, A_shape, initial_pivots, half=half,
                                 graph=True)  # fixme for some reason doesn't work with graph=False
runtime = time.time() - t
print(runtime)

if args.test:
    f = open('results.pkl', 'wb')
    pkl.dump(runtime, f)
print(max_err)
print(int_list[-1])

if args.graph:
    plt.semilogy(chi_mean, err_max)
    plt.xlabel("Average bond dimension")
    plt.ylabel("Max relative error")
    plt.savefig("chi.pdf")
    plt.clf()

    plt.semilogy(range(len(err_max)), err_max)
    plt.xlabel("Sweeps")
    plt.ylabel("Max relative error")
    plt.savefig("sweeps.pdf")
    plt.clf()

    # print(int_list)
    plt.semilogy(chi_mean, int_list)
    plt.xlabel("Average bond dimension")
    plt.ylabel("Integral relative error")
    plt.savefig("int.pdf")
    plt.clf()
