import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

modes = ['tensor', 'function', 'function_quantics', 'tensor_quantics']
float_type = tf.float64
int_type = tf.int64


@tf.function(experimental_relax_shapes=True, jit_compile=True)
def quantics_to_cartesian(coordinates, x_min, x_max, dim, R, float_type):
     r = tf.cast(tf.reduce_sum(
        tf.reshape(coordinates, [tf.shape(coordinates, out_type=int_type)[0], R, dim]) / (2 ** tf.expand_dims(tf.range(1, R + 1), 1)),
        axis=1) * (x_max - x_min) + x_min, float_type)
     return r




# @tf.function(experimental_relax_shapes=True, jit_compile=True)
def cartesian_to_quantics(coordinates, dim, R):
    binary = []
    for i in range(dim):
        binary.append([tf.math.mod(tf.bitwise.right_shift(coordinates[:, i:i+1], tf.range(R, dtype=int_type)), 2)[:, ::-1]])
    binary = tf.transpose(tf.concat(binary, axis=0), [1,0,2])
    return tf.reshape(tf.transpose(binary, [0,2,1]), [tf.shape(coordinates, out_type=int_type)[0], dim*R])

@tf.function(experimental_relax_shapes=True)
def uniform_shift(indices, x_min, x_max, R):
    return (indices / (2 ** R - 1)) * (x_max - x_min) + x_min  # fixme do we want the minus 1 be there?


# @tf.function(experimental_relax_shapes=True, jit_compile=True)
def evaluate(A, indices, mode, x_min=None, x_max=None, dim=None, R=None, float_type=None):
    size = tf.shape(indices, out_type=int_type)[-1]
    final_shape = tf.shape(indices, out_type=int_type)[:-1]
    flattened = tf.reshape(indices, [tf.size(indices, out_type=int_type) // size, size])
    if mode == modes[0] or mode == modes[3]:
        transformed = flattened
        values = A(transformed)
    if mode == modes[1]:
        transformed = uniform_shift(flattened, 0., 1., R)
        values = A(transformed, x_min, x_max)
    if mode == modes[2]:
        transformed = quantics_to_cartesian(flattened, 0., 1., dim, R, float_type)
        values = A(transformed, x_min, x_max)
    return tf.reshape(values, final_shape)

@tf.function(experimental_relax_shapes=True)
def evaluate_points(parts, I, points):
    r = tf.gather(tf.squeeze(parts[len(parts) - 1], axis=2), points[:, -1], axis=1)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = tf.gather(QR(parts[i], I[i + 1], I[i]), points[:, i], axis=1)
        r = tf.einsum('ijk,kj->ij', TP, r)
    TP = tf.gather(QR(parts[0], I[1], I[0]), points[:, 0], axis=1)
    r = tf.einsum('ijk,kj->ij', TP, r)
    return tf.transpose(r)

def initial(mode, d, N=1, R=None, shape=None, type='random'):
    if type == 'random':
        if mode == modes[0]:
            initial_pivots = []
            for i in shape:
                initial_pivots.append(tf.random.uniform([N, 1], maxval=i, dtype=int_type))
            initial_pivots = tf.concat(initial_pivots, axis=1)
        if mode == modes[1]:
            initial_pivots = tf.random.uniform([N, d], maxval=2 ** R, dtype=int_type)
        if mode == modes[2] or mode == modes[3]:
            initial_pivots = tf.random.uniform([N, d], maxval=2, dtype=int_type)
    if type == 'middle': #only one pivot
        if mode == modes[0]:
            initial_pivots = []
            for i in shape:
                initial_pivots.append(tf.random.uniform([1, 1], minval=i//2 ,maxval=i//2+1, dtype=int_type))
            initial_pivots = tf.concat(initial_pivots, axis=1)
        if mode == modes[1]:
            initial_pivots = tf.random.uniform([1, d], minval=2 ** R//2, maxval=2 ** R//2+1, dtype=int_type)
        if mode == modes[2] or mode == modes[3]:
            dim = d//R
            ones = tf.ones([1, dim], dtype=int_type)
            zeros = tf.zeros([1, d-dim], dtype=int_type)
            initial_pivots = tf.concat([ones, zeros], axis=1)
    return initial_pivots


def matrix_k(left, right, A_shape, k):
    if tf.shape(left, out_type=int_type)[1] == 0:
        left = left[0:1, :]
    if tf.shape(right, out_type=int_type)[1] == 0:
        right = right[0:1, :]
    l_s = tf.shape(left, out_type=int_type)
    r_s = tf.shape(right, out_type=int_type)
    all_k = tf.transpose([tf.range(A_shape[k])])
    first = tf.tile(right, [A_shape[k], 1])
    second = tf.concat([tf.repeat(all_k, r_s[0], axis=0), first], axis=1)
    batched_Tk = tf.concat([tf.repeat(left, r_s[0] * A_shape[k], axis=0), tf.tile(second, [l_s[0], 1])], axis=1)
    Tk = tf.reshape(batched_Tk, [l_s[0], A_shape[k], r_s[0], tf.size(A_shape, out_type=int_type)])
    return Tk


def inital_Tk(A, I, J, A_shape, mode, x_min=None, x_max=None, dim=None, R=None, float_type=None):
    parts = []
    for k in range(len(A_shape)):
        left = I[k]
        right = J[k]
        Tk = matrix_k(left, right, A_shape, k)
        parts.append(evaluate(A, Tk, mode, x_min, x_max, dim, R, float_type))
    return parts



@tf.function(experimental_relax_shapes=True)
def half_pi_alpha_exact(I_used, J_used, all_k, all_k_plus, A_shape):
    first = tf.concat([tf.repeat(all_k_plus, tf.shape(J_used, out_type=int_type)[0], axis=0), tf.tile(J_used, [tf.size(all_k_plus, out_type=int_type), 1])],
                      axis=1)
    second = tf.concat([tf.repeat(all_k, tf.shape(first, out_type=int_type)[0], axis=0), tf.tile(first, [tf.size(all_k), 1])], axis=1)
    batched_PIk = tf.concat([tf.repeat(I_used, tf.shape(second, out_type=int_type)[0], axis=0), tf.tile(second, [tf.shape(I_used, out_type=int_type)[0], 1])],
                            axis=1)
    PIk = tf.reshape(batched_PIk,
                     [tf.shape(I_used, out_type=int_type)[0], tf.size(all_k, out_type=int_type), tf.size(all_k_plus, out_type=int_type), tf.shape(J_used, out_type=int_type)[0], tf.size(A_shape, out_type=int_type)])
    return PIk

@tf.function(experimental_relax_shapes=True)
def full_pi_alpha_exact(I, J_plus, A_shape, k):
    all_k = tf.transpose([tf.range(A_shape[k])])
    all_k_plus = tf.transpose([tf.range(A_shape[k+1])])
    first = tf.concat([tf.repeat(all_k_plus, tf.shape(J_plus, out_type=int_type)[0], axis=0), tf.tile(J_plus, [A_shape[k+1], 1])],
                      axis=1)
    second = tf.concat([tf.repeat(all_k, tf.shape(first, out_type=int_type)[0], axis=0), tf.tile(first, [A_shape[k], 1])], axis=1)
    # second = tf.concat([tf.tile(all_k, [tf.shape(first, out_type=int_type)[0], 1]), tf.repeat(first, A_shape[k], axis=0)], axis=1)
    batched_PIk = tf.concat([tf.repeat(I, tf.shape(second, out_type=int_type)[0], axis=0), tf.tile(second, [tf.shape(I, out_type=int_type)[0], 1])],
                            axis=1)
    # tf.print(tf.concat([tf.repeat(all_k, tf.shape(first, out_type=int_type)[0], axis=0), tf.tile(all_k_plus, [A_shape[k], 1])], axis=1))
    # tf.print(tf.concat([tf.tile(all_k, [A_shape[k+1], 1]), tf.repeat(all_k_plus, tf.shape(first, out_type=int_type)[0], axis=0)], axis=1))

    PIk = tf.reshape(batched_PIk,
                     [tf.shape(I, out_type=int_type)[0], A_shape[k], A_shape[k + 1], tf.shape(J_plus, out_type=int_type)[0], tf.size(A_shape, out_type=int_type)])
    return PIk

# @tf.function(experimental_relax_shapes=True)
# def QR(Tk, I_plus, I):
#     size = tf.shape(Tk, out_type=int_type)
#     Pk_inxs_right = I_plus[:, -1]
#     Pk_inxs_left = I_plus[:, :-1]
#     Tk_changed = tf.reshape(Tk, [size[0] * size[1], size[2]])
#     first_inx = tf.cast(tf.where(tf.reduce_all(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2))[:, 1],
#                         int_type)
#     if tf.reduce_all(tf.reduce_all(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2)):#fixme there's a chance that something is VERY wrong
#         first_inx = Pk_inxs_right*0
#
#
#     Pk_changed_inxs = (first_inx * size[1] + Pk_inxs_right)
#
#     r = tf.matmul(Tk_changed, tf.linalg.inv(tf.gather(Tk_changed, Pk_changed_inxs)))
#
#     return tf.reshape(r, tf.shape(Tk, out_type=int_type))

@tf.function()
def QR(Tk, I_plus, I):
    size = tf.shape(Tk, out_type=int_type)
    # if size[2] == 1:
    #     return Tk
    Pk_inxs_right = I_plus[:, -1]
    Pk_inxs_left = I_plus[:, :-1]
    Tk_changed = tf.reshape(Tk, [size[0] * size[1], size[2]])
    full_inxs = tf.transpose([tf.range(size[0] * size[1])])
    first_inx = tf.cast(tf.where(tf.reduce_all(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2))[:, 1],
                        int_type)


    # size = tf.shape(Tk, out_type=int_type)
    # Pk_inxs_right = J[:, 0]
    # Pk_inxs_left = J[:, 1:]
    # Tk_changed = tf.reshape(Tk, [size[0] * size[1], size[2]])
    # full_inxs = tf.transpose([tf.range(size[0] * size[1])])
    # first_inx = tf.cast(tf.where(tf.reduce_all(tf.expand_dims(J_plus, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2))[:, 1],
    #                     int_type)

    # print(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1))
    # print(tf.reduce_all(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2))
    # print()
    if tf.reduce_all(tf.reduce_all(tf.expand_dims(I, 0) == tf.expand_dims(Pk_inxs_left, 1), axis=2)):#fixme there's a chance that something is VERY wrong
        first_inx = Pk_inxs_right*0



    Pk_changed_inxs = (first_inx * size[1] + Pk_inxs_right)
    # Pk_changed_inxs = (Pk_inxs_right * size[1] + first_inx) # bad!!!

    all_but_Pk = tf.reshape(tf.where(tf.logical_not(tf.reduce_any(full_inxs == Pk_changed_inxs, axis=1))), [-1])
    # Tk_changed = tf.ensure_shape(Tk_changed, [size[0] * size[1], size[2]])
    full_Q = tf.linalg.qr(Tk_changed)[0]


    Q = tf.gather(full_Q, Pk_changed_inxs)
    Q_tag = tf.gather(full_Q, all_but_Pk)

    # tf.print(Pk_inxs_right)
    # tf.print(Pk_changed_inxs)
    # tf.print(Tk_changed)
    # tf.print(Q)
    Q_inv = tf.linalg.pinv(Q)
    bottom = tf.matmul(Q_tag, Q_inv)

    top = tf.eye(tf.shape(Pk_inxs_right, out_type=int_type)[0], dtype=float_type)
    r = tf.scatter_nd(tf.transpose([all_but_Pk]), bottom, [size[0] * size[1], size[2]])
    r = tf.tensor_scatter_nd_add(r, tf.transpose([Pk_changed_inxs]), top)
    # r = tf.matmul(Tk_changed, tf.linalg.inv(tf.gather(Tk_changed, Pk_changed_inxs)))

    return tf.reshape(r, tf.shape(Tk, out_type=int_type))

# @tf.function(experimental_relax_shapes=True)
def find_pivot(A, Tk, Tk_plus, I, I_plus, J_plus, half, A_shape, k, mode, x_min, x_max, dim, R, float_type, threshold):
    if half:
        I_used = I
        all_k = tf.transpose([tf.range(A_shape[k])])
        all_k_plus = tf.transpose([tf.range(A_shape[k+1])])
        J_used = J_plus
        if tf.cast(tf.random.uniform([1], maxval=2, dtype=int_type), tf.bool):
            r1 = tf.random.uniform([1], maxval=tf.shape(I_used, out_type=int_type)[0], dtype=int_type)[0]
            I_used = I_used[r1:r1 + 1]
            r2 = tf.random.uniform([1], maxval=tf.shape(all_k, out_type=int_type)[0], dtype=int_type)[0]
            all_k = all_k[r2:r2 + 1]
            PI_approx = tf.einsum('ijk,kb...->ijb...', QR(Tk, I_plus, I)[r1:r1 + 1, r2:r2 + 1, :], Tk_plus)
        else:
            r1 = tf.random.uniform([1], maxval=tf.shape(all_k_plus, out_type=int_type)[0], dtype=int_type)[0]
            all_k_plus = all_k_plus[r1:r1 + 1]
            r2 = tf.random.uniform([1], maxval=tf.shape(J_used, out_type=int_type)[0], dtype=int_type)[0]
            J_used = J_used[r2:r2 + 1]
            PI_approx = tf.einsum('ijk,kb...->ijb...', QR(Tk, I_plus, I), Tk_plus[:, r1:r1 + 1, r2:r2 + 1])

        PI_ind = half_pi_alpha_exact(I_used, J_used, all_k, all_k_plus, A_shape)
        PI = evaluate(A, PI_ind, mode, x_min, x_max, dim, R, float_type)

    else:

        PI_ind = full_pi_alpha_exact(I, J_plus, A_shape, k)
        PI = evaluate(A, PI_ind, mode, x_min, x_max, dim, R, float_type)
        PI_approx = tf.einsum('ijk,kb...->ijb...', QR(Tk, I_plus, I), Tk_plus)

    # tf.print(PI_approx)
    # tf.print(PI)
    diff = tf.abs(PI - PI_approx)/ (threshold + tf.reduce_max(tf.abs(PI))) #fixme do we want this normalization?
    # print(diff)
    # tf.print((tf.reduce_max(tf.abs(PI_approx)+ tf.abs(PI)) + threshold))
    # tf.print(diff)
    # shift_diff = diff - tf.reduce_max(diff)
    new_pivot = tf.gather_nd(PI_ind, tf.where(diff == tf.reduce_max(diff)))
    # tf.print(PI_ind)
    # tf.print(new_pivot)
    # tf.print(tf.reduce_max(diff))
    # choice = tf.random.uniform([1], maxval=tf.shape(new_pivot)[0], dtype=tf.int32)[0]
    return new_pivot[0:1, :], tf.reduce_max(diff)

# @tf.function(experimental_relax_shapes=True) #fixme make sure all tile and repeat are correct
def add_pivot(A, Tk, Tk_plus, I, J_plus, k, pivot, A_shape, mode, x_min, x_max, dim, R, float_type):
    piv_left = pivot[:, :k + 1]
    piv_right = pivot[:, k + 1:]
    new_Tk = tf.scatter_nd(tf.cast(tf.where(tf.ones_like(Tk)), int_type), tf.reshape(Tk, [-1]),
                           tf.cast(tf.shape(Tk, out_type=int_type) + tf.constant([0, 0, 1], int_type), int_type))
    first_k = tf.tile(piv_right, [A_shape[k], 1])
    second_k = tf.concat([tf.transpose([tf.range(A_shape[k])]), first_k], axis=1)
    new_Tk_part = tf.concat([tf.repeat(I, A_shape[k], axis=0), tf.tile(second_k, [tf.shape(I)[0], 1])], axis=1)
    new_Tk_part_values = evaluate(A, new_Tk_part, mode, x_min, x_max, dim, R, float_type)
    ##
    left_ind = tf.transpose([tf.repeat(tf.range(tf.shape(Tk, out_type=int_type)[0]), A_shape[k])])
    mid_ind = tf.transpose([tf.tile(tf.range(A_shape[k]), [tf.shape(Tk, out_type=int_type)[0]])])
    right_ind = tf.ones([A_shape[k]*tf.shape(Tk, out_type=int_type)[0], 1], dtype=int_type)*(tf.shape(Tk, out_type=int_type)[2])
    indices_to_update = tf.concat([left_ind, mid_ind, right_ind], axis=1)
    ##
    # indices_to_update = tf.boolean_mask(tf.cast(tf.where(tf.ones_like(new_Tk)), int_type),
    #                                     tf.logical_not(tf.reduce_any(tf.reduce_all(
    #                                         tf.expand_dims(tf.where(tf.ones_like(new_Tk)), 0) == tf.expand_dims(
    #                                             tf.where(tf.ones_like(Tk)), 1), axis=2), axis=0)))
    new_Tk = tf.tensor_scatter_nd_update(new_Tk, indices_to_update, new_Tk_part_values)

    new_Tk_plus = tf.scatter_nd(tf.cast(tf.where(tf.ones_like(Tk_plus)), int_type), tf.reshape(Tk_plus, [-1]),
                                tf.cast(tf.shape(Tk_plus, out_type=int_type) + tf.constant([1, 0, 0], int_type), int_type))
    first_k_plus = tf.tile(J_plus, [A_shape[k+1], 1])
    second_k_plus = tf.concat(
        [tf.repeat(tf.transpose([tf.range(A_shape[k+1])]), tf.shape(J_plus, out_type=int_type)[0], axis=0), first_k_plus], axis=1)
    new_Tk_part_plus = tf.concat([tf.repeat(piv_left, A_shape[k+1] * tf.shape(J_plus, out_type=int_type)[0], axis=0), second_k_plus],
                                 axis=1)
    new_Tk_part_values_plus = evaluate(A, new_Tk_part_plus, mode, x_min, x_max, dim, R, float_type)
    ##
    left_ind_plus = tf.ones([A_shape[k+1]*tf.shape(Tk_plus, out_type=int_type)[2], 1], dtype=int_type)*tf.shape(Tk_plus, out_type=int_type)[0]
    mid_ind_plus = tf.transpose([tf.repeat(tf.range(A_shape[k+1]), tf.shape(Tk_plus, out_type=int_type)[2])])
    right_ind_plus = tf.transpose([tf.tile(tf.range(tf.shape(Tk_plus, out_type=int_type)[2]), [A_shape[k+1]])])
    indices_to_update_plus = tf.concat([left_ind_plus, mid_ind_plus, right_ind_plus], axis=1)
    ##
    # indices_to_update_plus = tf.boolean_mask(tf.cast(tf.where(tf.ones_like(new_Tk_plus)), int_type),
    #                                          tf.logical_not(tf.reduce_any(tf.reduce_all(
    #                                              tf.expand_dims(tf.where(tf.ones_like(new_Tk_plus)),
    #                                                             0) == tf.expand_dims(tf.where(tf.ones_like(Tk_plus)),
    #                                                                                  1), axis=2), axis=0)))
    # tf.print(tf.sort(tf.abs(new_Tk_part_values_plus)))
    new_Tk_plus = tf.tensor_scatter_nd_update(new_Tk_plus, indices_to_update_plus, new_Tk_part_values_plus)
    return new_Tk, new_Tk_plus


# @tf.function(experimental_relax_shapes=True)
def one_step(A, part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k, pivot, A_shape, mode, x_min, x_max, dim, R, float_type):
    piv_left = pivot[:, :k + 1]
    piv_right = pivot[:, k + 1:]
    J_k_update = tf.concat([Jk, piv_right], axis=0)
    I_k_plus_update = tf.concat([Ik_plus, piv_left], axis=0)
    pparts, pparts_plus = add_pivot(A, part_k, part_k_plus, Ik, Jk_plus, k, pivot, A_shape, mode, x_min, x_max, dim, R, float_type)
    pivots = tf.concat([pivots_k, pivot], axis=0)
    return pparts, pparts_plus, J_k_update, I_k_plus_update, pivots

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
def evaluate_full_mps(parts):
    r = tf.squeeze(parts[len(parts) - 1], axis=2)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = parts[i]
        r = tf.einsum('ijk,kb...->ijb...', TP, r)
    TP = tf.squeeze(parts[0], axis=0)
    r = tf.einsum('ij,jb...->ib...', TP, r)
    return r

# @tf.function(experimental_relax_shapes=True)
# def evaluate_integral(parts, I):
#     r = tf.reduce_sum(tf.squeeze(parts[len(parts) - 1], axis=2), axis=1, keepdims=True)
#     for i in range(len(parts) - 1)[::-1][:-1]:
#         TP = tf.reduce_sum(QR(parts[i], I[i + 1], I[i]), axis=1, keepdims=True)
#         r = tf.einsum('ijk,kb...->ijb...', TP, r)
#     TP = tf.reduce_sum(tf.squeeze(QR(parts[0], I[1], I[0]), axis=0), axis=0, keepdims=True)
#     r = tf.einsum('ij,jb...->ib...', TP, r)
#     return tf.reshape(r, [-1]) * tf.cast((x)**d, float_type)

@tf.function(experimental_relax_shapes=True)
def evaluate_integral(parts, I, mode, R, x_min, x_max, d):
    if mode == modes[0]:
        wk = 1.
        shift_size = 1.
    if mode == modes[1]:
        wk = 1.#tf.cast(1. / tf.cast(2 ** R - 1, float_type), float_type) * tf.expand_dims(
            #tf.expand_dims(tf.concat([[0.5], tf.ones(2 ** R - 2, float_type), [0.5]], axis=0), axis=0), axis=2)
        shift_size = 1.
    if mode == modes[2]:
        wk = 1.
        shift_size = 0.5# * (x_max - x_min) ** (1./float(R))
    if mode == modes[3]:
        wk = 1.
        shift_size = 0.5
    r = tf.reduce_sum(tf.squeeze(wk*parts[len(parts) - 1], axis=2), axis=1, keepdims=True)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = tf.reduce_sum(wk*QR(parts[i], I[i + 1], I[i]), axis=1, keepdims=True)
        r = tf.einsum('ijk,kb...->ijb...', TP, r)[:, 0]
    TP = tf.reduce_sum(tf.squeeze(wk*QR(parts[0], I[1], I[0]), axis=0), axis=0, keepdims=True)
    r = tf.einsum('ij,jb...->ib...', TP, r)
    vol = tf.cast((x_max-x_min)**(d/R), float_type)
    return vol * tf.reshape(r, [-1]) * tf.cast(tf.pow(tf.cast(shift_size, float_type), tf.cast(d, float_type)), float_type)

@tf.function(experimental_relax_shapes=True)
def evaluate_mps_integral(parts, mode, R, x_min, x_max, d):
    if mode == modes[0]:
        wk = 1.
        shift_size = 1.
    if mode == modes[1]:
        wk = 1.#tf.cast(1. / tf.cast(2 ** R - 1, float_type), float_type) * tf.expand_dims(
            #tf.expand_dims(tf.concat([[0.5], tf.ones(2 ** R - 2, float_type), [0.5]], axis=0), axis=0), axis=2)
        shift_size = 1.
    if mode == modes[2]:
        wk = 1.
        shift_size = 0.5# * (x_max - x_min) ** (1./float(R))
    if mode == modes[3]:
        wk = 1.
        shift_size = 0.5
    r = tf.reduce_sum(tf.squeeze(wk*parts[len(parts) - 1], axis=2), axis=1, keepdims=True)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = tf.reduce_sum(wk*parts[i], axis=1, keepdims=True)
        r = tf.einsum('ijk,kb...->ijb...', TP, r)[:, 0]
    TP = tf.reduce_sum(tf.squeeze(wk*parts[0], axis=0), axis=0, keepdims=True)
    r = tf.einsum('ij,jb...->ib...', TP, r)
    vol = tf.cast((x_max-x_min)**(d/R), float_type)
    return vol* tf.reshape(r, [-1]) * tf.cast(tf.pow(tf.cast(shift_size, float_type), tf.cast(d, float_type)), float_type)


@tf.function(experimental_relax_shapes=True)
def create_mps(parts, I, const=1.):
    mps = []
    for i in range(len(parts) - 1):
        mps.append(QR(parts[i]*const, I[i + 1], I[i]))
    mps.append(parts[-1]*const)
    return mps



@tf.function(experimental_relax_shapes=True, jit_compile=True)
def mps_multi(M1, M2):
    M12 = []
    for i in range(len(M1)):
        M12_i = []
        i1 = M1[i]
        i2 = M2[i]
        shape_1 = tf.shape(i1)
        shape_2 = tf.shape(i2)
        M12_i.append(tf.repeat(tf.repeat(i1, shape_2[2], axis=2), shape_2[0], axis=0) * tf.tile(i2, [shape_1[0], 1, shape_1[2]]))
        M12.append(tf.concat(M12_i, axis=0))
    return M12


@tf.function(experimental_relax_shapes=True, jit_compile=True)
def mps_add(M1, M2):
    M12 = []
    M12.append(tf.concat([M1[0], M2[0]], axis=2))
    for i in range(1, len(M1)-1):
        i1 = M1[i]
        i2 = M2[i]
        shape_1 = tf.shape(i1)
        shape_2 = tf.shape(i2)
        b11 = tf.zeros(shape_1, dtype=float_type)
        b22 = tf.zeros(shape_2, dtype=float_type)
        b12 = tf.zeros([shape_1[0], shape_1[1], shape_2[2]], dtype=float_type)
        b21 = tf.zeros([shape_2[0], shape_1[1], shape_1[2]], dtype=float_type)
        first = tf.concat([i1, b12], axis=2)
        first = tf.concat([first, tf.concat([b21, b22], axis=2)], axis=0)
        second = tf.concat([b11, b12], axis=2)
        second = tf.concat([second, tf.concat([b21, i2], axis=2)], axis=0)
        M12.append(first+second)
    M12.append(tf.concat([M1[-1], M2[-1]], axis=0))
    return M12


@tf.function(experimental_relax_shapes=True, jit_compile=True)
def create_mpo(mpo_mps):
    result = []
    for i in range(len(mpo_mps)//2):
        result.append(tf.einsum('abc,cjk->abjk', mpo_mps[i*2], mpo_mps[i*2+1]))
    return result

@tf.function(experimental_relax_shapes=True, jit_compile=True)
def tt_pow(M1, pow):
    M2 = []
    for i in range(len(M1)):
        M2.append(tf.math.pow(M1[i], pow))
    return M2


@tf.function(experimental_relax_shapes=True, jit_compile=True)
def mps_mpo(M1, M12, mode, R, x_min, x_max, d):
    result = []
    for i in range(len(M1)):
        m1 = M1[i]
        m12_1 = M12[2*i]
        m12_2 = M12[2*i+1]
        ri = tf.einsum('abc,cdj->abdj', m12_1, m12_2)
        ri = tf.einsum('abdj,idk->aibjk', ri, m1)
        s = tf.shape(ri)
        ri = tf.reshape(ri, [s[0]*s[1], s[2], s[3]*s[4]])
        result.append(ri)
    return result #* tf.cast(shift_size**d, float_type)

@tf.function(experimental_relax_shapes=True, jit_compile=True)
def mps_mpo_mps_full_int(mps1, mps2, mpo, R, x_min, x_max, d):
    r0 = mpo[0]
    r1 = mpo[1]
    r01 = tf.einsum('abcd,dijk->abcijk', r0, r1)
    mps10 = mps1[0]
    mps11 = mps1[1]
    mps101 = tf.einsum('abc,cij->abij', mps10, mps11)

    mps20 = mps2[0]
    mps21 = mps2[1]
    mps201 = tf.einsum('abc,cij->abij', mps20, mps21)

    initial_chain = tf.einsum('aijklb,ajlc->aikcb', r01, mps101)
    initial_chain = tf.einsum('aikbc,aikd->adbc', initial_chain, mps201)

    for i in range(2, len(mps1)):
        ri = mpo[i]
        mps1i = mps1[i]
        mps2i = mps2[i]

        initial_chain = tf.einsum('aijk,knml->aijnml', initial_chain, ri)
        initial_chain = tf.einsum('aijnml,jmk->aiknl', initial_chain, mps1i)
        initial_chain = tf.einsum('aiknl,inj->ajkl', initial_chain, mps2i)
    vol = tf.cast((x_max-x_min)**(2*d/R), float_type)
    return vol * tf.squeeze(initial_chain) * tf.cast(tf.pow(tf.cast(0.5, float_type), tf.cast(2*d, float_type)), float_type)

@tf.function(experimental_relax_shapes=True, jit_compile=True)
def mpo_shrink_to_k(mpo, k, direction):
    r0 = mpo[0]
    r1 = mpo[1]
    r01 = tf.einsum('abcd,dijk->abcijk', r0, r1)
    for i in range(1, len(mpo[1:])):
        ri = tf.einsum('...ib,bjc->...ijc', ri, mpo[i])
    return


class tensor_train:
    def __init__(self, I, J, pivots, parts, dim, R, d, x_min, x_max, shape, mode, dtype):
        self.I = I
        self.J = J
        self.pivots = pivots
        self.parts = parts
        self.dim = dim
        self.R = R
        self.d = d
        self.x_min = x_min
        self.x_max = x_max
        self.shape = shape
        self.mode = mode
        self.dtype = dtype

    def integral(self):
        return evaluate_integral(self.parts, self.I, self.mode, self.R, self.x_min, self.x_max, self.d)

    def scalar(self, const):
        new_parts = []
        for i in self.parts:
            new_parts.append(i*const)
        return tensor_train(self.I, self.J, self.pivots, new_parts, self.dim, self.R, self.d, self.x_min, self.x_max, self.shape, self.mode, self.dtype)

    def mps(self, const=1.):
        return mps(create_mps(self.parts, self.I, const), self.dim, self.R, self.d, self.x_min, self.x_max, self.shape, self.mode, self.dtype)

    def __pow__(self, power):
        return tensor_train(self.I, self.J, self.pivots, tt_pow(self.parts, power), self.dim, self.R, self.d, self.x_min, self.x_max, self.shape, self.mode, self.dtype)

    def operation(self, operation):
        new_parts = []
        for i in self.parts:
            new_parts.append(operation(i))
        return tensor_train(self.I, self.J, self.pivots, new_parts, self.dim, self.R, self.d, self.x_min, self.x_max, self.shape, self.mode, self.dtype)

    def assign_part(self, part, k):
        self.parts[k] = part

    def full(self):
        return evaluate_full(self.parts, self.I)

class mps:
    def __init__(self, parts, dim, R, d, x_min, x_max, shape, mode, dtype):
        self.parts = parts
        self.dim = dim
        self.R = R
        self.d = d
        self.x_min = x_min
        self.x_max = x_max
        self.shape = shape
        self.mode = mode
        self.dtype = dtype

    def integral(self):
        return evaluate_mps_integral(self.parts, self.mode, self.R, self.x_min, self.x_max, self.d)

    def __mul__(self, other):
        assert self.dim in [other.dim, 2*other.dim] , 'dimensions not compatible'
        assert self.R == other.R, 'R must be the same'
        if self.dim == other.dim:
            return mps(mps_multi(self.parts, other.parts), self.dim, self.R, self.d, self.x_min, self.x_max, self.shape, self.mode, self.dtype)
        elif self.dim == 2*other.dim:
            # mpo = create_mpo(self.parts)
            r = mps_mpo(other.parts, self.parts, other.mode, other.R, other.x_min, other.x_max, other.d)
            # r = mps_multi(mpo, other.parts)
            return mps(r, other.dim, other.R, other.d, other.x_min, other.x_max, other.shape, other.mode, other.dtype)

    def __add__(self, other):
        assert self.dim == other.dim, 'dimensions must be the same'
        assert self.R == other.R, 'R must be the same'
        return mps(mps_add(self.parts, other.parts), self.dim, self.R, self.d, self.x_min, self.x_max, self.shape, self.mode, self.dtype)

    def __pow__(self, power):
        assert type(power==int), 'only natural powers are supported'
        r = self
        for i in range(power-1):
            r = r*self
        return r

    def full(self):
        return evaluate_full_mps(self.parts)
