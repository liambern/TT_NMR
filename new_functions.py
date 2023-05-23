import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def evaluate(A, indices):
    size = tf.shape(indices)[-1]
    final_shape = tf.shape(indices)[:-1]
    flattened = tf.reshape(indices, [tf.size(indices) // size, size])
    values = tf.gather_nd(A, flattened)
    return tf.reshape(values, final_shape)

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

def inital_Tk(A, I, J, A_shape):
    parts = []
    for k in range(len(A_shape)):
        left = I[k]
        right = J[k]
        Tk = matrix_k(left, right, A_shape, k)
        parts.append(evaluate(A, Tk))
    return parts


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
    top = tf.eye(tf.shape(Pk_inxs_right)[0], dtype=tf.float32)
    r = tf.scatter_nd(tf.transpose([all_but_Pk]), bottom, [size[0] * size[1], size[2]])
    r = tf.tensor_scatter_nd_add(r, tf.transpose([Pk_changed_inxs]), top)
    return tf.reshape(r, tf.shape(Tk))

def half_pi_alpha_exact(I_used, J_used, all_k, all_k_plus, A_shape):
    first = tf.concat([tf.repeat(all_k_plus, tf.shape(J_used)[0], axis=0), tf.tile(J_used, [tf.size(all_k_plus), 1])],
                      axis=1)
    second = tf.concat([tf.repeat(all_k, tf.shape(first)[0], axis=0), tf.tile(first, [tf.size(all_k), 1])], axis=1)
    batched_PIk = tf.concat([tf.repeat(I_used, tf.shape(second)[0], axis=0), tf.tile(second, [tf.shape(I_used)[0], 1])],
                            axis=1)
    PIk = tf.reshape(batched_PIk,
                     [tf.shape(I_used)[0], tf.size(all_k), tf.size(all_k_plus), tf.shape(J_used)[0], tf.size(A_shape)])
    return PIk

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

def exact_search(A, ind):
    max_err = tf.reduce_max(A)
    shift_diff = (A - tf.reduce_max(A))
    new_pivot = tf.gather_nd(ind, tf.where(shift_diff == 0))
    return max_err, new_pivot[0:1, :]

def find_pivot(A, parts, fullI, Tk, Tk_plus, I, I_plus, J_plus, k, half, err_wanted):
    A_shape = tf.shape(A)
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

def add_pivot(A, Tk, Tk_plus, I, J_plus, k, pivot):
    A_shape = tf.shape(A)
    piv_left = pivot[:, :k + 1]
    piv_right = pivot[:, k + 1:]
    new_Tk = tf.scatter_nd(tf.cast(tf.where(tf.ones_like(Tk)), tf.int32), tf.reshape(Tk, [-1]),
                           tf.cast(tf.shape(Tk) + tf.constant([0, 0, 1]), tf.int32))
    first_k = tf.tile(piv_right, [A_shape[k], 1])
    second_k = tf.concat([tf.transpose([tf.range(A_shape[k])]), first_k], axis=1)
    new_Tk_part = tf.concat([tf.repeat(I, A_shape[k], axis=0), tf.tile(second_k, [tf.shape(I)[0], 1])], axis=1)
    new_Tk_part_values = evaluate(A, new_Tk_part)
    ##
    sd1 = tf.shape(Tk)[1]
    # sd2 = tf.shape(Tk_plus)[1]
    left_ind = tf.transpose([tf.repeat(tf.range(tf.shape(Tk)[0]), sd1)])
    mid_ind = tf.transpose([tf.tile(tf.range(sd1), [tf.shape(Tk)[0]])])
    right_ind = tf.ones([sd1*tf.shape(Tk)[0], 1], dtype=tf.int32)*tf.shape(Tk)[2]
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
    left_ind_plus = tf.ones([sd1*tf.shape(Tk_plus)[2], 1], dtype=tf.int32)*tf.shape(Tk_plus)[0]
    mid_ind_plus = tf.transpose([tf.repeat(tf.range(sd1), tf.shape(Tk_plus)[2])])
    right_ind_plus = tf.transpose([tf.tile(tf.range(tf.shape(Tk_plus)[2]), [sd1])])
    indices_to_update_plus = tf.concat([left_ind_plus, mid_ind_plus, right_ind_plus], axis=1)
    ##
    # indices_to_update_plus = tf.boolean_mask(tf.cast(tf.where(tf.ones_like(new_Tk_plus)), tf.int32),
    #                                          tf.logical_not(tf.reduce_any(tf.reduce_all(
    #                                              tf.expand_dims(tf.where(tf.ones_like(new_Tk_plus)),
    #                                                             0) == tf.expand_dims(tf.where(tf.ones_like(Tk_plus)),
    #                                                                                  1), axis=2), axis=0)))
    new_Tk_plus = tf.tensor_scatter_nd_update(new_Tk_plus, indices_to_update_plus, new_Tk_part_values_plus)
    return new_Tk, new_Tk_plus

def one_step(A, part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k, pivot):
    piv_left = pivot[:, :k + 1]
    piv_right = pivot[:, k + 1:]
    J_k_update = tf.concat([Jk, piv_right], axis=0)
    I_k_plus_update = tf.concat([Ik_plus, piv_left], axis=0)
    pparts, pparts_plus = add_pivot(A, part_k, part_k_plus, Ik, Jk_plus, k, pivot)
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
def evaluate_points(parts, I, points):
    r = tf.gather(tf.squeeze(parts[len(parts) - 1], axis=2), points[:, -1], axis=1)
    for i in range(len(parts) - 1)[::-1][:-1]:
        TP = tf.gather(QR(parts[i], I[i + 1], I[i]), points[:, i], axis=1)
        r = tf.einsum('ijk,kj->ij', TP, r)
    TP = tf.gather(QR(parts[0], I[1], I[0]), points[:, 0], axis=1)
    r = tf.einsum('ijk,kj->ij', TP, r)
    return tf.transpose(r)


@tf.function(experimental_relax_shapes=True)
def test(A, parts, I, n, err_wanted):
    A_shape = tf.shape(A)
    random_points = tf.random.uniform([n, tf.size(A_shape)], maxval=A_shape[0], dtype=tf.int32)
    approx = evaluate_points(parts, I, random_points)
    exact = evaluate(A, random_points)
    err = tf.reduce_max(tf.abs(exact-approx))#/(tf.abs(exact)+err_wanted))
    return err