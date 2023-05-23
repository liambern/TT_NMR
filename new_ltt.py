import numpy as np
import scipy.io
import tensorflow as tf
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
from new_functions import *
import matplotlib.pyplot as plt

err_wanted = 1.e-5
graph = True
half = False

with open("ser_final_bin_real", "rb") as f:
    re_data = np.load(f)
with open("ser_final_bin_im", "rb") as f:
    im_data = np.load(f)

tensor3D = tf.cast(tf.constant(np.reshape(re_data+im_data, (120, 84, 2560)).astype(float)), tf.float32)
tensor2D = tf.cast(tf.constant(np.imag(scipy.io.loadmat("p2dnmr.mat")['data'])+np.real(scipy.io.loadmat("p2dnmr.mat")['data'])), tf.float32)

A = tensor2D
A_shape = A.shape
d = len(A_shape)

initial_pivots = tf.random.uniform([1, d], maxval=A_shape[0], dtype=tf.int32)
I = []
J = []
for i in range(len(A_shape)):
    I.append(initial_pivots[0:1, :i])
    J.append(initial_pivots[0:1, i + 1:])

err_max_list = [1.]
mean_chi_list = []


parts = inital_Tk(A, I, J, A_shape)
pivots = [initial_pivots] * (len(A_shape) - 1)
all_pivots = tf.concat(pivots, axis=0)
#mean_chi_list.append(tf.shape(all_pivots)[0] / (tf.shape(all_pivots)[1] - 1))
sweeps = 0

try:
    while sweeps<50:#while err_max_list[-1] > err_wanted:
        # print(parts)
        max_err_i = 0
        # print(I)
        for k in (list(range(len(A_shape) - 1)) + list(range(len(A_shape) - 1)[::-1])):
            part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k = parts[k], parts[k + 1], I[k], I[k + 1], J[k], \
                                                                         J[k + 1], pivots[k], k
            pivot, max_err_pi = find_pivot(A, parts, I, part_k, part_k_plus, Ik, Ik_plus, Jk_plus, k, half, err_wanted)
            if half:
                for halfi in range(5):
                    test_pivot, test_max_err_pi = find_pivot(A, parts, I, part_k, part_k_plus, Ik, Ik_plus, Jk_plus, k,
                                                             half)
                    if test_max_err_pi > max_err_pi:
                        pivot = test_pivot
                        max_err_pi = test_max_err_pi
            if max_err_pi > max_err_i:
                max_err_i = max_err_pi
            if max_err_pi > err_wanted:
                pparts, pparts_plus, J_k_update, I_k_plus_update, pivots_new = one_step(A, part_k, part_k_plus, Ik,
                                                                                        Ik_plus,
                                                                                        Jk,
                                                                                        Jk_plus, pivots_k, k, pivot)
                add_condJ = tf.logical_not(tf.reduce_any(
                    tf.reduce_all(tf.expand_dims(Jk, axis=0) == tf.expand_dims(J_k_update[-1:], axis=1), axis=2),
                    axis=1))
                add_condI = tf.logical_not(tf.reduce_any(
                    tf.reduce_all(tf.expand_dims(Ik_plus, axis=0) == tf.expand_dims(I_k_plus_update[-1:], axis=1),
                                  axis=2),
                    axis=1))
                if add_condJ and add_condI:  # just in case
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
            # err_int_list.append(evaluate_integral(parts, I))
            all_pivots = tf.concat(pivots, axis=0)
            mean_chi_list.append(tf.shape(all_pivots)[0] / (tf.shape(all_pivots)[1] - 1))
        sweeps += 1
except KeyboardInterrupt:
    pass

err_max, chi_mean, max_err = err_max_list[1:-1], mean_chi_list[:-1], max_err
print(max_err)
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