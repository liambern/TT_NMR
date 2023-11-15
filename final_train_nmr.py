import numpy as np
from functions_nmr import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import argparse
import pickle as pkl

tf.autograph.set_verbosity(1)
float_type = tf.float64
int_type = tf.int64
parser = argparse.ArgumentParser(description='run')
parser.add_argument(
    '-threshold',
    '--threshold',
    type=float,
    help='error threshold',
    required=False,
    default=1.e-10)
parser.add_argument(
    '-half',
    '--half',
    type=int,
    help='half search? 0 or 1',
    required=True)
parser.add_argument(
    '-m',
    '--mode',
    type=str,
    help='tensor or function or function_quantics',
    required=True)
parser.add_argument(
    '-shape',
    '--shape',
    type=str,
    help='shape of tensor if m=tensor: x,y,z',
    required=False)
parser.add_argument(
    '-min',
    '--xmin',
    type=float,
    help='min limit of function def area',
    required=False)
parser.add_argument(
    '-max',
    '--xmax',
    type=float,
    help='max limit of function def area',
    required=False)
parser.add_argument(
    '-d',
    '--dim',
    type=int,
    help='dim of function',
    required=False)
parser.add_argument(
    '-R',
    '--R',
    type=int,
    help='2^R grid points for each dimension',
    required=False)
parser.add_argument(
    '-func',
    '--function',
    type=str,
    help='the name of the function in eval.py to use',
    required=True)
parser.add_argument(
    '-name',
    '--filename',
    type=str,
    help='the name of the function in eval.py to use',
    required=False,
    default='')
args = parser.parse_args()

threshold = args.threshold
half = bool(args.half)
mode = args.mode

modes = ['tensor', 'function', 'function_quantics', 'tensor_quantics']

x_max = None
x_min = None
R = None
B = None
# exec("from eval import "+args.function+" as B")
from nmr import eval_nmr
def B(x):
    return eval_nmr(x, args.filename)

if mode == modes[0] or mode == modes[3]:
    if mode == modes[0]:
        shape_l = (args.shape).split(',')
        shape = []
        for i in shape_l:
            shape.append(int(i))
        shape = tf.constant(shape, int_type)

        A = B
        d = dim = tf.cast(len(shape), int_type)

    elif mode == modes[3]: #integral not implmanted, and you cant have different Rs for different dimensions
        import math
        def closest_power_of_2(number):
            if number <= 1:
                return 0  # 2^0 is 1, so the closest power of 2 under 1 is 2^0

            exponent = math.floor(math.log2(number))
            return exponent


        dim = args.dim
        R = args.R #notice that this R is limited by the smallest dimension of the shape
        shape = [2**R]*dim
        ranges = []
        for i in shape:
            ranges.append(2**tf.expand_dims(tf.range(closest_power_of_2(i)-1, -1, -1)[:R+1], 1))
        ranges = tf.cast(tf.concat(ranges, axis=1), int_type)
        # print(ranges)
        dim = tf.cast(len(shape), int_type)
        d = R*dim


        @tf.function(experimental_relax_shapes=True, jit_compile=True)
        def other_quantics_to_cartesian(coordinates):
            return tf.cast(tf.reduce_sum(
                tf.reshape(coordinates, [tf.shape(coordinates, out_type=int_type)[0], R, dim]) * ranges, axis=1), int_type)
        @tf.function(experimental_relax_shapes=True, jit_compile=True)
        def A(coordinates):
            r = other_quantics_to_cartesian(coordinates)
            # tf.print(tf.reshape(coordinates, [tf.shape(coordinates, out_type=int_type)[0], R, dim]))
            return B(r)

        shape = 2*tf.ones(d, dtype=int_type)


elif mode == modes[1] or mode == modes[2]:
    A = B
    x_min = args.xmin
    x_max = args.xmax
    dim = args.dim
    R = args.R
    R = tf.cast(R, int_type)
    dim = tf.cast(dim, int_type)

    if mode == modes[1]:
        d = dim
        shape = 2**R*tf.ones(d, dtype=int_type)

    elif mode == modes[2]:
        d = R*dim
        shape = 2*tf.ones(d, dtype=int_type)

if mode != modes[3]:
    initial_pivots = tf.constant([[0, 0, 2559]], dtype=tf.int64)
    # initial_pivots = tf.constant([[16, 500]], dtype=tf.int64)
    extra_pivs = tf.constant([], dtype=tf.int64)



else:
    direc = args.filename[2]
    D1_pivs = []

    # initial_pivots = initial(mode, d, N=1, R=R, shape=shape, type='middle')
    if direc == 'x':
        initial_pivots = tf.constant([int(d-dim)*[0]+[0]+[1]+[0]+[0]+[0]+[0]], dtype=tf.int64)
        for i in range(0, R):
            D1_pivs.append([2 ** i - 1, 2 ** i, 0, 0, 0, 0])
            D1_pivs.append([2 ** i, 2 ** i - 1, 0, 0, 0, 0])

    elif direc == 'y':
        initial_pivots = tf.constant([int(d-dim)*[0]+[0]+[0]+[0]+[1]+[0]+[0]], dtype=tf.int64)
        for i in range(0, R):
            D1_pivs.append([0,0,2 ** i - 1, 2 ** i,0,0])
            D1_pivs.append([0,0,2 ** i, 2 ** i - 1,0,0])

    elif direc == 'z':
        initial_pivots = tf.constant([int(d-dim)*[0]+[0]+[0]+[0]+[0]+[0]+[1]], dtype=tf.int64)
        for i in range(0, R):
            D1_pivs.append([0,0,0,0,2 ** i - 1, 2 ** i])
            D1_pivs.append([0,0,0,0,2 ** i, 2 ** i - 1])


    extra_pivs = cartesian_to_quantics(tf.constant(D1_pivs, dtype=int_type), dim, R)

# initial_pivots = tf.constant([int(d-dim)*[0]+[1]+[0]], dtype=tf.int64)
# initial_pivots = tf.constant([[0]*int(d-dim)+[0]+[0]], dtype=tf.int64)
# initial_pivots = tf.constant([[15, 15]], dtype=tf.int64)
# print(A(initial_pivots))
# exit()
I = []
J = []
for i in range(d):
    I.append(initial_pivots[:, :i])
    J.append(initial_pivots[:, i + 1:])
parts = inital_Tk(A, I, J, shape, mode, x_min, x_max, dim, R, float_type)

err_max_list = [1]
err_int_list = []
mean_chi_list = []
pivots = []
for i in range(d - 1):
    pivots.append(initial_pivots)
# def max_p(parts):
#     a = 0
#     for i in parts:
#         max_i = tf.reduce_max(tf.abs(i))
#         if a < max_i:
#             a = max_i
#     return a

# #delete?
# extra_pivs = tf.constant([[0]*int(d-dim)+[0]+[1]], dtype=tf.int64)
# extra_pivs = tf.constant([int(d-dim)*[1]+[1]+[0], int(d-dim)*[1]+[1]+[1], int(d-dim)*[0]+[0]+[0]], dtype=tf.int64)


for i in range(len(extra_pivs)):
    max_err_i = 0
    for k in (list(range(d-1)) + list(range(d - 1)[::-1][1:-1])):
        k = tf.cast(k, int_type)
        part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k = parts[k], parts[k + 1], I[k], I[k + 1], J[k], \
                                                                     J[k + 1], pivots[k], k
        pivot = extra_pivs[i:i+1]
        pparts, pparts_plus, J_k_update, I_k_plus_update, pivots_new = one_step(A, part_k, part_k_plus, Ik, Ik_plus,
                                                                                Jk,
                                                                                Jk_plus, pivots_k, k, pivot, shape, mode, x_min, x_max, dim, R, float_type)
        add_condJ = tf.logical_not(tf.reduce_any(
            tf.reduce_all(tf.expand_dims(Jk, axis=0) == tf.expand_dims(J_k_update[-1:], axis=1), axis=2),
            axis=1))
        add_condI = tf.logical_not(tf.reduce_any(
            tf.reduce_all(tf.expand_dims(Ik_plus, axis=0) == tf.expand_dims(I_k_plus_update[-1:], axis=1),
                          axis=2),
            axis=1))
        if add_condJ and add_condI:  # just in case
            # print("pivot added")
            parts[k] = pparts
            parts[k + 1] = pparts_plus
            J[k] = J_k_update
            I[k + 1] = I_k_plus_update
            pivots[k] = pivots_new
        else:
            continue

#delete?


step = 0
max_step = 1000
min_step = 5
try:
    while np.mean(err_max_list[-5:]) > threshold and step<max_step:
        max_err_i = 0
        for k in (list(range(d-1)) + list(range(d - 1)[::-1][1:-1])):
            k = tf.cast(k, int_type)
            part_k, part_k_plus, Ik, Ik_plus, Jk, Jk_plus, pivots_k, k = parts[k], parts[k + 1], I[k], I[k + 1], J[k], \
                                                                         J[k + 1], pivots[k], k
            pivot, max_err_pi = find_pivot(A, part_k, part_k_plus, Ik, Ik_plus, Jk_plus, half, shape, k, mode, x_min, x_max, dim, R, float_type, threshold)
            # print(len(pivot))
            # pivot=pivot[0:1]
            # print(max_err_pi)
            if max_err_pi < threshold and step > min_step:
                continue
            if max_err_pi > max_err_i:
                max_err_i = max_err_pi

            piv_left = pivot[:, :k + 1]
            piv_right = pivot[:, k + 1:]

            add_condJ = tf.logical_not(tf.reduce_any(
                tf.reduce_all(tf.expand_dims(Jk, axis=0) == tf.expand_dims(piv_right, axis=1), axis=2),
                axis=1))
            add_condI = tf.logical_not(tf.reduce_any(
                tf.reduce_all(tf.expand_dims(Ik_plus, axis=0) == tf.expand_dims(piv_left, axis=1),
                              axis=2),
                axis=1))
            # print(add_condJ and add_condI)
            # if add_condJ and add_condI:
            #     print("hey")

            if add_condJ and add_condI:  # just in case
                pparts, pparts_plus, J_k_update, I_k_plus_update, pivots_new = one_step(A, part_k, part_k_plus, Ik,
                                                                                        Ik_plus,
                                                                                        Jk,
                                                                                        Jk_plus, pivots_k, k, pivot,
                                                                                        shape, mode, x_min, x_max, dim,
                                                                                        R, float_type)
                # print("pivot added")
                parts[k] = pparts
                parts[k + 1] = pparts_plus
                J[k] = J_k_update
                I[k + 1] = I_k_plus_update
                pivots[k] = pivots_new
            else:
                # print('got that pivot thing')
                continue
            # print(max_err_i)
        err_max_list.append(float(max_err_i))
        print(float(max_err_i))
        step+=1
        print(step)
except KeyboardInterrupt:
    pass
err_max_list = err_max_list[1:]

# pivs = pivots[0]

##
result = tensor_train(I, J, pivots, parts, dim, R, d, x_min, x_max, shape, mode, float_type)

if len(args.filename) > 0:
    with open(args.filename+".pkl", "wb") as f:
        pkl.dump(result, f)
##
bonds = []
for i in pivots:
    bonds.append(len(i))
print("bonds:")
print(bonds)

sum0 = 0.
print("pivots error:")
for i in range(len(pivots)):
    pivs = pivots[i]
    exacts = evaluate(A, pivs, mode, x_min, x_max, dim, R, float_type)
    sum0 += tf.reduce_mean(tf.abs((tf.reshape(evaluate_points(parts, I, pivs),-1) - exacts))/tf.reduce_max(threshold+tf.abs(exacts)))
    # print(pivs)
    # print(other_quantics_to_cartesian(pivs))
    # print(exacts)

# for i in range(len(I)):
#     print(I[i])
# for i in range(len(I)):
#     print(J[i])

print((sum0/len(pivots)).numpy())
print("random error:")
pivs = initial(mode, d, N=1000, R=R, shape=shape, type='random')
exacts = evaluate(A, pivs, mode, x_min, x_max, dim, R, float_type)
print(tf.reduce_mean(tf.abs((tf.reshape(evaluate_points(parts, I, pivs),-1) - exacts))/tf.reduce_max(threshold+tf.abs(exacts))).numpy())

# parts_2 = []
# for i in parts:
#     parts_2.append(i**0.5)
x_min = 0.
x_max = 1.
R = 1
print(evaluate_integral(parts, I, mode, R, x_min, x_max, d).numpy()[0])
# print(evaluate_integral(parts_2, I, mode, R, x_min, x_max, d).numpy()[0])
# print(mps(parts, I))
#

# deserialize the dictionary and print it out


# if mode == modes[0]:
#     import matplotlib
#     # matplotlib.use('WebAgg')
#     import matplotlib.pyplot as plt
#     from eval import plot
#     from matplotlib.widgets import Slider
#     full_full = evaluate_full(parts, I)
#     maxv = 5000#np.max(np.abs(full_full))
#
#     fig, axes = plt.subplots(2)
#     data0 = axes[0].imshow(plot(0), aspect='auto', cmap='turbo', vmin=-maxv, vmax=maxv)
#     data1 = axes[1].imshow(full_full[0, :,:], aspect='auto', cmap='turbo', vmin=-maxv, vmax=maxv)
#     ax_amp = fig.add_axes([0.25, 0.15, 0.65, 0.03])
#     slider = Slider(
#         ax=ax_amp,
#         label="N",
#         valmin=0,
#         valmax=119,
#         valinit=0,
#         orientation="horizontal"
#     )
#
#
#     def update(val):
#         data1.set_data(full_full[int(val), :, :])
#         data0.set_data(plot(int(val)))
#         fig.canvas.draw_idle()
#
#
#     # register the update function with each slider
#     slider.on_changed(update)
#
#
#     axes[0].set_title('exact')
#     axes[1].set_title('tensor train')
#     plt.show()
#     # print('plot')
#     # plt.savefig('tensor_train.pdf')
