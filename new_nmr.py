import numpy as np
import scipy.io
import tensorflow as tf
import os
import pickle as pkl
os.environ["MPLBACKEND"] = 'WebAgg'
os.environ['TMP']='./tmptest'

import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from goodenoughTCI.TCI import TCI, mps
from nmr import *



float_type = tf.complex128
int_type = tf.int64
# x_min = -5.
# x_max = 5.
threshold = 0.2
dim = 2
test = True
rerr =True
tci = TCI(float_type=float_type, int_type=int_type, jit_compile=None, NMR=True)


data = tf.constant(scipy.io.loadmat("gb1_10424.mat")['data'], float_type)

# data = tf.constant(scipy.io.loadmat("p2dnmr.mat")['data'], float_type)
# data = tf.constant(scipy.io.loadmat("gVp.mat")['data'], float_type)
# data = tf.cast(tf.math.real(scipy.io.loadmat("p2dnmr.mat")['data']), float_type)

#3d
# data = tf.constant(np.load('ser_final_bin_real').astype(float) + 1.j * np.load('ser_final_bin_im').astype(float), float_type)


shape = np.array(data.shape)

# for i in range(dim):
#     data = np.repeat(data, np.ceil(shape.max()/shape)[i], axis=i)
#     data = [:]
# shape = np.array(data.shape)
#a = [0.01, 0.05, 0.1, 0.2, 0.5]
#for i in a:
#    import pickle; figx = pickle.load(open('test'+str(i)+'_plt0', 'rb'));figx.show();figx = pickle.load(open('test'+str(i)+'_plt1', 'rb'));figx.show();figx = pickle.load(open('test'+str(i)+'_plt2', 'rb'));figx.show()




def quantics2normal(mps):
    if dim == 1:
        return tf.reshape(mps.full(), [2 ** mps.R])
    elif dim == 2:
        mid = mps.full()
        mid = tf.transpose(mid, list(range(0, mps.R * mps.dim, 2)) + list(range(1, mps.R * mps.dim, 2)))
        return tf.reshape(mid, [2 ** mps.R, 2 ** mps.R])

import math
def closest_power_of_2(number):
    if number <= 1:
        return 0  # 2^0 is 1, so the closest power of 2 under 1 is 2^0

    exponent = math.floor(math.log2(number))
    return exponent

def closest_power_of_2_max(number):
    if number <= 1:
        return 0  # 2^0 is 1, so the closest power of 2 under 1 is 2^0

    exponent = math.ceil(math.log2(number))
    return exponent
# R = tf.cast(closest_power_of_2_max(tf.reduce_max(shape)), int_type)
# R_extras = []
# for i in shape:
#     R_extras.append(tf.cast(closest_power_of_2(i), int_type).numpy())
# print(R_extras)
# slices = []
# for i in range(len(R_extras)):
#     slices.append(slice(2**max(R_extras)))
# data = data[slices]
# shape = np.array(data.shape)
# R = max(R_extras)
# tile = 2**(np.max(R_extras)-np.array(R_extras))

def machine(x):
    # print(x)
    p_add = float(np.unique(x, axis=0).shape[0]) / np.prod(shape)
    try:
        p_current = pkl.load(open(str(threshold)+'a_test', 'rb'))
        pkl.dump(float(p_current) + p_add, open(str(threshold)+'a_test', 'wb'))
    except FileNotFoundError:
        pkl.dump(p_add, open(str(threshold)+'a_test', 'wb'))
    return tf.gather_nd(data, x)


# def extra_func():
#     extra_pivs = []
#     for i in range(0, max(R_extras)):
#         new_piv = []
#         new_piv.append(2 ** i -1)
#         extra_pivs.append(new_piv)
#     extra_pivs = tci.cartesian_to_quantics(tf.constant(extra_pivs, dtype=int_type), dim, R)
#     return extra_pivs

dummy = np.ones(shape, dtype=np.cdouble)*6.62607
for i in [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9]:
    try:
        os.remove(str(threshold)+'a_test')
    except:
        pass

    NMR_file = str(threshold)+'/test' + str(i)
    print(shape)
    def measure(x):
        return tf.cast(eval_nmr(x, NMR_file, machine, shape, tile), float_type)
    # data2 = np.zeros(shape, dtype=np.cdouble)
    # def measure_fake(x, percent=False):
    #     if percent:
    #         return np.sum(data2 != 0.)
    #     else:
    #         return tf.cast(eval_nmr(x, data2, machine, shape), float_type)
    tile = tf.ones_like([1]*2, dtype=int_type)
    tci.cross_interpolation(measure, 'tensor', 2, shape=shape, initial=tf.constant([shape], dtype=int_type)*0,
                        save_mps=NMR_file, test=test, threshold=threshold, half=True, relative_error=rerr,
                            NMR_file=NMR_file, NMR_percentage=i)



    # last_pivot = tci.cartesian_to_quantics(coordinates, dim, R)
    # tci.cross_interpolation(measure, 'tensor_quantics', 2, R=R, initial=tf.zeros([1, R*dim], int_type),
    #                         save_mps=NMR_file, test=test, threshold=threshold, relative_error=rerr,
    #                         NMR_file=NMR_file, NMR_percentage=i, extra_function=extra_func)

    nmr_mps = tci.load(NMR_file).full()
    # nmr_mps = quantics2normal(tci.load(NMR_file))
    # nmr_mps = nmr_mps[::2, :shape[1]]
    np.save(NMR_file, nmr_mps.numpy())
    new_shape = [-1, shape[-1]] # if more than 2d

    # nmr_mps = nmr_mps.numpy().reshape(new_shape)
    data_np = data.numpy().reshape(new_shape)
    fig,ax = plt.subplots()

    # plt.imshow(tf.math.real(data), aspect='auto')
    cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])

    im = ax.imshow(tci.approx_error(data_np, nmr_mps, threshold, rerr), aspect='auto')

    # fig.colorbar(im, cax=cax, orientation='vertical')
    # ax.set_title('Error '+str(i))
    # pkl.dump(fig, open(NMR_file+'_plt0', 'wb'))
    # plt.savefig(NMR_file+'_plt0'+'.pdf')
    # plt.clf()
    #
    # fig,ax = plt.subplots()
    #
    # # plt.imshow(tf.math.real(data), aspect='auto')
    # cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
    # im = ax.imshow(np.abs(data_np), aspect='auto')
    #
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # ax.set_title('Exact '+str(i))
    # pkl.dump(fig, open(NMR_file+'_plt1', 'wb'))
    # plt.savefig(NMR_file+'_plt1'+'.pdf')
    # plt.clf()

    fig,ax = plt.subplots()
    cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
    im = ax.imshow(np.abs(nmr_mps), aspect='auto')
    ax.set_title('Approx '+str(i))
    fig.colorbar(im, cax=cax, orientation='vertical')
    # pkl.dump(fig, open(NMR_file+'_plt2', 'wb'))
    plt.savefig(NMR_file+'_plt2'+'.pdf')
    plt.clf()


    #
    lines = tci.count_lines(NMR_file)
    p_current = pkl.load(open(str(threshold)+'a_test', 'rb'))
    print('Used only ' + str(float(pkl.load(open(str(threshold)+'a_test', 'rb'))) * 100)[:6], '% of the data!')

    # print('Used only ' + str(np.sum(data2 != 0.) / np.prod(shape) * 100)[:6], '% of the data!')

    # print('Max error: ', np.max(tci.approx_error(data, nmr_mps, threshold, rerr)))

# tci.cross_interpolation(measure, 'tensor_quantics', 2, shape=shape, initial=tf.zeros([1, 24], int_type), save_mps='test', test=test, threshold=threshold, half=False, relative_error=True)
# tci.cross_interpolation(measure, 'tensor', 3, shape=shape, save_mps='test', test=test, threshold=threshold, half=True, relative_error=rerr)

# plt.imshow(np.abs(nmr_mps-data)/(np.abs(data)+threshold), aspect='auto')



# nmr_mps = quantics2normal(tci.load('test'))

# plt.clf()
# # plt.imshow(tci.approx_error(data, nmr_mps, threshold, rerr), aspect='auto')
# plt.imshow(tf.math.real(nmr_mps), aspect='auto')
# plt.colorbar()
# # plt.show()
# plt.savefig('test.pdf')

# plt.savefig('test1.pdf')
# plt.clf()
# plt.imshow(tf.math.abs(data), aspect='auto', cmap='magma')
# # plt.imshow(tf.math.real(data), aspect='auto')
# plt.colorbar()
# plt.show()
# # plt.savefig('test2.pdf')
#
# plt.clf()
# plt.imshow(tf.math.abs(data), aspect='auto', cmap='magma')
# # plt.imshow(tf.math.real(data), aspect='auto')
# # plt.colorbar()
# plt.show()
# # fig = plt.savefig('test3.pdf')


