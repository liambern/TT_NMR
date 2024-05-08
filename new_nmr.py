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
threshold = 500
dim = 2
test = True
rerr = False
tci = TCI(float_type=float_type, int_type=int_type, jit_compile=None, NMR=True)


data = tf.constant(scipy.io.loadmat("gb1_10424.mat")['data'], float_type)
tf.print(tf.reduce_max(tf.abs(data)))

# data = tf.constant(scipy.io.loadmat("p2dnmr.mat")['data'], float_type)
# data = tf.constant(scipy.io.loadmat("gVp.mat")['data'], float_type)
# data = tf.cast(tf.math.real(scipy.io.loadmat("p2dnmr.mat")['data']), float_type)

#3d
# data = tf.constant(np.load('ser_final_bin_real').astype(float) + 1.j * np.load('ser_final_bin_im').astype(float), float_type)


shape = data.shape
NMR_file = 'test'
print(shape)

#a = [0.01, 0.05, 0.1, 0.2, 0.5]
#for i in a:
#    import pickle; figx = pickle.load(open('test'+str(i)+'_plt0', 'rb'));figx.show();figx = pickle.load(open('test'+str(i)+'_plt1', 'rb'));figx.show();figx = pickle.load(open('test'+str(i)+'_plt2', 'rb'));figx.show()


def machine(x):
    return tf.gather_nd(data, x)

def quantics2normal(mps):
    if dim == 1:
        return tf.reshape(mps.full(), [2 ** mps.R])
    elif dim == 2:
        mid = mps.full()
        mid = tf.transpose(mid, list(range(0, mps.R * mps.dim, 2)) + list(range(1, mps.R * mps.dim, 2)))
        return tf.reshape(mid, [2 ** mps.R, 2 ** mps.R])

dummy = np.ones(shape, dtype=np.cdouble)*6.62607
for i in [0.01, 0.05, 0.1, 0.2, 0.5]:
    NMR_file = 'test' + str(i)
    def measure(x):
        return eval_nmr(x, NMR_file, machine, shape)
    # tci.cross_interpolation(measure, 'tensor', 2, shape=shape, initial=tf.constant([shape], dtype=int_type)*0,
    #                     save_mps=NMR_file, test=test, threshold=threshold, half=True, relative_error=rerr,
    #                         NMR_file=NMR_file, NMR_percentage=i)

    tci.cross_interpolation(measure, 'tensor_quantics', 2, shape=shape, initial=tf.zeros([1, 24], int_type),
                            save_mps=NMR_file, test=test, threshold=threshold, half=False, relative_error=rerr,
                            NMR_file=NMR_file, NMR_percentage=i)

    # nmr_mps = tci.load(NMR_file).full()
    nmr_mps = quantics2normal(tci.load(NMR_file))
    nmr_mps = nmr_mps[:shape[0], :shape[1]]

    new_shape = [-1, shape[-1]] # if more than 2d

    # nmr_mps = nmr_mps.numpy().reshape(new_shape)
    data_np = data.numpy().reshape(new_shape)
    fig,ax = plt.subplots()

    # plt.imshow(tf.math.real(data), aspect='auto')
    cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])

    im = ax.imshow(tci.approx_error(data_np, nmr_mps, threshold, rerr), aspect='auto')

    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('Error '+str(i))
    pkl.dump(fig, open(NMR_file+'_plt0', 'wb'))
    plt.clf()

    fig,ax = plt.subplots()

    # plt.imshow(tf.math.real(data), aspect='auto')
    cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
    im = ax.imshow(np.abs(data_np), aspect='auto')

    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('Exact '+str(i))
    pkl.dump(fig, open(NMR_file+'_plt1', 'wb'))
    plt.clf()

    fig,ax = plt.subplots()
    cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
    im = ax.imshow(np.abs(nmr_mps), aspect='auto')
    ax.set_title('Approx '+str(i))
    fig.colorbar(im, cax=cax, orientation='vertical')
    pkl.dump(fig, open(NMR_file+'_plt2', 'wb'))
    plt.clf()


    #
    lines = tci.count_lines(NMR_file)
    print('Used only ' + str(lines / np.prod(shape) * 100)[:6], '% of the data!')
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


