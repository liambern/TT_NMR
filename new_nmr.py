import numpy as np
import scipy.io
import tensorflow as tf
import os
import pickle as pkl
os.environ["MPLBACKEND"] = 'WebAgg'
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from goodenoughTCI.TCI import TCI, mps
from nmr import *



float_type = tf.complex128
int_type = tf.int64
# x_min = -5.
# x_max = 5.
threshold = 0.
dim = 2
test = True

tci = TCI(float_type=float_type, int_type=int_type, jit_compile=None, NMR=True)


# data = tf.constant(scipy.io.loadmat("gb1_10424.mat")['data'], float_type)
data = tf.constant(scipy.io.loadmat("p2dnmr.mat")['data'], float_type)

shape = data.shape
print(shape)




def machine(x):
    return tf.gather_nd(data, x)

def measure(x):
    return eval_nmr(x, 'test', machine)

tci.cross_interpolation(measure, 'tensor', 2, shape=shape, initial=tf.constant([shape], dtype=int_type)*0, save_mps='test', test=test, threshold=threshold, half=True, relative_error=True)
nmr_mps = tci.load('test').full().numpy()
# plt.imshow(np.abs(nmr_mps-data)/(np.abs(data)+threshold), aspect='auto')
plt.imshow(tf.math.abs(nmr_mps), aspect='auto')
plt.colorbar()
plt.savefig('test.pdf')

def count_lines(filename):
    with open(filename, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

lines = count_lines('test')
print('Used only ' + str(lines / (shape[0]*shape[1]) * 100)[:6], '% of the data!')
print('Max error: ', np.max(np.abs(nmr_mps-data)/(np.abs(data)+threshold)))