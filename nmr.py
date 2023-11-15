import numpy as np
import tensorflow as tf
import tempfile, shutil

import os
import scipy.io
def save_data(x, y, name): #makesure x is int
    with open("file1.txt", "ab") as f:
        np.savetxt(f, x.numpy(), delimiter=",", fmt='%i')
        f.close()
    with open("file2.txt", "ab") as f:
        np.savetxt(f, y.numpy())
        f.close()
    with open('file1.txt', 'r') as file1:
        lines1 = file1.readlines()
        file1.close()
    # Read the second file
    with open('file2.txt', 'r') as file2:
        lines2 = file2.readlines()
        file2.close()
    combined_lines = [f"{line1.strip()} {line2.strip()}" for line1, line2 in zip(lines1, lines2)]
    result = '\n'.join(combined_lines)
    os.remove('file1.txt')
    os.remove('file2.txt')
    with open(name, 'a') as f:
        f.write(result)
        f.write('\n')
        f.close()

data = scipy.io.loadmat("newnoy/data.mat")['data']
# data = scipy.io.loadmat("newnoy/p2dnmr.mat")['data']


#shape = (46,32,2560)
real_data3D = np.real(data)
imag_data3D = np.imag(data)
shape = imag_data3D.shape
def real_part3D(x):
    return tf.gather_nd(real_data3D, x)

# @tf.function(experimental_relax_shapes=True)
def imag_part3D(x):
    return tf.transpose([tf.gather_nd(imag_data3D, x)])

def send_to_machine(x):
    return imag_part3D(x)



def create_temporary_copy(path):
  tmp = tempfile.NamedTemporaryFile(delete=False)
  shutil.copy2(path, tmp.name)
  return tmp.name


def eval_nmr(x, name):
    if os.path.isfile(name):
        f = create_temporary_copy(name)
        init = tf.lookup.TextFileInitializer(
            filename=f,
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.float64, value_index=1,
            delimiter=" ")
        dic = tf.lookup.StaticHashTable(init, default_value=6.62607) #just a specific number that we can find
        x_look = tf.constant(np.array([[f'{",".join(map(str, row))}'] for row in x.numpy()]))
        save_results = dic.lookup(x_look)
        unkown_coordinates = tf.where((save_results == 6.62607)[:, 0])
        print(unkown_coordinates.shape)
        if tf.shape(unkown_coordinates)[0] == 0:
            r = save_results
        else:
            to_measure = tf.gather_nd(x, unkown_coordinates)
            new_y = send_to_machine(to_measure)
            save_data(to_measure, new_y, name)
            r = tf.tensor_scatter_nd_update(save_results, unkown_coordinates, new_y)
    else:
        to_measure = x
        new_y = send_to_machine(to_measure)
        save_data(to_measure, new_y, name)
        r = new_y
    return r
