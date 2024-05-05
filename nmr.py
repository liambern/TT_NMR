import numpy as np
import tensorflow as tf
import tempfile, shutil

import os
import scipy.io
# def save_data(x, y, name): #makesure x is int
#     with open("file1.txt", "ab") as f:
#         np.savetxt(f, x.numpy(), delimiter=",", fmt='%i')
#         f.close()
#     with open("file2.txt", "ab") as f:
#         np.savetxt(f, y, fmt='%s')
#         f.close()
#     with open('file1.txt', 'r') as file1:
#         lines1 = file1.readlines()
#         file1.close()
#     # Read the second file
#     with open('file2.txt', 'r') as file2:
#         lines2 = file2.readlines()
#         file2.close()
#     combined_lines = [f"{line1.strip()} {line2.strip()}" for line1, line2 in zip(lines1, lines2)]
#     result = '\n'.join(combined_lines)
#     os.remove('file1.txt')
#     os.remove('file2.txt')
#     with open(name, 'a') as f:
#         f.write(result)
#         f.write('\n')
#         f.close()


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


def send_to_machine(x ,machine):
    return machine(x)



def create_temporary_copy(path):
  tmp = tempfile.NamedTemporaryFile(delete=False)
  shutil.copy2(path, tmp.name)
  return tmp.name


def eval_nmr(x, name, machine):
    if os.path.isfile(name+'_real'):
        f1 = create_temporary_copy(name+'_real')
        f2 = create_temporary_copy(name+'_imag')
        init_real = tf.lookup.TextFileInitializer(
            filename=f1,
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.float64, value_index=1,
            delimiter=" ")
        init_imag = tf.lookup.TextFileInitializer(
            filename=f2,
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.float64, value_index=1,
            delimiter=" ")
        dic_real = tf.lookup.StaticHashTable(init_real, default_value=6.62607) #just a specific number that we can find
        dic_imag = tf.lookup.StaticHashTable(init_imag, default_value=6.62607)
        x_look = tf.constant(np.array([[f'{",".join(map(str, row))}'] for row in x.numpy()]))
        save_results_real = dic_real.lookup(x_look)
        save_results_imag = dic_imag.lookup(x_look)
        unkown_coordinates = tf.where((save_results_real == 6.62607)[:, 0]) #real or imag doesn't matter
        print(unkown_coordinates.shape)
        # save_results = tf.cast(save_results_real, tf.complex128)
        save_results = tf.cast(save_results_real, tf.complex128) + 1.j * tf.cast(save_results_imag, tf.complex128)
        save_results = save_results[:, 0]
        if tf.shape(unkown_coordinates)[0] == 0:
            r = save_results
        else:
            to_measure = tf.gather_nd(x, unkown_coordinates)
            new_y = send_to_machine(to_measure, machine)
            r = tf.tensor_scatter_nd_update(save_results, unkown_coordinates, new_y)
            r = tf.expand_dims(r, 1)
            # new_y = tf.expand_dims(new_y, 1)
            # new_y = tf.transpose(tf.concat(
            #     [tf.strings.as_string(tf.math.real(new_y)), tf.strings.as_string(tf.math.imag(new_y))],
            #     axis=1))
            # new_y = tf.reshape(tf.strings.join(new_y, separator=','), [tf.shape(new_y)[1], 1]).numpy().astype(str)
            save_data(to_measure, tf.math.real(new_y), name+'_real')
            save_data(to_measure, tf.math.imag(new_y), name+'_imag')
    else:
        to_measure = x
        new_y = tf.expand_dims(send_to_machine(to_measure, machine), 1)
        print(new_y)
        r = new_y
        # new_y = tf.transpose(tf.concat(
        #     [tf.strings.as_string(tf.math.real(new_y)), tf.strings.as_string(tf.math.imag(new_y))],
        #     axis=1))
        # new_y = tf.reshape(tf.strings.join(new_y, separator=','), [tf.shape(new_y)[1], 1]).numpy().astype(str)
        save_data(to_measure, tf.math.real(new_y), name + '_real')
        save_data(to_measure, tf.math.imag(new_y), name + '_imag')

    return r
