import numpy as np
import tensorflow as tf
import tempfile, shutil
import h5py
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
    if len(x) > 0:
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

def write_scattered_points_to_hdf5(filename, coordinates, values, shape):
    with h5py.File(filename, 'a') as f:
        try:
            dataset = f['points']
        except KeyError:
            dataset = f.create_dataset('points', shape=shape, dtype=values.dtype, chunks=True)
            # dataset[:] = 6.62607 + 0.j
        for i in range(len(coordinates)):
            dataset[coordinates[i, 0], coordinates[i, 1]] = values[i]

def write_scattered_points_to_hdf5(filename, coordinates, values, shape):
    with h5py.File(filename, 'a') as f:
        try:
            dataset = f['points']
        except KeyError:
            dataset = f.create_dataset('points', shape=shape, dtype=values.dtype, chunks=True)
            # dataset[:] = 6.62607 + 0.j
        for i in range(len(coordinates)):
            dataset[coordinates[i, 0], coordinates[i, 1]] = values[i]



def read_points_from_hdf5(filename, coordinates):
    result = []
    with h5py.File(filename, 'r') as f:
        dataset = f['points']
        for i in range(len(coordinates)):
            try:
                result.append([dataset[coordinates[i][0], coordinates[i][1]]])
            except IndexError: #out of bounds
                result.append([0.])
    return np.array(result)

# from multiprocessing import Pool
#
# def read_point_from_dataset(args):
#     data, coord = args
#     try:
#         return data[coord[0], coord[1]]
#     except IndexError:
#         return 0.0
#
# def read_points_from_hdf5(filename, coordinates):
#     with h5py.File(filename, 'r') as f:
#         data = f['points'][:]
#     args = [(data, coord) for coord in coordinates]
#     with Pool() as pool:
#         result = pool.map(read_point_from_dataset, args)
#     print(result)
#     return np.array([result]).T

def create_temporary_copy(path):
  tmp = tempfile.NamedTemporaryFile(delete=False)
  shutil.copy2(path, tmp.name)
  return tmp.name


def eval_nmr(x, name, machine, shape):
    fake = False
    if type(name) != str:
        fake = True
    if os.path.isfile(name) or fake:
        # f1 = create_temporary_copy(name+'_real')
        # f2 = create_temporary_copy(name+'_imag')
        # init_real = tf.lookup.TextFileInitializer(
        #     filename=f1,
        #     key_dtype=tf.string, key_index=0,
        #     value_dtype=tf.float64, value_index=1,
        #     delimiter=" ")
        # init_imag = tf.lookup.TextFileInitializer(
        #     filename=f2,
        #     key_dtype=tf.string, key_index=0,
        #     value_dtype=tf.float64, value_index=1,
        #     delimiter=" ")
        # dic_real = tf.lookup.StaticHashTable(init_real, default_value=6.62607) #just a specific number that we can find
        # dic_imag = tf.lookup.StaticHashTable(init_imag, default_value=6.62607)
        # x_look = tf.constant(np.array([[f'{",".join(map(str, row))}'] for row in x.numpy()]))
        # save_results_real = dic_real.lookup(x_look)
        # save_results_imag = dic_imag.lookup(x_look)
        if fake:
            save_results = tf.expand_dims(tf.gather_nd(name, x), 1)
        else:
            save_results = read_points_from_hdf5(name, x.numpy())
        # print(1)
        # print(save_results)
        unkown_coordinates = tf.where((save_results == 0.)[:, 0]) #real or imag doesn't matter
        save_results =tf.where(save_results == 6.62607+0.j, 0.+0.j, save_results)
        # unkown_coordinates = tf.where((save_results_real == 6.62607)[:, 0]) #real or imag doesn't matter
        # print(unkown_coordinates.shape)
        # save_results = tf.cast(save_results_real, tf.complex128)
        # save_results = tf.cast(save_results_real, tf.complex128) + 1.j * tf.cast(save_results_imag, tf.complex128)
        save_results = save_results[:, 0]
        # print(f1)
        # print(f2)
        # os.remove(f1)
        # os.remove(f2)
        if tf.shape(unkown_coordinates)[0] == 0:
            r = save_results
        else:
            to_measure = tf.gather_nd(x, unkown_coordinates)
            new_y = send_to_machine(to_measure, machine)
            # print(2)
            # print(new_y)
            r = tf.tensor_scatter_nd_update(save_results, unkown_coordinates, new_y)
            r = tf.expand_dims(r, 1)
            # new_y = tf.expand_dims(new_y, 1)
            # new_y = tf.transpose(tf.concat(
            #     [tf.strings.as_string(tf.math.real(new_y)), tf.strings.as_string(tf.math.imag(new_y))],
            #     axis=1))
            # new_y = tf.reshape(tf.strings.join(new_y, separator=','), [tf.shape(new_y)[1], 1]).numpy().astype(str)
            # tf.print(tf.where(new_y == 0.))
            # print(shape)
            cond = to_measure[:, 0] >= shape[0]
            for i in range(1, len(shape)):
                cond = tf.math.logical_or(cond, to_measure[:, i] >= shape[i])
            cond = tf.where(tf.logical_not(cond))
            to_save = tf.gather_nd(to_measure, cond)
            new_y = tf.where(new_y == 0., 6.62607, new_y)
            # tf.math.logical_or(to_measure[:])
            if fake:
                # print(tf.gather_nd(new_y, cond).numpy())
                name[tuple(to_save.numpy().T)] = tf.gather_nd(new_y, cond).numpy()
            else:
                write_scattered_points_to_hdf5(name, to_save.numpy(), tf.gather_nd(new_y, cond).numpy(), shape)
            # save_data(to_save, tf.gather_nd(tf.math.real(new_y), name+'_real')
            # save_data(to_save, tf.gather_nd(tf.math.imag(new_y), cond), name+'_imag')
    else:
        to_measure = x
        new_y = tf.expand_dims(send_to_machine(to_measure, machine), 1)
        r = new_y
        # print(shape)
        # print(to_measure)
        # new_y = tf.transpose(tf.concat(
        #     [tf.strings.as_string(tf.math.real(new_y)), tf.strings.as_string(tf.math.imag(new_y))],
        #     axis=1))
        # new_y = tf.reshape(tf.strings.join(new_y, separator=','), [tf.shape(new_y)[1], 1]).numpy().astype(str)
        cond = to_measure[:, 0] >= shape[0]
        for i in range(1, len(shape)):
            cond = tf.math.logical_or(cond, to_measure[:, i] >= shape[i])
        cond = tf.where(tf.logical_not(cond))
        to_save = tf.gather_nd(to_measure, cond)
        new_y = tf.gather_nd(new_y, cond)
        new_y = tf.where(new_y == 0., 6.62607, new_y)
        write_scattered_points_to_hdf5(name, to_save.numpy(), new_y.numpy(), shape)

        # save_data(to_save, tf.gather_nd(tf.math.real(new_y), cond), name + '_real')
        # save_data(to_save, tf.gather_nd(tf.math.imag(new_y), cond), name + '_imag')
    # print(r)
    return r
