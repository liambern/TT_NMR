#!/usr/bin/env python
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensor_train import *

THRESHOLD = 1e-12


def new_random_matrix_pivot(N, M, existing_pivots):
    """ Returns a new random pivot for a matrix of dimension N, M that does not
    share an index with any pivot in a list of existing pivots."""
    pivot = random_matrix_pivot(N, M)
    while any([(pivot == p).any() for p in existing_pivots]):
        pivot = random_matrix_pivot(N, M)
    return pivot


def random_matrix_pivot(N, M):
    """ Returns a random pivot for a matrix of dimension N, M."""
    return np.array(
        [np.random.randint(low=0, high=N),
         np.random.randint(low=0, high=M)])


def add_pivot_to_matrix(tt, pivot):
    """ Adds a pivot to a 2D tensor train."""
    location = 0
    bond_indices = pivot
    # Add new pivot indices
    tt.a_tensors[location].add_left_index(pivot[:location + 1],
                                          bond_indices[0])
    tt.a_tensors[location + 1].add_right_index(pivot[location + 1:],
                                               bond_indices[1])

    # Update right indices of T[location]
    shape = tt.a_tensors[location].data.shape
    dtype = tt.a_tensors[location].data.dtype
    bond_dim = shape[-1]
    shape = (shape[0], shape[1] + 1)
    new_data = np.zeros(shape, dtype=dtype)
    new_data[:, :bond_dim] = tt.a_tensors[location].data[...]
    new_data[:, bond_dim] = tt.evaluator(pivot, location)
    tt.a_tensors[location].data = new_data

    # Update left indices of T[location + 1]
    shape = tt.a_tensors[location + 1].data.shape
    bond_dim = shape[1]
    shape = (shape[0], shape[1] + 1)
    new_data = np.zeros(shape, dtype=dtype)
    new_data[:, :bond_dim] = tt.a_tensors[location + 1].data[...]
    new_data[:, bond_dim] = tt.evaluator(pivot, location + 1)
    tt.a_tensors[location + 1].data = new_data


def matrix_pivot_search_1d(N, M, existing_pivots, evaluator, tt):
    """ Finds a new pivot by performing a search only along the second dimension of a matrix."""
    initial_guess = new_random_matrix_pivot(N, M, existing_pivots)
    guess_n = initial_guess[0]

    real_data = evaluator(initial_guess, 1)
    predicted_data = np.array(
        [tt.evaluate(np.array([guess_n, m], dtype=int)) for m in range(M)])
    best_m = np.argmax(np.abs(predicted_data - real_data))

    error = np.abs(predicted_data[best_m] - real_data[best_m])
    if error > THRESHOLD:
        # print(f"Adding pivot with error {error}.")
        return np.array([guess_n, best_m], dtype=int)
    else:
        return None


def matrix_pivot_search(N, M, data, tt):
    """ Finds a new pivot by performing a full search for a matrix."""
    predicted_data = tt.contract()
    res = np.unravel_index(np.argmax(np.abs(predicted_data - data)),
                           data.shape)
    return np.array(res, dtype=int)


def main():
    N_ITER = 90

    # Read in data
    data = scipy.io.loadmat("p2dnmr.mat")['data']
    N = data.shape[0]
    M = data.shape[1]

    # Plot data
    # plt.figure()
    # plt.imshow(data.real)
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(data.imag)
    # plt.colorbar()
    # plt.show()

    # Matrix cross interpolation
    ############################

    # Construct initial train
    evaluator = lambda p, s=None: eval_from_full(data, p, s)
    initial_pivot = np.zeros(2, dtype=int)
    tt = TensorTrain(evaluator, initial_pivot)

    # Sweep
    pivots = [initial_pivot]
    for iter in range(N_ITER):
        # pivot = new_random_matrix_pivot(N, M, pivots)
        pivot = matrix_pivot_search_1d(N, M, pivots, evaluator, tt)
        # pivot = matrix_pivot_search(N, M, data, tt)
        if pivot is not None:
            pivots.append(pivot)
            add_pivot_to_matrix(tt, pivot)

        data_approx = tt.contract()
        print(f"Iteration {iter} max error: ",
              np.max(np.abs(data_approx.real - data.real)))

    approximation = tt.contract()

    # Plot data
    plt.figure()
    plt.imshow(data.real, aspect='auto', cmap='turbo')
    plt.colorbar()
    plt.figure()
    plt.imshow(approximation.real, aspect='auto', cmap='turbo')
    plt.colorbar()
    plt.show()

    scipy.io.savemat(f"approx_rank{N_ITER}.dat", {"data": approximation})

    # Plot errors
    # plt.figure()
    # plt.imshow(np.abs(approximation.real - data.real) / np.abs(data.real),
    #            aspect='auto',
    #            cmap='turbo',
    #            vmax=10.0)
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    main()
