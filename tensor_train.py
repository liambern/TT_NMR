""" TT Construction

Trying out sparse construction of TT from complete tensor, using CI
decomposition.
"""
import itertools
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def random_pivot(dim, dim_size):
    """ Returns a random pivot with `dim` dimensions, each having `dim_size` possible values."""
    return np.random.randint(low=0, high=dim_size, size=dim)


def find_local_maximum(pivot, dim_size, evaluator):
    """ Given an initial pivot and a function, find a local maximum."""
    dim = pivot.shape[0]
    value = evaluator(pivot)
    converged = False
    while not converged:
        converged = True
        for i in range(dim):
            pivot_plus = np.copy(pivot)
            if pivot_plus[i] < dim_size:
                pivot_plus[i] += 1
            value_plus = evaluator(pivot_plus)
            pivot_minus = np.copy(pivot)
            if pivot_minus[i] > 1:
                pivot_minus[i] -= 1
            value_minus = evaluator(pivot_minus)
            if value_plus > value:
                value = value_plus
                pivot = pivot_plus
                converged = False
            if value_minus > value:
                value = value_minus
                pivot = pivot_minus
                converged = False
    return pivot


def eval_from_full(tensor, pivot, slice_dim=None):
    """ Evaluates source tensor at either a single point, or at a slice
    along a given dimension that passes through that point.
    """
    if slice_dim is None:
        return tensor[tuple(pivot)]
    res = np.zeros(tensor.shape[slice_dim], dtype=tensor.dtype)
    iterator = np.nditer(res, op_flags=['readwrite'], flags=['f_index'])
    for element in iterator:
        element_pivot = np.copy(pivot)
        element_pivot[slice_dim] = iterator.index
        element[...] = tensor[tuple(element_pivot)]
    return res


def hilbert_tensor(dim, dim_size):
    """ Returns a Hilbert tensor 1 / (u_1 + u_2 + ... + u_L)."""
    u_mesh = np.meshgrid(*([list(np.arange(1, dim_size + 1))] * dim))
    return 1.0 / sum(u_mesh)


def gaussian_tensor(dim, dim_size, sigma=1):
    """ Returns a tensor made from Gaussian filtered noise (trivial at high
    dimensions)"""
    shape = [dim_size] * dim
    return gaussian_filter(np.random.uniform(size=shape), sigma=sigma)


def sweeps(dim, iterations):
    """ Iterator for performing DMRG-like sweeps."""

    def get_cycle():
        return itertools.chain(range(dim - 1), range(dim - 3, 0, -1))

    cycles = [] if iterations == 0 else get_cycle()
    for _ in range(1, iterations):
        cycles = itertools.chain(cycles, get_cycle())
    return cycles


def print_sweep(dim, location):
    """ Prints a visual representation of a stage in a sweep, to display
    progress on the command line."""
    line = ["-"] * dim
    line[location] = "○"
    line[location + 1] = "○"
    print("".join(line))


class ATensor:
    """ An `A` tensor in a tensor train, constructed from pivots."""

    def __init__(self, location, pivot, evaluator):
        self._dim = len(pivot)
        self.location = location
        self.left_pivot_indices = pivot[:location + 1].reshape(location + 1, 1)
        self.right_pivot_indices = pivot[location:].reshape(
            self.dim - location, 1)
        self.left_pivot_origins = np.zeros(1, dtype=int)
        self.right_pivot_origins = np.zeros(1, dtype=int)
        pval_slice = evaluator(pivot, location)
        self._dim_size = pval_slice.shape[0]
        self.data = pval_slice.reshape(self.dim_size, 1, 1)
        if location in (0, self.dim - 1):
            self.data = self.data.reshape(self._dim_size, 1)

    @property
    def dim(self):
        """ Returns the number of physical dimensions."""
        return self._dim

    @property
    def dim_size(self):
        """ Returns the number of possible coordinate values in the `A`
        tensor's physical dimension."""
        return self._dim_size

    @property
    def left_bond_dim(self):
        """ Returns the number of left pivot indices stored in the the `A`
        tensor."""
        return self.left_pivot_indices.shape[1]

    @property
    def right_bond_dim(self):
        """ Returns the number of right pivot indices stored in the the `A`
        tensor."""
        return self.right_pivot_indices.shape[1]

    def add_left_index(self, index, origin_index):
        """ Adds a left pivot index (u1,u2,...,u_{location}) to the `A` tensor."""
        self.left_pivot_indices = np.hstack(
            [self.left_pivot_indices,
             index.reshape(self.location + 1, 1)])
        self.left_pivot_origins = np.append(self.left_pivot_origins,
                                            origin_index)

    def add_right_index(self, index, origin_index):
        """ Adds a right pivot index (u_{location},...,u_d) to the `A` tensor."""
        self.right_pivot_indices = np.hstack([
            self.right_pivot_indices,
            index.reshape(self.dim - self.location, 1)
        ])
        self.right_pivot_origins = np.append(self.right_pivot_origins,
                                             origin_index)


class TensorTrain:
    """ A tensor train representing a `dim`-dimensional tensor."""

    def __init__(self, evaluator, pivot, weight_function=None):
        self.a_tensors = []
        self._dim = len(pivot)
        self.evaluator = evaluator
        if weight_function is None:
            self.weight_function = lambda _: 1.0
        else:
            self.weight_function = weight_function
        for i in range(self.dim):
            self.a_tensors.append(ATensor(i, pivot, evaluator))

    @property
    def dim(self):
        """ Returns the number of physical dimensions."""
        return self._dim

    def contract_direct(self):
        """ Returns the estimated tensor using the explicit pseudoinverse of
        the `P` matrices. This will generally be much too expensive to use
        except in very small test cases."""
        result = self.a_tensors[0].data @ self.inverse_pivot_matrix(0)
        for i in range(1, self.dim - 1):
            result = np.tensordot(result, self.a_tensors[i].data, axes=(i, 1))
            result = np.tensordot(result,
                                  self.inverse_pivot_matrix(i),
                                  axes=(i + 1, 0))
        result = np.tensordot(result,
                              self.a_tensors[self.dim - 1].data,
                              axes=(self.dim - 1, 1))
        return result

    def contract(self):
        """ Returns the estimated tensor. This will generally be much too
        expensive to use except in very small test cases."""
        result = self.a_tensor_times_inverse_pivot_matrix(0)
        for i in range(1, self.dim - 1):
            result = np.tensordot(result,
                                  self.a_tensor_times_inverse_pivot_matrix(i),
                                  axes=(i, 1))
        result = np.tensordot(result,
                              self.a_tensors[self.dim - 1].data,
                              axes=(self.dim - 1, 1))
        return result

    def weighted_sum(self):
        """ Returns the sum over the estimated tensor, weighted by its weight
        function if the latter exists."""
        u = np.arange(self.a_tensors[0].dim_size)
        result = self.a_tensor_times_inverse_pivot_matrix(0)
        result = np.sum(result * self.weight_function(u)[:, None], axis=0)
        for i in range(1, self.dim - 1):
            u = np.arange(self.a_tensors[i].dim_size)
            result = np.tensordot(result,
                                  self.a_tensor_times_inverse_pivot_matrix(i),
                                  axes=(0, 1))
            result = np.sum(result * self.weight_function(u)[:, None], axis=0)
        u = np.arange(self.a_tensors[self.dim - 1].dim_size)
        result = np.tensordot(result,
                              self.a_tensors[self.dim - 1].data,
                              axes=(0, 1))
        result = np.sum(result * self.weight_function(u), axis=0)
        return result

    def evaluate_direct(self, indices):
        """ Returns the estimated value of the tensor at a given set of index
        values, using the explicit pseudoinverse of the `P` matrix."""
        assert len(indices) == self.dim
        result = self.a_tensors[0].data[
            indices[0], :] @ self.inverse_pivot_matrix(0)
        for i in range(1, self.dim - 1):
            result = result @ self.a_tensors[i].data[indices[i], ...]
            result = result @ self.inverse_pivot_matrix(i)
        result = result @ self.a_tensors[self.dim - 1].data[indices[self.dim -
                                                                    1], ...]
        return result

    def evaluate(self, indices):
        """ Returns the estimated value of the tensor at a given set of index
        values."""
        assert len(indices) == self.dim
        result = self.a_tensor_times_inverse_pivot_matrix(0)[indices[0], :]
        for i in range(1, self.dim - 1):
            result = result @ self.a_tensor_times_inverse_pivot_matrix(i)[
                indices[i], ...]
        result = result @ self.a_tensors[self.dim - 1].data[indices[self.dim -
                                                                    1], ...]
        return result

    def piv_from_local(self, location, bond_indices, node_indices):
        """ Obtains the set of indices (u1, u2, ..., u_d) associated with a
        left bond index at T(location-1); u(location) and u(location+1); and a
        right bond index at T(location+1)."""
        (left_pivot_index, right_pivot_index) = bond_indices
        (left_index, right_index) = node_indices
        pivot_indices = np.zeros(self.dim, dtype=int)
        if location > 0:
            pivot_indices[:location] = self.a_tensors[
                location - 1].left_pivot_indices[:, left_pivot_index]
        if location < self.dim - 2:
            pivot_indices[location + 2:] = self.a_tensors[
                location + 2].right_pivot_indices[:, right_pivot_index]
        pivot_indices[location] = left_index
        pivot_indices[location + 1] = right_index
        return pivot_indices

    def eval_from_local_direct(self, location, bond_indices, node_indices):
        """ Obtains the estimated value associated with a left bond index at
        T(location-1); u(location) and u(location+1); and a right bond index at
        T(location+1). Uses the explicit pseudoinverse of the `P` matrices."""
        assert 0 <= location < self.dim - 1
        if location == 0:
            result = self.inverse_pivot_matrix(location) @ self.a_tensors[
                location + 1].data[node_indices[1], :, bond_indices[1]]
            result = self.a_tensors[location].data[node_indices[0], :] @ result
        elif location == self.dim - 2:
            result = self.inverse_pivot_matrix(location) @ self.a_tensors[
                location + 1].data[node_indices[1], :]
            result = self.a_tensors[location].data[node_indices[0],
                                                   bond_indices[0], :] @ result
        else:
            result = self.inverse_pivot_matrix(location) @ self.a_tensors[
                location + 1].data[node_indices[1], :, bond_indices[1]]
            result = self.a_tensors[location].data[node_indices[0],
                                                   bond_indices[0], :] @ result
        return result

    def eval_from_local(self, location, bond_indices, node_indices):
        """ Obtains the estimated value associated with a left bond index at
        T(location-1); u(location) and u(location+1); and a right bond index at
        T(location+1)."""
        assert 0 <= location < self.dim - 1
        if location == 0:
            result = self.a_tensor_times_inverse_pivot_matrix(location)[
                node_indices[0], :]
            result = result @ self.a_tensors[location +
                                             1].data[node_indices[1], :,
                                                     bond_indices[1]]
        elif location == self.dim - 2:
            result = self.a_tensor_times_inverse_pivot_matrix(location)[
                node_indices[0], bond_indices[0], :]
            result = result @ self.a_tensors[location +
                                             1].data[node_indices[1], :]
        else:
            result = self.a_tensor_times_inverse_pivot_matrix(location)[
                node_indices[0], bond_indices[0], :]
            result = result @ self.a_tensors[location +
                                             1].data[node_indices[1], :,
                                                     bond_indices[1]]
        return result

    def pivot_search(self, location, full_search=False, iterations=3):
        """ Returns the pivot with maximal error at a particular location. By
        default this is approximate, because we only search half the
        dimensions."""
        assert 0 <= location < self.dim - 1
        a_loc = self.a_tensors[location]
        a_right = self.a_tensors[location + 1]
        i_dim = 1
        if location > 0:
            i_dim = self.a_tensors[location - 1].left_bond_dim
        j_dim = 1
        if location < self.dim - 2:
            j_dim = self.a_tensors[location + 2].right_bond_dim

        # Guess a random initial pivot
        result = (np.random.randint(i_dim),
                  np.random.randint(j_dim)),\
            (np.random.randint(a_loc.dim_size),
             np.random.randint(a_right.dim_size))

        # Handle full and partial search variants
        if full_search:
            result, max_err_val = self.pivot_search_stage(
                result, location, (i_dim, j_dim), (True, True))
        else:
            search_left = np.random.choice(a=[False, True])
            search_right = not search_left
            for _ in range(iterations):
                result, max_err_val = self.pivot_search_stage(
                    result, location, (i_dim, j_dim),
                    (search_left, search_right))
                search_left ^= True
                search_right ^= True
                result, max_err_val = self.pivot_search_stage(
                    result, location, (i_dim, j_dim),
                    (search_left, search_right))

        return result, max_err_val

    def pivot_search_stage(self, initial_guess, location, bond_dims,
                           search_direction):
        """ Helper function for performing stages of the pivot search."""
        (i_vals, j_vals), (u_1_vals, u_2_vals) = initial_guess
        if search_direction[0]:
            i_vals = np.arange(bond_dims[0], dtype=int)
            u_1_vals = np.arange(self.a_tensors[location].dim_size, dtype=int)
        else:
            i_vals = np.array([i_vals])
            u_1_vals = np.array([u_1_vals])
        if search_direction[1]:
            j_vals = np.arange(bond_dims[1], dtype=int)
            u_2_vals = np.arange(self.a_tensors[location].dim_size, dtype=int)
        else:
            j_vals = np.array([j_vals])
            u_2_vals = np.array([u_2_vals])

        # Perform search
        max_err_val = 0.0
        result = (i_vals[0], j_vals[0]), (u_1_vals[0], u_2_vals[0])
        for i, j, u_1, u_2 in itertools.product(i_vals, j_vals, u_1_vals,
                                                u_2_vals):
            piv = self.piv_from_local(location, (i, j), (u_1, u_2))
            piv_weight = np.prod(self.weight_function(piv))
            approx_val = self.eval_from_local(location, (i, j), (u_1, u_2))
            err_val = np.abs(approx_val - self.evaluator(piv)) * piv_weight
            if err_val >= max_err_val:
                max_err_val = err_val
                result = (i, j), (u_1, u_2)
        return result, max_err_val

    def add_pivot(self, location, bond_indices, node_indices):
        """ Updates the tensor train by the addition of an additional pivot."""
        assert 0 <= location < self.dim - 1, "Invalid location"
        # Find all the indices of the new pivot
        pivot = self.piv_from_local(location, bond_indices, node_indices)

        # Add new pivot indices
        self.a_tensors[location].add_left_index(pivot[:location + 1],
                                                bond_indices[0])
        self.a_tensors[location + 1].add_right_index(pivot[location + 1:],
                                                     bond_indices[1])

        # Update right indices of T[location]
        shape = self.a_tensors[location].data.shape
        bond_dim = shape[-1]
        if location == 0:
            shape = (shape[0], shape[1] + 1)
            new_data = np.zeros(shape)
            new_data[:, :bond_dim] = self.a_tensors[location].data[...]
            new_data[:, bond_dim] = self.evaluator(pivot, location)
        else:
            shape = (shape[0], shape[1], shape[2] + 1)
            new_data = np.zeros(shape)
            new_data[..., :bond_dim] = self.a_tensors[location].data[...]
            for i in range(shape[1]):
                i_pivot = self.piv_from_local(location, (i, bond_indices[1]),
                                              node_indices)
                new_data[:, i, bond_dim] = self.evaluator(i_pivot, location)
        self.a_tensors[location].data = new_data

        # Update left indices of T[location + 1]
        shape = self.a_tensors[location + 1].data.shape
        bond_dim = shape[1]
        if location + 1 == self.dim - 1:
            shape = (shape[0], shape[1] + 1)
            new_data = np.zeros(shape)
            new_data[:, :bond_dim] = self.a_tensors[location + 1].data[...]
            new_data[:, bond_dim] = self.evaluator(pivot, location + 1)
        else:
            shape = (shape[0], shape[1] + 1, shape[2])
            new_data = np.zeros(shape)
            new_data[:, :bond_dim, :] = self.a_tensors[location + 1].data[...]
            for j in range(shape[2]):
                j_pivot = self.piv_from_local(location, (bond_indices[0], j),
                                              node_indices)
                new_data[:, bond_dim,
                         j] = self.evaluator(j_pivot, location + 1)
        self.a_tensors[location + 1].data = new_data

    def inverse_pivot_matrix(self, location):
        """ Returns the inverse of the `P` matrix at a particular location."""
        a_loc = self.a_tensors[location]
        a_right = self.a_tensors[location + 1]
        bond_dim = a_loc.left_bond_dim
        assert bond_dim == a_right.right_bond_dim, "Bond dim mismatch"

        if location in (0, self.dim - 1):
            u_indices = a_loc.left_pivot_indices[location, :].reshape(bond_dim)
            p_indices = np.ix_(u_indices, range(bond_dim))
            p_matrix = a_loc.data[p_indices]
        else:
            left_indices = a_loc.left_pivot_indices[location, :]
            loc_indices = a_loc.left_pivot_origins
            p_matrix = a_loc.data[left_indices, loc_indices, :]

        return np.linalg.pinv(p_matrix)

    def a_tensor_times_inverse_pivot_matrix(self, location):
        """ Returns the product of the `A` tensor at a particular location with
        the inverse of the `P` matrix immediately to its right."""
        a_loc = self.a_tensors[location]
        a_right = self.a_tensors[location + 1]
        bond_dim = a_loc.left_bond_dim
        assert bond_dim == a_right.right_bond_dim, "Bond dim mismatch"

        a_reshaped = a_loc.data.reshape(-1, bond_dim)
        q_q_prime, _r = np.linalg.qr(a_reshaped)
        result = np.zeros(q_q_prime.shape, dtype=a_reshaped.dtype)
        if location == 0:
            u_indices = a_loc.left_pivot_indices[location, :].reshape(bond_dim)
            other_indices = np.ones(q_q_prime.shape[0], bool)
            other_indices[u_indices] = False
            q_mat = q_q_prime[u_indices, :]
            q_prime_mat = q_q_prime[other_indices, :]
            result[u_indices, :] = np.eye(bond_dim)
            result[other_indices, :] = q_prime_mat @ np.linalg.inv(q_mat)
        else:
            u_indices = a_loc.left_pivot_indices[location, :]
            other_indices = np.ones(a_loc.data.shape[0], bool)
            other_indices[u_indices] = False
            left_indices = a_loc.left_pivot_origins
            other_left_indices = np.ones(a_loc.data.shape[1], bool)
            other_left_indices[left_indices] = False

            inds = np.vstack([u_indices, left_indices])
            raveled_indices = np.ravel_multi_index(inds, a_loc.data.shape[:2])
            other_raveled_indices = np.ones(q_q_prime.shape[0], bool)
            other_raveled_indices[raveled_indices] = False
            q_mat = q_q_prime[raveled_indices, :]
            q_prime_mat = q_q_prime[other_raveled_indices, :]
            result[raveled_indices, :] = np.eye(bond_dim)
            result[
                other_raveled_indices, :] = q_prime_mat @ np.linalg.inv(q_mat)

        if location > 0:
            result = result.reshape(a_loc.data.shape)
        return result
