# Taken from Lectures on Polytopes, Page 155, Example 6.3
import matplotlib.pyplot as plt
import numpy as np


from itertools import product
from spmlbl.math import orthogonal_complement
from spmlbl.verify import num_feasible_labels
from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.signvector import compute_sign_vectors, compute_potential_circuits


def test_general_position_example(plot=False):

    np.random.seed(15)
    N, D = 6, 2
    W = np.random.uniform(-1, 1, (N, D))

    # Homogenise to get affine dependencies
    W = np.hstack([np.ones((N, 1)), W])
    N, D = W.shape

    if plot:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*W.T)
        for i, w in enumerate(W):
            ax.text(*w, str(i+1))
        plt.show()

    ## Feasible
    W_feas, W_infs = compute_sign_vectors(W)

    ## Feasible for orthogonal complement

    W_orth = orthogonal_complement(W)

    W_orth_feas, W_orth_infs = compute_sign_vectors(W_orth)

    assert len(set(W_feas).intersection(set(W_orth_feas))) == 0

    num_feas = len(W_feas)
    num_dual_feas = len(W_orth_feas)
    infeas_everywhere = set(W_orth_infs).intersection(set(W_infs))
    num_infeas_everywhere = len(infeas_everywhere)
    print('%d feasible sign vectors' % num_feas)
    for feas in W_feas:
        print('\t%s' % feas)
    print('%d dual feasible sign vectors' % num_dual_feas)
    for feas in W_orth_feas:
        print('\t%s' % feas)
    print('%d sign vectors infeasible in both W and orth(W)' % num_infeas_everywhere)
    for inf in infeas_everywhere:
        print('\t%s' % inf)

    # If W is in general position, the relative interior of an orthant is either:
    #  1. Intersected by the subspace defined by $W$
    #  2. Intersected by the subspace defined by $W^\perp$ (the orthogonal complement of W)

    # Check that each orthant is either intersected by W or orth(W)
    assert(num_infeas_everywhere == 0)

    assert (num_feas + num_dual_feas + num_infeas_everywhere) == (2**N)
    print('Check: %d + %d + %d = %d' % (num_feas, num_dual_feas, num_infeas_everywhere, 2**N))

    # Check that the number of feasible and infeasible regions
    # agrees with analytical result (since general position)
    assert (num_feas == num_feasible_labels(N, D))
    assert (num_dual_feas == num_feasible_labels(N, N-D))

    print('The circuits of W are:')
    for v in compute_potential_circuits(W).vectors:
        print('\t%s' % v)

    # print(compute_potential_circuits(W_orth).vectors)
    print('The cocircuits of W are:')
    for v in compute_potential_circuits(W_orth).minimal_support().vectors:
        print('\t%s' % v)


if __name__ == "__main__":

    test_general_position_example(plot=True)
