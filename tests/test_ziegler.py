# Taken from Lectures on Polytopes, Page 155, Example 6.3
import matplotlib.pyplot as plt
import numpy as np


from itertools import product
from spmlbl.math import orthogonal_complement
from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.signvector import compute_sign_vectors, compute_potential_circuits


def test_ziegler_example():
    # This example departs from the situation we generally
    # think about in the paper, since while W is a polytope
    # the points are not in general position.
    # E.g. there are 4 points on a hyperplane in 3d.
    # Thinking in terms of co-vectors and co-circuits is still
    # very useful.

    W = np.array([[1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1],
                  [0, 0, -1]])
    N = W.shape[0]

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(*W.T)
    # for i, w in enumerate(W):
    #     ax.text(*w, str(i+1))
    # plt.show()

    # Homogenise - since the example is affine
    W = np.hstack([np.ones((N, 1)), W])
    N, D = W.shape

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

    # For W not in general position, the relative interior of an orthant is either:
    #  1. Intersected by the subspace defined by $W$
    #  2. Intersected by the subspace defined by $W^\perp$ (the orthogonal complement of W)
    #  3. Not intersected by either, since $W$ and $W^\perp$ are "tangent" to the orthant

    # If W is in general position, 3. cannot occur, so not being intersected by $W$
    # implies intersection by $W^\perp$.
    assert (num_feas + num_dual_feas + num_infeas_everywhere) == (2**N)
    print('Check: %d + %d + %d = %d' % (num_feas, num_dual_feas, num_infeas_everywhere, 2**N))

    print('The circuits of W are:')
    for v in compute_potential_circuits(W).vectors:
        print('\t%s' % v)

    # print(compute_potential_circuits(W_orth).vectors)
    print('The cocircuits of W are:')
    for v in compute_potential_circuits(W_orth).minimal_support().vectors:
        print('\t%s' % v)

if __name__ == "__main__":

    test_ziegler_example()
