import os
import numpy as np

from itertools import combinations
from spmlbl.verify import idxs_are_feasible_cyclic, num_feasible_labels
from spmlbl.verifier import ArgmaxableSubsetVerifier


def test_feasibility_invariant_to_permutations(N=10, C=1):
    # These t_i are not in order + have arbitrary upper and lower lim.
    # yet this will still guarantee that cardinality C labels are feasible
    # since cardinality does not depend on the order of the bits.
    t = np.random.uniform(-10, 10, N)

    D = 2 * C
    d = np.arange(1, (D // 2) + 1)
    d = np.repeat(d, 2)

    # prepare to outer product
    t = t.reshape(-1, 1)
    d = d.reshape(1, -1)

    a = t.dot(d)
    a[:, ::2] = np.cos(a[:, ::2])
    a[:, 1::2] = np.sin(a[:, 1::2])

    W = np.hstack([np.ones(N).reshape(-1, 1), a])
    W[:, 0] = W[:, 0] * np.sqrt(1/N)
    W[:, 1:] = W[:, 1:] * np.sqrt(2/N)


    v = ArgmaxableSubsetVerifier(W=W, num_processes=int(os.environ['MLBL_NUM_PROC']))
    # Check all possible labels
    samples = [list(c) for c in combinations(range(N), C)]

    res = v(samples, lb=-1e4, ub=1e4)
    
    num_feasible = 0
    for r in res:
        feasible = r['is_feasible']
        idxs = r['pos_idxs']
        assert feasible == idxs_are_feasible_cyclic(C, N, idxs)
        num_feasible += int(feasible)


if __name__ == "__main__":

    test_feasibility_invariant_to_permutations(N=100, C=1)
    test_feasibility_invariant_to_permutations(N=100, C=2)

    test_feasibility_invariant_to_permutations(N=20, C=1)
    test_feasibility_invariant_to_permutations(N=20, C=2)
    test_feasibility_invariant_to_permutations(N=20, C=3)
