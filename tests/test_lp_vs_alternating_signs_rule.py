import os
import numpy as np

from itertools import product
from spmlbl.verify import idxs_are_feasible_cyclic, num_feasible_labels
from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.modules import vander


def test_alternating_signs(N=10, C=2):
    W = vander(N, 2*C+1)
    D = W.shape[1]

    v = ArgmaxableSubsetVerifier(W=W, num_processes=int(os.environ['MLBL_NUM_PROC']))
    # Check all possible labels
    samples = [[i for i, j in enumerate(p) if j] for p in product((0, 1), repeat=N)]

    res = v(samples, lb=-1e4, ub=1e4)
    
    num_feasible = 0
    for r in res:
        feasible = r['is_feasible']
        idxs = r['pos_idxs']
        assert feasible == idxs_are_feasible_cyclic(C, N, idxs)
        num_feasible += int(feasible)
    theory_feasible = num_feasible_labels(N, D)
    assert num_feasible == theory_feasible


if __name__ == "__main__":

    test_alternating_signs(N=8, C=1)
    test_alternating_signs(N=8, C=2)
    test_alternating_signs(N=8, C=3)

    test_alternating_signs(N=10, C=1)
    test_alternating_signs(N=10, C=2)
    test_alternating_signs(N=10, C=3)
    test_alternating_signs(N=10, C=4)

    # Below works - but too slow for a test.
    # test_alternating_signs(N=20, C=1)
