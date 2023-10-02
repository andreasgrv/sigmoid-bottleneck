import numpy as np
from collections import Counter
from scipy.linalg import null_space

from spmlbl.modules import vander
from spmlbl.verifier import ArgmaxableSubsetVerifier


def test_numerical_issues():
    N = 1000
    C = 5
    D = 2 * C

    np.random.seed(11)

    W = vander(N, D+1)

    other = np.random.uniform(-1, 1, (N, 5))

    W = np.hstack([W, other])
    # W = W / np.linalg.norm(W, axis=0, keepdims=True)

    ver = ArgmaxableSubsetVerifier(W)

    samples = [[0, 2, 5, 10, 13, 18, 23, 25, 32, 45]]

    BOUND = 1e5

    res = ver(samples, lb=-BOUND, ub=BOUND)

    # This example should be feasible
    # but we found it to be infeasible
    assert res[0]['status'] in (3, 4)
    print(res[0]['status'])

    # There would definitely be a problem
    # If we found this region to exist in the nullspace arrangement
    # We therefore search for it there
    Wp = null_space(W.T)

    ver = ArgmaxableSubsetVerifier(Wp)

    res = ver(samples, lb=-BOUND, ub=BOUND)

    # We do not find it there either
    # hence we believe such corner cases exist
    # this empirical result does not invalidate the proof
    # it is just the case that the region is too small to detect.
    assert res[0]['status'] in (3, 4)
    print(res[0]['status'])


if __name__ == "__main__":

    test_numerical_issues()
