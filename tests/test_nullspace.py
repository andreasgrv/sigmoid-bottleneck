import numpy as np
from collections import Counter
from scipy.linalg import null_space

from spmlbl.modules import vander
from spmlbl.verifier import ArgmaxableSubsetVerifier


def test_nullspace():

    N = 20
    C = 5
    D = 2 * C

    W = vander(N, D+1)
    print(W.shape)

    Wp = null_space(W.T)
    print(Wp.shape)

    N_SAMPLES = 200000
    samples = np.random.uniform(-1, 1, (N_SAMPLES, Wp.shape[1]))

    out = samples.dot(Wp.T)

    counts = Counter((out > 0).sum(axis=1))
    for k, v in counts.most_common():
        print(k, v)

    for c in range(C):
        assert c not in counts
        assert (N - c) not in counts

    # TBH below may fail - that would be ok
    for c in range(C + 1, N - C):
        assert c in counts


if __name__ == "__main__":

    test_nullspace()
