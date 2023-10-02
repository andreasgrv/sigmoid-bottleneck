import numpy as np
from itertools import combinations

from spmlbl.verifier import ArgmaxableSubsetVerifier


def braid_hyperplanes_top_k(top_k, W):
    N, D = W.shape

    inv_sel = [i for i in range(N) if i not in top_k]

    Ws = []
    for i in top_k:
        Ws.append(W[i] - W[inv_sel])
    Wb = np.vstack(Ws)
    return Wb


def test_argmaxable():
    N = 200
    D = 4

    # NOTE: Cyclic init taught us that we only need to normalise
    # The first two dimensions of W
    W = np.random.uniform(-1, 1, (N, D))
    W[:, :2] = W[:, :2] / np.linalg.norm(W[:, :2], axis=1, keepdims=True)

    for c in range(N):
        top_k = [c]

        wp = braid_hyperplanes_top_k(top_k, W)

        ver = ArgmaxableSubsetVerifier(wp)
        f = ver([range(len(wp))])[0]['is_feasible']
        assert f


if __name__ == "__main__":

    test_argmaxable()
