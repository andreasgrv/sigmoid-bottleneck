import numpy as np
from itertools import combinations

from spmlbl.modules import vander
from spmlbl.verifier import ArgmaxableSubsetVerifier


# Check if top-k is feasible
def braid_hyperplanes_top_k(top_k, W):
    N, D = W.shape

    inv_sel = [i for i in range(N) if i not in top_k]

    Ws = []
    for i in top_k:
        Ws.append(W[i] - W[inv_sel])
    Wb = np.vstack(Ws)
    return Wb


def test_ranking():
    N = 10
    C = 2
    D = 2 * C

    # NOTE: We drop the constant since otherwise our matrix has a 0 col.
    W = vander(N, D + 1)[:, 1:]
    # W = np.random.uniform(-1, 1, (N, D))
    # W = W / np.linalg.norm(W, axis=1, keepdims=True)


    for cc in range(1, C+1):
        for c in combinations(range(N), cc):
            print('### For Combination %s' % str(c))
            top_k = c

            Wp = braid_hyperplanes_top_k(top_k, W)

            ver = ArgmaxableSubsetVerifier(Wp)
            f = ver([range(len(Wp))])[0]
            assert f['is_feasible']


if __name__ == "__main__":

    test_ranking()
