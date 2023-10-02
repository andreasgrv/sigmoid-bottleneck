import torch
import numpy as np
from spmlbl.modules import gale, vander


def test_equivalence():
    N = 51
    CARD = 25

    W = gale(N, 2*CARD + 1)
    assert np.allclose(W.dot(W.T), W.T.dot(W), atol=1e-07)
    assert np.allclose(W.dot(W.T), np.eye(N), atol=1e-07)

    W = vander(N, 2*CARD + 1)
    assert np.allclose(W.dot(W.T), W.T.dot(W))
    assert np.allclose(W.dot(W.T), np.eye(N))


if __name__ == "__main__":

    test_equivalence()
