import torch
import torch.nn.functional as F
import numpy as np

from scipy.special import logit


def torch_interleave_columns(a, b):
    assert a.shape == b.shape
    assert len(a.shape) == 2
    w = torch.vstack([a, b]).T
    return w.reshape(2*a.shape[1], a.shape[0]).T


# If these types of polytopes are useful, the definition below
# should be more numerically stable than exponentiating to the D dimension?
# https://images.math.cnrs.fr/IMG/pdf/1963-gale-neighborly-cyclic-polytopes.pdf
def cyclic_polytope_trig(N, D):
    assert D % 2 == 0
    t = 2 * np.pi * np.arange(N) / N
    d = np.arange(1, (D // 2) + 1)
    d = np.repeat(d, 2)

    # prepare to outer product
    t = t.reshape(-1, 1)
    d = d.reshape(1, -1)

    a = t.dot(d)
    a[:, ::2] = np.cos(a[:, ::2])
    a[:, 1::2] = np.sin(a[:, 1::2])
    return a


def affinise(W):
    N, D = W.shape
    # Make the vector configuration W a point configuration
    # via affinisation
    W = np.hstack([np.ones(N).reshape(-1, 1), W])
    return W


def gale(N, D):
    assert D % 2 == 1
    # Gale parametrisation of a cyclic polytope
    # reserve one of the dimensions for the affinisation
    cp = cyclic_polytope_trig(N, D-1)

    # Add a dimension that is the all ones vector
    W = affinise(cp)
    # Improve conditioning of matrix by scaling the columns such that
    # when if W was full rank we would have:
    # W.T(W) = W.T(W) = I
    # E.g. see https://www.math.kent.edu/~reichel/publications/optvan.pdf
    # For low-rank we have:
    # W.T(W) = I  (e.g. identity dxd matrix)
    W[:, 0] = W[:, 0] * np.sqrt(1/N)
    W[:, 1:] = W[:, 1:] * np.sqrt(2/N)
    return W


def vander(N, D, use_qr=True):
    # This parametrisation is not a good idea numerically
    # but it guarantees the geometrical type of the arrangement
    # is the uniform alternating matroid

    # Choose t such that t^D doesn't get out of hand
    MAX_OUT = 1e4
    # What number when raised to the power D gives us MAX_OUT?
    t_max = np.power(MAX_OUT, 1/D)

    # t = np.linspace(-1.5, 1.5, N)
    t = np.linspace(-t_max, t_max, N)
    W = np.vander(t, D, increasing=True)

    # The matrix above is not well conditioned
    # Use QR to get "better" basis
    if use_qr:
        W, s = np.linalg.qr(W)
    return W
