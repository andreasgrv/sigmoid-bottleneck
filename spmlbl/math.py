import numpy as np


def orthogonal_complement(W):
    N, D = W.shape
    # Take the nullspace vectors
    basis_cols, r = np.linalg.qr(W, mode='complete')
    W_orth = basis_cols[:, D:]
    return W_orth


def linear_dependence(W):
    # Any number with abs value less than TOL will be mapped to 0
    TOL = 1e-12
    assert len(W.shape) == 2
    N, D = W.shape
    # Try to write one row in terms of the others
    solution, error, r, _ = np.linalg.lstsq(W.T[:, :-1], W.T[:, -1], rcond=None)

    solution[np.abs(solution) < TOL] = 0.
    solution = solution.tolist()
    # We found a y:  y^T W[:-1] = W[-1]
    # The linear dependence is:
    # So:  y^T W[:-1] - W[-1] = 0
    # the coefficient of the right side is a -1.
    solution.append(-1.)
    solution = np.array(solution)

    residuals = solution.dot(W)

    is_dependence = np.allclose(residuals, np.zeros_like(residuals))
    # if len(error) == 0:
    #     is_dependence = False
    # elif np.isclose(error[0], 0.):
    #     is_dependence = True
    return dict(is_dependence=is_dependence,
                solution=solution)
