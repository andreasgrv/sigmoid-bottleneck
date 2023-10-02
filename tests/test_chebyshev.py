import numpy as np

from spmlbl.verify import lp_chebyshev


# Adapted version of numpy code from
# https://mlai.cs.uni-bonn.de/lecturenotes/ml2r-cn-linearprogramming1.pdf
# NOTE: The solver below is much much slower than Gurobi when N and D are large.
# def lp_chebyshev_np(positive_labels, W, b=None, lb=LB, ub=UB):
#     """Linear programme that computes maximum bounded sphere."""
#     num_classes, dim = W.shape
#
#     MIN_RADIUS = 1e-10
#
#     sign = -np.ones(num_classes)
#     # The labels we want to be present need to be positive
#     for l in positive_labels:
#         sign[l] = 1.
#
#     ww = -W * sign.reshape(-1, 1)
#
#     if b is None:
#         bb = np.zeros(num_classes)
#     else:
#         bb = b * sign
#
#     cheby = np.linalg.norm(ww, axis=1, keepdims=True)
#
#     A_ub = np.hstack([cheby, ww])
#     b_ub = -bb
#
#     c = np.zeros(A_ub.shape[1])
#     c[0] = -1
#
#     # Set bounds
#     low_b = (np.ones(dim + 1) * LB).tolist()
#     low_b[0] = MIN_RADIUS
#
#     up_b = (np.ones(dim + 1) * UB).tolist()
#     up_b[0] = None
#
#     bounds = tuple(zip(low_b, up_b))
#
#     result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub,
#                          bounds=bounds,
#                          options=dict(tol=1e-12))
#     if result['success'] == True:
#         point = result['x'][1:]
#
#         result = dict(is_feasible=True,
#                       status=result['status'],
#                       point=point,
#                       radius=result['x'][0])
#     else:
#         result = dict(is_feasible=False,
#                       status=result['status'],
#                       radius=None)
#     return result


def test_cheby():
    # We verify we get the same result as the following tutorial
    # https://mlai.cs.uni-bonn.de/lecturenotes/ml2r-cn-linearprogramming1.pdf

    matW = np.array([[ -0.26 , 0.42 , 0.91 , -0.82],
                     [ 0.97 , -0.91 , 0.42 , -0.57]]).T
    vecT = np.array([5.0 , 1.0 , 8.0 , -1.5])

    # Our LP solves:  Ax + b <= 0. For this case we want Ax <= b, so we need to pass -b
    result = lp_chebyshev([], matW, b=-vecT)
    print(result)

    expected_out = np.array([2.87626447, 3.16518324])
    assert np.allclose(result['point'], expected_out)

    assert np.sum(matW.dot(expected_out) - vecT < 0 ) == len(matW)


if __name__ == "__main__":

    test_cheby()
