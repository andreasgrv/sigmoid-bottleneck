import tqdm
import math
import torch
import random
import argparse
import numpy as np
import scipy as sp
import gurobipy as gp
import scipy.optimize as opt


from gurobipy import GRB
from itertools import combinations


LB = -1e5
UB = 1e5


def k_hot_to_dense(sv, N):
    dense = [0,] * N
    for s in sv:
        dense[s] = 1
    return dense


def count_alternating_bits(sequence):
    count = 0
    for i in range(len(sequence) - 1):
        if sequence[i] != sequence[i + 1]:
            count += 1
    # count is the number of positions between bits where there has been
    # a bit change
    return count


def pattern_is_feasible_cyclic(C, pat):
    alt_bits = count_alternating_bits(pat)
    if alt_bits <= 2*(C):
        return True
    return False


def idxs_are_feasible_cyclic(C, N, idxs):
    pat = k_hot_to_dense(idxs, N)
    return pattern_is_feasible_cyclic(C, pat)


def num_feasible_labels(N, D):
    # NOTE: This is also the number of ways to segment a sequence of integers
    # that is sorted in increasing order into D-1 non-empty groups of integers
    # that are still in increasing order, and colour the groups
    # in an alternating binary pattern.

    # See Eqn (40.8) in chapter 40 of MacKay for a derivation
    # http://www.inference.org.uk/mackay/itprnn/ps/482.491.pdf
    # This derivation does not depend on understanding Zaslavski's
    # more general theorem via characteristic polynomials of arrangements.

    # N = the number of hyperplanes (or classes)
    # D = the dimensionality of the class embeddings
    # of the sigmoid classifier
    assert 0 <= D <= N
    return sum(2 * math.comb(N - 1, d)
               for d in range(D))


def num_ksparse(n, k):
    return sum(math.comb(n, i) for i in range(k+1))


def num_necklace_partitions_alt_colour(n, k):
    ss = 0
    for i in range(k+1):
        ss += math.comb(n, 2*i)
    return 2 * ss


# There is a better way to compute this, see MacKay page 490.
def label_coverage(N, D):
    return num_feasible_labels(N, D) / (2 ** N)


def min_num_steps_infeasible(N, D):
    # The minimum number of rules needed to fully define the infeasible set
    # *IF* it is possible to have non overlapping rules
    num_inf = 2**N - num_feasible_labels(N, D)
    # num_inf = num_feasible_labels(N-D, N)
    return num_inf / 2**(N - D)


def min_num_steps_feasible(N, D):
    # The minimum number of rules needed to fully define the feasible set
    # *IF* it is possible to have non overlapping rules
    num_feas = num_feasible_labels(N, D)
    return num_feas / 2**(D)


def compute_pattern(infeasible_face, comb, N):
    # A "*" in a pattern means we do not care about the value at that position
    # since that input does not affect the result
    # pattern = ['*' for i in range(N)]
    pattern = dict()
    for value, c in zip(infeasible_face, comb):
        pattern[int(c)] = int(value)
    return pattern


def pattern_to_str(pattern, n):
    l = ['*'] * n
    for idx, v in pattern.items():
        l[idx] = str(v)
    return ''.join(l)


def expand_pattern(pat):
    # Expands * to 0 or 1 in all possible ways
    # E.g.  1*0* -> [1101, 1100, 1001, 1000]
    l = []
    def _expand_pattern(pat):
        if '*' in pat:
            _expand_pattern(pat.replace('*', '1', 1))
            _expand_pattern(pat.replace('*', '0', 1))
        else:
            l.append(pat)
    _expand_pattern(pat)
    return l


def sample_feasible_regions(W, num_samples):
    # Sample points and classify them
    N, D = W.shape
    # Super dumb idea - evaluate a grid of points
    # x = np.linspace(-.99, .99, 52)
    # grid = np.meshgrid(*[x for i in range(W.shape[0])])
    points = np.random.uniform(-1, 1, (num_samples, D))
    # points = np.hstack([g.reshape(-1, 1) for g in grid])
    acts = (points.dot(W.T) > 0).astype(int)
    feasible = set(map(lambda x: ''.join(map(str, x)), acts))
    return feasible


def compute_infeasible_patterns(W, comb_source='default', num_iters=np.inf):
    # NOTE: WARNING - this method assumes W in general position.
    N, D = W.shape
    patterns = []

    # Choose how to generate combinations of indices to use
    # in order to obtain infeasible regions
    if comb_source == 'default':
        # NOTE: Below is not a good idea because samples are not uniform
        # some indices get picked a lot more if we limit to first S approx samples
        # below is not memory bounded, however
        combs = combinations(range(N), D+1)
    elif comb_source == 'random':
        # TODO: There may be a strategy for picking which order
        # to choose combinations such that we get the most information
        if num_iters != np.inf:
            # all_combs = combinations(range(N), D+1)
            # combs = [next(all_combs) for i in range(100 * num_iters)]
            combs = [np.random.choice(N, D+1, replace=False)
                     for i in range(num_iters)]
        else:
            combs = list(combinations(range(N), D+1))
            random.shuffle(combs)
    elif comb_source == 'sparse':
        assert num_iters != np.inf
        combs = [sparse_subset_selection(W, np.random.randint(N))
                 for i in range(num_iters)]
    else:
        raise ValueError('Unknown comb_source "%s"' % comb_source)

    if num_iters != np.inf:
        desc = 'Querying %d combs' % num_iters
    else:
        desc = 'Querying (%d choose %d) combs' % (N, D+1)

    it = 1
    num_it = num_iters if num_iters != np.inf else int(math.comb(N, D+1))
    pbar = tqdm.tqdm(desc=desc, total=num_it)
    for comb in combs:
        assert W.shape[1] == len(comb) - 1
        # basis_cols, s, basis_rows = np.linalg.svd(W[comb, :])
        basis_cols, r = np.linalg.qr(W[comb, :], mode='complete')
        nullspace_basis = basis_cols[:, -1]

        infeasible_pos = nullspace_basis > 0.
        # Compute positive pattern
        pos_pat = compute_pattern(infeasible_pos, comb, N)
        # A negative pattern
        neg_pat = compute_pattern(~infeasible_pos, comb, N)

        # Postman Pat, Postman Pat..
        patterns.append(pos_pat)
        patterns.append(neg_pat)
        if it >= num_iters:
            pbar.update(1)
            break
        it += 1
        pbar.update(1)
    pbar.close()
    return patterns


def compute_feasible_patterns(W, comb_source='default', num_iters=np.inf):
    # NOTE: WARNING - this method assumes W in general position.
    # TODO: Compute the nullspace of W and use it in W's place.
    # May need to worry about loss of accuracy in methods that are not
    # numerically stable.
    N, D = W.shape
    # basis_cols, s, basis_rows = np.linalg.svd(W)
    basis_cols, r = np.linalg.qr(W, mode='complete')
    # Take the nullspace vectors
    Wn = basis_cols[:, D:]
    return compute_infeasible_patterns(Wn,
                                       comb_source=comb_source,
                                       num_iters=num_iters)


def sparse_subset_selection(W, idx):
    # If a row of W can be written as a conic combination of D other rows
    # the produced dword is guaranteed to be one-hot.
    # Therefore, given an row index *idx* we attempt to find D other rows
    # that when multiplied by positive coefficients approximately give us
    # the row of W with index idx. To this end we use
    # approximate subset selection via sparse non-negative linear regression
    # Sparsity is encouraged using l1 regularisation.
    from sklearn.linear_model import Lasso, LassoLars

    model = LassoLars(alpha=1e-12,
                      fit_intercept=False,
                      normalize=False,
                      positive=True,
                      max_iter=5000)
    # model = Lasso(alpha=1e-10,
    #               fit_intercept=False,
    #               positive=True,
    #               tol=1e-4,
    #               max_iter=1000)

    n, d = W.shape

    X = W[np.arange(n) != idx, :]
    y = W[idx, :]

    model.fit(X.T, y)
    # print(model.coef_)
    pred = model.coef_

    # print('Number of non-neg coefficients: %d' % (pred > 0).sum())
    pred_idxs = np.argsort(pred)[-d:]
    # print(pred.dot(X)[:20])
    # print(y[:20])
    idxs = [idx]
    for i in pred_idxs:
        if i < idx:
            idxs.append(int(i))
        else:
            idxs.append(int(i + 1))
    return idxs


def lp_chebyshev(positive_labels, W, b=None, lb=LB, ub=UB):
    """Linear programme that computes maximum bounded sphere."""

    # Since in most neural networks we compute y = Ax + b
    # This LP is framed in terms of Wx + b <= 0
    assert lb < ub
    assert lb != -np.inf
    assert ub != np.inf
    num_classes, dim = W.shape
    # Feasibility tolerance is a gray zone
    # E.g. if violation is up to TOL may be regarded
    # as feasible. Therefore we set the radius to be
    # EPSILON and Tolerance to .1 * EPSILON,
    # to avoid finding a solution when
    # our reconstruction may give a different answer.
    EPSILON = 1e-8
    TOL = EPSILON * 1e-1

    sign = -np.ones(num_classes)
    # The labels we want to be present need to be positive
    for l in positive_labels:
        sign[l] = 1.

    ww = W * -sign.reshape(-1, 1)

    if b is None:
        bb = np.zeros(num_classes)
    else:
        bb = b * -sign

    # Use Gurobi
    m = gp.Model('m')
    m.setParam('OutputFlag', 0)
    # m.setParam('Presolve', 1)
    m.setParam('Method', 2)
    m.setParam('ScaleFlag', 2)
    # m.setParam('NumericFocus', 3)
    # Radius lower bound is above this
    m.setParam('FeasibilityTol', TOL)
    m.setParam('OptimalityTol', TOL)
    m.params.threads = 1
    # NOTE: We need to specify ub and lb here otherwise there are definitely
    # unbounded regions.
    x = m.addMVar(lb=lb, ub=ub, shape=dim, name='xx')

    r = m.addMVar(lb=EPSILON, shape=1, name='r')
    # Add Chebyshev column
    cheby = np.linalg.norm(ww, axis=1, keepdims=True)

    m.addConstr(ww @ x + bb + cheby @ r <= 0., name='cc')
    m.setObjective(r, GRB.MAXIMIZE)
    m.update()

    try:
        m.optimize()
    except Exception as e:
        print(e)
    # If we find a feasible solution, class is not bounded.
    if m.status == GRB.OPTIMAL:
        var_names = ['xx[%d]' % i for i in range(dim)]
        point = np.array([m.getVarByName(n).X for n in var_names],
                         dtype=np.float64)

        result = dict(is_feasible=True,
                      status=m.status,
                      point=point,
                      radius=getattr(m, 'objval', None))
    else:
        result = dict(is_feasible=False,
                      status=m.status,
                      radius=getattr(m, 'objval', None))
    return result


def search_sgd(positive_labels, W, b=None, steps=1000, init_x=None, verbose=False):
    N, D = W.shape

    # Target label
    yy = np.zeros((N, 1), dtype=np.float64)
    for l in positive_labels:
        yy[l] = 1.

    yyt = torch.tensor(yy)

    if init_x is None:
        xxpt = torch.empty(D, 1, dtype=torch.float64, requires_grad=True)
        xxpt = torch.nn.init.kaiming_normal_(xxpt)
    else:
        xxpt = torch.tensor(init_x.reshape(-1, 1),
                            dtype=torch.float64,
                            requires_grad=True)

    # Train the weights to make the regions feasible
    # Inputs xx are not tied to anything - so learn freely
    if isinstance(W, torch.Tensor):
        Wt = W
    else:
        Wt = torch.tensor(W, dtype=torch.float64, requires_grad=False)
    if b is not None:
        if isinstance(b, torch.Tensor):
            bt = b
        else:
            bt = torch.tensor(b, dtype=torch.float64, requires_grad=False)

    lossfn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([xxpt, Wt], lr=.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=0.5, max_lr=5.)
    # optimizer = torch.optim.SGD([xxpt, Wt], lr=.1)

    # path = list()
    found = False

    if b is not None:
        logits = Wt @ xxpt + bt.view(-1, 1)
    else:
        logits = Wt @ xxpt
    decisions = (logits > 0.).detach().numpy()

    # path.append(tuple(yi for yi in decisions.ravel()))

    if verbose:
        pbar = tqdm.tqdm(range(steps))
    else:
        pbar = range(steps)
    for i in pbar:

        # if i % 100 == 0:
        #     if b is None:
        #         xx = torch.nn.functional.normalize(xxpt, dim=0)
        #     else:
        #         xx = xxpt
        # else:
        xx = xxpt

        if b is not None:
            logits = Wt @ xx + bt.view(-1, 1)
        else:
            logits = Wt @ xx

        decisions = (logits > 0.).detach().numpy()
        # path.append(tuple(yi for yi in decisions.ravel()))

        loss = lossfn(logits, yyt)
        if verbose:
            pbar.set_description('Loss: %.4f, card: %d, mag: %.2f' % (loss, decisions.sum(), torch.linalg.vector_norm(xx).detach().item()))

        if np.all(decisions.astype(int) == yy.astype(int)):
            found = True
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return dict(is_feasible=found,
                point=xx.detach().numpy().ravel(),
                # path=path,
                steps=i
                )
